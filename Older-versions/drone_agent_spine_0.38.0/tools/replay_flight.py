#!/usr/bin/env python3
"""
Replay a recorded flight through the real skills and the real arbiter, with
no aircraft.

Faithful replay — did the decisions reproduce?

    python tools/replay_flight.py logs/<flight>.jsonl --skill rotate_search

Dose-response — how much range error before the outcome changes?

    python tools/replay_flight.py logs/<flight>.jsonl --skill hover \
        --sweep range --values -0.6,-0.3,0,0.3,0.6 --repeats 5

Degrade one thing at a time:

    --sweep range   constant bias on the downward rangefinder, metres
    --sweep pose    gaussian noise on x/y, metres (sigma)
    --sweep latency extra pose age, seconds
    --sweep flow    flow-quality scale factor

Every run is seeded, so --repeats with a sweep separates decision variance
from perception variance: identical input repeated N times tells you how much
of the spread is the stack itself.

Results print as a table and optionally land in a CSV you can put straight
into the evaluation chapter.
"""

from __future__ import annotations

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_agent import __version__
from drone_agent.config import Config
from drone_agent.faults import FAULTS
from drone_agent.interfaces.replay import (Degradation, load_meta,
                                           load_states,
                                           run_skill_on_states)

SWEEPS = {
    "range": lambda v, s: Degradation(f"range{v:+.2f}m", seed=s,
                                      range_bias_m=v),
    "pose": lambda v, s: Degradation(f"pose_sigma{v:.2f}m", seed=s,
                                     pose_noise_m=v),
    "latency": lambda v, s: Degradation(f"latency+{v:.2f}s", seed=s,
                                        pose_extra_age_s=v),
    "flow": lambda v, s: Degradation(f"flow_x{v:.2f}", seed=s,
                                     flow_quality_scale=v),
    "drift": lambda v, s: Degradation(f"drift{v:+.3f}mps", seed=s,
                                      pose_drift_mps=v),
}


def skill_params_from(meta: Dict[str, Any], skill_name: str,
                      override: Dict[str, Any]) -> Dict[str, Any]:
    """Inherit the recorded parameters only when replaying the same skill —
    rotate_search's leg_deg means nothing to hover."""
    params: Dict[str, Any] = {}
    if meta.get("skill") == skill_name:
        params.update(meta.get("skill_params") or {})
    params.update(override)
    return params


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--skill", help="skill to run (default: the one recorded)")
    ap.add_argument("--filter-skill", help="use only ticks from this skill")
    ap.add_argument("--closed-loop", action="store_true",
                    help="advance pose with an approximate model instead of "
                         "replaying it (dynamics are not the real aircraft)")
    ap.add_argument("--sweep", choices=sorted(SWEEPS),
                    help="degradation to sweep")
    ap.add_argument("--values", help="comma-separated sweep values")
    ap.add_argument("--repeats", type=int, default=1,
                    help="runs per value, each with a different seed")
    ap.add_argument("--param", action="append", default=[],
                    metavar="K=V", help="override a skill parameter")
    ap.add_argument("--inject", help="fault to inject mid-skill "
                                     "(see --list-faults)")
    ap.add_argument("--inject-at-tick", type=int, default=20)
    ap.add_argument("--inject-repeat", type=int, default=1,
                    help="fire the fault on this many consecutive ticks. "
                         "Needed to trip the arbiter's escalation: one "
                         "gross violation is blocked and the mission "
                         "continues, a run of them lands the aircraft.")
    ap.add_argument("--inject-sweep", metavar="A:B",
                    help="sweep the injection tick over this inclusive "
                         "range, e.g. 15:25. Shows which ticks the uplink "
                         "rate limiter swallows.")
    ap.add_argument("--inject-not-immediate", action="store_true",
                    help="let the uplink rate limiter swallow the injected "
                         "command — reproduces the hollow test, for showing "
                         "why the immediate flag matters")
    ap.add_argument("--list-faults", action="store_true")
    ap.add_argument("--csv", help="write results here")
    ap.add_argument("--default-config", action="store_true",
                    help="use package defaults instead of the config "
                         "recorded in the log (changes skill timing)")
    ap.add_argument("--alt-max", type=float,
                    help="override the envelope ceiling for this replay")
    ap.add_argument("--fence", type=float,
                    help="override the geofence radius for this replay")
    ap.add_argument("--max-step", type=float,
                    help="override the step limit for this replay")
    ap.add_argument("--set", action="append", default=[],
                    metavar="SECTION.KEY=VALUE", dest="config_set",
                    help="override one config field for this replay, e.g. "
                         "skills.yaw_confirm_samples=2. This is the "
                         "what-if: the recorded sensor stream, flown "
                         "through a changed decision rule.")
    ap.add_argument("--version", action="version",
                    version=f"drone_agent {__version__}")
    args = ap.parse_args()

    if args.list_faults:
        for k, v in sorted(FAULTS.items()):
            print(f"  {k:<12} {v}")
        return 0

    meta = load_meta(args.log)
    if meta.get("skill") == "plan":
        # A flight from an instruction: the skills are the plan's steps, in
        # the log as skill-start records. Replay one step at a time.
        steps = [json.loads(l) for l in open(args.log) if '"skill_start"' in l
                 or ('"type": "skill"' in l and '"phase": "start"' in l)]
        if not steps:
            ap.error("this instruction flight never started a skill")
        pick = next((x for x in steps if x.get("skill") == args.skill),
                    steps[0])
        if len(steps) > 1:
            print(f"  this plan had {len(steps)} steps "
                  f"({', '.join(x['skill'] for x in steps)}); replaying "
                  f"{pick['skill']}. Use --skill for another.")
        meta = dict(meta, skill=pick["skill"],
                    skill_params={k: v for k, v in
                                  (pick.get("params") or {}).items()
                                  if v is not None})
    skill_name = args.skill or meta.get("skill")
    if not skill_name:
        ap.error("no skill recorded in the log; pass --skill")

    override: Dict[str, Any] = {}
    for kv in args.param:
        k, _, v = kv.partition("=")
        try:
            override[k] = float(v)
        except ValueError:
            override[k] = v
    params = skill_params_from(meta, skill_name, override)

    # Fly the replay under the configuration the flight was flown with, not
    # today's defaults — skill timing lives in the config.
    if meta.get("config") and not args.default_config:
        cfg = Config.from_dict(meta["config"])
        src = "from the log"
    else:
        cfg = Config()
        src = "package defaults"
    if args.alt_max is not None:
        cfg.envelope.alt_max_m = args.alt_max
    if args.fence is not None:
        cfg.envelope.fence_radius_m = args.fence
    if args.max_step is not None:
        cfg.envelope.max_step_m = args.max_step
    for item in args.config_set:
        key, sep, raw = item.partition("=")
        section, dot, field_name = key.partition(".")
        target = getattr(cfg, section, None) if dot else None
        if not sep or target is None or not hasattr(target, field_name):
            ap.error(f"--set {item!r}: expected SECTION.KEY=VALUE naming an "
                     f"existing config field")
        old_val = getattr(target, field_name)
        # Keep the field's own type: "2" for an int field must not become
        # 2.0, and a bool field must not become the string "false".
        if isinstance(old_val, bool):
            val: Any = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(old_val, int):
            val = int(float(raw))
        elif isinstance(old_val, float):
            val = float(raw)
        else:
            val = raw
        setattr(target, field_name, val)
        src += f", {key}={val!r} (was {old_val!r})"

    base_states = load_states(args.log, skill=args.filter_skill)
    if not base_states:
        print(f"no tick records in {args.log}"
              f"{' for skill ' + args.filter_skill if args.filter_skill else ''}")
        return 1

    print(f"drone_agent {__version__}")
    print(f"{args.log}")
    print(f"  {len(base_states)} ticks, replaying skill {skill_name!r} "
          f"with {params}")
    print(f"  mode: {'closed loop (approximate dynamics)' if args.closed_loop else 'open loop (pose from the log)'}")
    print(f"  config: {src}")
    print()

    degradations: List[Degradation] = []
    if args.sweep:
        if not args.values:
            ap.error("--sweep needs --values")
        vals = [float(v) for v in args.values.split(",")]
        for v in vals:
            for seed in range(args.repeats):
                degradations.append(SWEEPS[args.sweep](v, seed))
    else:
        for seed in range(args.repeats):
            degradations.append(Degradation(name="none", seed=seed))

    if args.inject and args.inject not in FAULTS:
        ap.error(f"unknown fault {args.inject!r}; "
                 f"have {sorted(FAULTS)}")

    if args.inject_sweep:
        a, _, b = args.inject_sweep.partition(":")
        inject_ticks = list(range(int(a), int(b) + 1))
        if not args.inject:
            ap.error("--inject-sweep needs --inject")
    else:
        inject_ticks = [args.inject_at_tick]

    rows: List[Dict[str, Any]] = []
    for deg in degradations:
        states = load_states(args.log, skill=args.filter_skill,
                             degradation=deg)
        for tick in inject_ticks:
            outcome = run_skill_on_states(
                skill_name, cfg, states, skill_params=params,
                closed_loop=args.closed_loop, degradation_name=deg.name,
                inject=args.inject, inject_at_tick=tick,
                inject_immediate=not args.inject_not_immediate,
                inject_repeat=args.inject_repeat)
            rows.append(outcome.to_row())

    cols = ["degradation", "status", "ticks", "interventions", "blocked"]
    if args.inject:
        cols += ["inject_tick", "inject_actions", "inject_transmitted"]
    cols += ["reason"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows))
              for c in cols}
    widths["reason"] = min(widths["reason"], 44)
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("-" * (sum(widths.values()) + 2 * (len(cols) - 1)))
    for r in rows:
        print("  ".join(str(r.get(c, ""))[:widths[c]].ljust(widths[c])
                        for c in cols))

    # Group by degradation value when repeating, so the spread is visible.
    if args.repeats > 1:
        print("\nby condition:")
        seen: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            seen.setdefault(r["degradation"], []).append(r)
        for name, group in seen.items():
            ok = sum(1 for g in group if g["status"] == "succeeded")
            iv = sum(g["interventions"] for g in group) / len(group)
            print(f"  {name:<22} {ok}/{len(group)} succeeded, "
                  f"{iv:.1f} interventions/run")

    if args.csv:
        keys: List[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
