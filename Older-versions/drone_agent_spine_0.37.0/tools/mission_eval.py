#!/usr/bin/env python3
"""
The end-to-end result: of N flights, how many did what they were told?

    python tools/mission_eval.py logs/*.jsonl
    python tools/mission_eval.py logs/*.jsonl --by-session --jsonl out.jsonl

The planner has a graded evaluation. The flights have been read by hand,
which does not scale past a few and cannot be reported as a number. This
grades them from the logs by declared rule: what the operator asked for is
read off the command line, what happened is read off the skill results, and
`drone_agent/outcome.py` holds the taxonomy so the same log always lands in
the same bucket.

Three outcomes, not two, because the difference matters in both directions.
A flight where `approach` refused because the person walked away is not a
fault -- the cross-check fired and the aircraft stopped -- and counting it
as one would mean the safest behaviour lowers the score. A flight the
arbiter aborted on height is also not a software fault, but the mission did
not happen, and hiding it inside "stopped safely" would flatter the
aircraft. So: completed, stopped safely by a guard that was right, failed.

Nothing here flies. It reads logs.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_agent import __version__
from drone_agent.grade import grade, load_cases
from drone_agent.outcome import Kind, Outcome, Stage, attrition, verdict
from drone_agent.planner import Step


def load(path: str, cases: Optional[Dict[str, Any]] = None
         ) -> Optional[Outcome]:
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    except (OSError, json.JSONDecodeError) as e:
        print(f"  {Path(path).name}: cannot read it ({e})")
        return None
    # The opening record is called `meta` in current logs and `header` in
    # older ones, and `argv` sits at the top level in the first and under
    # `extra` in the second. Both shapes are read rather than one being
    # declared correct, because the older flights are evidence too.
    hdr = next((x for x in rows
                if x.get("type") in ("meta", "header")
                or x.get("drone_agent_version")), {})
    argv = hdr.get("argv") or (hdr.get("extra") or {}).get("argv") or []
    if not argv and hdr.get("skill"):
        # A log written without argv still records the skill it was asked
        # for; the chained stages are then read off the skills that ran.
        argv = ["run_mission.py", hdr["skill"]]
        for flag, key in (("--then-face", "face_target"),
                          ("--then-measure", "measure_range"),
                          ("--then-approach", "approach")):
            if any(x.get("type") == "skill" and x.get("skill") == key
                   for x in rows):
                argv.append(flag)
    stages = [Stage(skill=x.get("skill", "?"), status=x.get("status", "?"),
                    reason=x.get("reason") or "",
                    detail=x.get("detail") or {})
              for x in rows
              if x.get("type") == "skill" and x.get("phase") == "result"]
    airborne = any((x.get("rngfnd_down_raw") or 0) > 0.3
                   for x in rows if x.get("type") == "tick")
    instr = hdr.get("instruction") or ""
    if not instr:
        for x in rows:
            if x.get("type") == "plan":
                instr = x.get("instruction") or ""
                break
    if not instr and "--instruction" in argv:
        instr = argv[argv.index("--instruction") + 1]
    # The plan the aircraft made for itself, if it was given words.
    plan = next((x for x in rows if x.get("type") == "event"
                 and x.get("kind") == "plan"), None)
    planned = [st["skill"] for st in (plan or {}).get("steps") or []]
    out = verdict(Path(path).name, argv, stages, instr, airborne,
                  planned=planned or None)
    if plan is not None and cases:
        # Was the plan right for the instruction? Matched by the
        # instruction's own text, so a flight only counts against a rubric
        # that was written for it.
        case = cases.get((instr or "").strip().lower())
        if case is not None:
            steps_obj = [Step(skill=d["skill"], params=d.get("params") or {})
                         for d in plan.get("steps") or []]
            out.plan_ok, out.plan_why = grade(
                case, steps_obj, bool(plan.get("ok")),
                plan.get("cannot_do") or "")
    # The version the flight actually ran, which is the only honest way to
    # group results: a fix is only evidence from the flights that had it.
    out_version = (hdr.get("drone_agent_version") or hdr.get("version")
                   or (hdr.get("extra") or {}).get("version") or "?")
    setattr(out, "version", out_version)
    setattr(out, "day", Path(path).name[:8])
    setattr(out, "clock", Path(path).name[9:15])
    return out


def table(outcomes: List[Outcome], title: str) -> None:
    n = len(outcomes)
    if not n:
        return
    done = [o for o in outcomes if o.kind is Kind.COMPLETED]
    stop = [o for o in outcomes if o.kind is Kind.STOPPED]
    bad = [o for o in outcomes if o.kind is Kind.FAILED]
    unf = [o for o in outcomes if o.kind is Kind.UNFLOWN]
    print(f"\n{title}  ({n} flights)")
    print(f"  completed what it was told   {len(done):3d}/{n} "
          f"({100.0 * len(done) / n:.0f}%)")
    print(f"  stopped safely by a guard    {len(stop):3d}/{n} "
          f"({100.0 * len(stop) / n:.0f}%)")
    print(f"  failed                       {len(bad):3d}/{n} "
          f"({100.0 * len(bad) / n:.0f}%)")
    if unf:
        print(f"  never left the ground        {len(unf):3d}/{n}")
    flown = n - len(unf)
    if flown:
        print(f"  completed or stopped safely  "
              f"{len(done) + len(stop):3d}/{flown} "
              f"({100.0 * (len(done) + len(stop)) / flown:.0f}% of the "
              f"flights that flew)")
    for kind, group in ((Kind.STOPPED, stop), (Kind.FAILED, bad)):
        if not group:
            continue
        print(f"\n  {kind.value}:")
        for label, k in Counter(o.label for o in group).most_common():
            print(f"    {k:3d}x {label}")
            if "unclassified" in label:
                for o in group:
                    if o.label == label:
                        print(f"          {o.clock}: {o.reason[:76]}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--by-session", action="store_true",
                    help="split by the stack version each flight ran, which "
                         "is the only honest grouping: a fix is evidence "
                         "only from the flights that had it")
    ap.add_argument("--stages", default="rotate_search,face_target,approach",
                    help="the stages to show attrition for")
    ap.add_argument("--eval-set",
                    help="the graded evaluation's rubrics (tools/"
                         "eval_set.jsonl). For a flight started from an "
                         "instruction, also grade the plan the aircraft "
                         "made, so plan errors and flight errors are "
                         "reported apart")
    ap.add_argument("--jsonl", help="write one line per flight here")
    args = ap.parse_args()
    print(f"drone_agent {__version__}")

    cases = None
    if args.eval_set:
        cases = {c.text.strip().lower(): c for c in load_cases(args.eval_set)}
        print(f"  {len(cases)} rubrics from {Path(args.eval_set).name}, for "
              f"grading the plans of instruction-started flights")
    outcomes = [o for o in (load(p, cases) for p in sorted(args.logs))
                if o is not None]
    if not outcomes:
        print("  no readable logs")
        return 1
    table(outcomes, "all flights")

    stages = [s for s in args.stages.split(",") if s]
    att = attrition(outcomes, stages)
    if any(got for _, got, _ in att):
        print(f"\n  where the flights got to:")
        print(f"    {'stage':14} {'reached':>8} {'through':>8}")
        for stage, got, through in att:
            if not got:
                continue
            print(f"    {stage:14} {got:8d} {through:8d}"
                  + (f"   {100.0 * through / got:.0f}% of those that "
                     f"reached it" if got else ""))

    if args.by_session:
        groups: Dict[str, List[Outcome]] = defaultdict(list)
        for o in outcomes:
            groups[str(getattr(o, "version", "?"))].append(o)
        for ver in sorted(groups):
            table(groups[ver], f"stack {ver}")

    instr = [o for o in outcomes if o.instruction]
    if not instr:
        print(f"\n  none of these {len(outcomes)} flights started from an "
              f"instruction in words: all named the skill")
        print(f"  directly. The end-to-end number for the natural-language "
              f"path is unmeasured.")
    else:
        print(f"\n  {len(instr)} started from an instruction in words. Two "
              f"questions, kept apart:")
        print(f"    {'':3} {'plan':>5} {'flight':>8}  instruction")
        for o in instr:
            pl = ("right" if o.plan_ok else "WRONG") if o.plan_ok is not None \
                else "-"
            print(f"    {'OK ' if o.end_to_end else '   '} {pl:>5} "
                  f"{('done' if o.completed else o.label[:8]):>8}  "
                  f"{o.instruction[:46]:46} {o.label if not o.completed else ''}")
            for w in o.plan_why[:1]:
                print(f"          plan: {w[:76]}")
        graded = [o for o in instr if o.plan_ok is not None]
        answered = [o for o in instr if o.end_to_end is not None]
        e2e = [o for o in answered if o.end_to_end]
        skipped = len(instr) - len(answered)
        if answered:
            print(f"\n    words in, right outcome  {len(e2e)}/"
                  f"{len(answered)} "
                  f"({100.0 * len(e2e) / len(answered):.0f}%)")
        if graded:
            print(f"    the plan was right       "
                  f"{sum(bool(o.plan_ok) for o in graded)}/{len(graded)}"
                  f"   (of those with a rubric)")
            flew = [o for o in graded if not o.plan_refused
                    and o.kind is not Kind.UNFLOWN]
            if flew:
                print(f"    the flight then ran it   "
                      f"{sum(o.completed for o in flew)}/{len(flew)}")
            ref = [o for o in graded if o.plan_refused]
            if ref:
                print(f"    correctly refused        "
                      f"{sum(bool(o.end_to_end) for o in ref)}/{len(ref)}"
                      f"   (nothing took off, which is the right outcome)")
        if skipped:
            print(f"    {skipped} left out: planned but never flown, so "
                  f"they answer neither question.")
        print("    A plan error and a flight error are different failures "
              "and are not added together.")

    if args.jsonl:
        with open(args.jsonl, "w") as fh:
            for o in outcomes:
                fh.write(json.dumps({**o.to_log(),
                                     "version": getattr(o, "version", "?"),
                                     "clock": getattr(o, "clock", "")})
                         + "\n")
        print(f"\n  per-flight verdicts written to {args.jsonl}")
    if cases:
        # Which flyable instructions have no flight yet. This is the only
        # part of this tool that says what to do next, and it says it from
        # the logs rather than from memory.
        flown = {(o.instruction or "").strip().lower() for o in outcomes}
        todo = [c for c in cases.values()
                if c.flyable and c.text.strip().lower() not in flown]
        did = [c for c in cases.values()
               if c.flyable and c.text.strip().lower() in flown]
        print(f"\n  of the {len(did) + len(todo)} instructions marked "
              f"flyable, {len(did)} have flown and {len(todo)} have not:")
        for c in sorted(todo, key=lambda c: c.id):
            print(f"    {c.id:20} {c.text[:56]}")
        if todo:
            print(f"\n    python run_mission.py --instruction "
                  f"\"{sorted(todo, key=lambda c: c.id)[0].text}\" \\")
            print(f"        --alt-max 1.2 --airframe 2.2in-cinewhoop "
                  f"--perception yoloe \\")
            print(f"        --camera 1 --intrinsics camera.json --classes "
                  f"person,chair \\")
            print(f"        --target-class person --target-height 1.73")

    unclassified = [o for o in outcomes if "unclassified" in o.label]
    if unclassified:
        print(f"\n  {len(unclassified)} flight(s) did not match any rule in "
              f"the taxonomy. Add a rule in")
        print(f"  drone_agent/outcome.py rather than widening an existing "
              f"one.")
    print("\n  Nothing was sent to any vehicle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
