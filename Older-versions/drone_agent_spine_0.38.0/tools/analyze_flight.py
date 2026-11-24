#!/usr/bin/env python3
"""
Load a flight log and print what happened. No pandas required; if pandas is
installed you also get a DataFrame dumped to CSV with --csv.

    python tools/analyze_flight.py logs/<flight>.jsonl
    python tools/analyze_flight.py logs/*.jsonl --compare
    python tools/analyze_flight.py logs/<flight>.jsonl --csv out.csv

What it reports, and why each number is here
--------------------------------------------
  interventions    ticks where setpoint_requested != setpoint_applied, by
                   cause. THE safety metric: the arbiter's whole job,
                   counted.
  skills           every skill's status, reason, duration, metrics. The
                   failure taxonomy starts here.
  link             uplink rate against the budget, RSSI/LQ if present,
                   heartbeat gaps.
  estimate         pose staleness distribution, flow quality, how often the
                   rangefinder was distrusted and for how long.
  timing           tick rate actually achieved, and logging latency
                   (t_mono - t_sample_mono).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_agent import __version__
from drone_agent.interfaces.replay import load_records


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _fmt(v: Optional[float], nd: int = 2, unit: str = "") -> str:
    return "n/a" if v is None else f"{v:.{nd}f}{unit}"


def _stats(vals: List[float]) -> str:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return "n/a"
    vals_sorted = sorted(vals)
    p95 = vals_sorted[min(len(vals) - 1, int(0.95 * len(vals)))]
    return (f"min {min(vals):.3f}  median {st.median(vals):.3f}  "
            f"p95 {p95:.3f}  max {max(vals):.3f}")


def _setpoint_modified(req: Dict[str, Any], app: Dict[str, Any]) -> bool:
    """Did the arbiter actually change the command? A breach or watchdog
    annotation does not, so asking whether it was transmitted is meaningless
    and the rate-limiter note does not apply to it."""
    keys = ("kind", "x", "y", "z", "yaw_deg", "heading_deg", "rate_dps")
    return any(req.get(k) != app.get(k) for k in keys)


def _was_transmitted(ticks: List[Dict[str, Any]], idx: int) -> bool:
    """Did the applied setpoint at this tick actually go out on the wire?

    The arbiter rate-limits position targets, so a clamped command can be
    computed, logged, and then dropped. sent_position incrementing across the
    tick is the only evidence it was transmitted.
    """
    if _get(ticks[idx], "arb.blocked"):
        return False
    kind = (ticks[idx].get("setpoint_applied") or {}).get("kind")
    key = {"position": "sent_position", "yaw": "sent_yaw"}.get(kind)
    if key is None or idx == 0:
        return False
    before = _get(ticks[idx - 1], f"arb.counters.{key}")
    after = _get(ticks[idx], f"arb.counters.{key}")
    if before is None or after is None:
        return False
    return after > before


# ──────────────────────────── sections ───────────────────────────

def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _linfit(xs: List[float], ys: List[float]) -> Optional[tuple]:
    """Least-squares slope, intercept and r, or None if x does not vary."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    slope = sxy / sxx
    return slope, my - slope * mx, sxy / math.sqrt(sxx * syy)


def _smooth(ts: List[float], xs: List[float], win_s: float) -> List[float]:
    """Centred moving average over a time window (samples are not evenly
    spaced, so the window is in seconds, not in samples)."""
    half = win_s / 2.0
    out: List[float] = []
    lo = hi = 0
    acc = 0.0
    n = len(ts)
    for i in range(n):
        while hi < n and ts[hi] <= ts[i] + half:
            acc += xs[hi]
            hi += 1
        while ts[lo] < ts[i] - half:
            acc -= xs[lo]
            lo += 1
        out.append(acc / max(1, hi - lo))
    return out


def altitude_hold(ticks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    How well the aircraft held its commanded height, in BOTH frames.

    There are two altitudes in every tick and they are not the same number.
    The EKF's is what the autopilot controls to and what its fence judges;
    the downward rangefinder's is where the aircraft physically is. On an
    airframe whose EKF height comes from that same rangefinder they should
    agree, and how far they do not is a measurement of the estimator, made
    for free on every flight.

    The EKF's vertical velocity is checked against its own altitude for the
    same reason. Integrated over the flight it must reproduce the altitude
    change; if it predicts metres of descent that did not happen, the
    velocity estimate is biased, and a position controller that trusts it
    holds the aircraft off its target to compensate.
    """
    rows = []
    for t in ticks:
        pose = t.get("pose") or {}
        if pose.get("z") is None:
            continue
        rows.append((t.get("t_sample_mono", t.get("t_mono")), -pose["z"],
                     t.get("rngfnd_down_raw", t.get("rngfnd_down")),
                     pose.get("vz")))
    if len(rows) < 10:
        return None

    # What was actually commanded: the z of the position setpoints that
    # went through the arbiter. The origin is only the fallback, because a
    # skill is free to hold somewhere else.
    cmd = [-s["z"] for s in (t.get("setpoint_applied") or {} for t in ticks)
           if s.get("kind") == "position" and s.get("z") is not None]
    if cmd:
        target = st.median(cmd)
    else:
        origin = next((t.get("origin") for t in ticks if t.get("origin")),
                      None)
        if not origin:
            return None
        target = -origin[2]

    ts = [r[0] for r in rows]
    ekf = [r[1] for r in rows]
    out: Dict[str, Any] = {
        "target_m": target, "duration_s": ts[-1] - ts[0],
        "ekf_mean": sum(ekf) / len(ekf), "ekf_min": min(ekf),
        "ekf_max": max(ekf), "ekf_range": max(ekf) - min(ekf),
        "ekf_above_mean": sum(ekf) / len(ekf) - target,
        "ekf_above_max": max(ekf) - target,
    }

    pairs = [(e, r) for _, e, r, _ in rows if r is not None]
    if len(pairs) >= 10:
        rf = [p[1] for p in pairs]
        gap = [r - e for e, r in pairs]
        out.update({
            "rf_mean": sum(rf) / len(rf), "rf_min": min(rf), "rf_max": max(rf),
            "rf_range": max(rf) - min(rf),
            "rf_above_mean": sum(rf) / len(rf) - target,
            "rf_above_max": max(rf) - target,
            "gap_mean": sum(gap) / len(gap), "gap_min": min(gap),
            "gap_max": max(gap),
        })
        fit = _linfit([p[0] for p in pairs], rf)
        if fit:
            out["rf_per_ekf"], _, out["rf_ekf_r"] = fit

    vzs = [(t, v) for t, _, _, v in rows if v is not None]
    if len(vzs) >= 10 and out["duration_s"] > 1.0:
        # vz is NED, positive DOWN, so its integral is a descent.
        integ = sum(0.5 * (v0 + v1) * (t1 - t0)
                    for (t0, v0), (t1, v1) in zip(vzs, vzs[1:]))
        ekf_change = ekf[-1] - ekf[0]
        out.update({
            "vz_mean": sum(v for _, v in vzs) / len(vzs),
            "vz_implied_change_m": -integ,
            "ekf_change_m": ekf_change,
            # what the velocity estimate says, minus what the altitude did
            "vz_bias_down_mps": (integ + ekf_change) / out["duration_s"],
        })

    # WHY the two altitudes differ. In a height hold the aircraft's real
    # vertical velocity averages to nothing over a few seconds, so what is
    # left in a smoothed EKF vz is the estimator's own error, b. Two loops
    # then each leave a gap proportional to it:
    #
    #   the position controller holds the EKF altitude b/P above its target,
    #   because that is where its demanded descent cancels the descent it
    #   believes in;
    #   the estimator holds the EKF altitude b/K below the rangefinder,
    #   because that is where its correction cancels the same error.
    #
    # If both fits are tight, the "rangefinder moves ~2 m per metre of EKF"
    # figure above is simply 1 + P/K -- a ratio of two loop gains -- and NOT
    # an EKF that sees half of every height change. The difference matters:
    # it says what the EKF will read in a real climb, which is what an
    # altitude fence is judged on.
    full = [(t, e, r, v) for t, e, r, v in rows
            if r is not None and v is not None]
    if len(full) >= 40 and out["duration_s"] >= 15.0:
        tt = [f[0] for f in full]
        vs = _smooth(tt, [f[3] for f in full], 4.0)
        ctrl = _linfit(vs, _smooth(tt, [f[1] - target for f in full], 4.0))
        est = _linfit(vs, _smooth(tt, [f[2] - f[1] for f in full], 4.0))
        if ctrl and est:
            out["bias_model"] = {
                "ctrl_s": ctrl[0], "ctrl_r": ctrl[2],
                "est_s": est[0], "est_r": est[2], "static_gap_m": est[1],
                "vz_min": min(vs), "vz_max": max(vs),
                "fits": (ctrl[2] >= 0.8 and est[2] >= 0.7
                         and ctrl[0] > 0.2 and est[0] > 0.2),
            }
    return out


def heading_steps(ticks: List[Dict[str, Any]],
                  events: Optional[List[Dict[str, Any]]] = None
                  ) -> List[Dict[str, Any]]:
    """
    Heading discontinuities: one sample-to-sample step far larger than the
    rotation in progress produces (83.4 -> 106.6 deg between two samples of a
    17 deg/s slew).

    What these are, on this airframe, is known. Once the autopilot's
    STATUSTEXT was being logged (0.15), every one of them coincided with

        EKF3 IMU0 MAG0 ground mag anomaly, yaw re-aligned

    The EKF's heading and the compass drift apart by 12-22 deg over one
    indoor rotation, and when the aircraft comes back round to the heading
    it took off on, the EKF snaps its yaw to the compass. That is why it was
    always the last leg of a 360 deg sweep. From that sample on, headings
    are in the re-aligned frame.

    An earlier version of this function called the step "a sample no
    rotation could have produced", dropped it, and reported the aircraft as
    "12.5 deg short" from the samples either side. The first part was true
    and the conclusion was not: the frame moved, not one sample.
    """
    samp: List[tuple] = []        # (first tick index, tick time, yaw)
    for i, t in enumerate(ticks):
        y = (t.get("pose") or {}).get("yaw_deg")
        if y is None:
            continue
        if not samp or y != samp[-1][2]:
            samp.append((i, t.get("t_mono", t.get("t_sample_mono")), y))
    out = []
    for k in range(4, len(samp)):
        a_, b_ = samp[k - 1], samp[k]
        d = _wrap(b_[2] - a_[2])
        dt = (b_[1] - a_[1]) if (a_[1] is not None and b_[1] is not None) \
            else 0.0
        prev = sorted(abs(_wrap(samp[j][2] - samp[j - 1][2]))
                      for j in range(k - 3, k))
        typical = prev[1]
        # Far beyond what the rotation is producing, and not simply the
        # first sample after a gap in the stream.
        if abs(d) >= 12.0 and abs(d) >= 2.5 * max(typical, 2.0) and dt <= 0.6:
            nxt = samp[k + 1] if k + 1 < len(samp) else None
            out.append({
                "tick": b_[0], "t_mono": b_[1],
                "before_deg": a_[2], "yaw_deg": b_[2],
                "after_deg": nxt[2] if nxt else None,
                "step_deg": d, "typical_step_deg": typical,
                "statustext": None,
            })
    for e in events or []:
        txt = e.get("text") or ""
        if e.get("kind") != "statustext" or "yaw" not in txt.lower():
            continue
        near = [st_ for st_ in out if st_["t_mono"] is not None
                and abs(st_["t_mono"] - e.get("t_mono", 1e18)) <= 1.0]
        if near:
            min(near, key=lambda st_: abs(st_["t_mono"] - e["t_mono"])
                )["statustext"] = txt
    return out


def yaw_legs(ticks: List[Dict[str, Any]],
             events: Optional[List[Dict[str, Any]]] = None
             ) -> Optional[Dict[str, Any]]:
    """
    Per-leg heading tracking for any skill that commands yaw legs.

    Every figure here comes from the attitude stream at a few hertz, so at
    20 deg/s the heading moves about 5 deg between samples: an "overshoot"
    of a few degrees on a single leg is within sampling. A pattern across
    legs or flights is not.
    """
    cmds = [i for i, t in enumerate(ticks)
            if (t.get("setpoint_requested") or {}).get("kind") == "yaw"
            and (t["setpoint_requested"].get("heading_deg") is not None)
            and (t.get("pose") or {}).get("yaw_deg") is not None]
    if not cmds:
        return None

    # A leg that contains an EKF yaw re-alignment is measured in two
    # heading frames. Its figures are kept, marked, and left out of the
    # pooled control statistics.
    steps = heading_steps(ticks, events)

    legs = []
    for n, i in enumerate(cmds):
        j = cmds[n + 1] if n + 1 < len(cmds) else len(ticks)
        seg = [ticks[k] for k in range(i, j)
               if (ticks[k].get("pose") or {}).get("yaw_deg") is not None]
        if len(seg) < 2:
            continue
        tgt = _wrap(ticks[i]["setpoint_requested"]["heading_deg"])
        t0 = seg[0].get("t_sample_mono", seg[0]["t_mono"])
        err = [_wrap(s["pose"]["yaw_deg"] - tgt) for s in seg]
        turn = _wrap(tgt - seg[0]["pose"]["yaw_deg"])
        sign = 1.0 if turn >= 0 else -1.0
        hit = next((k for k, e in enumerate(err) if abs(e) <= 3.0), None)
        slew = (seg[hit].get("t_sample_mono", seg[hit]["t_mono"]) - t0
                if hit is not None else None)

        # The moment the skill declared the leg complete: the first position
        # setpoint after the yaw command. The heading sample in force on the
        # tick before it is the one that satisfied the tolerance.
        done = next((k for k in range(i + 1, j)
                     if (ticks[k].get("setpoint_requested") or {})
                     .get("kind") == "position"), None)
        short = None
        if done is not None and done - 1 >= i:
            k = done - 1
            if (ticks[k].get("pose") or {}).get("yaw_deg") is not None:
                short = abs(_wrap(tgt - ticks[k]["pose"]["yaw_deg"]))
        reset = next((st_ for st_ in steps if i <= st_["tick"] < j), None)
        legs.append({
            "leg": n + 1, "target_deg": tgt, "turn_deg": turn,
            "slew_s": slew,
            "rate_dps": (abs(turn) / slew if slew else None),
            # past the target in the direction of travel
            "overshoot_deg": max(0.0, max(sign * e for e in err)),
            "end_err_deg": err[-1],
            "leg_s": seg[-1].get("t_sample_mono", seg[-1]["t_mono"]) - t0,
            # how far from the target the heading read when the skill
            # called the leg done
            "short_deg": short,
            # an EKF yaw re-alignment inside this leg, if there was one
            "yaw_reset": reset,
        })
    if not legs:
        return None
    first = next(t["pose"]["yaw_deg"] for t in ticks
                 if (t.get("pose") or {}).get("yaw_deg") is not None)
    last = next(t["pose"]["yaw_deg"] for t in reversed(ticks)
                if (t.get("pose") or {}).get("yaw_deg") is not None)
    total = sum(leg["turn_deg"] for leg in legs)
    clean = [leg for leg in legs if not leg.get("yaw_reset")] or legs
    abs_end = [abs(leg["end_err_deg"]) for leg in clean]
    rates = [leg["rate_dps"] for leg in clean if leg["rate_dps"]]
    return {
        "legs": legs, "n": len(legs), "start_deg": first, "end_deg": last,
        "turned_deg": total,
        # heading at the end, against the heading the last leg asked for
        "final_err_deg": legs[-1]["end_err_deg"],
        "end_err_abs_mean": sum(abs_end) / len(abs_end),
        "end_err_abs_max": max(abs_end),
        "overshoot_max": max(leg["overshoot_deg"] for leg in clean),
        "yaw_resets": steps,
        "rate_mean": (sum(rates) / len(rates) if rates else None),
        "commanded_rate": ticks[cmds[0]]["setpoint_requested"].get("rate_dps"),
    }


def perception_summary(ticks: List[Dict[str, Any]],
                       events: List[Dict[str, Any]],
                       skills: List[Dict[str, Any]]
                       ) -> Optional[Dict[str, Any]]:
    """
    What the detector contributed to a flight, from the flight log alone.

    None for a flight that had no perception, so older logs print nothing.

    "direction" is the aircraft's heading plus the object's bearing in the
    picture: where the object is, in the frame the aircraft turns in. For
    something standing still that number should not move while the aircraft
    turns, so its spread is a direct check on the calibration and on how old
    the picture is. A picture L seconds old on an aircraft turning at w deg/s
    is paired with a heading that has since moved on by L*w, and that shows
    up as a shift between the turning and the holding parts of the sweep.
    """
    started = next((e for e in events
                    if e.get("kind") == "perception_started"), None)
    with_obj = [t for t in ticks if t.get("objects")]
    if started is None and not with_obj:
        return None
    t0 = ticks[0]["t_sample_mono"]
    out: Dict[str, Any] = {
        "started": started,
        "health": next((e for e in events
                        if e.get("kind") == "perception_health"), None),
        "ticks": len(ticks), "ticks_with_objects": len(with_obj),
        "labels": {}}
    seen: Dict[str, List[tuple]] = {}
    prev_yaw: Optional[float] = None
    prev_t: Optional[float] = None
    for t in ticks:
        yaw = _get(t, "pose.yaw_deg")
        rate = None
        if (yaw is not None and prev_yaw is not None
                and t["t_sample_mono"] > prev_t):
            rate = _wrap(yaw - prev_yaw) / (t["t_sample_mono"] - prev_t)
        if yaw is not None:
            prev_yaw, prev_t = yaw, t["t_sample_mono"]
        for o in t.get("objects") or []:
            seen.setdefault(o.get("label", "?"), []).append(
                (t["t_sample_mono"] - t0, yaw, rate, o))
    for label, rows in seen.items():
        dirs = [_wrap(y + o["bearing_deg"]) for _, y, _, o in rows
                if y is not None and o.get("bearing_deg") is not None]
        # Directions wrap, so average them as vectors.
        mean_dir = spread = None
        if dirs:
            sx = sum(math.cos(math.radians(d)) for d in dirs)
            sy = sum(math.sin(math.radians(d)) for d in dirs)
            mean_dir = math.degrees(math.atan2(sy, sx))
            dev = [_wrap(d - mean_dir) for d in dirs]
            spread = (sum(x * x for x in dev) / len(dev)) ** 0.5
        turning = [_wrap(y + o["bearing_deg"] - (mean_dir or 0.0))
                   for _, y, r, o in rows
                   if y is not None and r is not None and abs(r) > 8.0
                   and o.get("bearing_deg") is not None]
        holding = [_wrap(y + o["bearing_deg"] - (mean_dir or 0.0))
                   for _, y, r, o in rows
                   if y is not None and r is not None and abs(r) < 2.0
                   and o.get("bearing_deg") is not None]
        turn_rates = [r for _, y, r, o in rows
                      if y is not None and r is not None and abs(r) > 8.0
                      and o.get("bearing_deg") is not None]
        shift = (st.mean(turning) - st.mean(holding)
                 if len(turning) >= 3 and len(holding) >= 3 else None)
        implied = (shift / st.mean(turn_rates)
                   if shift is not None and abs(st.mean(turn_rates)) > 1.0
                   else None)
        rngs = [o["range_m"] for *_, o in rows
                if o.get("range_m") is not None]
        out["labels"][label] = {
            "ticks": len(rows),
            "first_t_s": rows[0][0], "first_heading_deg": rows[0][1],
            "first_bearing_deg": rows[0][3].get("bearing_deg"),
            "ids": len({o.get("track_id") for *_, o in rows
                        if o.get("track_id") is not None}),
            "direction_deg": mean_dir, "direction_spread_deg": spread,
            "turning_minus_holding_deg": shift,
            "implied_picture_age_s": implied,
            "range_median_m": st.median(rngs) if rngs else None,
            "range_sources": sorted({o.get("range_source") for *_, o in rows
                                     if o.get("range_source")}),
        }
    acq = next((s for s in skills if s.get("phase") == "result"
                and s.get("reason") == "target acquired"), None)
    if acq is not None:
        tc = (acq.get("detail") or {}).get("target_class")
        first = out["labels"].get(tc, {}).get("first_t_s")
        t_stop = acq["t_mono"] - t0
        out["acquired"] = {"target_class": tc, "t_s": t_stop,
                           "after_first_sighting_s": (
                               t_stop - first if first is not None else None),
                           "legs_completed": (acq.get("detail") or {}).get(
                               "legs_completed")}
    return out


def position_hold(ticks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Distance from the hold point, BY THE EKF'S OWN ESTIMATE. On a
    flow-only airframe there is no external reference, so this measures
    how still the estimator believes the aircraft was, not where it was."""
    d = []
    for t in ticks:
        pose, origin = t.get("pose") or {}, t.get("origin")
        if origin and pose.get("x") is not None:
            d.append(math.hypot(pose["x"] - origin[0], pose["y"] - origin[1]))
    if len(d) < 10:
        return None
    s = sorted(d)
    return {"mean": sum(d) / len(d), "p95": s[int(0.95 * (len(s) - 1))],
            "max": s[-1]}


def report(path: str, verbose: bool = True) -> Dict[str, Any]:
    rows = load_records(path)
    meta = next((r for r in rows if r["type"] == "meta"), {})
    # 0.38 records the descent as ticks with skill "land". They are kept out
    # of every statistic about the skills and reported on their own below.
    land_ticks = [r for r in rows if r["type"] == "tick"
                  and r.get("skill") == "land"]
    ticks = [r for r in rows if r["type"] == "tick"
             and r.get("skill") != "land"]
    skills = [r for r in rows if r["type"] == "skill"]
    events = [r for r in rows if r["type"] == "event"]
    summary = next((r for r in rows if r["type"] == "summary"), {})

    out: Dict[str, Any] = {"path": path, "ticks": len(ticks)}

    if verbose:
        print("=" * 68)
        print(Path(path).name)
        print("=" * 68)
        label = meta.get("label") or "(no label)"
        print(f"  label      {label}")
        logged = meta.get("drone_agent_version", "pre-0.5.0")
        mismatch = ("" if logged == __version__
                    else f"   (this tool is {__version__})")
        print(f"  built by   drone_agent {logged}{mismatch}")
        print(f"  host       {meta.get('host')}  git {meta.get('git_rev')}")
        print(f"  skill      {meta.get('skill')} {meta.get('skill_params')}")
        print(f"  records    {len(rows)}  ({len(ticks)} ticks)")
        print(f"  outcome    {summary.get('outcome', 'unknown')}")

    if not ticks:
        if verbose:
            print("\n  no tick records - the flight ended before any skill ran")
            for e in events:
                print(f"    event: {e.get('kind')}  "
                      f"{ {k: v for k, v in e.items()
                           if k not in ('type', 'seq', 't_mono', 't_wall',
                                        'kind')} }")
        return out

    # ---- interventions ------------------------------------------------
    intervened = [t for t in ticks if _get(t, "arb.intervened")]
    blocked = [t for t in ticks if _get(t, "arb.blocked")]
    causes: Dict[str, int] = {}
    for t in intervened:
        for a in _get(t, "arb.actions", []) or []:
            causes[a.split()[0]] = causes.get(a.split()[0], 0) + 1

    modified = sum(1 for t in ticks
                   if _get(t, "arb.intervened")
                   and t.get("setpoint_requested") and t.get("setpoint_applied")
                   and _setpoint_modified(t["setpoint_requested"],
                                          t["setpoint_applied"]))
    out.update({"interventions": len(intervened), "blocked": len(blocked),
                "causes": causes, "modified": modified,
                "observations": len(intervened) - modified})

    # Where the aircraft actually went, as opposed to what it was told. A
    # clamped setpoint does not bound the vehicle; its controller overshoots.
    arb_sum = _get(summary, "arbiter", {}) or {}
    peak = arb_sum.get("peak_alt_breach_m")
    peak_ekf = arb_sum.get("peak_alt_breach_ekf_m")
    out["peak_alt_breach_m"] = peak
    if verbose:
        pct = 100.0 * len(intervened) / len(ticks)
        print(f"\nSAFETY")
        print(f"  interventions   {len(intervened)}/{len(ticks)} ticks "
              f"({pct:.1f}%)")
        print(f"  blocked         {len(blocked)}")
        if causes:
            for cause, n in sorted(causes.items(), key=lambda kv: -kv[1]):
                print(f"    {cause:<22} {n}")
        else:
            print("    (none - the envelope was never challenged)")
        if peak is not None and (peak > 0 or (peak_ekf or 0) > 0):
            print(f"  aircraft past the ceiling  +{peak:.2f} m "
                  f"(+{peak_ekf:.2f} m in the commanded frame)")
            print(f"    the clamp bounds the COMMAND; the vehicle's own "
                  f"controller overshoots it")
        elif peak is not None:
            print("  aircraft stayed inside the envelope throughout")

        # Show the actual divergence for the first few interventions, and
        # say whether the applied command actually reached the aircraft.
        # A clamp that the rate limiter swallowed proves nothing about the
        # envelope, so the distinction matters.
        # Two kinds of intervention, and they are not interchangeable. A
        # MODIFIED setpoint (requested != applied) is the arbiter changing a
        # command, and whether it was transmitted decides whether anything
        # was actually proven. An OBSERVATION (breach, watchdog) annotates
        # the aircraft's state and modifies nothing, so "was it transmitted"
        # is not a meaningful question to ask of it.
        shown = 0
        collapsed = 0
        for idx, t in enumerate(ticks):
            if not _get(t, "arb.intervened"):
                continue
            req, app = t.get("setpoint_requested"), t.get("setpoint_applied")
            if not (req and app):
                continue
            modified = _setpoint_modified(req, app)
            if not modified:
                collapsed += 1
                continue
            if shown >= 6:
                collapsed += 1
                continue
            shown += 1
            sent = _was_transmitted(ticks, idx)
            mark = "TRANSMITTED" if sent else "suppressed by the rate limiter"
            if req.get("kind") == "position":
                print(f"    t={t['t_mono']:.1f}  "
                      f"req ({req.get('x')},{req.get('y')},{req.get('z')}) "
                      f"-> app ({app.get('x')},{app.get('y')},{app.get('z')})")
            print(f"              {_get(t, 'arb.actions')}  [{mark}]")
            if not sent and not _get(t, "arb.blocked"):
                print("              NOTE: the modified command never reached "
                      "the aircraft, so this tick does not demonstrate the "
                      "envelope held — only that the rate limiter fired "
                      "first. Re-run with an immediate request.")
        if collapsed:
            print(f"    ({collapsed} further intervention tick(s) not shown — "
                  f"observations that modified no setpoint, or repeats)")

    # ---- skills --------------------------------------------------------
    results = [s for s in skills if s.get("phase") == "result"]
    out["skills"] = [{"skill": s.get("skill"), "status": s.get("status"),
                      "reason": s.get("reason"),
                      "elapsed_s": s.get("elapsed_s")} for s in results]
    if verbose:
        print(f"\nSKILLS")
        for s in skills:
            if s.get("phase") == "start":
                pre = s.get("precondition", {})
                flag = "" if pre.get("ok") else f"  PRECONDITION FAILED: {pre.get('reason')}"
                print(f"  start   {s.get('skill')} {s.get('params')}{flag}")
            else:
                print(f"  result  {s.get('skill')}: {s.get('status')}"
                      f"{' - ' + s['reason'] if s.get('reason') else ''}"
                      f"  ({s.get('elapsed_s')}s, {s.get('ticks')} ticks)")
                if s.get("metrics"):
                    print(f"          metrics {s['metrics']}")
                if s.get("detail"):
                    print(f"          detail  {s['detail']}")

    # ---- estimate health ----------------------------------------------
    pose_age = [t.get("pose_age_s") for t in ticks]
    flow = [t.get("flow_quality") for t in ticks
            if t.get("flow_quality") is not None]
    distrust = [t for t in ticks if t.get("rngfnd_down_raw") is not None
                and not t.get("rngfnd_down_trusted")]
    no_pose = [t for t in ticks if not t.get("pose")]
    rf_vals = [t.get("rngfnd_down_raw") for t in ticks
               if t.get("rngfnd_down_raw") is not None]

    _res = [t.get("rngfnd_down_residual") for t in ticks
            if t.get("rngfnd_down_residual") is not None]
    _allw = [t.get("rngfnd_down_allowed") for t in ticks
             if t.get("rngfnd_down_allowed") is not None]
    _marg = [a - abs(x) for x, a in zip(_res, _allw)] if _res and _allw else []
    out.update({"pose_missing": len(no_pose),
                "rf_distrusted_ticks": len(distrust),
                "rf_margin_worst": (min(_marg) if _marg else None)})
    if verbose:
        print(f"\nESTIMATE")
        print(f"  pose age (s)    {_stats([p for p in pose_age if p])}")
        print(f"  pose missing    {len(no_pose)}/{len(ticks)} ticks")
        if flow:
            print(f"  flow quality    min {min(flow)}  median "
                  f"{st.median(flow):.0f}  max {max(flow)}")
        if rf_vals:
            spread = max(rf_vals) - min(rf_vals)
            print(f"  rangefinder     {min(rf_vals):.2f}-{max(rf_vals):.2f} m "
                  f"(spread {spread:.3f} m)")
            if spread < 0.01 and len(rf_vals) > 20:
                print("    WARNING: zero variation across the whole flight. "
                      "A pinned reading usually means the sensor is clamped "
                      "at a limit, not measuring. Check RNGFND1_MIN / "
                      "RNGFND1_GNDCLEAR.")
        print(f"  RF distrusted   {len(distrust)}/{len(ticks)} ticks")
        resid = [t.get("rngfnd_down_residual") for t in ticks
                 if t.get("rngfnd_down_residual") is not None]
        if resid:
            allowed = [t.get("rngfnd_down_allowed") for t in ticks
                       if t.get("rngfnd_down_allowed") is not None]
            print(f"  RF residual     {_stats([abs(r) for r in resid])}")
            if allowed:
                print(f"  RF threshold    {min(allowed):.2f}-{max(allowed):.2f} m "
                      f"(widens with recent |vz|)")
                # How close the trust test came to firing. A thin margin means
                # the next flight may distrust a healthy sensor, which flips
                # the altitude source mid-flight.
                margins = [a - abs(r) for r, a in
                           zip(resid, allowed) if a is not None]
                if margins:
                    worst = min(margins)
                    print(f"  RF trust margin worst {worst:+.3f} m")
                    if worst < 0.10:
                        print(f"    WARNING: the residual came within "
                              f"{worst:.3f} m of distrusting a healthy "
                              f"sensor. The 0.35 m base threshold was "
                              f"calibrated from a stable hold; residuals on "
                              f"this airframe reach "
                              f"{max(abs(r) for r in resid):.2f} m in normal "
                              f"flight. Recalibrate from a dedicated vertical "
                              f"profile rather than inferring it.")
        elif distrust:
            print("    NOTE: this log predates residual logging, so the "
                  "trust decisions cannot be audited. Re-fly to capture "
                  "rngfnd_down_residual/_allowed.")

    # ---- how well it actually flew --------------------------------------
    alt = altitude_hold(ticks)
    yaw = yaw_legs(ticks, events)
    pos = position_hold(ticks)
    out.update({"alt_hold": alt, "yaw": yaw, "pos_hold": pos})

    # Two things that decide whether a flight belongs in a repeat set at all,
    # and that no flight-quality figure shows: how long the takeoff took,
    # and what the autopilot said about the battery.
    lb = next((e for e in events if e.get("kind") == "launch_begin"), None)
    le = next((e for e in events if e.get("kind") == "launch_end"), None)
    launch = None
    if lb and le and lb.get("t_mono") is not None:
        launch = {"s": le["t_mono"] - lb["t_mono"],
                  "timed_out": bool(le.get("climb_timed_out")),
                  "frame": le.get("climb_frame"),
                  "ekf_climb_m": le.get("ekf_climb_m"),
                  "alt_best_m": le.get("alt_best_m")}
    batt_msgs = [e.get("text", "") for e in events
                 if e.get("kind") == "statustext"
                 and "attery" in e.get("text", "")]
    out.update({"launch": launch, "battery_msgs": batt_msgs})
    if verbose and launch:
        line = f"\nTAKEOFF  {launch['s']:.1f} s from command to handover"
        if launch["timed_out"]:
            line += (f"  -- THE CLIMB WAIT TIMED OUT. It was judged on "
                     f"{launch['frame'] or '?'}: EKF climbed "
                     f"{launch['ekf_climb_m']} m, best estimate "
                     f"{launch['alt_best_m']} m")
        print(line)
    if verbose and batt_msgs:
        print(f"BATTERY  the autopilot reported, during this flight:")
        for m in batt_msgs:
            if "Failsafe" not in m:
                print(f"    {m}")
        print(f"    A flight on a pack in this state is not comparable with "
              f"one on a charged pack.")
    if verbose and alt:
        print(f"\nALTITUDE HOLD")
        print(f"  commanded       {alt['target_m']:.2f} m")
        print(f"  EKF             mean {alt['ekf_mean']:.2f}  "
              f"range {alt['ekf_min']:.2f}-{alt['ekf_max']:.2f} m  "
              f"(mean {alt['ekf_above_mean']:+.2f}, peak "
              f"{alt['ekf_above_max']:+.2f} vs commanded)")
        if "rf_mean" in alt:
            print(f"  rangefinder     mean {alt['rf_mean']:.2f}  "
                  f"range {alt['rf_min']:.2f}-{alt['rf_max']:.2f} m  "
                  f"(mean {alt['rf_above_mean']:+.2f}, peak "
                  f"{alt['rf_above_max']:+.2f} vs commanded)")
            print(f"  RF minus EKF    mean {alt['gap_mean']:+.2f}  "
                  f"range {alt['gap_min']:+.2f} to {alt['gap_max']:+.2f} m")
            if alt["rf_range"] > 0.25:
                print(f"    the aircraft wandered {alt['rf_range']:.2f} m in "
                      f"height by its rangefinder while holding one "
                      f"commanded altitude")
            k, r = alt.get("rf_per_ekf"), alt.get("rf_ekf_r")
            bm = alt.get("bias_model") or {}
            if bm.get("fits"):
                cs, es, sg = bm["ctrl_s"], bm["est_s"], bm["static_gap_m"]
                lo = sg + (cs + es) * bm["vz_min"]
                hi = sg + (cs + es) * bm["vz_max"]
                print(f"  cause           the height error follows the EKF's "
                      f"own vertical-velocity error, vz")
                print(f"    EKF above commanded = {cs:.2f} s x vz"
                      f"            (r={bm['ctrl_r']:.2f})  a position loop "
                      f"of gain {1 / cs:.2f}/s cancelling a")
                print(f"                                                    "
                      f"     descent that is not happening")
                print(f"    RF above EKF        = {es:.2f} s x vz + "
                      f"{sg:.2f} m   (r={bm['est_r']:.2f})  the estimator "
                      f"pulling toward the rangefinder")
                print(f"                                                    "
                      f"     at {1 / es:.2f}/s")
                print(f"    smoothed vz ran {bm['vz_min']:+.2f} to "
                      f"{bm['vz_max']:+.2f} m/s (+ is down), which puts the "
                      f"aircraft {lo:+.2f} to {hi:+.2f} m above")
                print(f"    what was commanded")
                if k is not None:
                    print(f"    the rangefinder moved {k:.2f} m per metre of "
                          f"EKF. That is the ratio of those two gaps "
                          f"(1 + {es:.2f}/{cs:.2f} = {1 + es / cs:.2f}),")
                    print(f"    not an EKF that sees half of every height "
                          f"change: both move with the same error")
                print(f"    the autopilot's altitude fence is judged on the "
                      f"EKF's number, which read up to "
                      f"{alt['gap_max']:.2f} m below the")
                print(f"    rangefinder in this flight")
            elif k is not None and r is not None and r > 0.8 and k > 1.3:
                print(f"    the rangefinder moved {k:.2f} m for every metre "
                      f"the EKF reported (r={r:.2f}). The autopilot's "
                      f"altitude fence is judged on the EKF's number")
        if "vz_bias_down_mps" in alt:
            print(f"  EKF vz          mean {alt['vz_mean']:+.3f} m/s (NED, "
                  f"+ is down); integrated it predicts "
                  f"{alt['vz_implied_change_m']:+.2f} m, the EKF altitude "
                  f"changed {alt['ekf_change_m']:+.2f} m")
            b = alt["vz_bias_down_mps"]
            if abs(b) > 0.03:
                print(f"    the velocity estimate disagrees with its own "
                      f"altitude by {b:+.3f} m/s on average. A controller "
                      f"that trusts it holds the aircraft off its target to "
                      f"cancel a motion that is not happening, which is the "
                      f"direction of the height error above")
    if verbose and pos:
        print(f"\nPOSITION HOLD (EKF frame - flow dead reckoning, no "
              f"external reference)")
        print(f"  from hold point mean {pos['mean']:.3f}  p95 "
              f"{pos['p95']:.3f}  max {pos['max']:.3f} m")
    if verbose and yaw:
        print(f"\nYAW LEGS")
        print(f"  {'leg':>3} {'target':>7} {'turn':>6} {'slew_s':>6} "
              f"{'deg/s':>6} {'overshoot':>9} {'end_err':>8} {'short':>6}")
        for leg in yaw["legs"]:
            slew = f"{leg['slew_s']:.2f}" if leg["slew_s"] is not None else "-"
            rate = f"{leg['rate_dps']:.1f}" if leg["rate_dps"] else "-"
            short = (f"{leg['short_deg']:.1f}"
                     if leg.get("short_deg") is not None else "-")
            print(f"  {leg['leg']:>3} {leg['target_deg']:7.1f} "
                  f"{leg['turn_deg']:+6.1f} {slew:>6} {rate:>6} "
                  f"{leg['overshoot_deg']:9.1f} {leg['end_err_deg']:+8.1f} "
                  f"{short:>6}{' *' if leg.get('yaw_reset') else ''}")
        print(f"    short = how far from its target the heading read when "
              f"the skill called the leg complete")
        for leg in yaw["legs"]:
            rs = leg.get("yaw_reset")
            if not rs:
                continue
            print(f"  * leg {leg['leg']}: the EKF re-aligned its heading to "
                  f"the compass mid-leg. It stepped {rs['step_deg']:+.1f} deg "
                  f"between two samples")
            print(f"      ({rs['before_deg']:.1f} -> {rs['yaw_deg']:.1f}), "
                  f"where the rotation was producing about "
                  f"{rs['typical_step_deg']:.0f} deg per sample.")
            if rs.get("statustext"):
                print(f"      autopilot, at that moment: "
                      f"\"{rs['statustext']}\"")
            else:
                print(f"      This log has no autopilot messages (recorded "
                      f"from 0.15 on); in every later flight this step")
                print(f"      came with \"EKF3 IMU0 MAG0 ground mag anomaly, "
                      f"yaw re-aligned\".")
            print(f"      From that sample on, headings are in the "
                  f"re-aligned frame, so this leg's figures are not control")
            print(f"      performance. It is left out of the summary lines "
                  f"below and of the pooled yaw statistics.")
        loose = [rs for rs in yaw.get("yaw_resets", [])
                 if not any(leg.get("yaw_reset") is rs
                            for leg in yaw["legs"])]
        for rs in loose:
            print(f"  heading step outside any leg: {rs['before_deg']:.1f} -> "
                  f"{rs['yaw_deg']:.1f} deg ({rs['step_deg']:+.1f})"
                  + (f"; autopilot: \"{rs['statustext']}\""
                     if rs.get("statustext") else ""))
        cr = yaw.get("commanded_rate")
        rate_txt = (f"{yaw['rate_mean']:.1f} deg/s achieved"
                    + (f" of {cr:g} commanded" if cr else "")
                    if yaw["rate_mean"] else "rate n/a")
        print(f"  {yaw['n']} legs, {yaw['turned_deg']:+.0f} deg; {rate_txt}")
        print(f"  heading error at end of leg  mean |{yaw['end_err_abs_mean']:.2f}|"
              f"  worst |{yaw['end_err_abs_max']:.2f}| deg;  "
              f"worst overshoot {yaw['overshoot_max']:.1f} deg")
        print(f"  final heading {yaw['end_deg']:.1f} vs "
              f"{yaw['legs'][-1]['target_deg']:.1f} asked: "
              f"{yaw['final_err_deg']:+.1f} deg")

    # ---- timing and link ----------------------------------------------
    t_mono = [t["t_mono"] for t in ticks]
    dts = [b - a for a, b in zip(t_mono, t_mono[1:]) if b > a]
    lat = [t["t_mono"] - t["t_sample_mono"] for t in ticks
           if "t_sample_mono" in t]
    uplink = [_get(t, "link.uplink_bytes") for t in ticks]
    uplink = [u for u in uplink if u is not None]
    perc = perception_summary(ticks, events, skills)
    out["perception"] = perc
    if verbose and perc is not None:
        print(f"\nPERCEPTION")
        ps = perc["started"]
        if ps:
            cs = ps.get("capture_size")
            print(f"  detector        {ps.get('detector')} "
                  f"({ps.get('model')}) on {ps.get('device_used')}, "
                  f"{ps.get('scene_rate_hz')} scenes/s on the ground, "
                  f"{_fmt(ps.get('inference_ms'), 0, ' ms')} per frame")
            print(f"  camera          "
                  f"{'x'.join(str(v) for v in cs) if cs else '?'}, HFOV "
                  f"{ps.get('hfov_deg')} deg"
                  f"{'' if ps.get('intrinsics_calibrated') else '  (ASSUMED, not calibrated)'}"
                  f", looking for {ps.get('classes')}")
        else:
            print("  no perception_started event: objects are in the log but "
                  "how the detector was started is not")
        print(f"  ticks carrying at least one object   "
              f"{perc['ticks_with_objects']} of {perc['ticks']} "
              f"({100.0 * perc['ticks_with_objects'] / perc['ticks']:.0f}%)")
        if not perc["labels"]:
            print("  nothing was detected at any tick of this flight")
        for label, L in sorted(perc["labels"].items(),
                               key=lambda kv: -kv[1]["ticks"]):
            print(f"  {label:<10} {L['ticks']} ticks, {L['ids']} track id(s); "
                  f"first seen {L['first_t_s']:.1f} s in at heading "
                  f"{_fmt(L['first_heading_deg'], 0)} deg, "
                  f"{_fmt(L['first_bearing_deg'], 0)} deg off the nose")
            print(f"             direction {_fmt(L['direction_deg'], 1)} deg, "
                  f"spread {_fmt(L['direction_spread_deg'], 1)} deg; range "
                  f"median {_fmt(L['range_median_m'], 2, ' m')} "
                  f"({', '.join(L['range_sources']) or 'no range'})")
            d = L["turning_minus_holding_deg"]
            if d is not None:
                age = L["implied_picture_age_s"]
                print(f"             while turning its direction reads "
                      f"{d:+.1f} deg against while holding"
                      + (f": a picture about {age:.2f} s old by the time "
                         f"it is used" if age is not None and age > 0
                         else ""))
                print("             (that holds for an object standing "
                      "still; one that moved during the sweep spoils it)")
        a = perc.get("acquired")
        if a:
            lag = a["after_first_sighting_s"]
            print(f"  TARGET ACQUIRED  {a['target_class']} at {a['t_s']:.1f} s"
                  f", after {a['legs_completed']} leg(s)"
                  + (f", {lag:.1f} s after it first appeared" if lag
                     is not None else ""))
        h = perc["health"]
        if h:
            bad = h.get("errors") or h.get("last_error") or h.get(
                "capture_warning")
            print(f"  at the end      {h.get('frames', '?')} frames, errors "
                  f"{h.get('errors', 0)}"
                  + (f"   <-- {h.get('last_error') or h.get('capture_warning')}"
                     if bad else ""))

    link_age = [_get(t, "link.age_s") for t in ticks]
    rssi = [_get(t, "link.rssi") for t in ticks if _get(t, "link.rssi")]

    if verbose:
        print(f"\nTIMING AND LINK")
        if dts:
            print(f"  tick rate       {1.0 / st.median(dts):.1f} Hz median "
                  f"(worst gap {max(dts):.3f}s)")
        if lat:
            print(f"  log latency     {_stats(lat)}")
        if len(uplink) >= 2 and t_mono[-1] > t_mono[0]:
            rate = (uplink[-1] - uplink[0]) / (t_mono[-1] - t_mono[0])
            budget = _get(meta, "config.link.uplink_bps", 735)
            print(f"  uplink          {rate:.0f} B/s of {budget} "
                  f"({100 * rate / budget:.0f}%)")
        gaps = [a for a in link_age if a is not None and a > 2.0]
        print(f"  heartbeat gaps  {len(gaps)} ticks over 2 s")
        if rssi:
            print(f"  RSSI            min {min(rssi)}  median "
                  f"{st.median(rssi):.0f}  max {max(rssi)}")
        else:
            print("  RSSI            not in the log — enable RADIO_STATUS or "
                  "RC_CHANNELS to log link quality alongside outcomes")

    # ---- landing (0.38) -------------------------------------------------
    if land_ticks and ticks:
        p0, p1 = ticks[-1].get("pose") or {}, land_ticks[-1].get("pose") or {}
        if p0 and p1:
            slid = math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])
            took = land_ticks[-1]["t_mono"] - land_ticks[0]["t_mono"]
            out["landing"] = {"ticks": len(land_ticks), "s": round(took, 1),
                              "ekf_slide_m": round(slid, 2)}
            if verbose:
                print(f"\nLANDING  {len(land_ticks)} ticks over {took:.1f} s; "
                      f"the EKF moved {slid:.2f} m horizontally on the way down")
                print(f"  (a slide the flow sensor cannot see is not in this "
                      f"number; a tape to the landing spot is)")

    # ---- events --------------------------------------------------------
    if verbose and events:
        print(f"\nEVENTS")
        for e in events:
            payload = {k: v for k, v in e.items()
                       if k not in ("type", "seq", "t_mono", "t_wall", "kind")}
            print(f"  {e['t_mono']:8.1f}  {e.get('kind')}  {payload or ''}")

    return out


def compare(paths: List[str]) -> None:
    """
    One row per flight, with the numbers a repeat test actually produces.

    Repeats of an identical command are the unit of the evaluation chapter, so
    the spread across them matters as much as any single value: three
    clamp_alt flights on 2026-11-13 overshot the ceiling by 0.50, 0.63 and
    0.85 m from the same command.
    """
    print("=" * 110)
    print("COMPARISON")
    print("=" * 110)
    hdr = (f"{'flight':<22} {'outcome':<15} {'ticks':>5} {'mod':>4} "
           f"{'obs':>4} {'brch_m':>7} {'brch_cmd':>8} {'rf_marg':>8} "
           f"{'dur_s':>6}  reason")
    print(hdr)
    print("-" * len(hdr))
    rows_out: List[Dict[str, Any]] = []
    for path in paths:
        r = report(path, verbose=False)
        recs = load_records(path)
        summary = next((x for x in recs if x["type"] == "summary"), {})
        res = [x for x in recs if x["type"] == "skill"
               and x.get("phase") == "result"]
        arb = _get(summary, "arbiter", {}) or {}
        meta = next((x for x in recs if x["type"] == "meta"), {})
        reason = (res[-1].get("reason", "") if res else "")
        row = {
            "flight": Path(path).name,
            "airframe": str(_get(meta, "config.airframe", "unknown")),
            "outcome": str(summary.get("outcome", "?")),
            "ticks": r.get("ticks", 0),
            "modified": r.get("modified", 0),
            "observations": r.get("observations", 0),
            "breach_m": arb.get("peak_alt_breach_m"),
            "breach_cmd_m": arb.get("peak_alt_breach_ekf_m"),
            "rf_margin_m": r.get("rf_margin_worst"),
            "duration_s": (res[-1].get("elapsed_s") if res else None),
            "reason": reason,
        }
        row["launch"] = r.get("launch")
        row["battery_msgs"] = r.get("battery_msgs") or []
        a, y, ph = r.get("alt_hold"), r.get("yaw"), r.get("pos_hold")
        bm = (a or {}).get("bias_model") or {}
        row.update({
            "alt_cmd": a and a.get("target_m"),
            "rf_mean": a and a.get("rf_mean"),
            "rf_max": a and a.get("rf_max"),
            "rf_range": a and a.get("rf_range"),
            "rf_above_mean": a and a.get("rf_above_mean"),
            "rf_above_max": a and a.get("rf_above_max"),
            "gap_max": a and a.get("gap_max"),
            "rf_per_ekf": a and a.get("rf_per_ekf"),
            "vz_bias": a and a.get("vz_bias_down_mps"),
            "alt_gain_s": (bm["ctrl_s"] + bm["est_s"]) if bm.get("fits")
            else None,
            "static_gap": bm["static_gap_m"] if bm.get("fits") else None,
            "yaw_legs": (y or {}).get("legs") or [],
            "yaw_err": y and y.get("end_err_abs_mean"),
            "yaw_err_max": y and y.get("end_err_abs_max"),
            "yaw_over": y and y.get("overshoot_max"),
            "yaw_final": y and y.get("final_err_deg"),
            "yaw_rate": y and y.get("rate_mean"),
            "xy_p95": ph and ph.get("p95"),
            "xy_max": ph and ph.get("max"),
        })
        rows_out.append(row)
        print(f"{row['flight'][:22]:<22} {row['outcome'][:15]:<15} "
              f"{row['ticks']:>5} {row['modified']:>4} "
              f"{row['observations']:>4} "
              f"{_fmt(row['breach_m']):>7} {_fmt(row['breach_cmd_m']):>8} "
              f"{_fmt(row['rf_margin_m'], 3):>8} "
              f"{_fmt(row['duration_s'], 1):>6}  {reason[:34]}")

    # How it flew, as opposed to whether the envelope held. This is the
    # table a baseline is: the same command repeated, and what the aircraft
    # did with it each time.
    if any(x.get("alt_cmd") is not None for x in rows_out):
        print()
        hdr2 = (f"{'flight':<22} {'alt_cmd':>7} {'rf_mean':>7} {'rf_max':>6} "
                f"{'rf_rng':>6} {'rf-ekf':>6} {'vz_bias':>7} {'xy_p95':>6} "
                f"{'xy_max':>6} {'yaw|e|':>6} {'yaw_ov':>6} {'final':>6}")
        print(hdr2)
        print("-" * len(hdr2))
        for x in rows_out:
            print(f"{x['flight'][:22]:<22} {_fmt(x['alt_cmd']):>7} "
                  f"{_fmt(x['rf_mean']):>7} {_fmt(x['rf_max']):>6} "
                  f"{_fmt(x['rf_range']):>6} {_fmt(x['gap_max']):>6} "
                  f"{_fmt(x['vz_bias'], 3):>7} {_fmt(x['xy_p95']):>6} "
                  f"{_fmt(x['xy_max']):>6} {_fmt(x['yaw_err']):>6} "
                  f"{_fmt(x['yaw_over'], 1):>6} {_fmt(x['yaw_final'], 1):>6}")
        print("  alt_cmd commanded height | rf_* rangefinder, m | rf-ekf "
              "worst gap, m | vz_bias EKF velocity vs its own")
        print("  altitude, m/s down | xy_* from the hold point, EKF frame, m "
              "| yaw|e| mean abs heading error at end")
        print("  of leg, deg | yaw_ov worst overshoot | final heading error "
              "after the last leg")

    # Spread across repeats is the number a single flight cannot give.
    print()
    for key, label in (("breach_m", "aircraft past ceiling (best est)"),
                       ("breach_cmd_m", "aircraft past ceiling (commanded)"),
                       ("rf_margin_m", "RF trust margin"),
                       ("duration_s", "duration to end"),
                       ("alt_cmd", "commanded height"),
                       ("rf_above_mean", "mean height above commanded (RF)"),
                       ("rf_above_max", "peak height above commanded (RF)"),
                       ("rf_range", "height wander (RF range)"),
                       ("rf_per_ekf", "RF change per metre of EKF"),
                       ("vz_bias", "EKF vz bias, m/s down"),
                       ("alt_gain_s", "height error per m/s of vz error, s"),
                       ("static_gap", "RF minus EKF at zero vz error, m"),
                       ("xy_p95", "horizontal p95 (EKF frame)"),
                       ("xy_max", "horizontal max (EKF frame)"),
                       ("yaw_final", "final heading error, deg")):
        vals = [x[key] for x in rows_out if x.get(key) is not None]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        print(f"  {label:<34} n={len(vals)}  mean {mean:+.3f}  "
              f"min {min(vals):+.3f}  max {max(vals):+.3f}  "
              f"spread {max(vals) - min(vals):.3f}")

    # Legs are the natural unit for yaw: eight per sweep, so three flights
    # are twenty-four samples of the same manoeuvre.
    all_legs = [leg for x in rows_out for leg in x.get("yaw_legs", [])]
    # A leg in which the EKF re-aligned its heading is measured in two
    # frames; it says nothing about how well the aircraft turns.
    legs = [leg for leg in all_legs if not leg.get("yaw_reset")]
    if len(legs) >= 4:
        def _pool(key: str, label: str, absolute: bool = False) -> None:
            v = [abs(leg[key]) if absolute else leg[key] for leg in legs
                 if leg.get(key) is not None]
            if len(v) < 2:
                return
            print(f"  {label:<34} n={len(v)}  mean {sum(v) / len(v):+.3f}  "
                  f"sd {st.stdev(v):.3f}  min {min(v):+.3f}  "
                  f"max {max(v):+.3f}")
        print()
        _pool("end_err_deg", "yaw |error| at end of leg, deg", absolute=True)
        _pool("overshoot_deg", "yaw overshoot, deg")
        _pool("rate_dps", "yaw rate achieved, deg/s")
        _pool("slew_s", "time to within 3 deg, s")
        _pool("leg_s", "leg duration, s")
        _pool("short_deg", "short of target at leg complete")
        hit = [(x["flight"], leg) for x in rows_out
               for leg in x.get("yaw_legs", []) if leg.get("yaw_reset")]
        if hit:
            flights = len({f for f, _ in hit})
            said = sum(1 for _, leg in hit if leg["yaw_reset"].get("statustext"))
            print(f"\n  EKF YAW RE-ALIGNMENT: in {flights} of {len(rows_out)} "
                  f"flights the EKF snapped its heading to the compass "
                  f"during a leg.")
            print(f"  Those {len(hit)} leg(s) are not in the yaw figures "
                  f"above (n={len(legs)} of {len(all_legs)}).")
            for f, leg in hit:
                rs = leg["yaw_reset"]
                print(f"    {f[:20]:<20} leg {leg['leg']}: "
                      f"{rs['before_deg']:.1f} -> {rs['yaw_deg']:.1f} deg, a "
                      f"step of {rs['step_deg']:+.1f}"
                      + ("  [autopilot: yaw re-aligned]"
                         if rs.get("statustext") else "  [no messages logged]"))
            sz = [abs(leg["yaw_reset"]["step_deg"]) for _, leg in hit]
            which = {leg["leg"] for _, leg in hit}
            print(f"  Step size {min(sz):.0f}-{max(sz):.0f} deg, of which "
                  f"about one sample of rotation ("
                  f"{sum(leg['yaw_reset']['typical_step_deg'] for _, leg in hit) / len(hit):.0f}"
                  f" deg) is the turn itself: the EKF heading and the "
                  f"compass")
            print(f"  had drifted that far apart by then."
                  + (f" Autopilot message logged for {said} of {len(hit)}."
                     if said else ""))
            if len(which) == 1 and flights > 1:
                print(f"  Always leg {which.pop()}, the leg that brings the "
                      f"aircraft back to its takeoff heading.")
        by_leg: Dict[int, List[float]] = {}
        for leg in legs:
            by_leg.setdefault(leg["leg"], []).append(leg["overshoot_deg"])
        if len(by_leg) > 2 and all(len(v) > 1 for v in by_leg.values()):
            means = {k: sum(v) / len(v) for k, v in by_leg.items()}
            worst = max(means, key=lambda k: means[k])
            rest = [m for k, m in means.items() if k != worst]
            if means[worst] > 2.0 * max(rest) and min(by_leg[worst]) > max(rest):
                print(f"    leg {worst} overshoots in EVERY flight "
                      f"({', '.join(f'{v:.1f}' for v in by_leg[worst])} deg) "
                      f"against at most {max(max(v) for k, v in by_leg.items() if k != worst):.1f} deg "
                      f"on any other leg: systematic, not sampling")

    # Height: compare what the aircraft DID, not the number it was given.
    # The commanded height is an EKF-frame value, and the aircraft flies
    # above it by an amount that follows the EKF's vertical-velocity error,
    # so two flights can be commanded 0.17 m apart and fly 0.10 m apart, or
    # be commanded 0.03 m apart and fly 0.17 m apart. Both have happened.
    cmds = [x["alt_cmd"] for x in rows_out if x.get("alt_cmd") is not None]
    rfm = [x["rf_mean"] for x in rows_out if x.get("rf_mean") is not None]
    if len(rfm) > 1 and max(rfm) - min(rfm) > 0.15:
        print(f"\n  WARNING: these flights were flown at different real "
              f"heights (rangefinder mean "
              f"{', '.join(f'{c:.2f}' for c in rfm)} m).")
        if cmds:
            print(f"  The commanded heights were "
                  f"{', '.join(f'{c:.2f}' for c in cmds)} m, in the EKF's "
                  f"frame. The altitude rows above mix that")
            print(f"  difference into their spread.")
    elif len(cmds) > 1 and max(cmds) - min(cmds) > 0.05 and rfm:
        print(f"\n  note: the commanded height differs "
              f"({', '.join(f'{c:.2f}' for c in cmds)} m, EKF frame) but "
              f"the real mean height is within "
              f"{max(rfm) - min(rfm):.2f} m")
        print(f"  ({', '.join(f'{c:.2f}' for c in rfm)} m by rangefinder). "
              f"The commanded number is not the height the aircraft flies "
              f"at.")

    slow = [(x["flight"], x["launch"]) for x in rows_out if x.get("launch")
            and (x["launch"]["timed_out"] or x["launch"]["s"] > 15.0)]
    for f, la in slow:
        print(f"\n  WARNING: {f[:24]} took {la['s']:.1f} s to take off"
              + (" (the climb wait timed out)" if la["timed_out"] else "")
              + ". It hovered that long before the skill began.")
    low = [(x["flight"], x["battery_msgs"]) for x in rows_out
           if x.get("battery_msgs")]
    for f, msgs in low:
        worst = next((m for m in msgs if "critical" in m),
                     next((m for m in msgs if "Failsafe" not in m), msgs[0]))
        print(f"\n  WARNING: {f[:24]} flew on a battery the autopilot "
              f"reported as: {worst}.")
        print(f"  Do not pool it with flights on a charged pack.")

    # Pooling across airframes is the one way this table lies. Hover
    # throttle, climb authority and sensor count differ between them, so a
    # spread computed over a mixed set measures the fleet and not the
    # repeatability of anything.
    frames = {r.get("airframe", "unknown") for r in rows_out}
    if len(frames) > 1:
        print(f"\n  WARNING: these flights are from {len(frames)} different "
              f"airframes: {', '.join(sorted(frames))}.")
        print(f"  The pooled spread above is NOT a repeatability figure. "
              f"Compare within one airframe.")
    elif frames == {"unknown"}:
        print(f"\n  note: no airframe recorded in these logs. Fly with "
              f"--airframe so a repeat")
        print(f"  set can be told apart from a mixed one.")
    else:
        print(f"\n  airframe: {frames.pop()}")


def to_csv(path: str, out_path: str) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed; pip install pandas")
        return
    rows = load_records(path)
    df = pd.json_normalize([r for r in rows if r["type"] == "tick"])
    df.to_csv(out_path, index=False)
    print(f"{len(df)} ticks, {len(df.columns)} columns -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--compare", action="store_true",
                    help="one-line summary per flight instead of full reports")
    ap.add_argument("--csv", help="also dump ticks to this CSV (needs pandas)")
    ap.add_argument("--version", action="version",
                    version=f"drone_agent {__version__}")
    args = ap.parse_args()

    if args.compare:
        compare(args.logs)
        return 0
    for p in args.logs:
        report(p)
        if args.csv:
            to_csv(p, args.csv)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
