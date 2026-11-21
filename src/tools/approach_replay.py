#!/usr/bin/env python3
"""
Replay the approach phase of a flight already flown and say what the
aircraft and the target really did, independently of what the skill
concluded at the time.

    python tools/approach_replay.py logs/*.jsonl

The aircraft knows its own displacement. Anchoring the target where the
first look put it, every later look can be predicted: this is where a
target that had not moved would appear. The gap between that and what was
measured is the target's own movement, and it is the number that separates
"the approach drifted" from "the person stepped aside".

Nothing here flies or changes a calibration. It reads logs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_agent import __version__

SCALE_MIN, SCALE_MAX = 0.9, 1.6


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def scale_of(tick: Dict[str, Any]) -> float:
    """Real metres per EKF metre, from this tick's two heights."""
    p = tick.get("pose")
    rf = tick.get("rngfnd_down_raw")
    if not p or rf is None:
        return 1.0
    ez = -float(p["z"])
    if ez <= 0.2 or rf <= 0.2:
        return 1.0
    return max(SCALE_MIN, min(SCALE_MAX, rf / ez))


def real_pose(tick: Dict[str, Any], first: Optional[Tuple[float, float]]
              ) -> Optional[Tuple[float, float, float]]:
    """
    Where the aircraft is relative to the first look, in real metres, plus
    heading.

    The height ratio corrects a DISPLACEMENT. Applying it to an absolute
    coordinate was a real bug: with the EKF's origin at (3648, -2060) a
    ratio moving from 1.014 to 1.074 between two looks moved the computed
    position by 209 m, and the skill concluded the person had walked away.
    """
    p = tick.get("pose")
    if not p:
        return None
    if first is None:
        first = (float(p["x"]), float(p["y"]))
    sc = scale_of(tick)
    return ((float(p["x"]) - first[0]) * sc,
            (float(p["y"]) - first[1]) * sc, float(p["yaw_deg"]))


def measurements(rows: List[Dict[str, Any]], label: str,
                 height_m: float = 1.73, min_rise: float = 0.35
                 ) -> List[Dict[str, Any]]:
    """One entry per MEASURE window of the approach."""
    out: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []
    for x in rows:
        if x.get("type") != "tick" or x.get("skill") != "approach":
            continue
        why = (x.get("setpoint_requested") or {}).get("reason") or ""
        if why.startswith("measuring"):
            cur.append(x)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    done = []
    for win in out:
        # The same rise floor the estimator applies: below it there is no
        # head range, so including those frames here reports garbage the
        # aircraft itself never acted on. Leaving this out made a flight
        # where the camera climbed to 1.49 m look like a target that had
        # moved 2.19 m, when the skill had correctly thrown those frames
        # away and measured 1.67 m on the rest.
        objs = [(t, o) for t in win for o in (t.get("objects") or [])
                if o.get("label") == label and o.get("range_m") is not None
                and t.get("rngfnd_down_raw") is not None
                and height_m - t["rngfnd_down_raw"] >= min_rise]
        if not objs:
            continue
        mid = win[len(win) // 2]
        done.append({"_mid": mid})
        done.pop()
        pose = None   # filled by the caller, which knows the first look
        skipped = sum(
            1 for t in win for o in (t.get("objects") or [])
            if o.get("label") == label and t.get("rngfnd_down_raw") is not None
            and height_m - t["rngfnd_down_raw"] < min_rise)
        done.append({
            "_mid": mid,
            "skipped": skipped,
            "range_m": st.median([o["range_m"] for _, o in objs]),
            "bearing_deg": st.median([o["bearing_deg"] for _, o in objs]),
            "sigma_m": st.median([o.get("range_sigma_m") or 0.0
                                  for _, o in objs]),
            "camera_m": st.median([t.get("rngfnd_down_raw") for t, _ in objs]),
            "u": st.median([(o["bbox"][0] + o["bbox"][2]) / 2
                            for _, o in objs if o.get("bbox")] or [0.5]),
            "frames": len(objs),
            "sources": sorted({str(o.get("range_source")) for _, o in objs}),
        })
    return done


def report(path: str, label: str, height_m: float) -> Optional[float]:
    rows = [json.loads(l) for l in open(path)]
    ms = measurements(rows, label, height_m)
    res = [x for x in rows if x.get("type") == "skill"
           and x.get("phase") == "result"]
    final = next((r for r in res if r.get("skill") == "approach"), None)
    print(f"\n{Path(path).name}")
    if not ms:
        end = res[-1] if res else {}
        print(f"  no approach measurements: last skill was "
              f"{end.get('skill')} ({end.get('status')}) - "
              f"{end.get('reason', '')[:60]}")
        return None
    anchor = None
    worst = 0.0
    first = None
    for i, m in enumerate(ms):
        if first is None:
            p0 = m["_mid"].get("pose")
            first = (float(p0["x"]), float(p0["y"])) if p0 else (0.0, 0.0)
        got = real_pose(m["_mid"], first)
        if got is None:
            continue
        px, py, yaw = got
        a = math.radians(yaw + m["bearing_deg"])
        here = (px + m["range_m"] * math.cos(a),
                py + m["range_m"] * math.sin(a))
        rise = height_m - m["camera_m"]
        e = math.degrees(math.atan2(rise, m["range_m"]))
        per_deg = (abs(rise) * math.radians(1.0)
                   / max(1e-9, math.sin(math.radians(max(1.0, e))) ** 2))
        line = (f"  look {i}: {m['range_m']:5.2f} m +-{m['sigma_m']:4.2f}  "
                f"bearing {m['bearing_deg']:+6.1f}  u {m['u']:.3f}  "
                f"camera {m['camera_m']:4.2f} m  rise {rise:4.2f} m  "
                f"{per_deg:4.2f} m/deg"
                + (f"  ({m['skipped']} frames dropped: camera too high)"
                   if m.get("skipped") else ""))
        if anchor is None:
            anchor = here
        else:
            dx, dy = anchor[0] - px, anchor[1] - py
            pr = math.hypot(dx, dy)
            pb = wrap(math.degrees(math.atan2(dy, dx)) - yaw)
            moved = math.hypot(here[0] - anchor[0], here[1] - anchor[1])
            worst = max(worst, moved)
            line += (f"\n            a target that had not moved would read "
                     f"{pr:5.2f} m {pb:+6.1f} deg: "
                     f"range {m['range_m'] - pr:+.2f} m, bearing "
                     f"{wrap(m['bearing_deg'] - pb):+.0f} deg, so it has "
                     f"moved {moved:.2f} m")
        print(line)
    if final:
        d = final.get("detail") or {}
        print(f"  the skill said: {final['status']} - {final.get('reason')}")
        if d.get("range_m"):
            print(f"    believed {d['range_m']} m against a "
                  f"{d.get('standoff_m')} m standoff, {d.get('steps')} steps, "
                  f"travelled {d.get('travelled_m')} m")
    print(f"  the target moved at most {worst:.2f} m during the approach")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--class", dest="label", default="person")
    ap.add_argument("--object-height", type=float, default=1.73,
                    help="how tall the target really is, for the rise")
    args = ap.parse_args()
    print(f"drone_agent {__version__}")
    moved = [report(p, args.label, args.object_height) for p in args.logs]
    got = [m for m in moved if m is not None]
    if len(got) > 1:
        print(f"\n  over {len(got)} approaches the target's own movement was "
              f"{min(got):.2f}-{max(got):.2f} m")
        print("  A still target is what this skill is for; the flights where "
              "that number is")
        print("  large are flights where the person stepped aside, not "
              "flights where the")
        print("  aircraft went wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
