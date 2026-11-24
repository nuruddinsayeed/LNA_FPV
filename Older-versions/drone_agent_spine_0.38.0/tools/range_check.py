#!/usr/bin/env python3
"""
Work out, from flights already flown, what the range to a target WOULD have
been if it had been measured from the top of its box instead of from the
size of the box or from parallax.

    python tools/range_check.py logs/*.jsonl --intrinsics camera.json \\
        --object-height 1.78 --truth 2.57

Three ways to a range, and this prints all three side by side:

  size prior   from how tall the box is, against a prior on how tall the
               object is. Wrong whenever part of the object is outside the
               frame, which for a person seen by a nose-up camera is always.
  parallax     from how far the bearing moved between two positions a
               measured distance apart. Needs the heading to be the same at
               both, to about a degree.
  head         from the elevation of the TOP of the box, the height of the
               camera above the floor, and how tall the object is. Needs the
               camera's tilt, which `calibrate_camera.py tilt` measures.
               Nothing else: one look, no movement, no heading.

Nothing here flies or changes a calibration. It reads logs.
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
from drone_agent.perception.geometry import CameraIntrinsics


def looks(rows: List[Dict[str, Any]], label: str) -> Dict[str, List[Any]]:
    """The ticks of each named phase of a measure_range, plus the rest."""
    out: Dict[str, List[Any]] = {"look A": [], "look B": [], "other": []}
    for x in rows:
        if x.get("type") != "tick":
            continue
        objs = [q for q in (x.get("objects") or []) if q.get("label") == label]
        if not objs:
            continue
        why = (x.get("setpoint_requested") or {}).get("reason") or ""
        key = ("look A" if "look A" in why else
               "look B" if "look B" in why else "other")
        out[key].append((x, min(objs, key=lambda q: abs(q["bearing_deg"]))))
    return out


def report(path: str, intr: CameraIntrinsics, label: str, height_m: float,
           truth: Optional[float]) -> List[float]:
    rows = [json.loads(l) for l in open(path)]
    res = [x for x in rows if x.get("type") == "skill"
           and x.get("phase") == "result"]
    print(f"\n{Path(path).name}")
    got: List[float] = []
    for name, ticks in looks(rows, label).items():
        if not ticks or name == "other":
            continue
        heads, priors, cams = [], [], []
        for tick, obj in ticks:
            h = tick.get("rngfnd_down_raw")
            if h is None:
                continue
            r = intr.height_range(obj["bbox"][1], height_m, h,
                                  u=(obj["bbox"][0] + obj["bbox"][2]) / 2,
                                  pitch_deg=(tick.get("pose") or {}).get(
                                      "pitch_deg") or 0.0)
            if r is not None:
                heads.append(r)
            if obj.get("range_m") is not None:
                priors.append(obj["range_m"])
            cams.append(h)
        if not heads:
            print(f"  {name:7} no head range: the camera's tilt is not in "
                  f"{intr.width_px}x{intr.height_px} calibration "
                  f"(run `calibrate_camera.py tilt`)")
            continue
        med = st.median(heads)
        got.append(med)
        spread = st.pstdev(heads) if len(heads) > 1 else 0.0
        line = (f"  {name:7} head {med:5.2f} m +-{spread:4.2f}   "
                f"size prior {st.median(priors):5.2f} m   "
                f"camera at {st.median(cams):4.2f} m, {len(heads)} frames")
        if truth:
            line += f"   head is {100 * (med / truth - 1):+5.0f}% of {truth} m"
        print(line)
    for r in res:
        if r.get("skill") == "measure_range" and r.get("detail", {}).get(
                "range_m") is not None:
            d = r["detail"]
            line = (f"  {'parallax':7} {d['range_m']:5.2f} m "
                    f"+-{d.get('uncertainty_m', 0):4.2f}   "
                    f"baseline {d.get('baseline_m')} m, parallax "
                    f"{d.get('parallax_deg')} deg")
            if truth:
                line += (f"   {100 * (d['range_m'] / truth - 1):+5.0f}% of "
                         f"{truth} m")
            print(line)
    return got


def quiet(path: str, intr: CameraIntrinsics, label: str,
          height_m: float) -> List[float]:
    """The head ranges of a log, printing nothing."""
    out: List[float] = []
    rows = [json.loads(l) for l in open(path)]
    for name, ticks in looks(rows, label).items():
        if name == "other":
            continue
        for tick, obj in ticks:
            h = tick.get("rngfnd_down_raw")
            if h is None:
                continue
            r = intr.height_range(obj["bbox"][1], height_m, h,
                                  u=(obj["bbox"][0] + obj["bbox"][2]) / 2,
                                  pitch_deg=(tick.get("pose") or {}).get(
                                      "pitch_deg") or 0.0)
            if r is not None:
                out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--intrinsics", default="camera.json")
    ap.add_argument("--object-height", type=float,
                    help="how tall the target really is, in metres. The "
                         "range is proportional to this minus the camera "
                         "height, so a guess here is a guess in the answer")
    ap.add_argument("--class", dest="label", default="person")
    ap.add_argument("--truth", type=float,
                    help="the tape-measured range, if you have it")
    ap.add_argument("--tilt", type=float,
                    help="override the camera tilt in the calibration, to "
                         "see what a different one would give")
    args = ap.parse_args()
    print(f"drone_agent {__version__}")
    intr = CameraIntrinsics.load(args.intrinsics)
    if args.tilt is not None:
        intr.tilt_up_deg = args.tilt
    print(f"  {args.intrinsics}: {intr.model}, "
          f"{intr.width_px}x{intr.height_px}, camera tilt "
          f"{intr.tilt_up_deg:+.2f} deg"
          + ("  <-- not measured; the head range needs it"
             if abs(intr.tilt_up_deg) < 1e-9 else ""))
    if args.object_height is None:
        print("  --object-height was not given, and the range scales with "
              "it. Measure the target.")
        print("  Using 1.75 m meanwhile, which is a guess, not a "
              "measurement.")
        args.object_height = 1.75
    print(f"  target: {args.label}, {args.object_height} m tall")
    all_heads: List[float] = []
    for path in args.logs:
        all_heads += report(path, intr, args.label, args.object_height,
                            args.truth)
    if args.truth and all_heads:
        # Tilt and target height are two unknowns and the tape pins the
        # answer, so say what the tilt would have to be. The wall
        # measurement and this number agreeing is the test of the method;
        # if they disagree there is a bias to find before building on it.
        lo, hi = -5.0, 40.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            intr.tilt_up_deg = mid
            got: List[float] = []
            for path in args.logs:
                got += quiet(path, intr, args.label, args.object_height)
            if got and st.median(got) < args.truth:
                hi = mid
            else:
                lo = mid
        stored = json.loads(Path(args.intrinsics).read_text()).get(
            "tilt_up_deg", 0.0)
        print(f"\n  For these looks to come out at {args.truth} m with a "
              f"{args.object_height} m target, the")
        print(f"  camera tilt would have to be {0.5 * (lo + hi):.1f} deg. "
              f"The calibration says {stored:.2f} deg.")
        print("  Those two agreeing is the test of this method; they are "
              "measured separately.")
    if len(all_heads) > 1:
        print(f"\n  head range over {len(all_heads)} looks: median "
              f"{st.median(all_heads):.2f} m, spread "
              f"{st.pstdev(all_heads):.2f} m, worst "
              f"{min(all_heads):.2f}-{max(all_heads):.2f} m")
        if args.truth:
            errs = [abs(r - args.truth) for r in all_heads]
            print(f"  against {args.truth} m: median error "
                  f"{st.median(errs):.2f} m, worst {max(errs):.2f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
