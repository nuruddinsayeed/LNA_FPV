#!/usr/bin/env python3
"""
Read an autopilot `.bin` log and say why the aircraft climbed.

    python tools/bin_vertical.py flight.bin

Twelve flights have ended with the arbiter aborting on real height while
nothing in the stack commanded a climb. The ground-station log shows the
symptom and cannot show the cause: it has no IMU, no vibration and no
estimator-health data. This reads the autopilot's own log instead.

What it does, and why in that order:

  1. Rotates the accelerometer into the world's down axis using the
     attitude the autopilot logged, which removes tilt. A hover leaning
     14 deg reads 0.30 m/s2 light on its body Z axis for purely geometric
     reasons and that is not an error; this is the step that tells the two
     apart.
  2. Finds a window where the aircraft is demonstrably motionless -- on
     the floor, rangefinder at its minimum -- but the motors are running.
     There the true vertical acceleration is zero by construction, so
     anything other than -9.81 is the sensor.
  3. Prints the vertical trace: rangefinder, EKF altitude, barometer,
     estimated vertical velocity, the vertical position innovation and its
     test ratio, and throttle.

Nothing here flies or changes a parameter. It reads a file.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_agent import __version__
from drone_agent.vertical import (G, Verdict, airborne_span, airborne_spans,
                                  grounded_spans, longest, mean_residual,
                                  nearest, runs, summarise)

WANT = ("IMU", "ATT", "VIBE", "ESC", "CTUN", "RFND", "XKF1", "XKF2", "XKF3",
        "XKF4", "XKF5", "BARO", "MODE", "MSG", "OF")


def read(path: str) -> Dict[str, List[Dict[str, Any]]]:
    try:
        from pymavlink import mavutil
    except ImportError:
        sys.exit("this needs pymavlink: pip install pymavlink")
    conn = mavutil.mavlink_connection(path)
    rows: Dict[str, List[Dict[str, Any]]] = {k: [] for k in WANT}
    while True:
        msg = conn.recv_match(type=list(WANT))
        if msg is None:
            break
        d = msg.to_dict()
        kind = msg.get_type()
        # One core, one instance: a second EKF core or a second IMU is a
        # different estimator and averaging them together means nothing.
        if d.get("C") not in (None, 0):
            continue
        if kind in ("IMU", "RFND", "BARO") and d.get(
                "I", d.get("Instance")) not in (None, 0):
            continue
        if kind == "VIBE" and d.get("IMU") not in (None, 0):
            continue
        if kind == "ESC" and d.get("Instance") not in (None, 0):
            continue
        rows[kind].append(d)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--floor", type=float, default=0.12,
                    help="rangefinder reading at or below which the "
                         "aircraft is on the floor (m)")
    ap.add_argument("--trace", action="store_true",
                    help="print the vertical trace tick by tick")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable summary on stdout")
    args = ap.parse_args()

    rows = read(args.log)
    if not rows["IMU"] or not rows["ATT"]:
        sys.exit("this log has no IMU or ATT records to work with")
    t0 = min(d["TimeUS"] for v in rows.values() for d in v if d.get("TimeUS"))
    air = None
    if not args.json:
        print(f"drone_agent {__version__}")
        print(f"  {Path(args.log).name}: "
              f"{(max(d['TimeUS'] for d in rows['IMU']) - t0) / 1e6:.1f} s, "
              f"{sum(len(v) for v in rows.values())} records")
        for m in rows["MSG"][:1]:
            print(f"  {m.get('Message')}")

    wins = []
    def win(name: str, span: Any, still: bool = False) -> None:
        wins.append(summarise(
            rows["IMU"], rows["ATT"], rows["VIBE"], rows["ESC"], rows["CTUN"],
            rows["RFND"], name, span[0], span[1], still=still))

    # Motors off: the zero of the whole measurement.
    quiet = longest(runs(rows["ESC"], lambda d: (d.get("RPM") or 0) < 3000,
                         1.0))
    if quiet:
        win("on the ground, motors off", quiet, still=True)
    # On the floor with the motors running: the control, because there the
    # true vertical acceleration is zero whatever the sensor says.
    spun = runs(rows["ESC"], lambda d: (d.get("RPM") or 0) >= 3000, 1.0)
    n = 0
    for a, b in grounded_spans(rows["RFND"], args.floor):
        for c, d in spun:
            lo, hi = max(a, c), min(b, d)
            if hi - lo < 2e6:
                continue
            n += 1
            win(f"on the ground, motors running ({n})", (lo, hi), still=True)
    # Each flight separately. The ends are trimmed: the climb-out and the
    # touchdown are real acceleration and belong to no window.
    # Airborne means off the floor AND under power: after a disarm the
    # aircraft gets carried, and a rangefinder reading 2 m with the motors
    # stopped is a hand, not a hover.
    flights = [f for f in airborne_spans(rows["RFND"], args.floor)
               if any(c <= f[0] and f[1] <= d for c, d in spun)
               or any(max(f[0], c) < min(f[1], d) - 1e6 for c, d in spun)]
    for i, span in enumerate(flights, 1):
        pad = 0.15 * (span[1] - span[0])
        win(f"airborne ({i})" if len(flights) > 1 else "airborne",
            (span[0] + pad, span[1] - pad))

    v = Verdict(windows=wins)
    # The verdict is read off the longest flight and the longest
    # motors-on-but-still window, not off whichever happened to be last.
    stills = [w for w in wins if w.still and w.vibe_z > 1.0
              and not math.isnan(w.error)]
    if stills:
        v.control_ms2 = max(stills, key=lambda w: w.n).error
    flying = [w for w in wins if not w.still and not math.isnan(w.error)]
    if flying:
        v.bias_ms2 = max(flying, key=lambda w: w.n).error
    air = longest(flights)
    if air:
        inside = [d for d in rows["XKF2"] if air[0] <= d["TimeUS"] <= air[1]]
        if inside:
            v.ekf_bias_est = max((d.get("AZ") or 0.0) for d in inside)
        i3 = [d for d in rows["XKF3"] if air[0] <= d["TimeUS"] <= air[1]]
        if i3:
            v.innovation_m = max(abs(d.get("IPD") or 0.0) for d in i3)
        i4 = [d for d in rows["XKF4"] if air[0] <= d["TimeUS"] <= air[1]]
        if i4:
            v.test_ratio = max((d.get("SH") or 0.0) for d in i4)
        up = [d["Dist"] for d in rows["RFND"]
              if air[0] <= d["TimeUS"] <= air[1]]
        if up:
            v.climb_m = max(up) - min(up)
            v.climb_from_to = (min(up), max(up))

    if args.json:
        print(json.dumps({"windows": [w.to_log() for w in wins],
                          "bias_ms2": round(v.bias_ms2, 3),
                          "control_ms2": round(v.control_ms2, 3),
                          "ekf_bias_est": round(v.ekf_bias_est, 3),
                          "innovation_m": round(v.innovation_m, 3),
                          "test_ratio": round(v.test_ratio, 3),
                          "climb_m": round(v.climb_m, 3)}, indent=1))
        return 0

    print(f"\n  the down axis, with tilt taken out. A motionless aircraft "
          f"must read {-G:.2f}")
    print(f"  {'window':34} {'n':>4} {'down g':>8} {'error':>7} "
          f"{'VibeZ':>6} {'RPM':>7} {'thr':>5} {'rngfnd':>7}")
    for w in wins:
        print(f"  {w.name:34} {w.n:4d} {w.down_g:8.3f} {w.error:+7.3f} "
              f"{w.vibe_z:6.1f} {w.rpm:7.0f} {w.throttle:5.2f} "
              f"{w.rangefinder:7.2f}"
              + ("   <- true acceleration is zero here" if w.still
                 and w.vibe_z > 1.0 else ""))
    print()
    for line in v.lines():
        print(f"  {line}")

    if args.trace and air:
        print(f"\n  {'t':>6} {'rngfnd':>7} {'EKF alt':>8} {'baro':>8} "
              f"{'VD':>6} {'IPD':>7} {'SH':>5} {'thr':>5} {'DAlt':>6}")
        for d in rows["RFND"]:
            if not air[0] <= d["TimeUS"] <= air[1]:
                continue
            if round((d["TimeUS"] - t0) / 1e5) % 5:
                continue
            c = nearest(rows["CTUN"], d["TimeUS"]) or {}
            x1 = nearest(rows["XKF1"], d["TimeUS"]) or {}
            x3 = nearest(rows["XKF3"], d["TimeUS"]) or {}
            x4 = nearest(rows["XKF4"], d["TimeUS"]) or {}
            print(f"  {(d['TimeUS'] - t0) / 1e6:6.2f} {d['Dist']:7.2f} "
                  f"{c.get('Alt', float('nan')):8.3f} "
                  f"{c.get('BAlt', float('nan')):8.2f} "
                  f"{x1.get('VD', float('nan')):6.2f} "
                  f"{x3.get('IPD', float('nan')):7.3f} "
                  f"{x4.get('SH', float('nan')):5.2f} "
                  f"{c.get('ThO', float('nan')):5.2f} "
                  f"{c.get('DAlt', float('nan')):6.2f}")

    # The horizontal axis, by the same logic: on the floor the aircraft is
    # translating at exactly zero, so a flow residual there is the sensor.
    if rows["OF"]:
        print(f"\n  the flow sensor, on the floor where the aircraft is "
              f"not moving at all")
        print(f"  {'window':34} {'n':>5} {'X rad/s':>9} {'Y rad/s':>9}"
              f" {'as m/s at 1 m':>14}")
        worst = 0.0
        for w in wins:
            if not w.still:
                continue
            got = mean_residual(rows["OF"], w.t0, w.t1)
            if got is None:
                continue
            x, y, n = got
            worst = max(worst, abs(x), abs(y)) if w.vibe_z > 1.0 else worst
            print(f"  {w.name:34} {n:5d} {x:+9.4f} {y:+9.4f} "
                  f"{max(abs(x), abs(y)):14.3f}")
        if worst > 0.05:
            print(f"\n  With the motors running and the aircraft held still "
                  f"by the floor, the flow")
            print(f"  sensor reports up to {worst:.3f} rad/s of motion that "
                  f"is not happening. The")
            print(f"  estimator has no way to know better: at a 1 m height "
                  f"that is {worst:.2f} m/s of")
            print(f"  velocity it believes in, and the position controller "
                  f"flies the aircraft")
            print(f"  sideways to cancel it. That is a real drift with no "
                  f"error signal behind it.")

    baro = [d["Alt"] for d in rows["BARO"] if air and
            air[0] <= d["TimeUS"] <= air[1]]
    if baro and air:
        rf = max(d["Dist"] for d in rows["RFND"]
                 if air[0] <= d["TimeUS"] <= air[1])
        print(f"\n  the barometer, for the record: it read up to "
              f"{max(baro):.1f} m while the rangefinder")
        print(f"  never passed {rf:.2f} m. Indoors it is not a height "
              f"source, which is worth knowing")
        print(f"  because EK3_SRC2_POSZ and EK3_SRC3_POSZ fall back to it.")
    print("\n  Nothing was sent to any vehicle and no parameter was changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
