#!/usr/bin/env python3
"""
Check a vehicle's ArduPilot parameters against what this stack assumes.

    python tools/check_params.py vehicle.param
    python tools/check_params.py vehicle.param --forward-rf
    python tools/check_params.py vehicle.param --source-system 254

Why this exists: on 2026-11-13 a flight finding was misdiagnosed because the
stack's own comments assumed EK3_SRC1_POSZ=1 (baro) while the airframe had 2
(rangefinder). The explanation was wrong, and the only reason it surfaced was
the pilot reading the parameter file. Assumptions about the vehicle belong in
a checker that reads the vehicle, not in a comment.

Severities:
  BLOCK  the stack will misbehave or a safety claim is false
  WARN   worth knowing; may explain observed behaviour
  INFO   recorded so a log can be interpreted later

Export the file from Mission Planner (Config -> Full Parameter List -> Save)
or `param download` in MAVProxy. Comment lines and a leading '#' are ignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_agent import __version__
from drone_agent.config import Config

BLOCK, WARN, INFO = "BLOCK", "WARN", "INFO"

# EK3_SRC*_POSZ / VELXY / YAW enumerations, for readable output.
POSZ = {0: "None", 1: "Baro", 2: "RangeFinder", 3: "GPS", 4: "Beacon",
        6: "ExternalNav"}
POSXY = {0: "None", 3: "GPS", 4: "Beacon", 6: "ExternalNav"}
VELXY = {0: "None", 3: "GPS", 4: "Beacon", 5: "OpticalFlow", 6: "ExternalNav",
         7: "WheelEncoder"}
YAW = {0: "None", 1: "Compass", 2: "GPS", 3: "GPS with compass fallback",
       6: "ExternalNav", 8: "GSF"}


def load(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "," if "," in line else None
        parts = line.split(sep) if sep else line.split()
        if len(parts) < 2:
            continue
        try:
            out[parts[0].strip().upper()] = float(parts[1].strip())
        except ValueError:
            continue
    return out


class Report:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str]] = []

    def add(self, sev: str, topic: str, text: str) -> None:
        self.rows.append((sev, topic, text))

    def counts(self) -> Dict[str, int]:
        c = {BLOCK: 0, WARN: 0, INFO: 0}
        for sev, _, _ in self.rows:
            c[sev] += 1
        return c

    def show(self) -> None:
        order = {BLOCK: 0, WARN: 1, INFO: 2}
        for sev, topic, text in sorted(self.rows, key=lambda r: order[r[0]]):
            print(f"  {sev:<5} {topic:<20} {text}")


def enum(p: Dict[str, float], key: str, table: Dict[int, str]) -> str:
    v = p.get(key)
    if v is None:
        return "(absent)"
    return f"{int(v)} ({table.get(int(v), 'unknown')})"


def check(p: Dict[str, float], cfg: Config, source_system: int,
          forward_rf: bool) -> Report:
    r = Report()
    g = p.get

    # ───────────────── height source and the residual test ────────────
    posz = g("EK3_SRC1_POSZ")
    rng_nse = g("EK3_RNG_M_NSE")
    rf_filt = g("RNGFND_FILT")
    if posz == 2:
        r.add(INFO, "height source",
              "EK3_SRC1_POSZ=2 (RangeFinder): the EKF height estimate IS "
              "derived from the rangefinder, so the residual test is a "
              "rate-of-divergence detector, NOT an independent cross-check")
        if rng_nse is not None and rng_nse >= 0.4:
            r.add(WARN, "residual test",
                  f"EK3_RNG_M_NSE={rng_nse:g} m makes the EKF smooth the "
                  f"rangefinder hard, so the estimate lags the reading during "
                  f"vertical motion. This is what the vz-scaled residual "
                  f"allowance (residual_vz_coeff="
                  f"{cfg.rangefinder.residual_vz_coeff:g}) compensates for")
        if g("EK3_SRC2_POSZ") == 1:
            r.add(INFO, "independent check",
                  "EK3_SRC2_POSZ=1 (Baro) is configured but inactive — an "
                  "independent height reference exists on the vehicle and "
                  "would have to be streamed separately to use it")
    elif posz == 1:
        r.add(INFO, "height source",
              "EK3_SRC1_POSZ=1 (Baro): EKF height and the rangefinder are "
              "independent, so the residual test is a true cross-check. Baro "
              "drifts indoors and cannot see the floor")
    else:
        r.add(WARN, "height source",
              f"EK3_SRC1_POSZ={enum(p, 'EK3_SRC1_POSZ', POSZ)} — the stack "
              f"assumes Baro or RangeFinder")
    if rf_filt is not None and 0 < rf_filt <= 1.0:
        r.add(INFO, "rangefinder filter",
              f"RNGFND_FILT={rf_filt:g} Hz (time constant "
              f"~{1/(6.2832*rf_filt):.2f} s): readings in the log are already "
              f"smoothed, which flattens the spread reported by "
              f"analyze_flight")

    # ───────────────── GPS-denied source set ──────────────────────────
    r.add(INFO, "EKF sources",
          f"POSXY={enum(p, 'EK3_SRC1_POSXY', POSXY)}  "
          f"VELXY={enum(p, 'EK3_SRC1_VELXY', VELXY)}  "
          f"POSZ={enum(p, 'EK3_SRC1_POSZ', POSZ)}  "
          f"YAW={enum(p, 'EK3_SRC1_YAW', YAW)}")
    if g("EK3_SRC1_VELXY") != 5:
        r.add(BLOCK, "optical flow",
              "EK3_SRC1_VELXY is not 5 (OpticalFlow): there is no horizontal "
              "velocity source for GPS-denied flight, so GUIDED position "
              "targets have nothing to track")
    if g("FLOW_TYPE", 0) == 0:
        r.add(BLOCK, "optical flow", "FLOW_TYPE=0: no flow sensor configured")
    if g("EK3_SRC1_POSXY", 0) != 0 and g("GPS1_TYPE", 0) == 0:
        r.add(WARN, "position source",
              "EK3_SRC1_POSXY expects a source that GPS1_TYPE=0 does not "
              "provide")
    if g("GPS1_TYPE", 0) != 0 and g("EK3_SRC1_POSXY", 0) == 0:
        r.add(INFO, "GPS",
              f"GPS1_TYPE={int(g('GPS1_TYPE'))} is enabled but POSXY=None, so "
              f"GPS is not used for position. Harmless indoors; it can still "
              f"affect pre-arm messages")

    # ───────────────── the onboard safety tier ────────────────────────
    # This is the tier that works with no ground station at all, so a claim
    # about two-tier safety is false if these are off.
    # Does the autopilot have a GLOBAL position? Without GPS and without an
    # origin given by hand it has none: no home, no GLOBAL_POSITION_INT, only
    # a position relative to the EKF origin. That decides which altitude a
    # fence can actually read.
    has_global = (g("EK3_SRC1_POSXY", 0) not in (0, None)
                  or (g("AHRS_ORIGIN_LAT", 0) or 0) != 0
                  or (g("AHRS_ORIGIN_LON", 0) or 0) != 0)

    if g("FENCE_ENABLE", 0) == 0:
        r.add(WARN, "onboard fence",
              "FENCE_ENABLE=0: no autopilot-level altitude limit. The "
              "arbiter's ceiling is ground-side only and dies with the "
              "script, so above it the cover is the pilot. An altitude fence "
              "is only usable here in the above-origin frame "
              "(FENCE_ALT_MAX_TP=2); see the fence frame check when it is on")
    else:
        # FENCE_TYPE is a bitmask: bit 0 max altitude, bit 1 circle centred
        # on home, bit 2 inclusion/exclusion zones. Each check below applies
        # only to a fence that is actually switched on -- warning about a
        # 300 m radius when the circle bit is off sends someone to fix a
        # parameter that is doing nothing.
        ftype = int(g("FENCE_TYPE", 0) or 0)
        alt_on, circle_on = bool(ftype & 1), bool(ftype & 2)
        alt_max = g("FENCE_ALT_MAX")
        radius = g("FENCE_RADIUS")
        margin = g("FENCE_MARGIN", 0) or 0.0
        abort_at = cfg.envelope.alt_max_m + cfg.envelope.breach_margin_m

        if not alt_on:
            r.add(BLOCK, "onboard fence",
                  f"FENCE_ENABLE=1 but FENCE_TYPE={ftype} has no altitude "
                  f"fence (bit 0). Altitude is the one axis this airframe "
                  f"estimates well, and the failure mode measured on it")
        elif alt_max is not None:
            # WHICH altitude. FENCE_ALT_MAX_TP picks the frame: 0 AMSL,
            # 1 above home, 2 above origin, 3 above terrain. "Above home" is
            # the default and it is the wrong one for a vehicle with no
            # global position: there is no home to measure from, so the
            # fence is not reading the EKF height this stack flies on. On
            # the 2.2in airframe (15 Nov) it tripped about 10 cm off the
            # ground with the limit at 1.4 m. "Above origin" is read with
            # get_relative_position_D_origin -- the same altitude that
            # arrives here as LOCAL_POSITION_NED z.
            tp = int(g("FENCE_ALT_MAX_TP", 1) or 0)
            if tp != 2 and not has_global:
                r.add(BLOCK, "fence frame",
                      f"FENCE_ALT_MAX_TP={tp} "
                      f"({ {0: 'AMSL', 1: 'above home', 3: 'above terrain'}.get(tp, '?')}) "
                      f"needs a home, and this vehicle has no position "
                      f"source to set one from (EK3_SRC1_POSXY=0, no "
                      f"AHRS_ORIGIN). The fence is then not judged on the "
                      f"EKF height: on the 2.2in airframe "
                      f"it tripped ~10 cm off the ground with the limit at "
                      f"1.4 m. FENCE_ALT_MAX_TP=2 (above origin) reads the "
                      f"EKF's own altitude instead; or set FENCE_ENABLE=0")
            elif tp == 2:
                r.add(INFO, "fence frame",
                      "FENCE_ALT_MAX_TP=2: measured from the EKF origin, the "
                      "same altitude this stack logs. The origin is not "
                      "re-zeroed at arming, so a ground reading that has "
                      "drifted moves the fence by the same amount")
            # The fence is judged on the EKF's altitude. The arbiter's
            # breach abort and the ceiling are real heights. Comparing the
            # numbers directly was the mistake: on this airframe the EKF
            # reads below the real height, by a little in a hover and by up
            # to half a metre in a fast climb, so the fence trips with the
            # aircraft HIGHER than the parameter says.
            e = cfg.envelope
            lo = alt_max + e.ekf_under_read_static_m
            hi = alt_max + e.ekf_under_read_climb_m
            clear = e.ceiling_m - hi
            need = 0.50     # ~0.3 m to stop from 0.75 m/s, plus prop wash
            where = (f"FENCE_ALT_MAX={alt_max:g} m is in the EKF's frame. "
                     f"The EKF read {e.ekf_under_read_static_m:g}-"
                     f"{e.ekf_under_read_climb_m:g} m below the real height "
                     f"(measured on the 2.2in airframe; unmeasured on any "
                     f"other), so the fence trips with the "
                     f"aircraft at about {lo:.2f} m (slow) to {hi:.2f} m "
                     f"(fast climb)")
            if alt_max > e.alt_max_m * 4:
                r.add(WARN, "onboard fence",
                      f"FENCE_ALT_MAX={alt_max:g} m against an arbiter "
                      f"ceiling of {e.alt_max_m:g} m — the onboard backstop "
                      f"is far outside the room")
            elif clear < need:
                top = e.ceiling_m - need - e.ekf_under_read_climb_m
                r.add(BLOCK, "onboard fence",
                      f"{where}. That leaves {clear:.2f} m under a "
                      f"{e.ceiling_m:g} m ceiling, and it needs about "
                      f"{need:g} m to stop: this fence does not protect the "
                      f"ceiling. Set FENCE_ALT_MAX to {top:.1f} or lower")
            else:
                r.add(INFO, "onboard fence",
                      f"{where}, {clear:.2f} m under the {e.ceiling_m:g} m "
                      f"ceiling. Arbiter: clamps setpoints at "
                      f"{e.alt_max_m:g} m (EKF frame), aborts at "
                      f"{abort_at:g} m (real)")
            if alt_max <= e.alt_max_m:
                r.add(WARN, "fence vs clamp",
                      f"FENCE_ALT_MAX={alt_max:g} m is not above the "
                      f"arbiter's setpoint ceiling of {e.alt_max_m:g} m "
                      f"(both EKF frame): a setpoint the arbiter allows "
                      f"would trip the fence. Lower --alt-max")
            elif alt_max - e.alt_max_m < 0.40:
                r.add(INFO, "fence vs clamp",
                      f"the fence is {alt_max - e.alt_max_m:.2f} m above "
                      f"the arbiter's setpoint ceiling (both EKF frame). "
                      f"The EKF overshot a clamped setpoint by 0.20-0.38 m "
                      f"on the 2.2in airframe, so a setpoint AT the ceiling "
                      f"can "
                      f"trip it. Both end in LAND; normal flights hold well "
                      f"below")
            if alt_max < 10:
                r.add(INFO, "fence range",
                      f"FENCE_ALT_MAX={alt_max:g} m is below the parameter's "
                      f"documented 10-1000 m range. The breach test is a "
                      f"plain comparison and the pre-arm check rejects only "
                      f"negative values, but some ground stations clamp "
                      f"out-of-range entries: this file says it stuck")
            if margin >= alt_max:
                r.add(BLOCK, "fence margin",
                      f"FENCE_MARGIN={margin:g} m is not smaller than "
                      f"FENCE_ALT_MAX={alt_max:g} m. The fence's safe "
                      f"altitude is ALT_MAX - MARGIN = {alt_max - margin:g} "
                      f"m, i.e. at or below the floor, so the aircraft is "
                      f"inside the margin for the whole flight. Set "
                      f"FENCE_MARGIN well below the fence height")
        if circle_on:
            if radius and radius > cfg.envelope.fence_radius_m * 4:
                r.add(WARN, "onboard fence",
                      f"FENCE_RADIUS={radius:g} m against an arbiter fence "
                      f"of {cfg.envelope.fence_radius_m:g} m")
            if g("EK3_SRC1_POSXY", 0) == 0:
                r.add(WARN, "fence circle",
                      "the circle fence (FENCE_TYPE bit 1) is on, but "
                      "horizontal position here is optical-flow dead "
                      "reckoning (EK3_SRC1_POSXY=0). It is judged against "
                      "an estimate that drifts, so it can fire spuriously "
                      "or not at all. FENCE_TYPE=1 keeps altitude only")
        action = int(g("FENCE_ACTION", 0) or 0)
        if action in (1, 3):
            r.add(BLOCK, "fence action",
                  f"FENCE_ACTION={action} can enter RTL, which climbs to "
                  f"RTL_ALT_M={g('RTL_ALT_M', 0):g} m first and needs a "
                  f"horizontal position this airframe does not reliably "
                  f"have. Indoors use 2 (Always Land)")
        elif action == 0:
            r.add(WARN, "fence action",
                  "FENCE_ACTION=0 only reports a breach; nothing happens")

    # A failsafe that RTLs is worse than none in a room: RTL climbs first.
    for name, val, bad in (
            ("FS_GCS_ENABLE", int(g("FS_GCS_ENABLE", 0) or 0), (1, 3, 6)),
            ("FS_THR_ENABLE", int(g("FS_THR_ENABLE", 0) or 0), (1, 4, 6))):
        if val in bad:
            r.add(BLOCK, f"{name} action",
                  f"{name}={val} can enter RTL, which climbs to "
                  f"RTL_ALT_M={g('RTL_ALT_M', 0):g} m before returning. "
                  f"Indoors use the Land option "
                  f"({'5' if name == 'FS_GCS_ENABLE' else '3'})")

    # Is the RC itself arriving as MAVLink? With ELRS in MAVLink mode there
    # is no RC serial port (protocol 23): the receiver "acts as a GCS" with
    # its own system id, 255 by default, and the sticks arrive as MAVLink
    # from that id. Two consequences, both learned the hard way on 15 Nov:
    #
    #   The autopilot takes RC input only from MAV_GCS_SYSID. Point that at
    #   this stack (254) and it stops listening to the receiver: "RC not
    #   found".
    #
    #   The GCS failsafe watches MAV_GCS_SYSID too, so it watches the
    #   RECEIVER, which keeps talking whether or not this script is alive.
    #   It cannot detect a dead script on this radio. It fires when the
    #   radio drops, which FS_THR_ENABLE already covers.
    rc_serial = any(int(g(f"SERIAL{i}_PROTOCOL", -1) or -1) == 23
                    for i in range(0, 10))
    mav_radio = any(int(g(f"SERIAL{i}_PROTOCOL", -1) or -1) in (1, 2)
                    and int(g(f"SERIAL{i}_BAUD", 0) or 0) == 460
                    for i in range(1, 10))
    rc_over_mavlink = (not rc_serial and mav_radio
                       and int(g("RSSI_TYPE", 0) or 0) == 5)
    rc_sysid = cfg.link.rc_source_system
    gcs_sysid = g("MAV_GCS_SYSID")
    gcs_hi = int(g("MAV_GCS_SYSID_HI", 0) or 0)
    fs_gcs = int(g("FS_GCS_ENABLE", 0) or 0)

    if rc_over_mavlink:
        r.add(INFO, "RC path",
              f"no RC serial port and a MAVLink radio at 460800 with "
              f"RSSI_TYPE=5: the sticks arrive as MAVLink from the ELRS "
              f"receiver, system {rc_sysid}")
        if gcs_sysid is not None:
            lo = int(gcs_sysid)
            accepted = (lo == rc_sysid
                        or (gcs_hi and lo <= rc_sysid <= gcs_hi))
            if not accepted:
                r.add(BLOCK, "GCS sysid",
                      f"MAV_GCS_SYSID={lo} but the RC arrives from system "
                      f"{rc_sysid}. The autopilot accepts RC input only from "
                      f"its GCS system id, so it will ignore the sticks: "
                      f"'RC not found'. Set MAV_GCS_SYSID={rc_sysid}. (An "
                      f"earlier version of this tool told you to set 254; "
                      f"that was wrong for this radio)")
        if fs_gcs == 0:
            r.add(INFO, "GCS failsafe",
                  "FS_GCS_ENABLE=0. On this radio it could not detect a dead "
                  "script anyway: the autopilot's GCS is the ELRS receiver, "
                  "which keeps talking while the script is dead. A script "
                  "that dies leaves the aircraft holding its last GUIDED "
                  "target, and the cover for that is the pilot")
        else:
            r.add(INFO, "GCS failsafe",
                  f"FS_GCS_ENABLE={fs_gcs} watches the ELRS receiver (system "
                  f"{rc_sysid}), not this stack. It fires when the radio "
                  f"drops, as FS_THR_ENABLE does, and not when the script "
                  f"dies")
    else:
        if fs_gcs == 0:
            r.add(BLOCK, "GCS failsafe",
                  "FS_GCS_ENABLE=0: if the ground station or the script dies "
                  "while the radio is still up, the autopilot does nothing "
                  "and holds the last GUIDED target. RC failsafe only covers "
                  "losing the radio itself")
        if gcs_sysid is not None and int(gcs_sysid) != source_system:
            sev = BLOCK if fs_gcs != 0 else WARN
            r.add(sev, "GCS sysid",
                  f"MAV_GCS_SYSID={int(gcs_sysid)} but this stack sends "
                  f"source_system={source_system}. The autopilot will not "
                  f"count our heartbeats as the GCS, so enabling "
                  f"FS_GCS_ENABLE would failsafe immediately. Set "
                  f"MAV_GCS_SYSID={source_system}")
    if g("FS_THR_ENABLE", 0) == 0:
        r.add(BLOCK, "RC failsafe",
              "FS_THR_ENABLE=0: losing the ELRS link triggers nothing. With "
              "RC and MAVLink on one radio this is the main safety net")
    else:
        r.add(INFO, "RC failsafe",
              "FS_THR_ENABLE on: losing the ELRS link triggers RC failsafe, "
              "which is the net for link loss since RC and MAVLink share the "
              "radio")

    # ───────────────── GUIDED behaviour ───────────────────────────────
    gt = g("GUID_TIMEOUT")
    if gt is not None:
        r.add(INFO, "GUIDED timeout",
              f"GUID_TIMEOUT={gt:g} s applies to VELOCITY and attitude "
              f"targets, not position targets. Harmless now; a velocity-based "
              f"follow skill must re-send faster than this")
    spd_up, acc_z = g("WP_SPD_UP"), g("WP_ACC_Z")
    if spd_up is not None and spd_up > cfg.envelope.max_speed_mps * 2:
        r.add(WARN, "vertical rate",
              f"WP_SPD_UP={spd_up:g} m/s bounds GUIDED climbs, against an "
              f"arbiter speed limit of {cfg.envelope.max_speed_mps:g} m/s. "
              f"Outdoor-scale limits are a likely contributor to the altitude "
              f"overshoot past a clamped setpoint")
    if acc_z is not None and acc_z >= 1.0:
        r.add(INFO, "vertical accel",
              f"WP_ACC_Z={acc_z:g} m/s^2 — lower it with WP_SPD_UP to reduce "
              f"overshoot on short indoor climbs")
    hover = g("MOT_THST_HOVER")
    if hover is not None and hover < 0.25:
        r.add(WARN, "hover throttle",
              f"MOT_THST_HOVER={hover:.3f} is very low (typical 0.30-0.50): "
              f"the airframe is overpowered for its mass, so small throttle "
              f"changes produce large vertical accelerations and altitude "
              f"overshoot is expected")

    # ───────────────── rangefinders ───────────────────────────────────
    n_rf = sum(1 for i in list(range(1, 10)) + ["A"]
               if g(f"RNGFND{i}_TYPE", 0) != 0)
    want = cfg.link.rf_streams_expected
    if n_rf != want:
        r.add(WARN, "rangefinder count",
              f"{n_rf} rangefinder backend(s) configured but "
              f"link.rf_streams_expected={want}. One DISTANCE_SENSOR stream "
              f"per backend, so the downlink budget is off by "
              f"{abs(n_rf - want)} x {cfg.link.distance_sensor_hz:g} Hz")
    # Every configured backend, with its orientation. With two rangefinders
    # on the airframe this stops being bookkeeping: ArduPilot picks the
    # height source by ORIENTATION, so two sensors both claiming "down"
    # makes the EKF height ambiguous -- and a FORWARD sensor mislabelled as
    # down feeds the altitude controller its distance to the nearest wall.
    # The symptom is an aircraft that will not climb: it believes it is
    # already high.
    downward = []
    for i in list(range(1, 10)) + ["A"]:
        t = g(f"RNGFND{i}_TYPE", 0)
        if not t:
            continue
        o = g(f"RNGFND{i}_ORIENT")
        oi = int(o) if o is not None else -1
        name = {25: "down", 0: "forward", 24: "up"}.get(oi, f"orient {oi}")
        r.add(INFO, f"RNGFND{i}",
              f"TYPE={int(t)}, ORIENT={oi} ({name}), "
              f"MIN={g(f'RNGFND{i}_MIN', 0):.2f} m, "
              f"MAX={g(f'RNGFND{i}_MAX', 0):g} m, "
              f"GNDCLR={g(f'RNGFND{i}_GNDCLR', 0):.3f} m")
        if oi == 25:
            downward.append(f"RNGFND{i}")
    if len(downward) > 1:
        r.add(BLOCK, "two downward RF",
              f"{' and '.join(downward)} are BOTH oriented down (25). "
              f"ArduPilot selects the height source by orientation, so with "
              f"EK3_SRC1_POSZ=2 the altitude estimate becomes ambiguous. If "
              f"one of these is the forward MT-15, set its ORIENT to 0 — a "
              f"forward sensor read as a height source reports the distance "
              f"to the nearest wall as altitude, and the aircraft will not "
              f"climb because it believes it is already there")
    if not downward:
        r.add(BLOCK, "no downward RF",
              f"no rangefinder is oriented down (25), but EK3_SRC1_POSZ=2 "
              f"asks the EKF to take height from one")

    orient = g("RNGFND1_ORIENT")
    if orient is not None and int(orient) != cfg.rangefinder.orientation:
        r.add(BLOCK, "rangefinder orient",
              f"RNGFND1_ORIENT={int(orient)} but the stack routes "
              f"orientation {cfg.rangefinder.orientation} as the downward "
              f"sensor, so its readings would be discarded")
    rmax = g("RNGFND1_MAX")
    if rmax is not None:
        if rmax < cfg.envelope.alt_max_m * 1.5:
            r.add(WARN, "rangefinder range",
                  f"RNGFND1_MAX={rmax:g} m leaves little headroom above the "
                  f"{cfg.envelope.alt_max_m:g} m arbiter ceiling; readings "
                  f"beyond it go unhealthy and the EKF loses its height "
                  f"source")

    if forward_rf:
        if g("RNGFND2_TYPE", 0) == 0:
            r.add(BLOCK, "forward rangefinder",
                  "--forward-rf was passed but RNGFND2_TYPE=0: no forward "
                  "sensor is configured on the vehicle")
        elif int(g("RNGFND2_ORIENT", -1)) != cfg.forward_rf.orientation:
            r.add(BLOCK, "forward rangefinder",
                  f"RNGFND2_ORIENT={int(g('RNGFND2_ORIENT', -1))} but the "
                  f"stack expects {cfg.forward_rf.orientation} (forward)")

    # ───────────────── the MAVLink port ───────────────────────────────
    port = None
    for i in range(8):
        if g(f"SERIAL{i}_PROTOCOL") == 2 and g(f"SERIAL{i}_BAUD") == 460:
            port = i
            break
    if port is None:
        r.add(WARN, "ELRS port",
              "no SERIAL port found with PROTOCOL=2 and BAUD=460 — ELRS "
              "MAVLink mode expects MAVLink2 at 460800")
    else:
        r.add(INFO, "ELRS port",
              f"SERIAL{port} is MAVLink2 at 460800, as ELRS MAVLink mode "
              f"expects")
    if g("RSSI_TYPE") is not None and int(g("RSSI_TYPE")) != 5:
        r.add(INFO, "RSSI",
              f"RSSI_TYPE={int(g('RSSI_TYPE'))}; ELRS MAVLink mode documents "
              f"5")

    # ───────────────── the handover path ──────────────────────────────
    dd = g("DISARM_DELAY")
    if dd is not None and dd > 0:
        r.add(INFO, "DISARM_DELAY",
              f"DISARM_DELAY={dd:g} s: in --handover the pilot must raise the "
              f"throttle within this long or the autopilot auto-disarms on "
              f"the ground")

    return r


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("param_file")
    ap.add_argument("--forward-rf", action="store_true",
                    help="also check the forward MT-15")
    ap.add_argument("--source-system", type=int,
                    help="override the stack's MAVLink source system")
    ap.add_argument("--alt-max", type=float,
                    help="check against this arbiter ceiling")
    ap.add_argument("--ceiling", type=float,
                    help="height of the room, m. The altitude fence is "
                         "checked against it in real height, not in the "
                         "EKF's frame")
    ap.add_argument("--version", action="version",
                    version=f"drone_agent {__version__}")
    args = ap.parse_args()

    cfg = Config()
    if args.forward_rf:
        cfg.forward_rf.enabled = True
        cfg.link.rf_streams_expected = 2
    if args.alt_max is not None:
        cfg.envelope.alt_max_m = args.alt_max
    if args.ceiling is not None:
        cfg.envelope.ceiling_m = args.ceiling
    src = (args.source_system if args.source_system is not None
           else cfg.link.source_system)

    p = load(args.param_file)
    print(f"drone_agent {__version__}")
    print(f"{args.param_file}: {len(p)} parameters")
    print(f"checked against alt_max={cfg.envelope.alt_max_m:g} m, "
          f"ceiling={cfg.envelope.ceiling_m:g} m, "
          f"fence={cfg.envelope.fence_radius_m:g} m, source_system={src}\n")

    r = check(p, cfg, src, args.forward_rf)
    r.show()
    c = r.counts()
    print(f"\n{c[BLOCK]} BLOCK, {c[WARN]} WARN, {c[INFO]} INFO")
    if c[BLOCK]:
        print("BLOCK items mean the stack will misbehave or a safety claim in "
              "the README is not true of this vehicle.")
    return 1 if c[BLOCK] else 0


if __name__ == "__main__":
    sys.exit(main())
