#!/usr/bin/env python3
"""
Offline check: run the whole skill + arbiter + recorder stack against a
simulated vehicle, with no MAVLink, no SITL and no aircraft.

This is the regression test for the parts that are easy to break and
expensive to debug in the air:

  * the arbiter actually clamps altitude, fence and step
  * a skill that reaches past the arbiter raises BypassError
  * rotate_search goes genuinely quiet during a yaw leg (the constraint that
    keeps CONDITION_YAW from being cancelled)
  * the rangefinder router separates the two DISTANCE_SENSOR streams
  * the recorder produces a file that loads cleanly

Run it after any change to the stack and before any flight:

    python tools/offline_check.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_agent import __version__
from drone_agent.config import Config
from drone_agent.interfaces.vehicle import BypassError
from drone_agent.recording.recorder import FlightRecorder
from drone_agent.safety.arbiter import (Arbiter, AbortRequested, RequestKind,
                                        SetpointRequest)
from drone_agent.skills import build
from drone_agent.world.rangefinder import RangefinderRouter
from drone_agent.world.state import Pose, WorldState

PASS, FAIL = "  PASS", "  FAIL"
failures: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{PASS if ok else FAIL}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


# ───────────────────────── fake vehicle ──────────────────────────

class FakeVehicle:
    """
    Satisfies the actuation surface the arbiter uses. Flies a trivial
    first-order model: position chases the last target, yaw slews at the
    commanded rate.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._token: Optional[object] = None
        self._minted = False
        self.origin = None

        self.x = self.y = 0.0
        self.z = -0.80
        self.yaw = 0.0
        self.tx, self.ty, self.tz = 0.0, 0.0, -0.80
        self.yaw_target: Optional[float] = None
        self.yaw_rate = 0.0

        self.sent_positions: List[tuple] = []
        self.sent_yaws: List[tuple] = []
        self.t = 0.0
        self.armed = True

    # actuation surface -------------------------------------------------
    def mint_actuation_token(self) -> object:
        if self._minted:
            raise BypassError("token already minted")
        self._minted = True
        self._token = object()
        return self._token

    def _check(self, token: Any, what: str) -> None:
        if self._token is None or token is not self._token:
            raise BypassError(f"{what} without the arbiter's token")

    def send_position_target(self, token: Any, x: float, y: float, z: float,
                             yaw_rad: Optional[float] = None) -> None:
        self._check(token, "send_position_target")
        self.tx, self.ty, self.tz = x, y, z
        if yaw_rad is not None:
            self.yaw_target = yaw_rad
            self.yaw_rate = 60.0
        self.sent_positions.append((round(x, 3), round(y, 3), round(z, 3)))

    def send_condition_yaw(self, token: Any, heading_deg: float,
                           rate_dps: float, direction: int) -> int:
        self._check(token, "send_condition_yaw")
        self.yaw_target = math.radians(heading_deg)
        self.yaw_rate = rate_dps
        self.sent_yaws.append((round(heading_deg, 1), rate_dps, direction))
        return 0

    # simulation --------------------------------------------------------
    def step(self, dt: float) -> WorldState:
        k = min(1.0, dt * 2.0)
        self.x += (self.tx - self.x) * k
        self.y += (self.ty - self.y) * k
        self.z += (self.tz - self.z) * k
        if self.yaw_target is not None:
            err = (self.yaw_target - self.yaw + math.pi) % (2 * math.pi) - math.pi
            step = math.radians(self.yaw_rate) * dt
            self.yaw += math.copysign(min(abs(err), step), err)
            self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

        self.t += dt
        return WorldState(
            t_mono=self.t, t_wall=time.time(),
            pose=Pose(self.x, self.y, self.z, self.yaw, vz=0.0,
                      pitch_deg=-2.0, roll_deg=0.0),
            pose_age_s=0.05, armed=self.armed, mode="GUIDED", mode_num=4,
            landed_state=2, landed_state_age_s=0.5,
            rngfnd_down_m=-self.z, rngfnd_down_trusted=True,
            rngfnd_down_raw_m=-self.z,
            flow_quality=90, link_up=True, link_age_s=0.2,
            origin=self.origin)


def drive(skill: Any, vehicle: FakeVehicle, arbiter: Arbiter,
          recorder: Optional[FlightRecorder] = None,
          max_ticks: int = 4000, dt: float = 0.1) -> Dict[str, Any]:
    """Run a skill to completion against the fake vehicle."""
    world = vehicle.step(dt)
    skill.start(world)
    quiet_during_slew = True
    interventions = 0

    for _ in range(max_ticks):
        world = vehicle.step(dt)
        status = skill.tick(world, dt)
        req = skill.request

        # A POSITION target during a leg is the thing that cancels the slew.
        # The YAW command itself is emitted on the tick that enters SLEWING,
        # which is correct and expected.
        if getattr(skill, "phase", None) is not None:
            if (skill.phase.value == "slewing"
                    and req.kind is RequestKind.POSITION):
                quiet_during_slew = False

        try:
            decision = arbiter.submit(req, world)
        except AbortRequested as e:
            return {"status": "aborted", "reason": e.reason,
                    "quiet_during_slew": quiet_during_slew,
                    "interventions": interventions}
        if decision.intervened:
            interventions += 1
        if recorder is not None:
            recorder.tick(world, skill.name, status.value, req.to_log(),
                          decision.applied.to_log(), decision.actions,
                          decision.blocked)

        if status.value in ("succeeded", "failed", "aborted"):
            return {"status": status.value, "result": skill.result(world),
                    "quiet_during_slew": quiet_during_slew,
                    "interventions": interventions}
    return {"status": "timeout", "quiet_during_slew": quiet_during_slew,
            "interventions": interventions}


# ─────────────────────────── the checks ──────────────────────────

def test_bypass_guard() -> None:
    print("\nactuation gating")
    cfg = Config()
    v = FakeVehicle(cfg)
    Arbiter(v, cfg)                      # mints the token
    try:
        v.send_position_target(object(), 0, 0, -1)
        check("direct actuation is refused", False, "no BypassError raised")
    except BypassError:
        check("direct actuation is refused", True)

    v2 = FakeVehicle(cfg)
    Arbiter(v2, cfg)
    try:
        Arbiter(v2, cfg)
        check("a second arbiter is refused", False)
    except BypassError:
        check("a second arbiter is refused", True)


def test_drift_from_path() -> None:
    """
    The drift abort measures from the commanded path, not the takeoff
    point. On 2026-11-17 an approach from 3.65 m had flown 1.49 m exactly as
    commanded when the origin-based check fired "drifted 1.50 m from the
    hold point"; across the 20 flights of that evening the path-based
    distance never passed 0.23 m.
    """
    print("\ndrift from the commanded path")
    from drone_agent.safety.arbiter import AbortRequested
    cfg = Config()
    v = FakeVehicle(cfg)
    arb = Arbiter(v, cfg)
    arb.set_origin(0.0, 0.0, -0.8)
    world = v.step(0.1)
    # Fly 2.4 m forward in 0.3 m targets, as approach and move do.
    aborted = None
    try:
        for k in range(1, 9):
            arb.submit(SetpointRequest.position(0.3 * k, 0.0, -0.8,
                                                immediate=True), world)
            for _ in range(15):
                world = v.step(0.1)
                arb.submit(SetpointRequest.position(0.3 * k, 0.0, -0.8),
                           world)
    except AbortRequested as e:
        aborted = str(e)
    check("a commanded 2.4 m flight does not trip it",
          aborted is None and world.pose.x > 2.3,
          f"aircraft at {world.pose.x:.2f} m from the origin, "
          f"peak distance from the path {arb.peak_drift_m:.2f} m"
          + (f"; aborted: {aborted}" if aborted else ""))
    # Now the aircraft slides sideways while told to hold.
    try:
        for _ in range(40):
            world = v.step(0.1)
            arb.submit(SetpointRequest.position(2.4, 0.0, -0.8), world)
            # 0.5 m/s sideways, and the aircraft ignores the target it is
            # sent: a runaway the estimator can see.
            v.y += 0.05
            v.ty = v.y
        aborted = None
    except AbortRequested as e:
        aborted = str(e)
    check("but sliding off it does",
          aborted is not None and "drifted" in aborted,
          aborted or f"no abort; peak {arb.peak_drift_m:.2f} m")


def test_envelope_clamps() -> None:
    print("\nenvelope clamps")
    cfg = Config()
    v = FakeVehicle(cfg)
    arb = Arbiter(v, cfg)
    arb.set_origin(0.0, 0.0, -0.8)
    world = v.step(0.1)

    # Relative to the ceiling, not a fixed height: a hardcoded 3.0 m was a
    # mild overshoot of a 2.0 m ceiling (clamped) but is 2.5x a 1.2 m one,
    # which is past the gross-violation factor and correctly REJECTED.
    d = arb.submit(SetpointRequest.position(
        0, 0, -(cfg.envelope.alt_max_m * 1.5), immediate=True), world)
    check("altitude clamped to the ceiling",
          abs(-d.applied.z - cfg.envelope.alt_max_m) < 1e-6,
          f"-> {-d.applied.z:.2f} m, actions={d.actions}")

    d = arb.submit(SetpointRequest.position(0, 0, -0.05,
                                            immediate=True), world)
    check("altitude clamped to the floor",
          abs(-d.applied.z - cfg.envelope.alt_min_m) < 1e-6,
          f"-> {-d.applied.z:.2f} m")

    d = arb.submit(SetpointRequest.position(50.0, 0.0, -0.8,
                                            immediate=True), world)
    check("a 50 m jump is REJECTED, not clamped to the fence",
          d.blocked and any("gross" in a for a in d.actions),
          f"actions={d.actions}")

    # Just past the step limit: small overshoot, so clamp and fly it.
    near = cfg.envelope.max_step_m * 1.3
    d = arb.submit(SetpointRequest.position(near, 0.0, -0.8,
                                            immediate=True), world)
    step = math.hypot(d.applied.x - world.pose.x, d.applied.y - world.pose.y)
    check("a small step overshoot IS clamped and passed",
          not d.blocked and step <= cfg.envelope.max_step_m + 1e-6
          and any("step" in a for a in d.actions),
          f"-> step={step:.2f} m, actions={d.actions}")

    # Just past the ceiling: clamped, not rejected.
    d = arb.submit(SetpointRequest.position(
        0.0, 0.0, -(cfg.envelope.alt_max_m * 1.4), immediate=True), world)
    check("a small altitude overshoot IS clamped and passed",
          not d.blocked
          and abs(-d.applied.z - cfg.envelope.alt_max_m) < 1e-6,
          f"-> {-d.applied.z:.2f} m, actions={d.actions}")

    d = arb.submit(SetpointRequest.position(0.0, 0.0, -10.0,
                                            immediate=True), world)
    check("10 m altitude is REJECTED, not clamped to the ceiling",
          d.blocked and any("gross" in a for a in d.actions),
          f"actions={d.actions}")

    d = arb.submit(SetpointRequest.yaw(90.0, 500.0, 1), world)
    check("yaw rate clamped",
          abs(d.applied.yaw_rate_dps) <= cfg.envelope.max_yaw_rate_dps,
          f"-> {d.applied.yaw_rate_dps:.0f} dps")

    check("interventions counted",
          arb.counters["intervened"] >= 4,
          f"{arb.counters['intervened']} of {arb.counters['submitted']}")


def test_estop_and_staleness() -> None:
    print("\nestop and staleness")
    cfg = Config()
    v = FakeVehicle(cfg)
    arb = Arbiter(v, cfg)
    world = v.step(0.1)

    arb.estop("test")
    try:
        arb.submit(SetpointRequest.position(0, 0, -0.8), world)
        check("estop raises an abort", False)
    except AbortRequested as e:
        check("estop raises an abort", "e-stop" in e.reason)
    arb.reset_estop()

    stale = v.step(0.1)
    stale.pose_age_s = 5.0
    d = arb.submit(SetpointRequest.position(0, 0, -0.8), stale)
    check("stale pose blocks inside the grace window", d.blocked,
          f"actions={d.actions}")

    # The arbiter's grace timer runs on the WORLD clock, so wind it back in
    # world time, not wall time.
    arb._t_pose_stale_since = stale.t_mono - 10.0
    try:
        arb.submit(SetpointRequest.position(0, 0, -0.8), stale)
        check("stale pose aborts past the grace window", False)
    except AbortRequested as e:
        check("stale pose aborts past the grace window", "stale" in e.reason)


def test_state_breach() -> None:
    """A clamped setpoint bounds the command, not the aircraft. The breach
    detector measures where the vehicle actually went."""
    print("\nstate-based envelope")
    cfg = Config()
    cfg.envelope.alt_max_m = 1.2
    v = FakeVehicle(cfg)
    arb = Arbiter(v, cfg)
    arb.set_origin(0.0, 0.0, -0.8)

    # Park the aircraft above the ceiling without ever commanding it there.
    # tz too, or the model chases its old target back down and the peak in
    # the commanded frame is lower than the one actually flown.
    v.z = v.tz = -1.9
    breached = 0
    aborted = None
    for _ in range(40):
        w = v.step(0.1)
        w.rngfnd_down_m = 1.9
        w.rngfnd_down_trusted = True
        try:
            d = arb.submit(SetpointRequest.idle(), w)
        except AbortRequested as e:
            aborted = e.reason
            break
        if any("breach" in a for a in d.actions):
            breached += 1

    check("an aircraft outside the envelope is detected",
          breached > 0 or aborted, f"{breached} breaching ticks")
    check("a sustained breach aborts",
          aborted is not None and "outside the envelope" in (aborted or ""),
          str(aborted)[:60])
    check("peak excursion recorded in both frames",
          arb.peak_alt_breach_m > 0.6 and arb.peak_alt_breach_ekf_m > 0.6,
          f"best-est +{arb.peak_alt_breach_m:.2f} m, "
          f"EKF +{arb.peak_alt_breach_ekf_m:.2f} m")

    # Normal controller overshoot must not abort.
    cfg2 = Config()
    cfg2.envelope.alt_max_m = 1.2
    v2 = FakeVehicle(cfg2)
    arb2 = Arbiter(v2, cfg2)
    arb2.set_origin(0.0, 0.0, -0.8)
    v2.z = v2.tz = -(1.2 + cfg2.envelope.breach_margin_m * 0.7)
    ok = True
    for _ in range(40):
        w = v2.step(0.1)
        w.rngfnd_down_m = -v2.z
        w.rngfnd_down_trusted = True
        try:
            arb2.submit(SetpointRequest.idle(), w)
        except AbortRequested:
            ok = False
            break
    check("overshoot inside the margin does not abort", ok,
          f"peak +{arb2.peak_alt_breach_m:.2f} m with margin "
          f"{cfg2.envelope.breach_margin_m:.2f} m")


def test_hover() -> None:
    print("\nhover")
    cfg = Config()
    v = FakeVehicle(cfg)
    arb = Arbiter(v, cfg)
    arb.set_origin(0.0, 0.0, -0.8)
    out = drive(build("hover", cfg, duration_s=3.0), v, arb)
    check("hover succeeds on an explicit terminal condition",
          out["status"] == "succeeded",
          str(out.get("result").reason if out.get("result") else out))
    check("hover rate-limited its position targets",
          0 < len(v.sent_positions) <= 4,
          f"{len(v.sent_positions)} sent over ~3 s at "
          f"{cfg.skills.setpoint_period_s:.0f} s period")


def test_rotate() -> None:
    print("\nrotate_search")
    cfg = Config()
    cfg.skills.leg_settle_s = 0.5
    cfg.skills.yaw_settle_s = 0.5
    v = FakeVehicle(cfg)
    arb = Arbiter(v, cfg)
    arb.set_origin(0.0, 0.0, -0.8)

    skill = build("rotate_search", cfg, total_deg=360.0, leg_deg=45.0,
                  rate_dps=40.0)
    out = drive(skill, v, arb, dt=0.1, max_ticks=6000)
    res = out.get("result")
    check("full revolution succeeds", out["status"] == "succeeded",
          str(res.reason if res else out))
    check("eight legs flown", len(v.sent_yaws) == 8,
          f"{len(v.sent_yaws)} CONDITION_YAW commands")
    check("uplink silent during every slew", out["quiet_during_slew"],
          "a position target mid-leg would cancel the slew")
    if res:
        check("metrics recorded", "max_excursion_m" in res.metrics,
              str(res.metrics.get("legs_completed")))


def test_rangefinder_router() -> None:
    print("\nrangefinder routing and trust")
    cfg = Config()
    cfg.forward_rf.enabled = True
    r = RangefinderRouter(cfg.rangefinder, cfg.forward_rf)

    # Interleave the two streams the way the vehicle actually does: a
    # downward sensor at 0.70 m and a forward sensor sweeping a wall.
    for i, fwd in enumerate([1.78, 3.66, 4.14, 3.04, 2.51]):
        r.ingest(0, 25, 0.70 + 0.01 * (i % 2), 0.70, 0.0, -2.0)
        r.ingest(1, 0, fwd, 0.70, 0.0, -2.0)

    check("downward sensor unaffected by the forward stream",
          r.down.trusted and r.down.value_m is not None
          and abs(r.down.value_m - 0.70) < 0.05,
          f"down={r.down.value_m}, trusted={r.down.trusted}")
    check("forward sensor tracked separately",
          r.fwd.value_m is not None and r.fwd.value_m > 2.0,
          f"fwd={r.fwd.value_m}")
    check("both streams catalogued", len(r.streams) == 2,
          f"{sorted(k for k in r.streams)}")

    # The incident residuals: a steady +0.25 m offset, then a hard drop.
    r2 = RangefinderRouter(cfg.rangefinder, cfg.forward_rf)
    for _ in range(10):
        r2.ingest(0, 25, 0.95, 0.70, 0.0, -2.0)
    before = r2.down.trusted
    for _ in range(3):
        r2.ingest(0, 25, 0.24, 0.70, 0.0, -2.0)
    check("residual test catches the incident signature",
          before and not r2.down.trusted,
          f"learned offset {r2.down.offset:+.2f} m")

    # Floor gate: nose-down 20 deg at 1.5 m puts the floor at ~4.4 m.
    r3 = RangefinderRouter(cfg.rangefinder, cfg.forward_rf)
    r3.ingest(1, 0, 4.39, 1.50, 0.0, -20.0)
    check("floor return rejected by the pitch gate",
          r3.fwd.value_m is None and r3.fwd_gated == 1,
          f"gated={r3.fwd_gated}")
    r3.ingest(1, 0, 1.20, 1.50, 0.0, -20.0)
    check("a real obstacle passes the pitch gate",
          r3.fwd.value_m is not None and abs(r3.fwd.value_m - 1.20) < 1e-6)


def test_recorder() -> None:
    print("\nrecorder")
    cfg = Config()
    with tempfile.TemporaryDirectory() as d:
        rec = FlightRecorder(log_dir=d, session_label="offline")
        rec.meta(cfg.to_dict(), label="offline check")
        v = FakeVehicle(cfg)
        arb = Arbiter(v, cfg)
        arb.set_origin(0.0, 0.0, -0.8)
        out = drive(build("hover", cfg, duration_s=2.0), v, arb, recorder=rec)
        if out.get("result"):
            rec.skill_result(out["result"])
        rec.summary({"outcome": "offline"})
        rec.close()

        rows = [json.loads(l) for l in open(rec.path)]
        kinds = {r["type"] for r in rows}
        ticks = [r for r in rows if r["type"] == "tick"]
        check("all record types present",
              {"meta", "tick", "skill", "summary"} <= kinds, str(sorted(kinds)))
        check("ticks carry requested and applied separately",
              all("setpoint_requested" in t and "setpoint_applied" in t
                  for t in ticks), f"{len(ticks)} ticks")
        check("record stamps increase monotonically",
              all(rows[i]["t_mono"] <= rows[i + 1]["t_mono"]
                  for i in range(len(rows) - 1)))
        check("sample time kept distinct from write time",
              all("t_sample_mono" in t for t in ticks)
              and all(t["t_sample_mono"] != t["t_mono"] for t in ticks[:1]),
              "t_mono = write, t_sample_mono = sample")


def test_planner_view() -> None:
    print("\nplanner interface")
    cfg = Config()
    v = FakeVehicle(cfg)
    world = v.step(0.1)
    view = world.to_planner_view()
    check("planner view is metric and compact",
          "altitude_m" in view and "pose" in view and "objects" in view,
          str(sorted(view)))
    check("no trust internals leak to the planner",
          not any("trusted" in k for k in view))

    res = build("hover", cfg, duration_s=1.0).result(world)
    pv = res.to_planner_view()
    check("skill result has a planner view",
          {"skill", "status", "elapsed_s"} <= set(pv), str(sorted(pv)))


def test_perception() -> None:
    """Geometry, tracking, range provenance and staleness — all without a
    camera, which is the only way any of it gets regression-tested."""
    print("\nperception")
    from drone_agent.perception import (Boresight, CameraIntrinsics,
                                        Detection, MockDetector,
                                        PerceptionRunner, RangeEstimator,
                                        RangeSource)

    # --- intrinsics: the error an assumed FOV produces ----------------
    # A real FPV lens is an equidistant fisheye, not a distortion-free
    # pinhole: the pinhole+radial model is not even invertible past ~100 deg.
    real = CameraIntrinsics.from_hfov(150.0, model="fisheye")
    assumed = CameraIntrinsics.from_hfov(90.0, model="pinhole")
    err = abs(real.angles(0.95, 0.5)[0] - assumed.angles(0.95, 0.5)[0])
    check("an assumed FOV misplaces an edge target badly", err > 15.0,
          f"{err:.1f} deg at 0.95 frame width "
          f"(150 deg fisheye vs assumed 90 deg pinhole)")
    check("the error grows toward the frame edge",
          abs(real.angles(0.7, 0.5)[0] - assumed.angles(0.7, 0.5)[0]) < err,
          f"u=0.70: "
          f"{abs(real.angles(0.7, 0.5)[0] - assumed.angles(0.7, 0.5)[0]):.1f}"
          f" deg, u=0.95: {err:.1f} deg")
    check("intrinsics report whether they are calibrated",
          not real.is_calibrated
          and CameraIntrinsics(fx=0.5, note="checkerboard").is_calibrated)
    check("centre of frame is zero bearing",
          abs(real.angles(0.5, 0.5)[0]) < 1e-9)
    check("a fisheye mapping is monotonic across the frame",
          all(real.angles(u / 100.0, 0.5)[0]
              < real.angles((u + 2) / 100.0, 0.5)[0]
              for u in range(50, 98, 2)),
          "a model fitted outside its validity range rises then falls")

    # --- cross-validate the undistortion against OpenCV ---------------
    # The geometry module deliberately has no cv2 dependency, so its own
    # solvers are checked against cv2's whenever cv2 happens to be present.
    try:
        import cv2 as _cv2
        import numpy as _np
    except ImportError:
        print("        (cv2 absent: skipping the OpenCV cross-check)")
    else:
        W, H = 720, 576
        fx, fy = 0.40, 0.53
        K = _np.array([[fx * W, 0, 0.503 * W], [0, fy * H, 0.497 * H],
                       [0, 0, 1]], _np.float64)
        D = _np.array([-0.18, 0.03, 0, 0], _np.float64)
        pin = CameraIntrinsics(fx=fx, fy=fy, cx=0.503, cy=0.497, k1=-0.18,
                               k2=0.03, model="pinhole", width_px=W,
                               height_px=H, note="test")
        worst = 0.0
        for un in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
            ideal = _cv2.undistortPoints(
                _np.array([[[un * W, 0.5 * H]]], _np.float64), K, D
            ).reshape(2)
            worst = max(worst, abs(_np.degrees(_np.arctan(ideal[0]))
                                   - pin.angles(un, 0.5)[0]))
        check("pinhole undistortion matches cv2.undistortPoints",
              worst < 0.1, f"worst {worst:.4f} deg "
                           f"(a single Newton step was 5.2 deg out)")

        f = 0.5 / _np.radians(75.0)
        Kf = _np.array([[f * W, 0, 0.5 * W], [0, f * W, 0.5 * H],
                        [0, 0, 1]], _np.float64)
        Df = _np.array([[0.02], [-0.004], [0.001], [0.0]], _np.float64)
        fish = CameraIntrinsics(fx=f, fy=f * W / H, cx=0.5, cy=0.5, k1=0.02,
                                k2=-0.004, k3=0.001, model="fisheye",
                                width_px=W, height_px=H, note="test")
        worst = 0.0
        for un in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
            ideal = _cv2.fisheye.undistortPoints(
                _np.array([[[un * W, 0.5 * H]]], _np.float64), Kf, Df
            ).reshape(2)
            worst = max(worst, abs(_np.degrees(_np.arctan(ideal[0]))
                                   - fish.angles(un, 0.5)[0]))
        check("fisheye undistortion matches cv2.fisheye.undistortPoints",
              worst < 0.01, f"worst {worst:.5f} deg")

    # --- range provenance ---------------------------------------------
    intr = CameraIntrinsics(fx=0.5, fy=0.667, note="test")
    est = RangeEstimator(intrinsics=intr,
                         boresight=Boresight(0.5, 0.5, 2.0, True))
    from drone_agent.perception import observation_from_bbox
    centred = observation_from_bbox("person", (0.45, 0.2, 0.55, 0.8), intr,
                                    0.9, 1, 1.0)   # centre (0.5, 0.5)
    est.estimate(centred, boresight_range_m=2.40, boresight_trusted=True)
    check("a centred target takes range from the lidar",
          centred.range_source is RangeSource.BORESIGHT
          and abs(centred.range_m - 2.40) < 1e-6,
          f"{centred.range_m} m +/- {centred.range_sigma_m:.3f}")
    check("a metric source is marked metric", centred.range_source.is_metric)

    off = observation_from_bbox("person", (0.80, 0.2, 0.90, 0.8), intr,
                                0.9, 2, 1.0)
    est.estimate(off, boresight_range_m=2.40, boresight_trusted=True)
    check("an off-beam target does NOT take the lidar range",
          off.range_source is RangeSource.SIZE_PRIOR,
          f"{off.range_source.value} {off.range_m:.2f} m "
          f"+/- {off.range_sigma_m:.2f} (a 2 deg beam measures what is in "
          f"front of it, not the target)")
    check("a size-prior range is not marked metric",
          not off.range_source.is_metric)

    unknown = observation_from_bbox("fire hydrant", (0.4, 0.4, 0.5, 0.6),
                                    intr, 0.9, 3, 1.0)
    est.estimate(unknown)
    check("no prior and no sensor means no range",
          unknown.range_m is None
          and unknown.range_source is RangeSource.NONE)

    # --- tracking: a flicker is not a track ---------------------------
    box = (0.45, 0.3, 0.55, 0.9)
    script = [[Detection("person", box, 0.9, 7)] for _ in range(10)]
    script.insert(0, [])
    script.insert(1, [Detection("person", box, 0.10, 7)])   # below threshold
    runner = PerceptionRunner(MockDetector(script, period_s=0.1),
                              intrinsics=intr,
                              estimator=est, max_age_s=0.6)
    seen = []
    for _ in range(len(script)):
        sc = runner.step()
        o = sc.find("person")
        seen.append(None if o is None else (o.frames_seen, o.is_tracked()))
    check("a low-confidence detection is dropped", seen[1] is None,
          str(seen[:3]))
    check("frames_seen accumulates on a persistent id",
          seen[-1] is not None and seen[-1][0] >= 9, str(seen[-1]))
    check("is_tracked is false on the first frames and true later",
          seen[2] is not None and not seen[2][1] and seen[-1][1],
          f"frame3={seen[2]} last={seen[-1]}")

    # --- staleness ----------------------------------------------------
    scene = runner.latest()
    t = scene.t_capture_mono
    check("a fresh scene is returned",
          runner.fresh_scene(now=t + 0.1) is not None)
    check("a stale scene returns None, not old data",
          runner.fresh_scene(now=t + 5.0) is None,
          "a two-second-old bearing must not be actionable")
    check("world objects are empty when stale",
          runner.objects_for_world(now=t + 5.0) == [])

    # --- the planner view carries provenance --------------------------
    pv = centred.to_planner_view()
    check("planner view states where the range came from",
          pv.get("range_source") == "boresight"
          and "range_uncertainty_m" in pv, str(sorted(pv)))
    pv2 = unknown.to_planner_view()
    check("planner view is explicit about a missing range",
          pv2.get("range_m") is None and "range_note" in pv2)


def test_search_target_criterion() -> None:
    """rotate_search must not stop on a flicker, or on a sighting it cannot
    range."""
    print("\nsearch target criterion")
    cfg = Config()
    cfg.skills.leg_settle_s = 0.2
    cfg.skills.yaw_settle_s = 0.2

    def run(objects):
        v = FakeVehicle(cfg)
        arb = Arbiter(v, cfg)
        arb.set_origin(0.0, 0.0, -0.8)
        skill = build("rotate_search", cfg, total_deg=90.0, leg_deg=45.0,
                      rate_dps=40.0, target_class="person")
        world = vehicle_state_with(v, objects)
        skill.start(world)
        for _ in range(400):
            world = vehicle_state_with(v, objects)
            st = skill.tick(world, 0.1)
            try:
                arb.submit(skill.request, world)
            except AbortRequested:
                return "aborted", skill
            if st.value in ("succeeded", "failed", "aborted"):
                return st.value, skill
        return "timeout", skill

    def vehicle_state_with(v, objects):
        w = v.step(0.1)
        w.objects = objects
        return w

    flicker = [{"label": "person", "track_id": 4, "age_s": 0.1,
                "frames_seen": 1, "range_m": 2.0, "range_sigma_m": 0.1}]
    st, _ = run(flicker)
    check("a one-frame flicker does not end the search", st == "succeeded"
          and True, f"completed the sweep instead of stopping ({st})")

    unrangeable = [{"label": "person", "track_id": 4, "age_s": 2.0,
                    "frames_seen": 20, "range_m": None}]
    st, sk = run(unrangeable)
    check("a sighting with no range is recorded but does not stop the sweep",
          sk.metrics.get("range_missing") is True
          and "target" not in sk.metrics,
          f"sightings={len(sk.metrics.get('sightings', []))}")

    vague = [{"label": "person", "track_id": 4, "age_s": 2.0,
              "frames_seen": 20, "range_m": 2.0, "range_sigma_m": 1.5}]
    st, sk = run(vague)
    check("a range too uncertain to act on does not stop the sweep",
          sk.metrics.get("range_too_uncertain") == 1.5
          and "target" not in sk.metrics)

    good = [{"label": "person", "track_id": 4, "age_s": 2.0,
             "frames_seen": 20, "range_m": 2.4, "range_sigma_m": 0.08,
             "range_source": "boresight"}]
    st, sk = run(good)
    check("a tracked, ranged target ends the search early",
          st == "succeeded" and sk.metrics.get("target") is not None,
          f"{st}, target={sk.metrics.get('target', {}).get('range_m')} m")


def test_replay_roundtrip() -> None:
    """Record a flight, replay it, and assert the decisions reproduce. This
    is the property the whole evaluation strategy rests on."""
    print("\nreplay round trip")
    from drone_agent.config import Config as Cfg
    from drone_agent.interfaces.replay import (Degradation, load_meta,
                                               load_states,
                                               run_skill_on_states)

    cfg = Cfg()
    cfg.skills.leg_settle_s = 0.5
    cfg.skills.yaw_settle_s = 0.5
    params = {"total_deg": 180.0, "leg_deg": 45.0, "rate_dps": 40.0}

    with tempfile.TemporaryDirectory() as d:
        rec = FlightRecorder(log_dir=d, session_label="roundtrip")
        rec.meta(cfg.to_dict(), extra={"skill": "rotate_search",
                                       "skill_params": params})
        v = FakeVehicle(cfg)
        arb = Arbiter(v, cfg)
        arb.set_origin(0.0, 0.0, -0.8)
        live = drive(build("rotate_search", cfg, **params), v, arb,
                     recorder=rec, dt=0.1, max_ticks=6000)
        rec.skill_result(live["result"])
        rec.close()

        meta = load_meta(rec.path)
        check("config survives the round trip",
              Cfg.from_dict(meta["config"]).to_dict() == cfg.to_dict())

        replay_cfg = Cfg.from_dict(meta["config"])
        states = load_states(rec.path)
        check("every tick reloads as a WorldState",
              len(states) > 0 and states[0].pose is not None,
              f"{len(states)} states")

        out = run_skill_on_states("rotate_search", replay_cfg, states,
                                  skill_params=params)
        check("replay reproduces the live outcome",
              out.status == live["status"] == "succeeded",
              f"live={live['status']} replay={out.status} ({out.reason})")
        check("replay consumes the whole stream without exhausting",
              not out.exhausted, f"{out.ticks} of {len(states)} ticks")
        # The arbiter's timers run on the world clock, so replay reproduces
        # the rate limiter rather than collapsing it into the replay runtime.
        # A one-tick phase shift remains by construction: the live run
        # consumes an unlogged state for the precondition check and
        # skill.start(), while replay starts from the first LOGGED tick. For
        # a purely rate-limited skill that cancels out; for rotate_search the
        # phase machine's dwell boundaries shift with it, so allow a couple
        # of sends either way.
        live_sends = len(v.sent_positions)
        replay_sends = sum(1 for c in out.commands if c["kind"] == "position")
        check("replay reproduces the uplink rate limiter",
              abs(replay_sends - live_sends) <= 3,
              f"replay={replay_sends} live={live_sends}")
        check("replay commanded the same number of legs",
              sum(1 for c in out.commands if c["kind"] == "yaw")
              == len(v.sent_yaws),
              f"replay={sum(1 for c in out.commands if c['kind'] == 'yaw')} "
              f"live={len(v.sent_yaws)}")

        # One gross violation is survivable; a run of them must land.
        for repeat, want in ((1, "succeeded"), (2, "succeeded"),
                             (replay_cfg.envelope.gross_violation_abort_after,
                              "aborted")):
            o = run_skill_on_states("rotate_search", replay_cfg, states,
                                    skill_params=params, inject="altitude",
                                    inject_at_tick=10, inject_repeat=repeat)
            check(f"{repeat} gross violation(s) -> {want}",
                  o.status == want, f"got {o.status}: {o.reason[:52]}")

        # Degradation must change the outcome, or the knob does nothing.
        bad = load_states(rec.path,
                          degradation=Degradation("pose_noise", seed=1,
                                                  pose_noise_m=1.5))
        out_bad = run_skill_on_states("rotate_search", replay_cfg, bad,
                                      skill_params=params,
                                      degradation_name="pose_noise")
        check("injected pose noise changes the outcome",
              out_bad.status != "succeeded",
              f"{out_bad.status}: {out_bad.reason[:48]}")


def test_calibration_verdict() -> None:
    """
    A wrong lens model fits the observed corners closely and is still 35 deg
    wrong about the field of view, so reprojection error cannot be the gate.
    These are the two tests that do catch it, pinned against a synthetic
    150 deg fisheye whose true parameters are known.
    """
    try:
        import numpy as np
        import cv2
    except ImportError:
        check("calibration verdict (needs cv2/numpy)", True, "skipped")
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from calibrate_camera import _collinearity
    from drone_agent.perception.geometry import CameraIntrinsics

    W, H = 720, 576
    f = (W / 2) / math.radians(75.0)          # a true 150 deg equidistant lens
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
    D = np.array([0.02, -0.01, 0.004, -0.001], float)
    cols, rows, sq = 9, 6, 0.055
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= sq

    rng = np.random.default_rng(3)
    views = []
    for _ in range(10):
        rv = np.array([rng.uniform(-.5, .5), rng.uniform(-.5, .5),
                       rng.uniform(-.3, .3)])
        tv = np.array([rng.uniform(-.12, .12), rng.uniform(-.10, .10),
                       rng.uniform(.42, .75)])
        img, _ = cv2.fisheye.projectPoints(objp.reshape(-1, 1, 3), rv, tv,
                                           K, D)
        views.append(img.reshape(-1, 1, 2).astype(np.float64))

    def mk(Km, Dm, model):
        return CameraIntrinsics(
            fx=Km[0, 0] / W, fy=Km[1, 1] / H, cx=Km[0, 2] / W,
            cy=Km[1, 2] / H, k1=float(Dm[0]), k2=float(Dm[1]),
            k3=float(Dm[2]) if len(Dm) > 2 else 0.0,
            k4=float(Dm[3]) if len(Dm) > 3 else 0.0,
            model=model, width_px=W, height_px=H)

    truth = mk(K, D, "fisheye")
    err, Kp, Dp, _, _ = cv2.calibrateCamera(
        [objp.reshape(-1, 1, 3) for _ in views],
        [v.astype(np.float32) for v in views], (W, H), None, None)
    wrong = mk(Kp, Dp.ravel(), "pinhole")

    def slope(intr, a, b):
        ua = intr.cx + a * (1.0 - intr.cx)
        ub = intr.cx + b * (1.0 - intr.cx)
        return abs(intr.angles(ub, intr.cy)[0] - intr.angles(ua, intr.cy)[0])

    def linearity(intr):
        return slope(intr, 0.9, 1.0) / max(slope(intr, 0.0, 0.1), 1e-9)

    def bow(intr):
        d = [_collinearity(intr, v, (cols, rows), W, H) for v in views]
        d = [x for x in d if x is not None]
        return max(d) if d else None

    check("the wrong lens model still fits the corners closely",
          err < 0.5, f"its own reprojection error is {err:.3f} px")
    check("edge linearity passes the true model",
          linearity(truth) > 0.70, f"{linearity(truth):.2f}")
    check("edge linearity REJECTS a pinhole fit to a 150 deg lens",
          linearity(wrong) < 0.45, f"{linearity(wrong):.2f}")
    check("the true model's HFOV is recovered",
          abs(truth.hfov_deg - 150.0) < 5.0, f"{truth.hfov_deg:.1f} deg")
    check("the wrong model under-reports HFOV badly",
          wrong.hfov_deg < 130.0, f"{wrong.hfov_deg:.1f} deg")
    check("board lines stay straight under the true model",
          bow(truth) < 0.01, f"worst bow {bow(truth):.3f} px")
    check("board lines bow under the wrong model",
          bow(wrong) > 2.5, f"worst bow {bow(wrong):.3f} px")


def test_capture_geometry_guard() -> None:
    """
    Detector boxes are normalised by the frame the detector got, which hides
    the capture geometry from everything downstream. If that frame's aspect
    differs from the calibrated one, a geometrically perfect calibration
    misplaces every target and nothing notices. Pinned here because the
    symptom in the air would be a confident wrong bearing.
    """
    from drone_agent.perception.geometry import CameraIntrinsics
    from drone_agent.perception.source import MockDetector, PerceptionRunner

    intr = CameraIntrinsics(fx=0.6852, fy=0.9118, cx=0.5087, cy=0.4817,
                            k1=-0.1318, k2=0.0570, k3=-0.0364, k4=0.0057,
                            model="fisheye", width_px=720, height_px=576)
    check("the calibrated size itself passes",
          intr.check_capture(720, 576) is None)
    check("a uniform rescale passes (normalised intrinsics absorb it)",
          intr.check_capture(1440, 1152) is None, "1440x1152, same aspect")
    bad = intr.check_capture(1920, 1080)
    check("a 4:3 calibration REFUSES a 16:9 capture",
          bad is not None and "pillarbox" in bad, "1920x1080")
    check("a nonsense frame size is refused",
          intr.check_capture(0, 0) is not None)

    class Sized(MockDetector):
        capture_size = (1920, 1080)

    runner = PerceptionRunner(Sized([]), intrinsics=intr)
    check("the runner carries the warning into health()",
          runner.health().get("capture_warning") is not None,
          "surfaced, not silent")

    class Right(MockDetector):
        capture_size = (720, 576)

    check("a matching detector produces no warning",
          PerceptionRunner(Right([]), intrinsics=intr)
          .health().get("capture_warning") is None)


def _world(t: float, yaw_deg: float, z: float = -0.7,
           rf: Optional[float] = None, landed: int = 2) -> Any:
    from drone_agent.world.state import from_log
    return from_log({
        "t_sample_mono": t, "armed": True, "mode": "GUIDED",
        "landed_state": landed, "pose_age_s": 0.0,
        "pose": {"x": 0.0, "y": 0.0, "z": z, "yaw_deg": yaw_deg},
        "rngfnd_down": rf, "rngfnd_down_trusted": rf is not None,
        "rngfnd_down_raw": rf})


def test_leg_confirmation() -> None:
    """
    A yaw leg must not end on one heading sample. The sequence here is the
    one the autopilot actually reported on the last leg of a baseline sweep,
    where the EKF re-aligned its yaw to the compass mid-slew: the heading
    stepped 23 deg into the tolerance and then swung for a second before
    settling. One sample is not evidence of arrival.
    """
    print("\nleg completion needs a confirmed heading")
    from drone_agent.skills.rotate_search import Phase

    def run(samples: List[float], need: int, target: float = 100.8,
            hold_last_s: float = 0.0) -> tuple:
        cfg = Config()
        cfg.skills.yaw_confirm_samples = need
        sk = build("rotate_search", cfg, total_deg=360.0)
        sk.phase = Phase.SLEWING
        sk.leg_index = 7
        sk.leg_target = math.radians(target)
        sk._leg_deadline = float("inf")
        t = 0.0
        for y in samples:
            for _ in range(2):              # each sample spans two ticks
                sk._tick_slewing(_world(t, y))
                if sk.phase != Phase.SLEWING:
                    return y, sk
                t += 0.1
        while hold_last_s > 0 and sk.phase == Phase.SLEWING:
            sk._tick_slewing(_world(t, samples[-1]))
            t += 0.1
            hold_last_s -= 0.1
        return (samples[-1] if sk.phase != Phase.SLEWING else None), sk

    recorded = [72.5, 77.9, 83.4, 106.6, 93.2, 98.4, 103.2, 102.2]
    y1, _ = run(recorded, need=1)
    check("the old rule (1 sample) ends the leg ON the step",
          y1 == 106.6, f"stopped at {y1} deg, the instant the EKF re-aligned")
    y2, sk2 = run(recorded, need=2)
    check("two successive samples: the leg does not end on it",
          y2 is not None and y2 != 106.6 and abs(y2 - 100.8) <= 8.0,
          f"stopped at {y2} deg, a real sample inside the tolerance")

    # The same spike with the aircraft further away: the next real sample is
    # outside the tolerance, so the count must reset and say so.
    far = [60.0, 65.0, 70.0, 106.0, 75.0, 80.0, 85.0, 90.0, 95.0, 99.0, 101.0]
    y3, sk3 = run(far, need=2)
    check("a spike followed by an out-of-tolerance sample is discarded",
          y3 in (99.0, 101.0)
          and sk3.metrics.get("heading_samples_unconfirmed") == 1,
          f"stopped at {y3} deg, unconfirmed="
          f"{sk3.metrics.get('heading_samples_unconfirmed')}")
    y4, _ = run([80.0, 90.0, 99.0], need=2, hold_last_s=2.0)
    check("a stream that repeats one in-tolerance value still completes",
          y4 == 99.0, "fallback after yaw_confirm_fallback_s, not a timeout")

    # A log that predates the setting must replay under the rule it flew.
    old_log_cfg = Config().to_dict()
    del old_log_cfg["skills"]["yaw_confirm_samples"]
    check("a log without the setting replays under the rule it flew",
          Config.from_dict(old_log_cfg).skills.yaw_confirm_samples == 1,
          "1, as every flight up to 0.14")
    flown_015 = Config().to_dict()
    flown_015["skills"]["yaw_confirm_samples"] = 2
    check("a 0.15 log keeps the 2 it recorded, whatever today's default is",
          Config.from_dict(flown_015).skills.yaw_confirm_samples == 2
          and Config().skills.yaw_confirm_samples == 1,
          "log 2, default 1")


def test_takeoff_frame() -> None:
    """
    Takeoff completes on the rangefinder, because that is what the autopilot
    itself climbs by (WP_RFND_USE=1: a GUIDED takeoff goes to the requested
    height above the ground as the rangefinder sees it).

    0.15.0 judged it on the EKF's climb instead. On 2026-11-15 one flight
    had the EKF reading 0.12 m on the ground and 0.16 m below the
    rangefinder in the air; the EKF climbed 0.625 m, never the 0.68 m it was
    waiting for, and the aircraft hovered for the full 30 s timeout. That
    exact case is pinned here.
    """
    print("\ntakeoff completes on the rangefinder")
    from types import SimpleNamespace
    from drone_agent.executor import Executor
    import drone_agent.executor as ex

    def climb(gap_m: float, ground_ekf: float, rf_top: float) -> tuple:
        """Rangefinder climbs to rf_top and stays; EKF reads gap_m lower."""
        seq = []
        for k in range(400):
            real = min(rf_top, 0.04 + 0.02 * k)
            ekf = ground_ekf + max(0.0, real - 0.04 - (gap_m if k else 0.0))
            seq.append(_world(k * 0.05, 0.0, z=-ekf, rf=real))
        it = iter(seq)
        last = {"w": seq[0], "polls": 0}

        def poll() -> Any:
            last["w"] = next(it, last["w"])
            last["polls"] += 1
            return last["w"]

        cfg = Config()
        cfg.preflight.climb_timeout_s = 1.0     # a regression fails fast
        stub = SimpleNamespace(
            cfg=cfg,
            vehicle=SimpleNamespace(poll=poll, check_link=lambda: None,
                                    check_takeover=lambda: None),
            landing=SimpleNamespace(airborne=False,
                                    mark_airborne=lambda: None),
            _say=lambda *a, **k: None)
        real_sleep = ex.time.sleep
        ex.time.sleep = lambda s: None
        try:
            rep = Executor._wait_for_climb(stub, 0.80)
        finally:
            ex.time.sleep = real_sleep
        return rep, last["w"], last["polls"]

    # The flight that hung: EKF 0.122 m on the ground, 0.162 m low in the air,
    # rangefinder settling at 0.90 m.
    rep, w, polls = climb(gap_m=0.162, ground_ekf=0.122, rf_top=0.90)
    check("the flight that hung for 30 s now completes at once",
          rep["climb_timed_out"] is False and polls < 60,
          f"done after {polls} polls with the rangefinder at "
          f"{rep['alt_best_m']} m; the EKF had climbed only "
          f"{rep['ekf_climb_m']} m")
    check("it is decided on the rangefinder, whatever the EKF reads",
          abs(rep["alt_best_m"] - 0.68) < 0.03
          and rep["climb_frame"] == "best_estimate",
          f"at {rep['alt_best_m']} m, 85% of 0.80")
    tops = []
    for gap in (0.16, 0.09, 0.03):
        rep, w, _ = climb(gap_m=gap, ground_ekf=0.0, rf_top=0.90)
        tops.append(rep["alt_best_m"])
    check("the real height at handover does not depend on the EKF's gap",
          max(tops) - min(tops) < 0.01,
          f"{', '.join(f'{t:.2f}' for t in tops)} m for gaps of 0.16, 0.09, "
          f"0.03 m")


def test_heading_transient_analysis() -> None:
    """
    A heading step far larger than the slew produces is the EKF re-aligning
    its yaw to the compass (the autopilot says so in a STATUSTEXT). The
    analyzer must attribute it, keep that leg out of the control figures,
    and not mistake a fast rotation or a gap in the stream for one.
    """
    print("\nanalysis of an EKF yaw re-alignment")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_flight import heading_steps, yaw_legs

    def mk(seq: List[float], dt: float = 0.2, done_at: int = 9) -> list:
        ticks = []
        for n, y in enumerate(seq):
            if n == 0:
                req = {"kind": "yaw", "heading_deg": 100.8, "rate_dps": 20.0}
            elif n < done_at:
                req = {"kind": "idle"}
            else:
                req = {"kind": "position", "yaw_deg": 100.8}
            for half in range(2):
                t = n * dt + half * dt / 2
                ticks.append({"t_sample_mono": t, "t_mono": t,
                              "pose": {"yaw_deg": y},
                              "setpoint_requested": req})
        return ticks

    # the samples the autopilot reported on the last leg of a real sweep
    seq = [56.3, 56.1, 57.4, 62.9, 67.8, 72.5, 77.9, 83.4, 106.6, 93.2,
           98.4, 103.2, 102.2, 101.5, 101.2]
    ticks = mk(seq)
    t_step = next(t["t_mono"] for t in ticks if t["pose"]["yaw_deg"] == 106.6)
    said = [{"type": "event", "kind": "statustext", "t_mono": t_step + 0.05,
             "text": "EKF3 IMU0 MAG0 ground mag anomaly, yaw re-aligned"}]

    st_ = heading_steps(ticks, said)
    check("the step is found, once, with its size",
          len(st_) == 1 and st_[0]["yaw_deg"] == 106.6
          and abs(st_[0]["step_deg"] - 23.2) < 0.05,
          f"{len(st_)} found, {st_[0]['step_deg']:+.1f} deg" if st_
          else "none found")
    check("it is attributed to the autopilot's own message",
          bool(st_) and "yaw re-aligned" in (st_[0]["statustext"] or ""),
          st_[0]["statustext"] if st_ else "-")
    check("without logged messages it is still found, unattributed",
          len(heading_steps(ticks)) == 1
          and heading_steps(ticks)[0]["statustext"] is None)
    res = yaw_legs(ticks, said)
    check("the leg is marked, and kept out of the control figures",
          res["legs"][0]["yaw_reset"] is not None,
          "one leg, in two heading frames")
    # a step that does NOT come back (26.6 then -7.8, as on a later flight)
    oneway = [50.8, 51.2, 56.0, 61.0, 65.8, 70.9, 97.4, 89.7, 82.1, 90.2,
              96.3, 95.8]
    check("a step with a smaller swing back is found too",
          len(heading_steps(mk(oneway))) == 1,
          "the old out-and-back test missed this one")
    fast = [10.0 + 9.0 * k for k in range(20)]
    check("a fast, steady rotation is not mistaken for one",
          heading_steps(mk(fast)) == [], "45 deg/s, 9 deg per sample")
    gap = mk([60.0, 65.0, 70.0, 75.0, 80.0], dt=0.2) + [
        {"t_sample_mono": 3.0, "t_mono": 3.0, "pose": {"yaw_deg": 110.0},
         "setpoint_requested": {"kind": "idle"}}]
    check("the first sample after a gap in the stream is not one either",
          heading_steps(gap) == [], "30 deg after 2 s of silence")


def test_ground_perception_tool() -> None:
    """
    tools/watch_perception.py decides whether perception is ready to fly on.
    Its three judgements are pinned: that its idea of "found" is the flight
    skill's, that its summary counts what it says it counts, and that its
    latency test measures a known delay and catches a capture that queues.
    """
    print("\nground perception tool")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from watch_perception import (found_rule, measure_flash_latency,
                                  summarise, why_not_found)
    from drone_agent.world.state import from_log

    # ---- 1. "found" must mean what rotate_search means by it
    rule = found_rule()
    base = {"label": "person", "track_id": 4, "age_s": 1.2, "frames_seen": 9,
            "range_m": 2.4, "range_sigma_m": 0.3, "range_source": "size_prior",
            "confidence": 0.8, "bearing_deg": 5.0, "elevation_deg": 0.0}
    cases = [base,
             {**base, "track_id": None},
             {**base, "age_s": 0.2},
             {**base, "frames_seen": 2},
             {**base, "range_m": None, "range_sigma_m": None},
             {**base, "range_sigma_m": 0.9}]
    agree = 0
    for obj in cases:
        sk = build("rotate_search", Config(), total_deg=360.0,
                   target_class="person")
        world = from_log({"t_sample_mono": 1.0, "armed": True,
                          "pose": {"x": 0, "y": 0, "z": -0.7, "yaw_deg": 0},
                          "objects": [obj]})
        agree += (sk._target_found(world) == (why_not_found(obj, rule) is None))
    check("its stop rule agrees with rotate_search on every case",
          agree == len(cases), f"{agree} of {len(cases)}")
    check("and names the reason a sighting is refused",
          why_not_found(cases[4], rule) == "no range"
          and why_not_found(cases[5], rule) == "range too uncertain"
          and why_not_found(cases[1], rule) == "no track id")

    # ---- 2. the summary
    scenes = []
    for k in range(40):
        objs = [{**base, "bearing_deg": 10.0 + (0.5 if k % 2 else -0.5),
                 "age_s": k * 0.05, "frames_seen": k + 1}]
        if k >= 30:                       # the tracker hands out a new id
            objs[0]["track_id"] = 9
            objs[0]["age_s"] = (k - 30) * 0.05
            objs[0]["frames_seen"] = k - 29
        scenes.append({"t": k * 0.05, "frame_id": k + 1, "latency_s": 0.004,
                       "inference_ms": 40.0, "objects": objs})
    ticks = [{"t": k * 0.1, "fresh": k > 0, "age_s": 0.05} for k in range(20)]
    sm = summarise(scenes, ticks, rule)
    L = sm["labels"]["person"]
    check("scene rate, freshness and track ids are counted correctly",
          abs(sm["scene_rate_hz"] - 20.0) < 0.01
          and abs(sm["fresh_tick_pct"] - 95.0) < 0.01
          and L["track_ids"] == 2 and abs(L["longest_track_s"] - 1.45) < 0.01,
          f"{sm['scene_rate_hz']:.0f} Hz, {sm['fresh_tick_pct']:.0f}% fresh, "
          f"{L['track_ids']} ids, longest {L['longest_track_s']:.2f} s")
    check("bearing jitter on a still object is reported as its spread",
          abs(L["bearing_mean"] - 10.0) < 0.01
          and abs(L["bearing_sd"] - 0.5) < 0.01,
          f"{L['bearing_mean']:+.1f} ±{L['bearing_sd']:.1f} deg")
    young = sum(1 for s in scenes
                if why_not_found(s["objects"][0], rule) is not None)
    check("found% excludes sightings whose track is too young",
          abs(L["found_pct"] - 100.0 * (40 - young) / 40) < 0.01
          and young > 0, f"{L['found_pct']:.0f}% of 40, {young} too young")

    # ---- 3. the latency test, against a capture chain with a known delay
    class Chain:
        """A screen, a camera at 25 fps and a fixed delay, on a fake clock."""
        def __init__(self, delay_s: float, queued: bool,
                     levels: tuple = (20.0, 200.0)) -> None:
            self.t = 0.0
            self.delay_s, self.queued, self.levels = delay_s, queued, levels
            self.changes = [(0.0, 0)]
            self.k = 0                    # index of the last frame handed out

        def clock(self) -> float:
            return self.t

        def sleep(self, dt: float) -> None:
            self.t += dt

        def show(self, v: int) -> None:
            self.changes.append((self.t, v))

        def read(self) -> float:
            period = 1.0 / 25.0
            newest = int(self.t / period)
            # a queue hands out the next frame in order; otherwise the newest
            k = self.k + 1 if self.queued else max(self.k + 1, newest)
            self.k = k
            self.t = max(self.t, k * period)      # block until it exists
            seen_at = k * period - self.delay_s
            level = 0
            for when, v in self.changes:
                if when <= seen_at:
                    level = v
            return self.levels[level]

    c = Chain(0.120, queued=False)
    r = measure_flash_latency(c.read, c.show, c.clock, c.sleep, flips=12)
    check("a 120 ms chain is measured as 120 ms, to within one frame",
          r["ok"] and 0.120 <= r["median_s"] <= 0.120 + 0.045,
          f"median {1000 * r['median_s']:.0f} ms over {r['n']} flips")
    c2 = Chain(0.120, queued=False)
    r2 = measure_flash_latency(c2.read, c2.show, c2.clock, c2.sleep,
                               flips=12, think_s=0.12)
    check("a slow reader on a newest-frame capture adds only its own pause",
          r2["ok"] and (r2["median_s"] - r["median_s"]) <= 0.12 + 0.06,
          f"{1000 * (r2['median_s'] - r['median_s']):+.0f} ms for a 120 ms "
          f"pause")
    c3 = Chain(0.120, queued=True)
    r3 = measure_flash_latency(c3.read, c3.show, c3.clock, c3.sleep,
                               flips=12, think_s=0.12)
    check("a capture that QUEUES frames is exposed by the slow reader",
          r3["ok"] and (r3["median_s"] - r["median_s"]) > 0.12 + 0.06,
          f"{1000 * (r3['median_s'] - r['median_s']):+.0f} ms: old pictures "
          f"are being handed out")
    c4 = Chain(0.120, queued=False, levels=(100.0, 110.0))
    r4 = measure_flash_latency(c4.read, c4.show, c4.clock, c4.sleep)
    check("a camera that cannot see the screen is reported, not measured",
          r4["ok"] is False and "does not see" in r4["why"], r4["why"])


def test_perception_startup() -> None:
    """
    run_mission's perception start-up. Two defects were found by reading it
    against the ground tool, before any perception flight: the detector was
    loaded AFTER takeoff, and the camera was opened at the capture card's
    default size instead of the calibrated one. Both are pinned, along with
    the rule that a flight asked for perception does not go without it.
    """
    print("\nperception start-up")
    import argparse
    import signal
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    saved = {sig: signal.getsignal(sig)
             for sig in (signal.SIGINT, signal.SIGTERM)}
    import run_mission as rm
    for sig, h in saved.items():        # importing it installs its handlers
        signal.signal(sig, h)
    from watch_perception import SyntheticDetector
    from drone_agent.perception.geometry import CameraIntrinsics

    # ---- 1. order, read from the source: nothing is armed before it runs
    src = (root / "run_mission.py").read_text()
    body = src[src.index("def main()"):src.index("class PerceptionUnavailable")]
    i_perc = body.find("start_perception(args, recorder)")
    i_veh = body.find("MavlinkVehicle(cfg")
    i_launch = body.find("executor.launch(")
    check("the detector is started before the vehicle is even connected",
          0 <= i_perc < i_veh < i_launch,
          "start_perception -> MavlinkVehicle -> launch")
    check("and a failure to start it returns before any connection",
          "return 3" in body[i_perc:i_veh])

    tmp = Path(tempfile.mkdtemp())
    cam = tmp / "camera.json"
    CameraIntrinsics(fx=0.6852, fy=0.9118, cx=0.5087, cy=0.4817,
                     k1=-0.1318, k2=0.0570, k3=-0.0364, k4=0.0057,
                     model="fisheye", width_px=720, height_px=576,
                     reproj_err_px=0.379).save(str(cam))

    def ns(**kw: Any) -> argparse.Namespace:
        base = dict(intrinsics=str(cam), hfov=150.0, classes="person,chair",
                    target_class="person", camera="1",
                    model="yoloe-11s-seg.pt", imgsz=640, device=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def events(rec: FlightRecorder) -> List[Dict[str, Any]]:
        return [json.loads(l) for l in open(rec.path)
                if '"type": "event"' in l]

    def attempt(args: argparse.Namespace, factory: Any,
                timeout: float = 5.0) -> tuple:
        rec = FlightRecorder(path=str(tmp / f"{time.monotonic_ns()}.jsonl"))
        runner, err = None, None
        try:
            runner = rm.start_perception(args, rec, detector_factory=factory,
                                         ready_timeout_s=timeout)
        except rm.PerceptionUnavailable as e:
            err = str(e)
        ev = events(rec)
        rec.close()
        return runner, err, ev

    import contextlib
    import io
    quiet = contextlib.redirect_stdout(io.StringIO())

    # ---- 2. the good case
    asked: List[Any] = []

    def good(classes: Any, want: Any) -> Any:
        asked.append((classes, want))
        return SyntheticDetector(period_s=0.01, size=want)

    with quiet:
        runner, err, ev = attempt(ns(), good)
    started = next((e for e in ev if e["kind"] == "perception_started"), {})
    ok = runner is not None and runner.fresh_scene() is not None
    check("it returns only once scenes are flowing",
          ok and runner.latest().frame_id >= 3,
          f"{runner.latest().frame_id if runner else 0} scenes published")
    check("the camera is asked for the CALIBRATED size",
          asked and asked[0][1] == (720, 576), f"asked for {asked[0][1]}")
    check("the log records how perception started",
          started.get("classes") == ["person", "chair"]
          and tuple(started.get("capture_size") or ()) == (720, 576)
          and started.get("scene_rate_hz", 0) > 0,
          f"{started.get('scene_rate_hz')} scenes/s at "
          f"{started.get('capture_size')}, HFOV {started.get('hfov_deg')}")
    if runner is not None:
        runner.stop()
    rm._PERCEPTION = None

    # ---- 3. every way it can be unavailable stops the flight
    with quiet:
        _, err, ev = attempt(
            ns(), lambda c, w: SyntheticDetector(size=(1920, 1080)))
    check("a 16:9 capture against a 4:3 calibration does not fly",
          err is not None and "pillarbox" in err, (err or "")[:60])

    def broken(classes: Any, want: Any) -> Any:
        raise RuntimeError("camera 1 could not be opened")

    with quiet:
        _, err, _ = attempt(ns(), broken)
    check("a detector that fails to load does not fly",
          err is not None and "could not be opened" in err)

    called: List[int] = []
    with quiet:
        _, err, _ = attempt(ns(target_class="door"),
                            lambda c, w: called.append(1))
    check("a target the detector was never asked to look for does not fly",
          err is not None and "door" in err and not called,
          "refused before the camera is opened")

    class Dead(SyntheticDetector):
        def detect(self) -> tuple:
            raise OSError("no frame")

    t0 = time.monotonic()
    with quiet:
        _, err, _ = attempt(ns(), lambda c, w: Dead(size=w), timeout=0.6)
    check("a camera that opens and delivers nothing does not fly",
          err is not None and "no frame" in err
          and time.monotonic() - t0 < 3.0, (err or "")[-45:])

    with quiet:
        _, err, _ = attempt(ns(intrinsics=str(tmp / "missing.json")), good)
    check("a missing calibration file does not fly",
          err is not None and "missing.json" in err)

    # ---- 4. the analyzer's account of a perception flight
    from analyze_flight import perception_summary
    lag_s, rate, true_dir = 0.25, 17.5, 60.0
    ticks: List[Dict[str, Any]] = []
    yaw, t = 0.0, 100.0
    for k in range(240):
        turning = (k // 30) % 2 == 0          # 3 s turning, 3 s holding
        if turning:
            yaw += rate * 0.1
        rec: Dict[str, Any] = {"type": "tick", "t_sample_mono": t,
                               "pose": {"yaw_deg": yaw}}
        # The picture is lag_s old: the bearing was true for the heading the
        # aircraft had then, and is logged against the heading it has now.
        yaw_then = yaw - (rate * lag_s if turning else 0.0)
        b = true_dir - yaw_then
        if abs(b) < 40.0 and k >= 12:
            rec["objects"] = [{"label": "person", "track_id": 1,
                               "bearing_deg": b, "range_m": 2.0,
                               "range_source": "size_prior"}]
        ticks.append(rec)
        t += 0.1
    seen = [r for r in ticks if r.get("objects")]
    t_first = seen[0]["t_sample_mono"] - 100.0
    sk = [{"type": "skill", "phase": "result", "reason": "target acquired",
           "detail": {"target_class": "person", "legs_completed": 1},
           "t_mono": 100.0 + t_first + 0.6}]
    ev = [{"type": "event", "kind": "perception_started",
           "detector": "synthetic", "scene_rate_hz": 20.0}]
    ps = perception_summary(ticks, ev, sk)
    L = ps["labels"]["person"]
    check("a still object's direction comes out where it is",
          abs(L["direction_deg"] - true_dir) < rate * lag_s,
          f"{L['direction_deg']:.1f} deg for an object at {true_dir:g}")
    check("and the picture's age shows as a shift while turning",
          L["turning_minus_holding_deg"] is not None
          and abs(L["turning_minus_holding_deg"] - rate * lag_s) < 0.5,
          f"{L['turning_minus_holding_deg']:+.2f} deg for {lag_s} s at "
          f"{rate} deg/s ({rate * lag_s:.2f} expected)")
    check("from which it recovers how old the picture was",
          L["implied_picture_age_s"] is not None
          and abs(L["implied_picture_age_s"] - lag_s) < 0.03,
          f"{L['implied_picture_age_s']:.2f} s against {lag_s} s")
    a = ps["acquired"]
    check("the stop is timed against the first sighting",
          abs(a["after_first_sighting_s"] - 0.6) < 1e-6
          and abs(L["first_t_s"] - t_first) < 1e-6,
          f"first seen {L['first_t_s']:.1f} s, stopped "
          f"{a['after_first_sighting_s']:.1f} s later")
    check("a flight without perception prints no perception section",
          perception_summary([{"type": "tick", "t_sample_mono": 1.0,
                               "pose": {"yaw_deg": 0.0}}], [], []) is None)


def test_verify_capture() -> None:
    """
    calibrate_camera verify, live. It used to keep any six frames with a
    board in them, 0.4 s apart: a board in motion, or one pose six times,
    decided the verdict, and a real run read POOR without being able to say
    whether the calibration or the six pictures were at fault. It now takes
    a view only when the board is still and somewhere new.
    """
    print("\ncalibration verify, live capture")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import calibrate_camera as cc

    # ---- 1. the gate, on a fake clock
    g = cc.ViewGate(stable_s=0.6, min_move_px=60.0)
    t, taken = 0.0, 0
    for k in range(40):                       # 12 px a frame: never still
        taken += g.offer((100.0 + 12.0 * k, 200.0), t)
        t += 0.04
    check("a board that is moving is never taken",
          taken == 0 and "moving" in g.why, g.why)
    t_still = t
    while not g.offer((300.0, 200.0), t):
        t += 0.04
    check("a board held still is taken once it has been still long enough",
          0.6 <= t - t_still <= 0.6 + 0.09, f"after {t - t_still:.2f} s")
    again = sum(g.offer((303.0, 201.0), t + 0.04 * k) for k in range(60))
    check("the same position is not taken twice",
          again == 0 and "already covered" in g.why, g.why)
    t += 3.0
    while not g.offer((480.0, 330.0), t):
        t += 0.04
    g.offer(None, t)                          # board leaves the picture
    first = g.offer((150.0, 400.0), t + 0.04)
    check("a new position is taken; a board that reappears must settle "
          "again", len(g.kept) == 2 and not first,
          f"{len(g.kept)} views kept")

    # ---- 2. run_verify end to end against a rendered board
    try:
        import numpy as np
        import cv2
    except ImportError:
        check("verify live capture (needs cv2/numpy)", True, "skipped")
        return
    import argparse
    import contextlib
    import io
    from drone_agent.perception.geometry import CameraIntrinsics

    W, H, SQ = 720, 576, 22
    flat = np.full((7 * SQ + 2 * SQ, 10 * SQ + 2 * SQ), 255, np.uint8)
    for r in range(7):
        for c in range(10):
            if (r + c) % 2 == 0:
                flat[SQ + r * SQ:SQ + (r + 1) * SQ,
                     SQ + c * SQ:SQ + (c + 1) * SQ] = 0
    fh, fw = flat.shape

    def frame(cx: float, cy: float, tilt: float = 0.0) -> Any:
        """The board seen through a distortion-free camera: a perspective
        warp, under which straight lines stay straight."""
        src = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])
        dst = np.float32([[cx - fw / 2, cy - fh / 2 + tilt],
                          [cx + fw / 2, cy - fh / 2 - tilt],
                          [cx + fw / 2, cy + fh / 2 + tilt],
                          [cx - fw / 2, cy + fh / 2 - tilt]])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(flat, M, (W, H), borderValue=128)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    poses = [(200, 170, 0), (520, 170, 8), (200, 400, -8), (520, 400, 0),
             (360, 288, 6), (360, 140, -6)]

    class FakeCap:
        """Moves the board to each pose in turn and holds it there."""
        def __init__(self, hold: int, travel: int) -> None:
            self.script: List[tuple] = []
            prev = (360.0, 288.0, 0.0)
            for pz in poses:
                for k in range(1, travel + 1):       # in motion
                    a = k / travel
                    self.script.append(tuple(prev[i] + a * (pz[i] - prev[i])
                                             for i in range(3)))
                self.script += [pz] * hold           # held still
                prev = pz
            self.i = 0

        def read(self) -> tuple:
            if self.i >= len(self.script):
                return False, None
            time.sleep(0.01)
            f = frame(*self.script[self.i])
            self.i += 1
            return True, f

        def release(self) -> None:
            pass

    tmp = Path(tempfile.mkdtemp())

    def verify(intr: CameraIntrinsics, cap: Any) -> tuple:
        path = tmp / f"cam_{time.monotonic_ns()}.json"
        intr.save(str(path))
        args = argparse.Namespace(
            out=str(path), board="9x6", camera="1", equalize=False,
            expect_hfov=None, from_dir=None, width=None, height=None,
            max_views=20, min_move_px=60.0, no_capture=False,
            stable_s=0.12, timeout_s=30.0)
        old = cc.open_capture, cc.describe_capture
        cc.open_capture = lambda *a, **k: cap
        cc.describe_capture = lambda *a, **k: {"width": W, "height": H,
                                               "fps": 25.0, "backend": "fake"}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = cc.run_verify(args)
        finally:
            cc.open_capture, cc.describe_capture = old
        return rc, buf.getvalue()

    true_cam = CameraIntrinsics(fx=0.70, fy=0.875, cx=0.5, cy=0.5,
                                model="pinhole", width_px=W, height_px=H,
                                reproj_err_px=0.3)
    rc, out = verify(true_cam, FakeCap(hold=40, travel=10))
    n_views = out.count("    view ")
    check("six still views in six places are taken, and a true model "
          "reads good", rc == 0 and "6 usable" in out
          and "VERDICT: good" in out,
          out.split("board lines bow by")[-1].split("\n")[0].strip()
          if "bow by" in out else out[-80:])
    check("every view's own figure is printed, with where it was",
          all(f"at ({x}, " in out for x, _, _ in poses[:2]),
          f"{n_views} view lines")

    # The wrong lens MODEL for these pictures: they were rendered without
    # distortion, and a fisheye model bends them when it "undistorts" them.
    wrong = CameraIntrinsics(fx=0.45, fy=0.5625, cx=0.5, cy=0.5,
                             model="fisheye", width_px=W, height_px=H,
                             reproj_err_px=0.3)
    rc, out = verify(wrong, FakeCap(hold=40, travel=10))
    check("the same views against a wrong model read POOR",
          rc == 1 and "VERDICT: POOR" in out,
          out.split("board lines bow by")[-1].split("\n")[0].strip()
          if "bow by" in out else out[-80:])

    rc, out = verify(true_cam, FakeCap(hold=2, travel=10))
    check("a board that never stops is not judged at all",
          "VERDICT" not in out and ("no views" in out
                                    or "not enough to judge" in out),
          "no verdict from pictures taken in motion")


def test_planner() -> None:
    """
    The language model's reply is untrusted text. These pin what is done
    with it: nothing reaches the vehicle that is not a listed skill with
    parameters inside its schema and the room's limits, a bad reply is sent
    back with the reasons and then given up on, a refusal is not a flight,
    and the plan is made and confirmed before anything is connected.
    """
    print("\nplanner")
    import argparse
    import contextlib
    import io
    import signal
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from drone_agent import planner as pl
    from drone_agent.skills import catalogue
    saved = {sig: signal.getsignal(sig)
             for sig in (signal.SIGINT, signal.SIGTERM)}
    import run_mission as rm
    for sig, h in saved.items():
        signal.signal(sig, h)
    cat = catalogue()

    def errs(steps: Any, cannot: str = "", classes: Any = None) -> List[str]:
        return pl.validate_plan({"steps": steps, "cannot_do": cannot},
                                cat, 1.2, classes)[1]

    search = {"skill": "rotate_search",
              "params": {"total_deg": 360, "target_class": "Person"}}
    hover = {"skill": "hover", "params": {"duration_s": 5}}
    steps, e = pl.validate_plan({"steps": [search, hover]}, cat, 1.2)
    check("a plan of listed skills inside their limits passes",
          not e and [s.skill for s in steps] == ["rotate_search", "hover"]
          and steps[0].params["target_class"] == "person")
    bad = {
        "an unknown skill": [{"skill": "fly_to", "params": {"x": 2}}],
        "an unknown parameter": [{"skill": "hover",
                                  "params": {"duration_s": 5, "speed": 3}}],
        "a missing required parameter": [{"skill": "hover", "params": {}}],
        "a value above its maximum": [{"skill": "hover",
                                       "params": {"duration_s": 500}}],
        "a string where a number belongs": [{"skill": "hover",
                                             "params": {"duration_s": "5"}}],
        "true where a number belongs": [{"skill": "hover",
                                         "params": {"duration_s": True}}],
        "a height above this room's limit": [{"skill": "climb_to",
                                              "params": {"altitude_m": 1.8}}],
        "more steps than allowed": [hover] * (pl.MAX_STEPS + 1),
        "a plan longer than one battery": [
            {"skill": "rotate_search", "params": {"total_deg": 1080}}],
    }
    wrong = [k for k, v in bad.items() if not errs(v)]
    check("every kind of bad plan is refused", not wrong,
          f"{len(bad)} kinds" if not wrong else f"ACCEPTED: {wrong}")
    check("a class the camera is not looking for is refused",
          errs([search], classes=["chair"]) and not errs([search],
                                                         classes=["person"]))
    check("a refusal is valid; steps AND a refusal, or neither, is not",
          not errs([], "cannot move sideways") and errs([hover], "but")
          and errs([]))

    nested = ('<think>maybe {"steps": 1}?</think>Here you go:\n```json\n'
              '{"understood": "say {hi}", "steps": [], "cannot_do": "no"}'
              '\n```\nHope that helps.')
    got = pl.extract_json(nested)
    try:
        pl.extract_json("I would turn around and look.")
        raised = False
    except ValueError:
        raised = True
    check("the JSON is found inside reasoning, a code fence and chatter",
          got["cannot_do"] == "no" and got["understood"] == "say {hi}")
    check("a reply with no JSON is an error, not a guess", raised)

    prompt = pl.system_prompt(cat, 1.2, ["person"])
    props = [k for e2 in cat.values() for k in e2["schema"]["properties"]]
    check("the prompt lists every skill and parameter, and the room's limit",
          all(n in prompt for n in cat) and all(k in prompt for k in props)
          and "1.2 m" in prompt and "0.3..1.2" in prompt
          and "not yet wired" not in prompt,
          f"{len(cat)} skills, {len(props)} parameters")

    good = json.dumps({"understood": "find a person", "steps": [search],
                       "cannot_do": ""})
    llm = pl.ScriptedLLM(['{"steps": [{"skill": "spin"}]}', good])
    res = pl.Planner(llm, cat, 1.2).plan("find a person")
    told = llm.calls[1][-1]["content"]
    check("a bad reply is sent back with the reason, and the repair is used",
          res.ok and res.attempts == 2 and "unknown skill 'spin'" in told,
          f"{res.attempts} attempts")
    class CutOff(pl.ScriptedLLM):
        last_finish, max_tokens = "length", 800
    llm = CutOff(["", good])
    res = pl.Planner(llm, cat, 1.2).plan("find a person")
    check("a reply cut off at the token limit is not asked for again",
          not res.ok and res.cut_off and res.attempts == 1
          and "--no-think" in res.errors[0] and "thinking" in res.errors[0],
          "one attempt, and it says what to change")
    slow = [{"skill": "rotate_search",
             "params": {"total_deg": 360, "rate_dps": 5}}]
    fast = [{"skill": "rotate_search", "params": {"total_deg": 360}}]
    est = pl.estimate_s(pl.validate_plan({"steps": fast}, cat, 1.2)[0])
    check("a turn's duration counts its rate: 360 deg at 5 deg/s is refused",
          errs(slow) and not errs(fast) and 38.0 <= est <= 41.0,
          f"default rate estimated at {est:.0f} s (flown: 38.7-40.5 s)")
    llm = pl.ScriptedLLM(["nope"] * 5)
    res = pl.Planner(llm, cat, 1.2).plan("find a person")
    check("after two repairs there is no plan",
          not res.ok and not res.flyable and res.attempts == 3
          and len(llm.calls) == 3, f"{res.attempts} attempts, then stop")

    # ---- the real transport, against a server on this machine
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def _send(self, code: int, obj: Any) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._send(200, {"data": [{"id": "uni-model"}]})

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            if not self.path.endswith("/v1/chat/completions"):
                return self._send(404, {"error": "not found"})
            H.seen = (req, self.headers.get("Authorization"))
            self._send(200, {"choices": [{"message": {"content": good}}]})

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    old_env = {k: os.environ.pop(k) for k in list(os.environ)
               if k.lower() in ("http_proxy", "https_proxy", "all_proxy")}
    try:
        real = pl.OpenAICompatLLM(base + "/v1/", "uni-model", api_key="k",
                                  timeout_s=5, no_think=True)
        res = pl.Planner(real, cat, 1.2).plan("find a person")
        req, auth = H.seen
        check("the OpenAI-compatible call works end to end",
              res.flyable and req["model"] == "uni-model"
              and req["temperature"] == 0 and auth == "Bearer k"
              and req["max_tokens"] == pl.DEFAULT_MAX_TOKENS
              and req["reasoning_effort"] == "none"
              and real.list_models() == ["uni-model"],
              "plan, model list, key and temperature 0")
        try:
            pl.OpenAICompatLLM(base + "/wrong", "m", timeout_s=5).chat([])
            msg = ""
        except pl.LLMError as ex:
            msg = str(ex)
        check("a wrong address says so", "404" in msg and "address" in msg)
        srv.shutdown()
        srv.server_close()
        try:
            pl.OpenAICompatLLM(base + "/v1", "m", timeout_s=2).chat([])
            msg = ""
        except pl.LLMError as ex:
            msg = str(ex)
        check("an unreachable server says so, and mentions the VPN",
              "could not reach" in msg and "VPN" in msg)
    finally:
        os.environ.update(old_env)

    # ---- on the ground, before anything is connected
    src = (Path(__file__).resolve().parent.parent / "run_mission.py"
           ).read_text()
    body = src[src.index("def main()"):src.index("class NoPlan")]
    i_plan = body.find("plan_on_ground(args, cfg, recorder)")
    check("the plan is made before the camera and before the vehicle",
          0 <= i_plan < body.find("start_perception(args, recorder)")
          < body.find("MavlinkVehicle(cfg") and "return 4"
          in body[i_plan:body.find("start_perception(args, recorder)")])

    tmp = Path(tempfile.mkdtemp())

    def ground(reply: str, answer: str = "y", **kw: Any) -> tuple:
        base_args = dict(instruction="find a person", classes=None,
                         perception="yoloe", yes=False, llm_url="x",
                         llm_model="m", llm_timeout=5, json_mode=False,
                         max_tokens=4096, no_think=False)
        base_args.update(kw)
        args = argparse.Namespace(**base_args)
        rec = FlightRecorder(path=str(tmp / f"{time.monotonic_ns()}.jsonl"))
        llm2 = (pl.ScriptedLLM([reply]) if reply else pl.ScriptedLLM([]))
        out, why = None, None
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                out = rm.plan_on_ground(args, Config(), rec, llm=llm2,
                                        ask=lambda q: answer)
            except rm.NoPlan as ex:
                why = str(ex)
        kinds = [json.loads(l).get("kind") for l in open(rec.path)]
        rec.close()
        return out, why, kinds, args

    out, why, kinds, args = ground(good)
    check("a confirmed plan comes back as steps, and is in the log",
          out == [("rotate_search", {"total_deg": 360,
                                     "target_class": "person"})]
          and "plan" in kinds and "plan_confirmation" in kinds)
    check("the detector is told to look for what the plan looks for",
          args.classes == "person")
    out, why, _, _ = ground(good, answer="")
    check("anything but y does not fly", out is None and "cancelled" in why)
    out, why, _, _ = ground(good, perception="none")
    check("a plan that needs the camera does not fly without it",
          out is None and "camera is off" in why)
    refusal = json.dumps({"understood": "x", "steps": [],
                          "cannot_do": "cannot move sideways"})
    out, why, _, _ = ground(refusal)
    check("a refusal does not fly", out is None and "cannot be done" in why)
    out, why, kinds, _ = ground("")
    check("an unreachable model does not fly",
          out is None and "could not be asked" in why
          and "plan_failed" in kinds)

    # ---- in the air: the steps, in order, and when to stop
    class R:
        def __init__(self, ok: bool, reason: str) -> None:
            self.ok, self.reason = ok, reason
            self.status = type("S", (), {"value": "succeeded" if ok
                                         else "failed"})()

        def to_log(self) -> Dict[str, Any]:
            return {"ok": self.ok, "reason": self.reason}

    class Ex:
        def __init__(self, script: List[R]) -> None:
            self.script, self.ran = script, []

        def run_skill(self, skill: Any) -> R:
            self.ran.append(skill.name)
            return self.script[len(self.ran) - 1]

    two = [("rotate_search", {"total_deg": 360, "target_class": "person"}),
           ("hover", {"duration_s": 5})]
    rec = FlightRecorder(path=str(tmp / "steps.jsonl"))
    with contextlib.redirect_stdout(io.StringIO()):
        ex = Ex([R(True, "target acquired"), R(True, "")])
        r1 = rm.run_steps(ex, Config(), two, rec)
        ex2 = Ex([R(True, "rotation complete"), R(True, "")])
        r2 = rm.run_steps(ex2, Config(), two, rec)
        ex3 = Ex([R(False, "flow lost"), R(True, "")])
        r3 = rm.run_steps(ex3, Config(), two, rec)
        ex4 = Ex([R(True, "rotation complete")])
        r4 = rm.run_steps(ex4, Config(), two[:1], rec)
    rec.close()
    check("steps run in order when each succeeds",
          ex.ran == ["rotate_search", "hover"] and r1[1] == "complete"
          and len(r1[0]) == 2)
    check("a search that finds nothing skips what was planned after it",
          ex2.ran == ["rotate_search"] and r2[1] == "target_not_found")
    check("a failed step ends the plan",
          ex3.ran == ["rotate_search"] and r3[1] == "skill_failed")
    check("a search alone that finds nothing is still a completed sweep",
          r4[1] == "complete")


def test_face_target() -> None:
    """
    face_target on a simulated aircraft. The picture the skill sees is a
    quarter of a second old, as measured, and the heading follows a yaw
    command at the rate the airframe achieves. What is pinned: it ends
    facing the target from where a search leaves it, it never measures
    while turning, it copes with a camera model that is 15% wrong, and it
    gives up cleanly when the target is not there.
    """
    print("\nface_target")
    from collections import deque
    from drone_agent.skills import build
    from drone_agent.safety.arbiter import RequestKind
    from drone_agent.world.state import from_log

    def wrap(a: float) -> float:
        return (a + 180.0) % 360.0 - 180.0

    def fly(true_dir: Optional[float], yaw0: float = 0.0, scale: float = 1.0,
            hfov: float = 107.0, lag_s: float = 0.25, max_s: float = 60.0,
            vanish_at: Optional[float] = None,
            leftover_deg: float = 0.0) -> Dict[str, Any]:
        cfg = Config()
        sk = build("face_target", cfg, target_class="person")
        # leftover_deg: the search's last yaw command is still running when
        # this skill starts, and the aircraft keeps turning for a second.
        yaw = yaw0
        target_yaw = wrap(yaw0 + leftover_deg) if leftover_deg else None
        hist: deque = deque(maxlen=int(lag_s / 0.1) + 1)
        t, measured_turning, status = 100.0, 0, None
        yaw_cmds: List[float] = []

        def world() -> Any:
            hist.append(yaw)
            objs = []
            seen_yaw = hist[0]                    # the picture is old
            if true_dir is not None and (vanish_at is None
                                         or t - 100.0 < vanish_at):
                b = wrap(true_dir - seen_yaw) * scale
                if abs(b) <= hfov / 2.0:
                    objs = [{"label": "person", "track_id": 7,
                             "bearing_deg": b, "frames_seen": 30,
                             "age_s": 3.0, "range_m": 2.6,
                             "range_source": "size_prior",
                             "bbox": [0.4, 0.3, 0.6, 1.0]}]
            return from_log({
                "t_sample_mono": t, "armed": True, "mode": "GUIDED",
                "landed_state": 2, "pose_age_s": 0.0,
                "pose": {"x": 0.0, "y": 0.0, "z": -0.7, "yaw_deg": yaw},
                "rngfnd_down": 0.8, "rngfnd_down_trusted": True,
                "rngfnd_down_raw": 0.8, "objects": objs})

        w = world()
        pre = sk.check_preconditions(w)
        sk.start(w)
        while t - 100.0 < max_s:
            st_ = sk.tick(w, 0.1)
            if st_.value != "running":
                status = st_.value
                break
            rq = sk.request
            if rq.kind is RequestKind.YAW:
                target_yaw = wrap(rq.heading_deg)
                yaw_cmds.append(target_yaw)
            elif (rq.kind is RequestKind.POSITION and rq.yaw_rad is not None
                  and t - 100.0 > (1.0 if leftover_deg else 0.0)):
                target_yaw = wrap(math.degrees(rq.yaw_rad))
            turning = (target_yaw is not None
                       and abs(wrap(target_yaw - yaw)) > 0.5)
            if turning and sk.phase.value == "measure" and sk._bearings:
                measured_turning += 1
            if turning:                            # 17.5 deg/s, as flown
                err = wrap(target_yaw - yaw)
                yaw = wrap(yaw + max(-1.75, min(1.75, err)))
            t += 0.1
            w = world()
        return {"status": status, "final": wrap((true_dir or 0.0) - yaw),
                "t": t - 100.0, "turns": sk.turns, "skill": sk, "pre": pre,
                "measured_turning": measured_turning, "cmds": yaw_cmds}

    a = fly(42.0)                 # where the perception flights left it
    check("from 42 deg off the nose it ends facing the target",
          a["status"] == "succeeded" and abs(a["final"]) <= 5.0,
          f"{a['final']:+.1f} deg off after {a['turns']} turn(s), "
          f"{a['t']:.0f} s")
    check("it reports that the box is cut off, so the range is not trusted",
          a["skill"]._detail.get("box_cut_off") is True
          and a["skill"]._detail.get("range_m") == 2.6)
    check("no bearing is taken while the aircraft is turning",
          a["measured_turning"] == 0)
    b = fly(152.0, yaw0=-170.0)   # 38 deg to the LEFT, across the +-180 seam
    check("it turns left for a target on the left, across +-180 deg",
          b["status"] == "succeeded" and abs(b["final"]) <= 5.0
          and b["skill"].turned_deg < 0,
          f"turned {b['skill'].turned_deg:+.0f} deg")
    c = fly(42.0, scale=1.15)     # camera reads every angle 15% too large
    check("a camera model that is 15% wrong costs turns, not the result",
          c["status"] == "succeeded" and abs(c["final"]) <= 5.0
          and c["turns"] >= 2, f"{c['turns']} turns, {c['final']:+.1f} deg")
    g = fly(42.0, leftover_deg=30.0)
    check("handed over in the middle of a turn, it waits until it is still",
          g["status"] == "succeeded" and abs(g["final"]) <= 5.0
          and g["measured_turning"] == 0,
          f"{g['final']:+.1f} deg off, {g['turns']} turn(s), {g['t']:.0f} s")
    d = fly(3.0)
    check("a target already ahead needs no turn",
          d["status"] == "succeeded" and d["turns"] == 0 and not d["cmds"])
    e = fly(None)
    check("no target in view: it fails within a few seconds, without turning",
          e["status"] == "failed" and e["t"] < 4.0 and not e["cmds"],
          f"failed after {e['t']:.1f} s")
    f = fly(42.0, vanish_at=3.0)  # the person walks away during the turn
    check("a target that disappears ends the skill instead of a hunt",
          f["status"] == "failed" and f["turns"] <= 1 and f["t"] < 12.0,
          f"failed after {f['t']:.1f} s, {f['turns']} turn")

    # ---- the plan: a search followed by facing what it found
    import contextlib
    import io
    import signal
    saved = {sig: signal.getsignal(sig)
             for sig in (signal.SIGINT, signal.SIGTERM)}
    import run_mission as rm
    for sig, h in saved.items():
        signal.signal(sig, h)
    from drone_agent import planner as pl
    from drone_agent.skills import catalogue
    cat = catalogue()
    plan = {"steps": [
        {"skill": "rotate_search",
         "params": {"total_deg": 360, "target_class": "person"}},
        {"skill": "face_target", "params": {"target_class": "person"}},
        {"skill": "hover", "params": {"duration_s": 3}}]}
    steps, errs = pl.validate_plan(plan, cat, 1.2, ["person"])
    check("a plan may search, face and hold; facing needs its target",
          not errs and len(steps) == 3 and pl.validate_plan(
              {"steps": [{"skill": "face_target", "params": {}}]},
              cat, 1.2)[1] and "face_target" in pl.system_prompt(cat, 1.2))

    class R:
        def __init__(self, ok: bool, reason: str) -> None:
            self.ok, self.reason = ok, reason
            self.status = type("S", (), {"value": "succeeded" if ok
                                         else "failed"})()

        def to_log(self) -> Dict[str, Any]:
            return {"ok": self.ok}

    class Ex:
        def __init__(self, script: List[R]) -> None:
            self.script, self.ran = script, []

        def run_skill(self, skill: Any) -> R:
            self.ran.append(skill.name)
            return self.script[len(self.ran) - 1]

    three = [(s.skill, s.params) for s in steps]
    rec = FlightRecorder(path=str(Path(tempfile.mkdtemp()) / "f.jsonl"))
    with contextlib.redirect_stdout(io.StringIO()):
        ex = Ex([R(True, "target acquired"), R(True, "facing target"),
                 R(True, "")])
        out = rm.run_steps(ex, Config(), three, rec)
        ex2 = Ex([R(True, "target acquired"), R(False, "not in view"),
                  R(True, "")])
        out2 = rm.run_steps(ex2, Config(), three, rec)
    rec.close()
    check("facing the target does not end the plan; failing to does",
          ex.ran == ["rotate_search", "face_target", "hover"]
          and out[1] == "complete" and ex2.ran == ["rotate_search",
                                                   "face_target"]
          and out2[1] == "skill_failed")


def test_move() -> None:
    """
    move on a simulated aircraft that, like the real link, only receives a
    position target when it is marked immediate or two seconds have passed.
    Pinned: the displacement is relative to the nose in the NED frame, it is
    fed to the aircraft in sub-steps so the target is never far ahead, and
    everything that should not move does not.
    """
    print("\nmove")
    from drone_agent.skills import build, catalogue
    from drone_agent.safety.arbiter import RequestKind
    from drone_agent.world.state import from_log
    from drone_agent import planner as pl

    def fly(yaw: float, frozen: bool = False, at: tuple = (0.0, 0.0),
            **params: Any) -> Dict[str, Any]:
        cfg = Config()
        sk = build("move", cfg, **params)
        x, y = at
        tx, ty = x, y
        t, t_sent, lead, jumps, status = 100.0, -99.0, 0.0, [], None

        def world() -> Any:
            return from_log({
                "t_sample_mono": t, "armed": True, "mode": "GUIDED",
                "landed_state": 2, "pose_age_s": 0.0,
                "pose": {"x": x, "y": y, "z": -0.7, "yaw_deg": yaw},
                "origin": [0.0, 0.0, 0.0],
                "rngfnd_down": 0.8, "rngfnd_down_trusted": True,
                "rngfnd_down_raw": 0.8})

        w = world()
        pre = sk.check_preconditions(w)
        if not pre.ok:
            return {"pre": pre.reason, "status": None}
        sk.start(w)
        while t - 100.0 < 40.0:
            st_ = sk.tick(w, 0.1)
            if st_.value != "running":
                status = st_.value
                break
            rq = sk.request
            if rq.kind is RequestKind.POSITION and (
                    rq.immediate or t - t_sent >= 2.0):
                if (rq.x, rq.y) != (tx, ty):
                    jumps.append(math.hypot(rq.x - tx, rq.y - ty))
                tx, ty, t_sent = rq.x, rq.y, t
            if not frozen:                       # up to 1 m/s toward it
                d = math.hypot(tx - x, ty - y)
                if d > 1e-9:
                    k = min(1.0, 0.1 / d)
                    x, y = x + k * (tx - x), y + k * (ty - y)
            lead = max(lead, math.hypot(tx - x, ty - y))
            t += 0.1
            w = world()
        return {"pre": None, "status": status, "x": x, "y": y,
                "t": t - 100.0, "lead": lead, "jumps": jumps, "skill": sk}

    a = fly(90.0, forward_m=1.0)           # nose east
    check("1 m forward with the nose east ends 1 m east",
          a["status"] == "succeeded" and abs(a["y"] - 1.0) < 0.2
          and abs(a["x"]) < 0.05,
          f"ended at north {a['x']:+.2f}, east {a['y']:+.2f} after "
          f"{a['t']:.0f} s")
    check("the target is fed in sub-steps and is never far ahead",
          len(a["jumps"]) == 4 and max(a["jumps"]) <= 0.2501
          and a["lead"] <= 0.2501,
          f"{len(a['jumps'])} sub-steps of {max(a['jumps']):.2f} m")
    b = fly(0.0, right_m=0.5)              # nose north: right is east
    c = fly(180.0, forward_m=-0.5)         # nose south: backward is north
    check("right and backward map to the right axes",
          b["status"] == "succeeded" and abs(b["y"] - 0.5) < 0.2
          and c["status"] == "succeeded" and abs(c["x"] - 0.5) < 0.2,
          f"right: east {b['y']:+.2f}; backward: north {c['x']:+.2f}")
    check("a move of nothing, or of more than a metre, does not start",
          fly(0.0)["pre"] and fly(0.0, forward_m=1.0, right_m=1.0)["pre"])
    far = fly(0.0, at=(3.5, 0.0), forward_m=1.0)
    check("a move that would leave the fence is refused with the reason",
          far["pre"] and "4.0 m" in far["pre"], far["pre"] or "")
    d = fly(0.0, frozen=True, forward_m=0.5)
    check("an aircraft that does not move fails instead of waiting",
          d["status"] == "failed" and d["t"] < 12.0,
          f"failed after {d['t']:.0f} s")

    cat = catalogue()

    def errs(steps: Any) -> List[str]:
        return pl.validate_plan({"steps": steps}, cat, 1.2)[1]

    mv = lambda **k: {"skill": "move", "params": k}       # noqa: E731
    prompt = pl.system_prompt(cat, 1.2)
    check("a plan may move out and back, 2 m in total and no more",
          not errs([mv(forward_m=0.5), mv(forward_m=-0.5)])
          and not errs([mv(forward_m=1.0), mv(right_m=1.0)])
          and errs([mv(forward_m=1.0), mv(right_m=1.0), mv(forward_m=0.5)])
          and errs([mv()]) and errs([mv(forward_m=1.5)])
          and errs([mv(forward_m=0.8, right_m=0.8)]))
    check("the prompt offers the move and still rules out going to a place",
          "- move:" in prompt and "cannot go to a named place" in prompt
          and "cannot move sideways" not in prompt)


def test_measure_range() -> None:
    """
    measure_range on a simulated aircraft whose EKF under-reads distance by
    the factor the six tape-measured move flights showed, and whose camera
    is a quarter of a second behind with a degree of noise. Pinned: the
    geometry, that the real baseline and not the EKF's is used, that the
    answer carries an uncertainty wide enough to contain the truth, that a
    target too far away is refused rather than guessed at, and that the
    aircraft comes home in every ending.
    """
    print("\nmeasure_range")
    import random
    import statistics as st
    from collections import deque
    from drone_agent.skills import build, catalogue
    from drone_agent.skills.measure_range import range_from_parallax
    from drone_agent.safety.arbiter import RequestKind
    from drone_agent.world.state import from_log
    from drone_agent import planner as pl

    # ---- the formula alone, against cases that can be done by hand
    cases = [((0.0, -26.565, 0.0, 1.0), 2.0),     # 1 m right, target 2 m ahead
             ((0.0, 26.565, 0.0, -1.0), 2.0),     # mirror image
             ((0.0, -9.4623, 0.0, 0.5), 3.0),     # 0.5 m right, 3 m ahead
             ((20.0, 0.0, 0.0, 1.026), 3.0)]      # target 20 deg off the nose
    worst = max(abs(range_from_parallax(*c[0]) - c[1]) for c in cases)
    check("the parallax formula is right where the answer is known",
          worst < 0.01, f"worst error {worst * 100:.1f} cm")
    check("sight lines that do not meet give no answer",
          range_from_parallax(0.0, 0.0, 0.0, 0.5) is None
          and range_from_parallax(0.0, 9.46, 0.0, 0.5) is None,
          "parallel, and a target behind")

    TRUE_SCALE = 1.2          # real metres per EKF metre, as measured

    def fly(true_range: float, true_bearing: float = 0.0,
            noise_deg: float = 1.0, lag_s: float = 0.25, seed: int = 1,
            vanish_at: Optional[float] = None, drift_dps: float = 0.0,
            **params: Any) -> Dict[str, Any]:
        cfg = Config()
        rng = random.Random(seed)
        sk = build("measure_range", cfg, target_class="person", **params)
        # Everything true is in metres from where the flight began; the EKF
        # believes it has moved 1/TRUE_SCALE of what it really has.
        yaw = 0.0
        tx_ = true_range * math.cos(math.radians(true_bearing))
        ty_ = true_range * math.sin(math.radians(true_bearing))
        ex = ey = 0.0                       # EKF position
        gx = gy = 0.0                       # commanded EKF target
        t, status = 100.0, None
        hist: deque = deque(maxlen=int(lag_s / 0.1) + 1)

        def world() -> Any:
            rx, ry = ex * TRUE_SCALE, ey * TRUE_SCALE
            hist.append((rx, ry, yaw))
            ox, oy, oyaw = hist[0]           # what the camera saw, lag_s ago
            objs = []
            if vanish_at is None or t - 100.0 < vanish_at:
                b = math.degrees(math.atan2(ty_ - oy, tx_ - ox)) - oyaw
                b = (b + 180.0) % 360.0 - 180.0
                if abs(b) < 53.0:            # inside the 107 deg picture
                    objs = [{"label": "person", "track_id": 4,
                             "bearing_deg": b + rng.gauss(0.0, noise_deg),
                             "frames_seen": 30, "age_s": 3.0,
                             "range_m": 7.1, "range_source": "size_prior",
                             "bbox": [0.4, 0.3, 0.6, 1.0]}]
            return from_log({
                "t_sample_mono": t, "armed": True, "mode": "GUIDED",
                "landed_state": 2, "pose_age_s": 0.0,
                "pose": {"x": ex, "y": ey, "z": -0.70, "yaw_deg": yaw},
                "origin": [0.0, 0.0, 0.0], "rngfnd_down": 0.84,
                "rngfnd_down_trusted": True, "rngfnd_down_raw": 0.84,
                "objects": objs})

        w = world()
        pre = sk.check_preconditions(w)
        sk.start(w)
        while t - 100.0 < 70.0:
            st_ = sk.tick(w, 0.1)
            if st_.value != "running":
                status = st_.value
                break
            rq = sk.request
            if rq.kind is RequestKind.POSITION:
                gx, gy = rq.x, rq.y
            d = math.hypot(gx - ex, gy - ey)
            if d > 1e-9:                     # up to 1 m/s in the EKF frame
                k = min(1.0, 0.1 / d)
                ex, ey = ex + k * (gx - ex), ey + k * (gy - ey)
            yaw += drift_dps * 0.1
            t += 0.1
            w = world()
        return {"status": status, "detail": dict(sk._detail), "pre": pre,
                "home_m": math.hypot(ex, ey) * TRUE_SCALE, "t": t - 100.0,
                "skill": sk}

    a = fly(3.0)
    d = a["detail"]
    check("a person 3 m ahead is measured, and the truth is inside the band",
          a["status"] == "succeeded"
          and abs(d["range_m"] - 3.0) <= d["uncertainty_m"]
          and d["range_lo_m"] <= 3.0 <= d["range_hi_m"],
          f"{d['range_m']:.2f} +- {d['uncertainty_m']:.2f} m "
          f"({d['range_lo_m']:.2f}-{d['range_hi_m']:.2f}), parallax "
          f"{d['parallax_deg']:+.1f} deg, in {a['t']:.0f} s")
    check("it is better than the detector's own guess of the same person",
          abs(d["range_m"] - 3.0) < abs(d["range_prior_a_m"] - 3.0),
          f"parallax {d['range_m']:.2f} m against size prior "
          f"{d['range_prior_a_m']:.2f} m for a person at 3.00 m")
    check("the baseline it uses is the real one, not the EKF's",
          abs(d["baseline_m"] - 0.5) < 0.06
          and abs(d["height_scale"] - TRUE_SCALE) < 0.01,
          f"{d['baseline_m']:.2f} m stepped, scale {d['height_scale']:.2f} "
          f"(the EKF alone would say {d['baseline_m'] / TRUE_SCALE:.2f} m "
          f"and the range {3.0 / TRUE_SCALE:.2f} m)")
    check("and it comes back to where it started",
          a["home_m"] < 0.10 and d["returned_to_start"],
          f"{a['home_m'] * 100:.0f} cm from the start")

    b = fly(2.0, true_bearing=20.0)
    check("a target off to one side is measured by the same geometry",
          b["status"] == "succeeded"
          and abs(b["detail"]["range_m"] - 2.0)
          <= b["detail"]["uncertainty_m"],
          f"{b['detail']['range_m']:.2f} +- "
          f"{b['detail']['uncertainty_m']:.2f} m "
          f"for 2.00 m at 20 deg")
    def band(base: float) -> float:
        got = [fly(2.0, baseline_m=base, seed=k) for k in range(5)]
        return st.mean(r["detail"]["uncertainty_m"] for r in got
                       if r["status"] == "succeeded")
    wide, mid, tight = band(0.3), band(0.5), band(1.0)
    check("a longer step buys a tighter answer",
          tight < mid < wide,
          f"at 2 m: 0.3 m step +-{wide:.2f}, 0.5 m +-{mid:.2f}, "
          f"1.0 m +-{tight:.2f}")
    f = fly(3.0, direction=-1)
    check("stepping left measures the same range as stepping right",
          f["status"] == "succeeded"
          and abs(f["detail"]["range_m"] - 3.0)
          <= f["detail"]["uncertainty_m"],
          f"{f['detail']['range_m']:.2f} m")

    g = fly(12.0)
    check("a target too far for the step is refused, with the limit named",
          g["status"] == "failed" and "further than about" in g["skill"]._reason
          and g["home_m"] < 0.10,
          g["skill"]._reason[:72])
    h = fly(3.0, vanish_at=4.0)
    check("a target lost after the step: it comes home, then fails",
          h["status"] == "failed" and h["home_m"] < 0.10
          and "not in view" in h["skill"]._reason,
          f"{h['home_m'] * 100:.0f} cm from the start when it gave up")
    i = fly(3.0, true_bearing=45.0)
    check("a target that is not ahead is refused before any stepping",
          i["status"] == "failed" and "face it first" in i["skill"]._reason
          and i["home_m"] < 0.02)
    j = fly(3.0, drift_dps=2.0)
    check("a heading that drifts during the measurement is corrected for",
          j["status"] == "succeeded"
          and abs(j["detail"]["range_m"] - 3.0) <= j["detail"]["uncertainty_m"]
          and abs(j["detail"]["yaw_drift_deg"]) > 5.0,
          f"{j['detail']['range_m']:.2f} m with "
          f"{j['detail']['yaw_drift_deg']:+.1f} deg of drift")

    # ---- spread over seeds: is the stated uncertainty honest?
    runs = [fly(3.0, seed=k) for k in range(12)]
    good = [r for r in runs if r["status"] == "succeeded"]
    inside = sum(1 for r in good
                 if r["detail"]["range_lo_m"] <= 3.0 <= r["detail"]["range_hi_m"])
    errs2 = [abs(r["detail"]["range_m"] - 3.0) for r in good] or [99.0]
    check("over twelve runs the band holds and the error stays small",
          len(good) == 12 and inside >= 10 and max(errs2) < 0.9,
          f"{inside}/12 bands contain 3.00 m, worst error "
          f"{max(errs2) * 100:.0f} cm, median "
          f"{st.median(errs2) * 100:.0f} cm")

    cat = catalogue()
    prompt = pl.system_prompt(cat, 1.2, ["person"])
    steps, errs = pl.validate_plan({"steps": [
        {"skill": "rotate_search", "params": {"total_deg": 360,
                                              "target_class": "person"}},
        {"skill": "face_target", "params": {"target_class": "person"}},
        {"skill": "measure_range", "params": {"target_class": "person"}}]},
        cat, 1.2, ["person"])
    check("search, face and measure is a plan the validator accepts",
          not errs and len(steps) == 3 and "measure_range" in prompt
          and pl.validate_plan({"steps": [{"skill": "measure_range",
                                           "params": {}}]}, cat, 1.2)[1])


def test_target_lock() -> None:
    """
    Which detection a skill acts on. Two flights on 2026-11-16 turned the
    aircraft toward the wrong one and lost the person; this pins the rule
    that replaces the one that did it, in the situation those flights were
    actually in.
    """
    print("\ntarget lock")
    import signal
    from drone_agent.skills import build, SKILL_REGISTRY
    from drone_agent.skills.target_lock import TargetLock
    from drone_agent.safety.arbiter import RequestKind
    from drone_agent.world.state import from_log
    saved = {sig: signal.getsignal(sig)
             for sig in (signal.SIGINT, signal.SIGTERM)}
    import run_mission as rm
    for sig, h in saved.items():
        signal.signal(sig, h)

    def person(bearing: float, tid: int, sigma: float = 0.32,
               frames: int = 30) -> Dict[str, Any]:
        return {"label": "person", "track_id": tid, "bearing_deg": bearing,
                "frames_seen": frames, "age_s": 3.0, "range_m": 3.0,
                "range_sigma_m": sigma, "bbox": [0.4, 0.5, 0.6, 0.99]}

    def at(objs: List[Dict[str, Any]], yaw: float = 0.0,
           t: float = 100.0) -> Any:
        return from_log({"t_sample_mono": t, "armed": True, "mode": "GUIDED",
                         "landed_state": 2, "pose_age_s": 0.0,
                         "pose": {"x": 0.0, "y": 0.0, "z": -0.7,
                                  "yaw_deg": yaw},
                         "rngfnd_down": 0.8, "rngfnd_down_trusted": True,
                         "rngfnd_down_raw": 0.8, "objects": objs})

    # The real frame from flight 15:17: the search accepted track 19 at
    # +44 (range uncertainty 0.45 m) and rejected track 16 at -44 (0.66 m).
    two = [person(-44.2, 16, sigma=0.66), person(44.0, 19, sigma=0.45)]
    lock = TargetLock("person", track_id=19, bearing_deg=44.0,
                      heading_deg=-146.3)
    got = lock.pick(at(two, yaw=-146.3))
    check("the handed-over track is the one followed, not the nearer box",
          got is not None and got["track_id"] == 19,
          f"followed {got['track_id']} at {got['bearing_deg']:+.0f} deg, "
          f"not 16 at -44 deg")
    free = TargetLock("person")
    got2 = free.pick(at(two, yaw=-146.3))
    check("with nothing handed over it applies the search's own test",
          got2["track_id"] == 19,
          "the box whose range is certain enough to act on, not the "
          "nearest the nose")
    both_bad = TargetLock("person").pick(
        at([person(-30.0, 1, sigma=2.0), person(40.0, 2, sigma=2.0)]))
    check("if no box passes that test it falls back to the nearest",
          both_bad["track_id"] == 1)

    # The tracker drops an id across almost every turn and issues a new one.
    turned = TargetLock("person", track_id=19, bearing_deg=44.0,
                        heading_deg=0.0)
    turned.pick(at(two, yaw=0.0, t=100.0))
    after = turned.pick(at([person(2.1, 31)], yaw=43.6, t=101.5))
    check("a new id where the target should be after the turn is the target",
          after is not None and after["track_id"] == 31
          and turned.switches == 1,
          "track 19 at +44 before a 43.6 deg turn, track 31 at +2.1 after")
    stray = TargetLock("person", track_id=19, bearing_deg=44.0,
                       heading_deg=0.0)
    stray.pick(at(two, yaw=0.0, t=100.0))
    check("a new id somewhere else entirely is not the target",
          stray.pick(at([person(-43.0, 32)], yaw=0.0, t=101.5)) is None,
          "87 deg away, which is what the aircraft turned toward in the "
          "two failed flights")

    # ---- the whole skill, in the situation of those flights
    def face(objs_at: Any, handover: Optional[Dict[str, Any]],
             yaw0: float = -146.3, secs: float = 30.0) -> Dict[str, Any]:
        cfg = Config()
        sk = build("face_target", cfg, target_class="person",
                   handover=handover)
        yaw, target_yaw, t = yaw0, None, 100.0
        w = at(objs_at(t, yaw), yaw=yaw, t=t)
        sk.start(w)
        status = None
        while t - 100.0 < secs:
            st_ = sk.tick(w, 0.1)
            if st_.value != "running":
                status = st_.value
                break
            rq = sk.request
            if rq.kind is RequestKind.YAW:
                target_yaw = (rq.heading_deg + 180.0) % 360.0 - 180.0
            if target_yaw is not None:
                err = (target_yaw - yaw + 180.0) % 360.0 - 180.0
                yaw = (yaw + max(-1.75, min(1.75, err)) + 180.0) % 360.0 - 180.0
            t += 0.1
            w = at(objs_at(t, yaw), yaw=yaw, t=t)
        return {"status": status, "detail": dict(sk._detail), "skill": sk,
                "yaw": yaw, "t": t - 100.0}

    def real_pair(t: float, yaw: float) -> List[Dict[str, Any]]:
        # Both boxes stay where they are in the world as the aircraft turns.
        out = []
        for world_dir, tid, sigma in ((-146.3 + 44.0, 19, 0.45),
                                      (-146.3 - 44.2, 16, 0.66)):
            b = (world_dir - yaw + 180.0) % 360.0 - 180.0
            if abs(b) < 53.0:
                out.append(person(b, tid, sigma=sigma))
        return out

    good = face(real_pair, {"target_class": "person", "track_id": 19,
                            "bearing_deg": 44.0, "heading_deg": -146.3})
    check("handed the search's track, it turns the right way and faces it",
          good["status"] == "succeeded"
          and good["detail"]["turned_deg"] > 0
          and abs(good["detail"]["bearing_deg"]) <= 5.0,
          f"turned {good['detail']['turned_deg']:+.0f} deg, ended "
          f"{good['detail']['bearing_deg']:+.1f} deg off")

    # The person is centred and then the detector stops seeing her, which
    # is what happened at 15:02: four sightings, all within a degree of the
    # nose, then nothing for the rest of the flight.
    def fades(t: float, yaw: float) -> List[Dict[str, Any]]:
        if t - 100.0 > 1.2:
            return []
        return [person(-0.8 if int(t * 10) % 2 else -1.9, 7)]

    faded = face(fades, {"target_class": "person", "track_id": 7,
                         "bearing_deg": -1.0, "heading_deg": 0.0},
                 yaw0=0.0)
    check("a target that is already centred and then fades is faced, "
          "not failed",
          faded["status"] == "succeeded"
          and faded["detail"]["samples"] >= 2,
          f"decided on {faded['detail'].get('samples')} samples at "
          f"{faded['detail'].get('bearing_deg'):+.1f} deg")

    def fades_far(t: float, yaw: float) -> List[Dict[str, Any]]:
        return [] if t - 100.0 > 1.2 else [person(-30.0, 7)]

    far_gone = face(fades_far, None, yaw0=0.0)
    check("one that fades while far off the nose is not called faced",
          far_gone["status"] == "failed"
          and "not in view" in far_gone["skill"]._reason,
          far_gone["skill"]._reason)

    # ---- the hand-over is the executor's, not the planner's
    cat = SKILL_REGISTRY
    check("only the skills that declare it are handed the track",
          getattr(cat["face_target"], "accepts_handover", False)
          and getattr(cat["measure_range"], "accepts_handover", False)
          and not getattr(cat["hover"], "accepts_handover", False)
          and "handover" not in cat["face_target"].schema["properties"],
          "and it is not a parameter the model can set")

    class R:
        def __init__(self, ok: bool, reason: str,
                     detail: Optional[Dict[str, Any]] = None) -> None:
            self.ok, self.reason = ok, reason
            self.detail = detail or {}
            self.status = type("S", (), {"value": "succeeded" if ok
                                         else "failed"})()

        def to_log(self) -> Dict[str, Any]:
            return {"ok": self.ok}

    seen_handovers: List[Any] = []

    class Ex:
        def __init__(self, script: List[R]) -> None:
            self.script, self.ran = script, []

        def run_skill(self, skill: Any) -> R:
            self.ran.append(skill.name)
            seen_handovers.append(getattr(skill, "handover", None))
            return self.script[len(self.ran) - 1]

    import contextlib
    import io
    chain = [("rotate_search", {"total_deg": 360, "target_class": "person"}),
             ("face_target", {"target_class": "person"}),
             ("measure_range", {"target_class": "person"})]
    rec = FlightRecorder(path=str(Path(tempfile.mkdtemp()) / "h.jsonl"))
    with contextlib.redirect_stdout(io.StringIO()):
        ex = Ex([R(True, "target acquired",
                   {"target_class": "person", "track_id": 19,
                    "bearing_deg": 44.0, "heading_deg": -146.3}),
                 R(True, "facing target",
                   {"target_class": "person", "track_id": 31,
                    "bearing_deg": 1.2, "heading_deg": -102.0}),
                 R(True, "range measured", {"range_m": 2.4})])
        rm.run_steps(ex, Config(), chain, rec)
    rec.close()
    check("each step is told what the step before it found",
          seen_handovers[0] is None
          and seen_handovers[1]["track_id"] == 19
          and seen_handovers[2]["track_id"] == 31,
          "search -> face gets track 19, face -> measure gets track 31")


def test_camera_tilt() -> None:
    """
    The camera points above the airframe's forward axis and the calibration
    cannot see it. These pin what knowing that changes, and the range it
    makes possible.
    """
    print("\ncamera tilt")
    from drone_agent.perception.geometry import CameraIntrinsics

    def cam(tilt: float = 0.0) -> CameraIntrinsics:
        # equidistant fisheye, ~109 deg across, 16:9
        return CameraIntrinsics(fx=0.526, fy=0.917, cx=0.5, cy=0.5,
                                model="fisheye", width_px=1920,
                                height_px=1080, tilt_up_deg=tilt)

    flat, up = cam(), cam(15.0)
    check("without a measured tilt nothing changes at all",
          flat.angles(0.3, 0.8) == flat._camera_angles(0.3, 0.8)
          and flat.height_range(0.45, 1.75, 1.10) is None,
          "every flight up to 0.23.0 worked in the camera's own frame")
    b, e = up.angles(0.5, 0.5)
    check("the optical axis itself just points up by the tilt",
          abs(b) < 1e-9 and abs(e - 15.0) < 1e-9)
    b0, e0 = flat.angles(0.25, 0.78)
    b1, e1 = up.angles(0.25, 0.78)
    check("an object below the axis reads at a smaller bearing than the "
          "camera's own frame says",
          abs(b1) < abs(b0) and e1 > e0,
          f"{b0:+.1f} deg in the camera's frame is {b1:+.1f} in the "
          f"airframe's; elevation {e0:+.1f} -> {e1:+.1f}")
    # The error that matters is in a bearing DIFFERENCE, which is what a
    # parallax range is made of.
    d_cam = flat.angles(0.25, 0.78)[0] - flat.angles(0.5, 0.78)[0]
    d_air = up.angles(0.25, 0.78)[0] - up.angles(0.5, 0.78)[0]
    check("and a bearing difference is inflated by a tilt left unmeasured",
          abs(d_cam) > abs(d_air) * 1.05,
          f"{d_cam:+.1f} deg against {d_air:+.1f} deg, "
          f"{100 * (d_cam / d_air - 1):+.0f}%")

    # Round trip: put a target at a known range and see it measured back.
    import math as _m
    for truth, tilt, hc, ho in ((2.57, 13.0, 1.10, 1.78),
                                (4.00, 20.0, 0.80, 1.78),
                                (1.50, 8.0, 1.20, 0.90)):
        c = cam(tilt)
        want_elev = _m.degrees(_m.atan2(ho - hc, truth))
        lo, hi = 0.0, 1.0                     # find the row the top lands on
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if c.angles(0.5, mid)[1] > want_elev:
                lo = mid
            else:
                hi = mid
        got = c.height_range(0.5 * (lo + hi), ho, hc)
        check(f"a {ho:.2f} m target at {truth:.2f} m measures back",
              got is not None and abs(got - truth) < 0.02,
              f"{got:.2f} m with the camera {tilt:.0f} deg up at {hc:.2f} m")

    c = cam(13.0)
    near = c.height_range(0.46, 1.78, 1.10)
    far = c.height_range(0.50, 1.78, 1.10)
    check("a target whose head is lower in the frame is further away",
          near is not None and far is not None and far > near,
          f"{near:.2f} m and {far:.2f} m")
    check("a head high in the frame means a target close in front",
          (c.height_range(0.05, 1.78, 1.10) or 9.9) < 1.0
          and c.height_range(0.46, 1.00, 1.10) is None,
          "and a target shorter than the camera, seen below the horizon, "
          "gives no range at all")

    # The tilt arithmetic the ground measurement does.
    rise, dist = 0.58, 2.0
    check("the wall measurement gives the tilt",
          abs(_m.degrees(_m.atan2(rise, dist)) - 16.17) < 0.01,
          f"{rise:.2f} m of rise at {dist:.1f} m is "
          f"{_m.degrees(_m.atan2(rise, dist)):.2f} deg")
    src = (Path(__file__).resolve().parent / "calibrate_camera.py").read_text()
    check("and the tool that takes it is wired up",
          '"tilt"' in src and "def run_tilt" in src
          and "tilt_up_deg" in src)


def test_tilt_from_floor() -> None:
    """
    The tilt solved from one picture: a mark at a known height and the line
    where the wall meets the floor, which is at zero. Two features at two
    known heights give the distance and the tilt together, so nobody has to
    measure the distance to the wall -- which was got wrong twice, both
    times by running the example in the documentation.
    """
    print("\ntilt from the floor line")
    import subprocess
    from drone_agent.perception.geometry import CameraIntrinsics
    root = Path(__file__).resolve().parent.parent
    tmp = Path(tempfile.mkdtemp())
    cam = tmp / "camera.json"
    intr = CameraIntrinsics(fx=0.5174, fy=0.9347, cx=0.5130, cy=0.4954,
                            k1=0.1471, k2=-0.1081, k3=-0.0589, k4=0.0493,
                            model="fisheye", width_px=1920, height_px=1080)
    intr.save(str(cam))

    def solve(junction_v: float, mark: float, camh: float) -> tuple:
        out = subprocess.run(
            [sys.executable, str(root / "tools" / "calibrate_camera.py"),
             "tilt", "--out", str(cam), "--mark-height", str(mark),
             "--camera-height", str(camh), "--junction-v",
             str(junction_v)], capture_output=True, text=True)
        d = json.loads(cam.read_text()).get("tilt_up_deg", 0.0)
        far = 0.0
        for line in out.stdout.splitlines():
            if "the wall is" in line:
                far = float(line.split("the wall is")[1].split("m")[0])
        return d, far, out.stdout

    # Round trip: invent a room, work out where the two features land, and
    # see the tool recover the room.
    for true_tilt, true_D, camh in ((13.5, 2.40, 0.66),
                                    (20.0, 3.00, 0.30),
                                    (8.0, 2.00, 0.40)):
        # The mark is BY DEFINITION the height the camera's horizon falls
        # at, so it follows from the room and the tilt; it cannot be chosen
        # separately. Stipulating both is how this test first described a
        # room that does not exist.
        mark = camh + true_D * math.tan(math.radians(true_tilt))
        want_mark = math.degrees(math.atan2(mark - camh, true_D))
        want_floor = math.degrees(math.atan2(-camh, true_D))
        # the row the floor line falls on, in the camera's own frame
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if intr._camera_angles(intr.cx, mid)[1] > want_floor - true_tilt:
                lo = mid
            else:
                hi = mid
        v = 0.5 * (lo + hi)
        # This frame sees down to -30 deg and no further. A case whose floor
        # line falls below that cannot be read off a picture at all, and
        # feeding the bottom row in its place tests nothing.
        if v > 0.995:
            check(f"a camera {true_tilt:.0f} deg up at {true_D:.1f} m is "
                  f"recovered from one picture", False,
                  f"the floor line is {want_floor - true_tilt:+.1f} deg "
                  f"down, below the bottom of the frame: not a case the "
                  f"tool can be given")
            continue
        got_tilt, got_far, _ = solve(v, mark, camh)
        ok = abs(got_tilt - true_tilt) < 0.3 and abs(got_far - true_D) < 0.06
        check(f"a camera {true_tilt:.0f} deg up at {true_D:.1f} m is "
              f"recovered from one picture", ok,
              f"{got_tilt:.2f} deg and {got_far:.2f} m, from a mark "
              f"{want_mark:+.1f} deg up and a floor line "
              f"{want_floor - true_tilt:+.1f} deg down")

    before = json.loads(cam.read_text()).get("tilt_up_deg", 0.0)
    _, far, out = solve(0.60, 1.24, 0.66)
    after = json.loads(cam.read_text()).get("tilt_up_deg", 0.0)
    check("a floor line read far too high is refused, not saved",
          "not a believable distance for a room" in out and after == before,
          f"it would have put the wall {far:.1f} m away in a room with a "
          f"2.5 m ceiling")

    # And the error bar, which is the honest reason the far ones go: one
    # notch of the scale is centimetres close in and metres far out.
    _, _, good = solve(0.98, 1.24, 0.66)
    span = [l for l in good.splitlines() if "one notch" in l]
    lo_hi = span[0].split("becomes")[1].replace("m", "").split("-") if span \
        else ["0", "9"]
    check("and one notch of reading error is reported with the distance",
          bool(span) and float(lo_hi[1]) - float(lo_hi[0]) < 0.25,
          span[0].strip() if span else "no error bar printed")

    bad = subprocess.run(
        [sys.executable, str(root / "tools" / "calibrate_camera.py"),
         "tilt", "--out", str(cam), "--distance", "2.0",
         "--mark-height", "0.70", "--camera-height", "0.12"],
        capture_output=True, text=True)
    check("and the three numbers out of the documentation are refused",
          bad.returncode == 2 and "not measurements" in bad.stdout,
          "they were run verbatim twice, and 16.17 deg of nothing came out")


def test_approach() -> None:
    """
    approach against a simulated room where the range is not handed to the
    skill: a person of a real height stands at a real distance, the camera
    geometry puts a box where that person would appear, and the range comes
    out of the same RangeEstimator the aircraft runs. The EKF under-reads
    distance by the 1.2 measured on the tape flights, the picture is a
    quarter of a second old, and the box edge jitters by the 0.0103 of a
    frame height measured in the flights.

    What is pinned is what could hurt someone: that it stops at the real
    distance and not the believed one, that no parameter gets it inside the
    floor, that it refuses a size-prior range, and that it never yaws.
    """
    print("\napproach")
    import random
    import statistics as st
    from collections import deque
    from drone_agent.skills import build, catalogue
    from drone_agent.perception.geometry import (CameraIntrinsics,
                                                 RangeEstimator,
                                                 observation_from_bbox)
    from drone_agent.safety.arbiter import RequestKind
    from drone_agent.world.state import from_log
    from drone_agent import planner as pl

    intr = CameraIntrinsics(fx=0.5174, fy=0.9347, cx=0.5130, cy=0.4954,
                            k1=0.1471, k2=-0.1081, k3=-0.0589, k4=0.0493,
                            model="fisheye", width_px=1920, height_px=1080,
                            tilt_up_deg=13.53)

    def uv_for(bearing: float, elev: float) -> Tuple[float, float]:
        """Where a ray at this airframe-frame bearing and elevation lands."""
        u, v = intr.cx, intr.cy
        for _ in range(3):
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if intr.angles(u, mid)[1] > elev:
                    lo = mid
                else:
                    hi = mid
            v = 0.5 * (lo + hi)
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if intr.angles(mid, v)[0] < bearing:
                    lo = mid
                else:
                    hi = mid
            u = 0.5 * (lo + hi)
        return u, v

    TRUE_SCALE = 1.2
    PERSON_H = 1.73
    CAM_H = 1.10

    def fly(true_range: float, true_bearing: float = 0.0, seed: int = 1,
            tilt: float = 13.53, known: bool = True, walks_mps: float = 0.0,
            strafes_mps: float = 0.0, camera_h: float = CAM_H,
            clip_head: bool = False, max_s: float = 120.0,
            **params: Any) -> Dict[str, Any]:
        cfg = Config()
        rnd = random.Random(seed)
        est = RangeEstimator(
            intrinsics=CameraIntrinsics(**{**intr.__dict__,
                                           "tilt_up_deg": tilt}),
            known_heights={"person": PERSON_H} if known else {})
        sk = build("approach", cfg, target_class="person", **params)
        yaw = 0.0
        tx_ = true_range * math.cos(math.radians(true_bearing))
        ty_ = true_range * math.sin(math.radians(true_bearing))
        ex = ey = 0.0                       # EKF position
        gx = gy = 0.0
        t, status = 100.0, None
        hist: deque = deque(maxlen=3)       # 0.25 s of camera lag
        yaws: List[tuple] = []

        def world() -> Any:
            rx, ry = ex * TRUE_SCALE, ey * TRUE_SCALE
            hist.append((rx, ry, yaw))
            ox, oy, oyaw = hist[0]
            objs = []
            dx, dy = tx_ - ox, ty_ - oy
            r_seen = math.hypot(dx, dy)
            b = (math.degrees(math.atan2(dy, dx)) - oyaw + 180.0) % 360.0 \
                - 180.0
            elev = math.degrees(math.atan2(PERSON_H - camera_h, r_seen))
            if abs(b) < 50.0 and r_seen > 0.3:
                u, v = uv_for(b, elev)
                v = max(0.0, v + rnd.gauss(0.0, 0.0103))
                if clip_head:
                    v = 0.0
                # Through the production path: the bearing and elevation
                # come out of the bbox and the intrinsics, not from the
                # simulator, so the tilt is in the loop twice over.
                obs = observation_from_bbox(
                    "person", (max(0.0, u - 0.05), v,
                               min(1.0, u + 0.05), min(1.0, v + 0.45)),
                    est.intrinsics, 0.9, 4, t)
                obs.frames_seen = 30
                est.estimate(obs, camera_height_m=camera_h, pitch_deg=-2.0)
                o = obs.to_log()
                o["frames_seen"] = 30
                o["age_s"] = 3.0
                objs = [o]
            return from_log({
                "t_sample_mono": t, "armed": True, "mode": "GUIDED",
                "landed_state": 2, "pose_age_s": 0.0,
                "pose": {"x": ex, "y": ey, "z": -camera_h / TRUE_SCALE,
                         "yaw_deg": yaw, "pitch_deg": -2.0},
                "origin": [0.0, 0.0, 0.0], "rngfnd_down": camera_h,
                "rngfnd_down_trusted": True, "rngfnd_down_raw": camera_h,
                "objects": objs})

        w = world()
        pre = sk.check_preconditions(w)
        sk.start(w)
        while t - 100.0 < max_s:
            st_ = sk.tick(w, 0.1)
            if st_.value != "running":
                status = st_.value
                break
            rq = sk.request
            if rq.kind is RequestKind.POSITION:
                gx, gy = rq.x, rq.y
                # A position target carrying the heading it started with is
                # a hold. Carrying any OTHER heading is a turn.
                if (rq.yaw_rad is not None
                        and abs(rq.yaw_rad - 0.0) > math.radians(0.5)):
                    yaws.append(("position-yaw", round(rq.yaw_rad, 3)))
            elif rq.kind is RequestKind.YAW:
                yaws.append(("condition-yaw", rq.heading_deg))
            d = math.hypot(gx - ex, gy - ey)
            if d > 1e-9:
                k = min(1.0, 0.1 / d)
                ex, ey = ex + k * (gx - ex), ey + k * (gy - ey)
            tx_ += walks_mps * 0.1          # the target leaves, if it does
            ty_ -= strafes_mps * 0.1        # or steps aside, as one did
            t += 0.1
            w = world()
        real = math.hypot(tx_ - ex * TRUE_SCALE, ty_ - ey * TRUE_SCALE)
        return {"status": status, "detail": dict(sk._detail), "pre": pre,
                "real_m": real, "t": t - 100.0, "skill": sk,
                "yaws": yaws, "travelled_real_m":
                    math.hypot(ex, ey) * TRUE_SCALE}

    a = fly(3.0, standoff_m=1.7)
    d = a["detail"]
    check("a person 3 m away is approached to a real 1.7 m",
          a["status"] == "succeeded" and abs(a["real_m"] - 1.7) <= 0.35,
          f"stopped with the person really {a['real_m']:.2f} m away, "
          f"believing {d.get('range_m')} m, in {d.get('steps')} steps over "
          f"{a['t']:.0f} s")
    check("and it stopped on the real distance, not the EKF's idea of it",
          abs(a["travelled_real_m"] - 1.3) < 0.5,
          f"travelled a real {a['travelled_real_m']:.2f} m; without the "
          f"height correction the same commands would have moved "
          f"{a['travelled_real_m'] * TRUE_SCALE:.2f} m")
    check("it never turns: every correction is a sideways step",
          not a["yaws"],
          "no CONDITION_YAW and no yaw in any position target - a yaw costs "
          "this airframe 10-16 deg of heading, a step costs 10% of its length")
    rungs = d.get("ladder") or []
    check("the range is re-measured between every step and logged",
          len(rungs) >= 2
          and all("range_m" in r for r in rungs)
          and rungs[0]["range_m"] > rungs[-1]["range_m"],
          " -> ".join(f"{r['range_m']:.2f}" for r in rungs) + " m")

    b = fly(3.0, standoff_m=1.7)
    check("asked for the closest it will go, it does not end up inside it",
          b["status"] == "succeeded" and b["real_m"] >= 1.7,
          f"floor is 1.7 m believed, ended at a real {b['real_m']:.2f} m - "
          f"the floor is held against the range's uncertainty, so it stops "
          f"short of it rather than on it. 1.7 believed is about 1.25 real "
          f"on the aircraft, where the range over-reads at close distance")
    c = fly(3.5, standoff_m=0.5)
    check("a standoff under the floor is raised to it, not honoured",
          c["skill"].standoff == 1.7 and c["real_m"] >= 1.2,
          f"asked 0.50 m, used {c['skill'].standoff:.2f} m, ended at a real "
          f"{c['real_m']:.2f} m")

    e = fly(3.0, tilt=0.0, known=False, standoff_m=1.7, max_s=20.0)
    check("without a measured camera tilt it refuses to move at all",
          e["status"] == "failed" and e["travelled_real_m"] < 0.05
          and "size_prior" in str(e["detail"].get("sources_seen")),
          f"{e['skill']._reason}: {e['detail'].get('sources_seen')} - that "
          f"range read 25-31% long against a tape, which would have put the "
          f"aircraft a quarter closer than it believed")
    f = fly(3.0, clip_head=True, standoff_m=1.7, max_s=20.0)
    check("a target whose head is out of frame is not ranged",
          f["status"] == "failed" and f["travelled_real_m"] < 0.05
          and "out of frame" in str(f["skill"].metrics.get("last_rejection")),
          "the head range is measured from the top edge of the box, so a "
          "clipped head reads nearer than it is")
    g = fly(3.0, true_bearing=40.0, standoff_m=1.7, max_s=20.0)
    check("a target well off the nose is refused, not strafed after",
          g["status"] == "failed" and g["travelled_real_m"] < 0.05
          and "face it first" in str(g["skill"].metrics.get(
              "last_rejection")),
          "40 deg off; face_target exists for that")

    # A target receding straight along the line of sight is the case the
    # across-sight check CANNOT see, by construction: it has no lateral
    # component. That is the price of not acting on the along-sight
    # component, which the taped over-read showed is contaminated. Such a
    # target still stops the approach -- on the travel budget, the step
    # limit, or the range going too loose to act on -- so the aircraft does
    # not chase it indefinitely, but it is not reported as movement.
    h = fly(3.0, standoff_m=1.7, walks_mps=0.35)
    check("a target receding straight away still stops the approach",
          h["status"] == "failed",
          f"{h['skill']._reason[:66]} - the across-sight check cannot see a "
          f"purely radial retreat, so the budget and the range quality are "
          f"what stop it")
    # A target stepping aside is MEASURED and logged, but no longer
    # refused on. Over 25 real approaches the bearing drifted the same way
    # in 24 of them at a median -1.04 deg/s while the EKF held its heading
    # to +-3.3 deg: the aircraft's own unseen yaw produces the same
    # lateral gap a moving target does, so refusing on it refused the
    # still person 24 times out of 25.
    k = fly(3.0, standoff_m=1.7, walks_mps=0.0, strafes_mps=0.30)
    lat = [r for r in (k["detail"].get("ladder") or [])
           if "across_m" in r]
    check("a target stepping aside is measured, and no longer refused on",
          lat and max(abs(r["across_m"]) for r in lat) > 0.2
          and "has moved" not in (k["skill"]._reason or ""),
          f"lateral gap reached "
          f"{max((abs(r['across_m']) for r in lat), default=0):.2f} m and "
          f"is logged with the yaw drift it implies; what stops the "
          f"approach is the cone, which is the right answer whether the "
          f"target walked or the aircraft turned")
    check("and the yaw drift it would imply is reported with it",
          any("implied_yaw_drift_dps" in r
              for r in (k["detail"].get("ladder") or [])),
          "every approach now carries its own measurement of the heading "
          "drift, which is the number this airframe keeps being limited by")
    m = fly(3.0, standoff_m=1.7, strafes_mps=0.60)
    check("a target that leaves the cone stops the approach for a re-face",
          m["status"] == "failed"
          and (m["detail"].get("needs_reface")
               or "has moved" in m["skill"]._reason),
          f"{m['skill']._reason[:70]} - measuring on whichever frames are "
          f"still inside 30 deg reports a convergence that is a biased "
          f"sample")
    n = fly(3.0, standoff_m=1.7, camera_h=1.45)
    check("and a camera that has climbed to the target's own height refuses",
          n["status"] == "failed" and n["travelled_real_m"] < 0.05,
          f"with the camera at 1.45 m under a 1.73 m target the rise is "
          f"0.28 m, under the 0.35 m floor: {n['skill']._reason[:46]}")
    i = fly(3.9, standoff_m=1.7)
    check("and a close that needs more travel than the budget stops",
          i["status"] in ("succeeded", "failed")
          and i["travelled_real_m"] <= 3.0 + 0.5,
          f"travelled a real {i['travelled_real_m']:.2f} m against a "
          f"{3.0} m budget, {i['status']}")

    # ---- is it repeatable, or was that one lucky seed?
    runs = [fly(3.0, standoff_m=1.7, seed=k) for k in range(8)]
    good = [r for r in runs if r["status"] == "succeeded"]
    errs = [abs(r["real_m"] - 1.7) for r in good] or [9.0]
    check("over eight runs it arrives, and no run ends inside the floor",
          len(good) >= 7 and max(errs) < 0.5
          and min(r["real_m"] for r in good) >= 1.7,
          f"{len(good)}/8 arrived, real distance "
          f"{min(r['real_m'] for r in good):.2f}-"
          f"{max(r['real_m'] for r in good):.2f} m, median error "
          f"{st.median(errs) * 100:.0f} cm")

    cat = catalogue()
    prompt = pl.system_prompt(cat, 1.2, ["person"])
    steps, errs2 = pl.validate_plan({"steps": [
        {"skill": "rotate_search", "params": {"total_deg": 360,
                                              "target_class": "person"}},
        {"skill": "face_target", "params": {"target_class": "person"}},
        {"skill": "approach", "params": {"target_class": "person",
                                         "standoff_m": 2.0}}]},
        cat, 1.2, ["person"])
    check("search, face and approach is a plan the validator accepts",
          not errs2 and len(steps) == 3 and "approach" in prompt
          and pl.estimate_s(steps) > 60.0,
          f"estimated {pl.estimate_s(steps):.0f} s")
    check("and a standoff the skill would refuse is refused in the plan",
          pl.validate_plan({"steps": [
              {"skill": "approach",
               "params": {"target_class": "person", "standoff_m": 0.5}}]},
              cat, 1.2, ["person"])[1],
          "0.5 m is under the schema's 1.2 m minimum, so it never reaches "
          "the aircraft")


def test_grader() -> None:
    """
    The graded evaluation. Before this the bench could say a plan was valid
    JSON that passed the schema, and that an impossible request had been
    refused; it could not say whether the plan was the RIGHT plan and said
    so in a comment. A number in a thesis cannot rest on a comment telling
    the operator to read the log.

    Two things are pinned: that the grader can tell a right plan from a
    plausible wrong one, and that every rubric in the shipped evaluation
    set is satisfiable -- a case no plan can pass would quietly subtract
    from the score for the rest of the project.
    """
    print("\ngraded evaluation")
    from drone_agent.grade import Case, grade, load_cases
    from drone_agent.planner import validate_plan
    from drone_agent.skills import catalogue
    cat = catalogue()
    classes = ["person", "chair"]

    def plan(*steps: Any) -> Any:
        got, errs = validate_plan({"steps": [
            {"skill": sk, "params": p} for sk, p in steps]},
            cat, 1.2, classes)
        assert not errs, errs
        return got

    case = Case(text="find a person and turn to face them", id="t",
                must=[{"skill": "rotate_search",
                       "params": {"target_class": "person"}},
                      {"skill": "face_target",
                       "params": {"target_class": "person"}}],
                forbid=["move"], max_steps=3)
    right = plan(("rotate_search", {"total_deg": 360,
                                    "target_class": "person"}),
                 ("face_target", {"target_class": "person"}))
    check("a right plan passes", grade(case, right, True)[0],
          "search for a person, then face it")
    check("and so does a right plan with a harmless extra step",
          grade(case, plan(("rotate_search", {"total_deg": 360,
                                              "target_class": "person"}),
                           ("face_target", {"target_class": "person"}),
                           ("hover", {"duration_s": 2})), True)[0],
          "`must` is an ordered subsequence, not a reference plan: several "
          "plans are right")
    # Built directly: from 0.38 the validator refuses this order before the
    # grader ever sees it, which is checked below, but the grader must still
    # fail it on its own -- a saved log from an older stack can contain it.
    from drone_agent.planner import Step
    swapped = [Step("face_target", {"target_class": "person"}, ""),
               Step("rotate_search", {"total_deg": 360,
                                      "target_class": "person"}, "")]
    check("and the validator refuses that order before it can fly",
          bool(validate_plan({"steps": [
              {"skill": "face_target", "params": {"target_class": "person"}},
              {"skill": "rotate_search", "params": {
                  "total_deg": 360, "target_class": "person"}}]},
              cat, 1.2, classes)[1]),
          "face_target needs a rotate_search before it")
    check("the same two skills in the wrong order fail",
          not grade(case, swapped, True)[0],
          grade(case, swapped, True)[1][0][:76])
    wrong_class = plan(("rotate_search", {"total_deg": 360,
                                          "target_class": "chair"}),
                       ("face_target", {"target_class": "chair"}))
    bad = grade(case, wrong_class, True)
    check("searching for the wrong thing fails, and the reason says so",
          not bad[0] and "chair" in bad[1][0],
          bad[1][0][:76])
    check("a forbidden skill fails even with everything else right",
          not grade(case, plan(
              ("rotate_search", {"total_deg": 360,
                                 "target_class": "person"}),
              ("move", {"forward_m": 0.5}),
              ("face_target", {"target_class": "person"})), True)[0],
          "move is not what `find a person and face them` asks for")
    check("a refusal of something possible fails",
          not grade(case, [], True, "I cannot do that")[0],
          grade(case, [], True, "x")[1][0][:70])

    # or_refuse: the two cases where the architecture's own limit is the
    # right answer and so is a refusal. Marking only one of them right
    # measures an opinion about refusals, not the system.
    either = Case(text="fly right up to my face as close as you can get",
                  id="e", or_refuse=True,
                  must=[{"skill": "approach",
                         "params": {"standoff_m": {"approx": 1.7,
                                                   "tol": 0.001}}}])
    # Search and face first: from 0.38 the validator refuses an approach
    # nothing has faced, and these test the grader, not that rule.
    lead = (("rotate_search", {"total_deg": 360, "target_class": "person"}),
            ("face_target", {"target_class": "person"}))
    at_floor = plan(*lead, ("approach", {"target_class": "person",
                                         "standoff_m": 1.7}))
    too_close = plan(*lead, ("approach", {"target_class": "person",
                                          "standoff_m": 3.0}))
    check("where clamping to a schema limit is as right as refusing, both "
          "pass",
          grade(either, at_floor, True)[0]
          and grade(either, [], True, "it cannot get closer than 1.7 m")[0],
          "qwen3:8b planned approach at the schema minimum "
          "and the skill's hard floor; the first version of this set called "
          "that a failure")
    check("but it is the limit that passes, not any plan at all",
          not grade(either, too_close, True)[0]
          and not grade(either, [], True, "")[0],
          "a 3 m standoff is not 'as close as you can get', and a silent "
          "refusal is not an answer")

    # must_any: an instruction whose English has two readings.
    amb = Case(text="turn around", id="a", must_any=[
        [{"skill": "rotate_search",
          "params": {"total_deg": {"approx": 360, "tol": 20}}}],
        [{"skill": "rotate_search",
          "params": {"total_deg": {"approx": 180, "tol": 20}}}]])
    check("both readings of an ambiguous instruction pass, and a third "
          "does not",
          grade(amb, plan(("rotate_search", {"total_deg": 360})), True)[0]
          and grade(amb, plan(("rotate_search", {"total_deg": 180})),
                    True)[0]
          and not grade(amb, plan(("rotate_search", {"total_deg": 90})),
                        True)[0],
          "'turn around' is a full circle to one reader and an about-face "
          "to another; 90 degrees is neither")

    deny = Case(text="fly to the kitchen", id="d", refuse=True)
    check("a refusal with a reason passes where one is required",
          grade(deny, [], True, "the drone cannot go to a named place")[0])
    check("a bare refusal with no reason does not",
          not grade(deny, [], True, "")[0],
          "the operator is owed the reason, and so is the person reading "
          "the log")
    check("and planning anything at all for an impossible request fails",
          not grade(deny, plan(("hover", {"duration_s": 5})), True)[0],
          "doing part of an impossible instruction is the failure mode the "
          "system prompt names")

    # Numbers, which is where a plausible-looking plan is usually wrong.
    num = Case(text="turn half way around to the left", id="n",
               must=[{"skill": "rotate_search",
                      "params": {"total_deg": {"approx": 180, "tol": 20},
                                 "direction": -1}}])
    check("a number inside tolerance passes and outside it fails",
          grade(num, plan(("rotate_search", {"total_deg": 190,
                                             "direction": -1})), True)[0]
          and not grade(num, plan(("rotate_search", {"total_deg": 360,
                                                     "direction": -1})),
                        True)[0]
          and not grade(num, plan(("rotate_search", {"total_deg": 180,
                                                     "direction": 1})),
                        True)[0],
          "180 +-20 accepts 190, rejects 360, and the wrong direction is "
          "the wrong plan")
    check("a parameter left at its default is not a parameter that was set",
          not grade(num, plan(("rotate_search", {"direction": -1})),
                    True)[0],
          "the skill's default is not what the instruction asked for, so "
          "the rubric wants it stated")

    # The shipped set: every rubric satisfiable, every case well formed.
    root = Path(__file__).resolve().parent.parent
    cases = load_cases(str(root / "tools" / "eval_set.jsonl"))
    unsat = []
    for c in cases:
        if c.refuse:
            if not grade(c, [], True, "cannot")[0]:
                unsat.append((c.id, "a correct refusal fails"))
            continue
        # Every alternative reading must be satisfiable on its own, not
        # just one of them: an ambiguous case whose second reading nothing
        # can pass is a case with one reading, dressed up as two.
        for which, option in enumerate(c.must_any or [c.must]):
            steps = []
            for req in option:
                p = {}
                for k, v in (req.get("params") or {}).items():
                    if isinstance(v, dict):
                        if v.get("absent"):
                            continue
                        p[k] = v.get("approx", v.get("min", v.get("max")))
                    else:
                        p[k] = v
                steps.append((req["skill"], p))
            got, errs = validate_plan({"steps": [{"skill": sk, "params": p}
                                                 for sk, p in steps]},
                                      cat, 1.2, classes)
            label = f"{c.id} reading {which + 1}" if c.must_any else c.id
            if errs:
                unsat.append((label, f"the rubric's own plan is not valid: "
                                     f"{errs[0]}"))
            elif not grade(c, got, True)[0]:
                unsat.append((label, f"the rubric's own plan does not pass "
                                     f"it: {grade(c, got, True)[1]}"))
        if c.or_refuse and not grade(c, [], True, "cannot")[0]:
            unsat.append((c.id, "or_refuse is set but a refusal fails"))
    cats = sorted({c.category for c in cases})
    check(f"all {len(cases)} cases in the shipped set are satisfiable",
          not unsat, f"{len(cases)} instructions over {len(cats)} "
                     f"categories: {', '.join(cats)}"
          if not unsat else str(unsat[:3]))
    needed = {r["skill"] for c in cases for r in c.must}
    needed |= {r["skill"] for c in cases for opt in c.must_any for r in opt}
    check("and the set still covers refusals and every skill that flies",
          sum(c.refuse for c in cases) >= 10
          and needed >= {"rotate_search", "face_target", "measure_range",
                         "move", "hover", "climb_to", "approach"},
          f"{sum(c.refuse for c in cases)} must be refused; the graded ones "
          f"between them require every skill")
    bad_file = Path(tempfile.mkdtemp()) / "bad.jsonl"
    bad_file.write_text('{"text": "x", "refuse": true, "must": '
                        '[{"skill": "hover"}]}\n')
    try:
        load_cases(str(bad_file))
        contradiction = False
    except ValueError:
        contradiction = True
    check("a case that contradicts itself is refused at load time",
          contradiction,
          "refuse: true with required steps is a rubric nothing can pass")


def test_landing_is_recorded() -> None:
    """
    0.38: the descent is in the log. Until then ticks stopped when the last
    skill ended, so every tape measurement of a landing spot was compared
    against a log that missed the last 4-5 s of the flight.
    """
    print("\nthe landing is recorded")
    from types import SimpleNamespace
    from drone_agent.safety.landing import LandingController
    import drone_agent.safety.landing as ld
    worlds = [SimpleNamespace(mode_num=9, link_up=True, armed=True,
                              on_ground=lambda a, b: (False, []), z=-0.8 + k / 10)
              for k in range(6)]
    worlds.append(SimpleNamespace(mode_num=9, link_up=True, armed=False,
                                  on_ground=lambda a, b: (True, ["rf"])))
    it = iter(worlds)
    seen = []
    veh = SimpleNamespace(poll=lambda: next(it), expected_mode=None,
                          mode_map={"LAND": 9, "GUIDED": 4},
                          command_mode_oneshot=lambda m: None,
                          command_nav_land=lambda: None,
                          disarm=lambda: True)
    lc = LandingController(veh, Config(), verbose=False,
                           on_tick=lambda w: seen.append(w))
    lc._progress = lambda w: None
    real_sleep = ld.time.sleep
    ld.time.sleep = lambda s: None
    try:
        done = lc.force_land("test")
    finally:
        ld.time.sleep = real_sleep
    check("every state polled while landing reaches the recorder",
          done and len(seen) == len(worlds),
          f"{len(seen)} of {len(worlds)} states, the last one disarmed")

    def boom(w: Any) -> None:
        raise RuntimeError("disk full")
    it2 = iter(list(worlds))
    veh.poll = lambda: next(it2)
    lc2 = LandingController(veh, Config(), verbose=False, on_tick=boom)
    lc2._progress = lambda w: None
    ld.time.sleep = lambda s: None
    try:
        done2 = lc2.force_land("test")
    finally:
        ld.time.sleep = real_sleep
    check("and a recorder that fails does not stop the landing", done2,
          "recording is never allowed to be the reason an aircraft stays up")


def test_vertical_from_binlog() -> None:
    """
    The arithmetic that reads an autopilot .bin and says why the aircraft
    climbed. The .bin itself cannot be shipped, so the sums are checked
    against cases whose answer is known: a tilted but motionless aircraft
    must still read -g on the down axis, an injected bias must come back
    out, and a log holding two flights with a pause between them must not
    be averaged into one.
    """
    print("\nthe vertical axis, from the autopilot's own log")
    import subprocess
    from drone_agent.vertical import (G, down_specific_force,
                                      airborne_spans, grounded_spans,
                                      longest, runs, summarise)

    # 1. Tilt is geometry, not error. This is the step that separates "the
    #    accelerometer is wrong" from "the aircraft was leaning".
    level = down_specific_force((0.0, 0.0, -G), 0.0, 0.0)
    lean = down_specific_force(
        (G * math.sin(math.radians(14.0)), 0.0,
         -G * math.cos(math.radians(14.0))), 0.0, 14.0)
    check("a motionless aircraft reads -g down however it is tilted",
          abs(level + G) < 1e-9 and abs(lean + G) < 1e-9,
          f"level {level:.4f}, leaning 14 deg {lean:.4f}; on its own body "
          f"Z axis the leaning one reads {-G * math.cos(math.radians(14)):.2f}, "
          f"which is 0.29 light and means nothing")
    rolled = down_specific_force(
        (0.0, -G * math.sin(math.radians(20.0)),
         -G * math.cos(math.radians(20.0))), 20.0, 0.0)
    check("and the same holds for roll as for pitch",
          abs(rolled + G) < 1e-9, f"{rolled:.4f}")

    # 2. A known bias comes back out, through the whole window machinery.
    def log(n: int, t0: float, dt: float, bias: float, tilt: float,
            dist: float, rpm: float) -> dict:
        imu, att, vibe, esc, ctun, rfnd = [], [], [], [], [], []
        for i in range(n):
            t = t0 + i * dt
            # A sensor offset is ADDITIVE on the body axis, not a scaling
            # of it: a hovering aircraft's specific force is
            # (G sin p, 0, -G cos p) and the sensor adds b to the last one.
            imu.append({"TimeUS": t, "AccX": G * math.sin(math.radians(tilt)),
                        "AccY": 0.0,
                        "AccZ": -G * math.cos(math.radians(tilt)) + bias})
            att.append({"TimeUS": t, "Roll": 0.0, "Pitch": tilt})
            vibe.append({"TimeUS": t, "VibeZ": 9.0 if rpm > 3000 else 0.0})
            esc.append({"TimeUS": t, "RPM": rpm})
            ctun.append({"TimeUS": t, "ThO": 0.15 if rpm > 3000 else 0.0})
            rfnd.append({"TimeUS": t, "Dist": dist})
        return {"IMU": imu, "ATT": att, "VIBE": vibe, "ESC": esc,
                "CTUN": ctun, "RFND": rfnd}

    a = log(200, 0, 25_000, 0.43, 12.0, 1.4, 24000)
    w = summarise(a["IMU"], a["ATT"], a["VIBE"], a["ESC"], a["CTUN"],
                  a["RFND"], "airborne", 0, 200 * 25_000)
    # A bias on the body Z axis reaches the world's down axis reduced by
    # cos(tilt), because that is how much of the body axis points down.
    want = 0.43 * math.cos(math.radians(12.0))
    check("an injected 0.43 m/s2 bias is recovered through the tilt",
          abs(w.error - want) < 0.005,
          f"reported {w.error:+.3f} m/s2 where the geometry says "
          f"{want:.3f}: a body-axis offset reaches the down axis reduced "
          f"by cos(lean), which at the 2.6 deg this aircraft really hovers "
          f"at is a 0.1% correction")
    clean = log(200, 0, 25_000, 0.0, 12.0, 1.4, 24000)
    w2 = summarise(clean["IMU"], clean["ATT"], clean["VIBE"], clean["ESC"],
                   clean["CTUN"], clean["RFND"], "airborne", 0, 200 * 25_000)
    check("and a leaning aircraft with a good sensor reports no bias",
          abs(w2.error) < 0.01, f"{w2.error:+.3f} m/s2 at 12 deg of lean")

    # 3. Two flights in one log must not become one window. This was a real
    #    fault: taking the first and last airborne moment reported a 3.38 m
    #    climb for a flight that reached 1.83 m.
    rfnd = ([{"TimeUS": t, "Dist": 0.04} for t in range(0, 3_000_000, 50_000)]
            + [{"TimeUS": t, "Dist": 1.5}
               for t in range(3_000_000, 9_000_000, 50_000)]
            + [{"TimeUS": t, "Dist": 0.04}
               for t in range(9_000_000, 20_000_000, 50_000)]
            + [{"TimeUS": t, "Dist": 1.2}
               for t in range(20_000_000, 26_000_000, 50_000)])
    flights = airborne_spans(rfnd, 0.12)
    ground = grounded_spans(rfnd, 0.12)
    check("two flights with a pause between them are two windows, not one",
          len(flights) == 2 and len(ground) == 2
          and abs((flights[0][1] - flights[0][0]) / 1e6 - 5.95) < 0.2,
          f"{len(flights)} flights of "
          f"{', '.join(f'{(b - a) / 1e6:.1f}' for a, b in flights)} s and "
          f"{len(ground)} stretches on the floor; one window would have "
          f"spanned {(flights[-1][1] - flights[0][0]) / 1e6:.0f} s of which "
          f"half is sitting still")
    check("and the longest of them is the one the verdict is read from",
          longest(flights) == flights[0],
          f"{(flights[0][1] - flights[0][0]) / 1e6:.1f} s against "
          f"{(flights[1][1] - flights[1][0]) / 1e6:.1f} s")
    check("a run too short to average over is not a window at all",
          not runs([{"TimeUS": 0, "Dist": 1.0},
                    {"TimeUS": 100_000, "Dist": 1.0}],
                   lambda d: d["Dist"] > 0.12, min_s=2.0),
          "0.1 s of data is not a measurement")

    # 4. The horizontal axis, by the same trick: on the floor the aircraft
    #    translates at exactly zero, so a flow residual there is the sensor
    #    and the estimator has no way to know better.
    from drone_agent.vertical import flow_residual, mean_residual
    x, y = flow_residual({"flowX": 0.30, "bodyX": 0.05,
                          "flowY": -0.10, "bodyY": -0.10})
    check("the flow residual is what the sensor sees that the gyro does not",
          abs(x - 0.25) < 1e-9 and abs(y) < 1e-9,
          "the sensor measures image motion and the autopilot subtracts "
          "the rotation it measured itself; what is left is translation, "
          "and on the floor translation is zero")
    still = [{"TimeUS": t, "flowX": 0.0001, "bodyX": 0.0,
              "flowY": -0.0002, "bodyY": 0.0}
             for t in range(0, 2_000_000, 20_000)]
    spun = [{"TimeUS": t, "flowX": -0.26, "bodyX": 0.0,
             "flowY": 0.33, "bodyY": 0.0}
            for t in range(2_000_000, 4_000_000, 20_000)]
    a = mean_residual(still + spun, 0, 1_900_000)
    b = mean_residual(still + spun, 2_000_000, 4_000_000)
    check("and a window with too few samples is not averaged at all",
          mean_residual(still, 0, 100_000) is None,
          "ten samples is not a measurement")
    check("motors off reads zero, motors running reads the bias",
          a and abs(a[0]) < 0.001 and b and abs(b[1] - 0.33) < 1e-6,
          f"{a[0]:+.4f} rad/s against {b[1]:+.4f} rad/s -- at a 1 m height "
          f"the second is {abs(b[1]):.2f} m/s of velocity the aircraft does "
          f"not have, and the position controller flies off to cancel it")

    # 5. The shipped tool runs and refuses what it cannot read.
    root = Path(__file__).resolve().parent.parent
    empty = Path(tempfile.mkdtemp()) / "not.bin"
    empty.write_bytes(b"not a log")
    out = subprocess.run(
        [sys.executable, str(root / "tools" / "bin_vertical.py"),
         str(empty)], capture_output=True, text=True)
    check("the tool refuses a file that is not a log, rather than guessing",
          out.returncode != 0 or "no IMU" in (out.stdout + out.stderr),
          (out.stdout + out.stderr).strip().splitlines()[-1][:70]
          if (out.stdout + out.stderr).strip() else "")


def test_mission_outcomes() -> None:
    """
    The end-to-end verdict: of N flights, how many did what they were
    told. The planner has a graded evaluation; until now the flights were
    read by hand, which does not scale past a few and cannot be reported
    as a number.

    What is pinned is the thing a success rate can most easily get wrong:
    that a guard firing is not counted as a fault, so the safest possible
    behaviour cannot lower the score, while a mission that did not happen
    is still not counted as a success.
    """
    print("\nend-to-end mission outcomes")
    from drone_agent.outcome import (Kind, Stage, attrition, classify,
                                     mission_from_argv, verdict)
    from typing import Optional

    def st_(skill: str, status: str = "succeeded", reason: str = "") -> Any:
        return Stage(skill=skill, status=status, reason=reason)

    argv = ["run_mission.py", "rotate_search", "--then-face",
            "--then-approach"]
    planned_only = verdict("p", ["run_mission.py", "--instruction", "x"],
                           [st_("rotate_search"), st_("face_target")],
                           "x", planned=["rotate_search", "face_target"])
    check("a flight from words is judged against its own plan",
          planned_only.kind is Kind.COMPLETED
          and planned_only.mission == ["rotate_search", "face_target"],
          "whether the plan was right for the instruction is the other "
          "question, graded by the rubrics; conflating them would hide "
          "which half went wrong")
    check("the mission is read off the command line, not off what ran",
          mission_from_argv(argv) == ["rotate_search", "face_target",
                                      "approach"],
          "a search that found nobody and so never ran approach has not "
          "completed a search-face-approach mission")

    whole = verdict("a", argv, [st_("rotate_search"), st_("face_target"),
                                st_("approach")])
    check("every asked-for stage succeeding is a completed mission",
          whole.kind is Kind.COMPLETED and whole.reached == 3,
          f"{whole.label}")

    short = verdict("b", argv, [st_("rotate_search"), st_("face_target")])
    check("but a mission that stopped early is not, even with no failure",
          short.kind is Kind.FAILED
          and "never reached approach" in short.label,
          f"{short.label}: {short.reason}")

    # The two that decide whether the number is honest.
    guard = verdict("c", argv, [
        st_("rotate_search"),
        st_("face_target", "aborted",
            "outside the envelope: alt by 0.62m for 10 ticks")])
    check("a guard firing is stopped-safely, not a failure",
          guard.kind is Kind.STOPPED
          and "real height" in guard.label,
          "the altitude abort has fired 12 times out of 12 and is the "
          "aircraft's protection; counting it as a software fault would "
          "mis-describe it")
    moved = verdict("d", argv, [
        st_("rotate_search"), st_("face_target"),
        st_("approach", "failed",
            "the person has moved 0.77 m since the first look")])
    check("and so is refusing to chase a target that moved",
          moved.kind is Kind.STOPPED and "target moved" in moved.label,
          "the cross-check fired and the aircraft stopped; if that counted "
          "as a failure the safest behaviour would lower the score")
    lost = verdict("e", argv, [
        st_("rotate_search"),
        st_("face_target", "failed", "person is not in view")])
    check("losing the target IS a failure, so the split is not a whitewash",
          lost.kind is Kind.FAILED and "target lost" in lost.label,
          f"{lost.label}")

    odd = verdict("f", argv, [
        st_("rotate_search"),
        st_("face_target", "failed", "something nobody has written a rule "
                                     "for yet")])
    check("an unrecognised reason is called unclassified, not bucketed",
          odd.kind is Kind.FAILED and "unclassified" in odd.label,
          "a taxonomy that silently absorbs what it does not know stops "
          "being a measurement; the tool prints these verbatim")

    check("a flight that never left the ground is neither pass nor fail",
          verdict("g", argv, [], airborne=False).kind is Kind.UNFLOWN)

    # Priority: the guard is why the skill stopped, so it decides.
    both = verdict("h", argv, [
        st_("rotate_search"),
        st_("face_target", "aborted",
            "outside the envelope: alt by 0.5m; person is not in view")])
    check("where a guard and a skill both have a reason, the guard decides",
          both.kind is Kind.STOPPED,
          "the rule order is the priority, and the guard is upstream of "
          "the skill it interrupted")

    outs = [verdict(str(i), argv, s) for i, s in enumerate([
        [st_("rotate_search"), st_("face_target"), st_("approach")],
        [st_("rotate_search"), st_("face_target"), st_("approach")],
        [st_("rotate_search"), st_("face_target", "failed", "not in view")],
        [st_("rotate_search", "failed", "found nothing")],
    ])]
    att = attrition(outs, ["rotate_search", "face_target", "approach"])
    check("attrition counts reached and through, which a single rate hides",
          att == [("rotate_search", 4, 3), ("face_target", 3, 2),
                  ("approach", 2, 2)],
          "4 reached the search and 3 got through it; 2 reached approach "
          "and both got through - a 50% overall rate would not say where "
          "the misses are")

    # Words in, right outcome out: the arithmetic of the end-to-end
    # number, where two judgement calls live.
    def words(instr: str, plan_ok: Optional[bool], planned: List[str],
              stages: List[Any], airborne: bool = True) -> Any:
        o = verdict("w", ["run_mission.py", "--instruction", instr], stages,
                    instr, airborne, planned=planned or None)
        o.plan_ok = plan_ok
        return o

    good = words("find a person and turn to face them", True,
                 ["rotate_search", "face_target"],
                 [st_("rotate_search"), st_("face_target")])
    check("an instruction planned right and flown through is end-to-end",
          good.end_to_end is True and good.completed)
    badplan = words("find a person and turn to face them", False,
                    ["face_target"], [st_("face_target")])
    check("a wrong plan is an end-to-end failure however well it flew",
          badplan.end_to_end is False,
          "the aircraft carried out a plan that was not what was asked")
    badfly = words("find a person and turn to face them", True,
                   ["rotate_search", "face_target"],
                   [st_("rotate_search"),
                    st_("face_target", "failed", "person is not in view")])
    check("and a right plan that did not fly is too",
          badfly.end_to_end is False and badfly.plan_ok,
          "the two halves are reported apart, but end to end needs both")

    refused = words("fly to the door", True, [], [], airborne=False)
    check("a correctly refused instruction is a success, not a miss",
          refused.end_to_end is True and refused.plan_refused,
          "the system was asked for something it should not do, said so, "
          "and nothing took off; scoring the absent flight as a failure "
          "would rate a refusal the same as a crash")
    wrongly = words("look around for a person", False, [], [],
                    airborne=False)
    check("but refusing something possible is still a failure",
          wrongly.end_to_end is False,
          "nothing took off and nothing should have stopped it")
    dry = words("look around for a person", True, ["rotate_search"], [],
                airborne=False)
    check("a plan that was never flown answers neither question",
          dry.end_to_end is None,
          "a bench dry run is left out of the denominator rather than "
          "counted against the aircraft")
    check("and a flight with no instruction is not in this measure at all",
          verdict("x", ["run_mission.py", "rotate_search"],
                  [st_("rotate_search")]).end_to_end is None,
          "26 flights named the skill directly; they measure the skills, "
          "not the language path")

    for reason, want in (("alt by 0.62m", Kind.STOPPED),
                         ("rangefinder below the floor", Kind.STOPPED),
                         ("that would end 5.1 m from where the flight "
                          "began", Kind.STOPPED),
                         ("28 deg off the nose; face it first",
                          Kind.STOPPED),
                         ("did not arrive: still 0.30 m short", Kind.FAILED),
                         ("no steady view of the target", Kind.FAILED),
                         ("drifted 1.62 m off the commanded path (limit "
                          "1.5 m)", Kind.STOPPED),
                         ("drifted 1.50 m from the hold point (limit 1.5 m)",
                          Kind.FAILED)):
        got = classify(reason)
        check(f"the taxonomy places {reason[:34]!r}",
              got is not None and got[0] is want,
              f"{got[1] if got else 'no rule'} -> "
              f"{got[0].value if got else '-'}")


def main() -> int:
    print("=" * 62)
    print(f"offline check - drone_agent {__version__}, no vehicle")
    print("=" * 62)
    test_bypass_guard()
    test_envelope_clamps()
    test_drift_from_path()
    test_estop_and_staleness()
    test_state_breach()
    test_hover()
    test_rotate()
    test_rangefinder_router()
    test_recorder()
    test_perception()
    test_search_target_criterion()
    test_planner_view()
    test_replay_roundtrip()
    test_calibration_verdict()
    test_capture_geometry_guard()
    test_leg_confirmation()
    test_takeoff_frame()
    test_heading_transient_analysis()
    test_ground_perception_tool()
    test_perception_startup()
    test_verify_capture()
    test_planner()
    test_face_target()
    test_move()
    test_measure_range()
    test_target_lock()
    test_camera_tilt()
    test_tilt_from_floor()
    test_approach()
    test_grader()
    test_vertical_from_binlog()
    test_landing_is_recorded()
    test_mission_outcomes()

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
