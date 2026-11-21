"""
MeasureRange — how far away a target is, by stepping sideways and watching
how far it slides across the picture.

The detector's own range is a guess from the object's apparent height
against a prior on how tall that kind of thing is. For a person seen by this
camera it is unusable: in four flights that ended facing the same standing
person it read 3.2, 4.0, 4.1 and 7.1 m, because the person's feet were
always below the bottom of the frame and the box height is therefore not the
person's height.

Parallax needs no prior and no complete object. It needs a baseline, and
since 0.19.0 the aircraft can produce one: hold the heading, step sideways a
measured distance, and see how much the bearing changed.

    look (still)  ->  step sideways  ->  look (still)  ->  step back

With the target ahead and a step d to the right, the bearing goes to b and

    R = d / tan|b|

which this skill computes in its general form, for any starting bearing and
any actual displacement. The accuracy follows from the bearing noise, so
every number here is reported with the uncertainty it earns: at 3 m with a
0.5 m baseline, one degree of bearing error is about 6% of the range, and
the bearings are repeated and the median taken to keep that small.

What it is not: a measurement of anything that moves. A person who walks
during the two looks changes the bearing by more than the parallax does.
The skill reports how much the picture disagreed with a static world, and
the caller is expected to distrust a large value.
"""

from __future__ import annotations

import math
import statistics
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config
from ..safety.arbiter import SetpointRequest
from ..world.state import WorldState, wrap_pi
from .base import PreconditionResult, SkillStatus, register
from .hover import HoldingSkill
from .move import Move
from .rotate_search import RotateSearch
from .target_lock import TargetLock


class RangePhase(str, Enum):
    LOOK_A = "look_a"
    STEP_OUT = "step_out"
    LOOK_B = "look_b"
    STEP_BACK = "step_back"


def range_from_parallax(b0_deg: float, b1_deg: float,
                        dx: float, dy: float) -> Optional[float]:
    """
    Range to a static target from two bearings and the displacement between
    them, all in the body frame the FIRST bearing was taken in: x toward the
    nose, y to the right, bearings clockwise from the nose.

    The target sits at R*u0 and, after the aircraft has moved by d, along
    u1 from there, so (R*u0 - d) is parallel to u1 and the cross products
    give R directly. None when the two sight lines are too close to
    parallel to say anything.
    """
    b0, b1 = math.radians(b0_deg), math.radians(b1_deg)
    denom = math.sin(b1 - b0)                       # u0 x u1
    if abs(denom) < 1e-4:
        return None
    r = (dx * math.sin(b1) - dy * math.cos(b1)) / denom
    return r if r > 0.0 else None


@register
class MeasureRange(HoldingSkill):
    """Measure how far away a target is by stepping sideways and watching how far it moves in the picture."""

    name = "measure_range"
    schema = {
        "type": "object",
        "properties": {
            "target_class": {"type": "string",
                             "description": "one noun, e.g. person. The "
                                            "object must already be in view "
                                            "and roughly ahead: use this "
                                            "after face_target"},
            "baseline_m": {"type": "number", "minimum": 0.3, "maximum": 1.0,
                           "description": "how far to step sideways; a "
                                          "longer step measures a longer "
                                          "range more accurately"},
            "direction": {"type": "integer", "enum": [-1, 1],
                          "description": "1 steps right, -1 steps left"},
        },
        "required": ["target_class"],
    }

    MIN_FRAMES = 3
    SAMPLES = 7               # bearings per look; the median is used
    SKIP_S = 0.6              # ignore what was seen while still moving
    STEADY_DEG = 1.5
    STEADY_WINDOW_S = 0.5
    LOOK_LIMIT_S = 6.0
    MAX_START_BEARING_DEG = 40.0
    MIN_PARALLAX_DEG = 3.0
    STEP_TOLERANCE_M = 0.10   # tighter than a plain move: the step IS the
    MIN_USABLE_STEP_M = 0.15  # measurement, but any real step can be used
    BEARING_FLOOR_DEG = 1.0   # no measurement is better than this
    DISPLACEMENT_REL_SIGMA = 0.10

    accepts_handover = True

    def __init__(self, cfg: Config, target_class: Optional[str] = None,
                 baseline_m: float = 0.5, direction: int = 1,
                 handover: Optional[Dict[str, Any]] = None,
                 **params: Any) -> None:
        super().__init__(cfg, target_class=target_class,
                         baseline_m=baseline_m, direction=direction, **params)
        self.handover = handover or {}
        self.target_class = (target_class or "").strip().lower()
        self.baseline = float(baseline_m)
        self.lock: Optional[TargetLock] = None
        self.direction = 1 if int(direction) >= 0 else -1
        self.phase = RangePhase.LOOK_A
        self._t_phase = 0.0
        self._yaw_hist: List[Tuple[float, float]] = []
        self._steady = False
        self._samples: List[float] = []
        self._look: Dict[str, Any] = {}
        self._pose_a: Optional[Tuple[float, float, float]] = None
        self._pose_b: Optional[Tuple[float, float, float]] = None
        self._mover: Optional[Move] = None
        self._seen: Optional[Dict[str, Any]] = None
        self._pending_fail: Optional[Tuple[str, Dict[str, Any]]] = None

    # ------------------------------------------------------------------

    def check_preconditions(self, world: WorldState) -> PreconditionResult:
        if not self.target_class:
            return PreconditionResult.failed("no target_class given")
        base = self.base_preconditions(world)
        if not base.ok:
            return base
        # The sideways step and the step back are each a Move, so whatever
        # a Move would refuse, this refuses here instead of halfway through.
        out = Move(self.cfg, right_m=self.direction * self.baseline)
        return out.check_preconditions(world)

    def on_start(self, world: WorldState) -> None:
        self.capture_hold(world)
        h = self.handover
        self.lock = TargetLock(
            self.target_class,
            track_id=h.get("track_id") if h.get("target_class")
            == self.target_class else None,
            bearing_deg=h.get("bearing_deg"),
            heading_deg=h.get("heading_deg"),
            max_range_sigma_m=RotateSearch.MAX_RANGE_SIGMA_M)
        self.metrics["lock"] = self.lock.to_log()
        self._enter(RangePhase.LOOK_A, world)
        if world.pose is not None:
            self.metrics["start_heading_deg"] = round(world.pose.yaw_deg, 1)

    def _enter(self, phase: RangePhase, world: WorldState) -> None:
        self.phase = phase
        self._t_phase = world.t_mono
        self._samples = []
        self._steady = False
        self._yaw_hist = []
        self._mover = None
        if phase in (RangePhase.LOOK_A, RangePhase.LOOK_B):
            # Hold where the aircraft IS, not where the skill began. Without
            # this the second look drags it back and measures no parallax at
            # all.
            self.capture_hold(world)

    def _update_steady(self, world: WorldState) -> None:
        yaw = world.pose.yaw_rad
        self._yaw_hist.append((world.t_mono, yaw))
        self._yaw_hist = [(t, y) for t, y in self._yaw_hist
                          if world.t_mono - t <= self.STEADY_WINDOW_S]
        span = world.t_mono - self._yaw_hist[0][0]
        spread = max(abs(math.degrees(wrap_pi(y - yaw)))
                     for _, y in self._yaw_hist)
        self._steady = (span >= 0.8 * self.STEADY_WINDOW_S
                        and spread <= self.STEADY_DEG)

    # ------------------------------------------------------------------

    def tick(self, world: WorldState, dt: float) -> SkillStatus:
        self.note_tick()
        self.track(world)
        if world.pose is None:
            self.request = SetpointRequest.idle("no pose")
            return SkillStatus.RUNNING
        if self.hold is None:
            self.capture_hold(world)
        self._update_steady(world)
        assert self.lock is not None
        self._seen = self.lock.pick(world)
        handler = {
            RangePhase.LOOK_A: self._tick_look,
            RangePhase.STEP_OUT: self._tick_step,
            RangePhase.LOOK_B: self._tick_look,
            RangePhase.STEP_BACK: self._tick_step,
        }[self.phase]
        return handler(world)

    # --- looking -------------------------------------------------------

    def _tick_look(self, world: WorldState) -> SkillStatus:
        first = self.phase is RangePhase.LOOK_A
        self.request = self.hold_request(
            self.hold_yaw,
            f"look {'A' if first else 'B'}: measuring the bearing of "
            f"{self.target_class}")
        obj = self._seen
        if self._steady and obj is not None \
                and self._t_phase + self.SKIP_S <= world.t_mono:
            self._samples.append(float(obj["bearing_deg"]))
            if len(self._samples) >= self.SAMPLES:
                return self._finish_look(world, obj)
        if world.t_mono - self._t_phase > self.LOOK_LIMIT_S:
            why = (f"no steady view of {self.target_class}"
                   if obj is not None else
                   f"{self.target_class} is not in view")
            detail = {"phase": self.phase.value,
                      "samples": len(self._samples)}
            if first:
                return self.fail(why, **detail)
            # Halfway out. Measure nothing, but come home before failing.
            self._pending_fail = (why, detail)
            self._begin_step(world, back=True)
        return SkillStatus.RUNNING

    def _finish_look(self, world: WorldState,
                     obj: Dict[str, Any]) -> SkillStatus:
        med = statistics.median(self._samples)
        spread = (statistics.pstdev(self._samples)
                  if len(self._samples) > 1 else 0.0)
        look = {"bearing_deg": round(med, 2),
                "spread_deg": round(spread, 2),
                "yaw_deg": world.pose.yaw_deg,
                "range_prior_m": obj.get("range_m"),
                "samples": len(self._samples)}
        if self.phase is RangePhase.LOOK_A:
            if abs(med) > self.MAX_START_BEARING_DEG:
                return self.fail(
                    f"{self.target_class} is {med:+.0f} deg off the nose; "
                    f"face it first", bearing_deg=round(med, 1))
            self._look["a"] = look
            self._pose_a = (world.pose.x, world.pose.y, world.pose.yaw_rad)
            self.metrics["look_a"] = look
            self._begin_step(world, back=False)
            return SkillStatus.RUNNING
        self._look["b"] = look
        self._pose_b = (world.pose.x, world.pose.y, world.pose.yaw_rad)
        self.metrics["look_b"] = look
        self._begin_step(world, back=True)
        return SkillStatus.RUNNING

    # --- stepping ------------------------------------------------------

    def _begin_step(self, world: WorldState, back: bool) -> None:
        self._enter(RangePhase.STEP_BACK if back else RangePhase.STEP_OUT,
                    world)
        if not back:
            self._mover = Move(self.cfg,
                               right_m=self.direction * self.baseline,
                               tolerance_m=self.STEP_TOLERANCE_M)
        else:
            # Go back to where look A was taken, by the EKF's own reckoning,
            # rather than stepping the nominal distance the other way: if
            # the step out fell short or ran long, this still comes home.
            fwd, rgt = 0.0, -self.direction * self.baseline
            if self._pose_a is not None and world.pose is not None:
                scale = Move(self.cfg).height_scale(world)
                dn = (self._pose_a[0] - world.pose.x) * scale
                de = (self._pose_a[1] - world.pose.y) * scale
                c, s = math.cos(world.pose.yaw_rad), math.sin(world.pose.yaw_rad)
                fwd, rgt = dn * c + de * s, -dn * s + de * c
                far = math.hypot(fwd, rgt)
                if far > Move.MAX_M:          # should not happen; cap anyway
                    fwd, rgt = fwd * Move.MAX_M / far, rgt * Move.MAX_M / far
            self._mover = Move(self.cfg, forward_m=fwd, right_m=rgt)
        self._mover.start(world)

    def _tick_step(self, world: WorldState) -> SkillStatus:
        assert self._mover is not None
        st = self._mover.tick(world, 0.0)
        self.request = self._mover.request
        self.request.reason = (
            f"{'stepping back' if self.phase is RangePhase.STEP_BACK else 'stepping aside'}"
            f" - {self.request.reason}")
        if st is SkillStatus.RUNNING:
            return SkillStatus.RUNNING
        detail = dict(self._mover._detail if hasattr(self._mover, "_detail")
                      else {})
        if self.phase is RangePhase.STEP_OUT:
            # A step that fell short or timed out is not a failure here:
            # the geometry uses the distance actually covered, which is
            # measured either way. Only a step that went nowhere is useless.
            gone = (math.hypot(world.pose.x - self._pose_a[0],
                               world.pose.y - self._pose_a[1])
                    * Move(self.cfg).height_scale(world)
                    if self._pose_a is not None else 0.0)
            if st is not SkillStatus.SUCCEEDED \
                    and gone < self.MIN_USABLE_STEP_M:
                self._pending_fail = (
                    f"could not step aside: {self._mover._reason}", detail)
                self._begin_step(world, back=True)
                return SkillStatus.RUNNING
            if st is not SkillStatus.SUCCEEDED:
                self.metrics["step_out_short"] = self._mover._reason
            self.metrics["step_out"] = detail
            self._enter(RangePhase.LOOK_B, world)
            return SkillStatus.RUNNING
        # Home again, whatever happened on the way out.
        self.metrics["step_back"] = detail
        if self._pending_fail is not None:
            why, extra = self._pending_fail
            return self.fail(why, returned=st is SkillStatus.SUCCEEDED,
                             **extra)
        return self._result(world, returned=st is SkillStatus.SUCCEEDED)

    # --- the answer ----------------------------------------------------

    def _result(self, world: WorldState, returned: bool) -> SkillStatus:
        a, b = self._look["a"], self._look["b"]
        assert self._pose_a is not None and self._pose_b is not None
        # Displacement in the body frame of look A, in REAL metres: the EKF
        # under-reads distance in exactly the proportion by which its height
        # reads low, and that is the number the baseline must be in.
        scale = Move(self.cfg).height_scale(world)
        dn = (self._pose_b[0] - self._pose_a[0]) * scale
        de = (self._pose_b[1] - self._pose_a[1]) * scale
        c, s = math.cos(self._pose_a[2]), math.sin(self._pose_a[2])
        dx = dn * c + de * s                  # toward the nose at look A
        dy = -dn * s + de * c                 # to the right at look A
        step = math.hypot(dx, dy)
        # The nose may have drifted between the looks; the camera measures
        # against the nose, so put look B's bearing back into A's frame.
        dyaw = math.degrees(wrap_pi(self._pose_b[2] - self._pose_a[2]))
        b1 = b["bearing_deg"] + dyaw
        parallax = b1 - a["bearing_deg"]

        base: Dict[str, Any] = {
            "target_class": self.target_class,
            "baseline_m": round(step, 3),
            "height_scale": round(scale, 3),
            "bearing_a_deg": a["bearing_deg"], "bearing_b_deg": b1,
            "parallax_deg": round(parallax, 2),
            "yaw_drift_deg": round(dyaw, 2),
            "returned_to_start": returned,
            "lock": self.lock.to_log() if self.lock else None,
            "range_prior_a_m": a["range_prior_m"],
            "range_prior_b_m": b["range_prior_m"],
        }
        if step < 0.15:
            return self.fail("the aircraft did not move far enough to "
                             "measure anything", **base)
        if abs(parallax) < self.MIN_PARALLAX_DEG:
            far = step / math.tan(math.radians(self.MIN_PARALLAX_DEG))
            return self.fail(
                f"{self.target_class} did not move enough in the picture: "
                f"it is further than about {far:.0f} m, which this "
                f"{step:.2f} m step cannot measure", **base)
        r = range_from_parallax(a["bearing_deg"], b1, dx, dy)
        if r is None:
            return self.fail("the two sight lines do not meet in front of "
                             "the aircraft", **base)

        # Uncertainty: push each input to the edge of what it is known to
        # within and see how far the answer moves.
        sa = max(a["spread_deg"], self.BEARING_FLOOR_DEG)
        sb = max(b["spread_deg"], self.BEARING_FLOOR_DEG)
        lo = hi = r
        for da in (-sa, sa):
            for db in (-sb, sb):
                for ds in (1.0 - self.DISPLACEMENT_REL_SIGMA,
                           1.0 + self.DISPLACEMENT_REL_SIGMA):
                    rr = range_from_parallax(a["bearing_deg"] + da, b1 + db,
                                             dx * ds, dy * ds)
                    if rr is None:
                        return self.fail(
                            "the measurement is too uncertain to use: a "
                            "degree either way changes the answer "
                            "completely", range_m=round(r, 2), **base)
                    lo, hi = min(lo, rr), max(hi, rr)
        return self.succeed(
            "range measured", range_m=round(r, 2),
            uncertainty_m=round(0.5 * (hi - lo), 2),
            range_lo_m=round(lo, 2), range_hi_m=round(hi, 2),
            range_source="parallax", **base)
