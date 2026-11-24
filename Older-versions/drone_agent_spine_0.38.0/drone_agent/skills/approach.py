"""
Approach — close on a target that is already in view until it is a chosen
distance away, measuring the distance again between every step.

This is the first skill on this airframe that acts on a range. Which range
matters more than anything else here, so it is worth being explicit about
what may drive it:

  head_height   the top of the box against the floor the rangefinder is
                measuring. Tape-measured on 2026-11-16 over six looks at
                2.57 m: median error 0.13 m, worst 0.36 m. Allowed.
  boresight     a forward lidar on the beam. Allowed, better.
  size_prior    the box's height against an assumed real height. Measured
                on the same looks: +25% to +31%, consistently LONG.
                **Refused.** A range that over-reads by a quarter is a
                stack that thinks it has 2.5 m when it has 1.9 m, and the
                target is a person.
  parallax      heading-dependent, and this aircraft's heading moved 10-16
                deg during a 13 s manoeuvre with the target standing still.
                Never wired in. Refused by not existing.

It does not yaw. The target is centred by *stepping sideways along its
bearing*, not by turning toward it: the one measurement this airframe is
known to be bad at is heading, and a yaw costs 10-16 deg of it. `move` is
known to be good to about 10% once corrected for the height ratio, so the
approach is built out of moves and nothing else. The target therefore stays
where it is in the frame, which is also what keeps the head range valid.

Each cycle:

  MEASURE   hold still, steady heading, take the median range and bearing
            over several frames of the locked track
  STEP      one `move`, capped, along the bearing -- delegated to the move
            skill itself, so the sub-steps, the height-ratio correction and
            the fence check are the ones that have flown
  then MEASURE again, and stop when the range is inside tolerance

The step is deliberately shorter than the distance to close: it takes the
measured move scatter (about 10%) and the measured range error (0.13 m) and
spends them on arriving slowly rather than on arriving close. STANDOFF_MIN_M
is a floor no parameter can get under.

A free cross-check falls out of this. The aircraft knows its own
displacement, so it can say where a target that had not moved would now
appear; the camera says where it is. The difference is how far the target
has moved, and approaching something that is moving is not what this skill
does.

That check started out comparing ranges only, and the first nine flights
(2026-11-17) showed why that is not enough. In two of them the person
stepped aside as the aircraft closed: the range came down roughly on
schedule while the bearing went from -7 deg to -34 deg, and a range-only
check missed it by 0.10 m. The whole vector is the signal. In the one
flight where the person stood still the implied movement stayed under
0.31 m; in the two where they moved it reached 0.76 and 1.25 m.
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


def wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class ApproachPhase(str, Enum):
    MEASURE = "measure"
    STEP = "step"


# Only these may drive the aircraft toward something. See the module note.
TRUSTED_SOURCES = ("head_height", "boresight", "depth")


@register
class Approach(HoldingSkill):
    """Close on a target already in view until it is a chosen distance away, re-measuring between steps."""

    name = "approach"
    # No parameter gets under this, and it is a BELIEVED range, so it has
    # to carry what the range is known to be wrong by. Seven tape-anchored
    # sessions give ratios of measured to true range from 0.61 to 1.37: the
    # same method, the same calibration file, read 39% short at 3.77 m and
    # 37% long at 1.30 m. Two causes are now identified -- a camera tilt
    # that is no longer what the file says, and detection boxes that are a
    # fragment of the person rather than all of it -- and until the tilt
    # has been re-measured and the method re-validated against a tape,
    # nothing here should drive an aircraft close to a person on the
    # strength of it. 1.7 m rather than 1.2 m is a margin against a range
    # that has been seen to read 37% long, not a modelled correction, and
    # it should come down when the method is trustworthy again rather than
    # when it looks tidier.
    #
    # An earlier version of this comment explained the 1.7 by a
    # "close-range over-read" of about 0.45 m. That reading was wrong: the
    # two flights it rested on had boxes covering about a third of the
    # person, so their tops were not heads and the range was not measuring
    # what it claimed. The guard in RangeEstimator now rejects most of
    # them.
    STANDOFF_MIN_M = 1.7
    STANDOFF_MAX_M = 4.0
    schema = {
        "type": "object",
        "properties": {
            "target_class": {"type": "string",
                             "description": "one noun, e.g. person. It must "
                                            "already be in view and roughly "
                                            "ahead: run face_target first"},
            "standoff_m": {"type": "number", "minimum": STANDOFF_MIN_M,
                           "maximum": STANDOFF_MAX_M,
                           "description": "how far from the target to stop, "
                                          "in metres"},
            "tolerance_m": {"type": "number", "minimum": 0.2, "maximum": 1.0,
                            "description": "how close to standoff_m counts "
                                           "as arrived"},
        },
        "required": ["target_class"],
    }

    MIN_FRAMES = 3
    MEASURE_SAMPLES = 5
    MEASURE_LIMIT_S = 8.0
    LOST_S = 2.5
    STEADY_DEG = 1.5
    STEADY_WINDOW_S = 0.5

    MAX_STEPS = 6
    MAX_STEP_M = 0.4          # one move, well inside move's own 1.0 m limit
    # Of the distance left to close. 0.6 was timid: the aircraft travels
    # about three quarters of what it is told (measured two ways), so
    # 0.6 x 0.75 closed under half the gap per step and the five flights
    # of 2026-11-17 01:00 all stopped on the far side of the tolerance,
    # at 1.66-1.76 m for a 1.5 m standoff -- a one-sided +0.20 m. The
    # floor is held against sigma, so being less timid cannot drive it
    # closer than the floor.
    STEP_FRACTION = 0.9
    MIN_STEP_M = 0.15         # shorter than this is inside the move scatter
    BEARING_MAX_DEG = 30.0    # further off the nose than this: face it first
    TOTAL_BUDGET_M = 3.0      # travelled, over all steps
    MAX_SIGMA_M = 0.80        # a sanity bound; the floor handles the rest
    # The cross-check, and WHICH PART of it is trustworthy.
    #
    # The aircraft knows its own displacement, so it can say where a target
    # that had not moved would now appear. The gap between that and the
    # sighting is the implied movement -- but only its component ACROSS the
    # line of sight is evidence that the target moved. The component ALONG
    # the line of sight is contaminated by the range's own error, which at
    # close distances is large and one-sided: taped on 2026-11-17, a
    # believed 1.89 m was really 1.50 m and a believed 1.77 m was really
    # 1.30 m. A range that over-reads as the aircraft closes looks exactly
    # like a target backing away, and refusing on it means refusing to
    # approach a person standing still.
    #
    # Across 19 approaches with the person's behaviour known: the three
    # where the person genuinely stepped aside had an across-sight
    # component of 0.65, 0.68 and 0.83 m; every flight where the person
    # stood still stayed at or under 0.44 m, including the two taped ones
    # whose ALONG component reached 0.65 and 0.73 m. So the across-sight
    # component separates them and the total does not.
    # NOTHING here refuses on apparent target movement any more. Measured
    # over 25 approaches on 2026-11-17: the target's bearing drifted the
    # SAME WAY in 24 of them, at a median rate of -1.04 deg/s, while the
    # EKF reported its own heading steady to +-3.3 deg. A person stepping
    # aside would be a coin toss; 24 of 25 is not. The aircraft is yawing
    # and does not know it, which is the heading weakness already measured
    # two other ways on this airframe, now measured a third time and with
    # a rate: 8 s of hovering walks the bearing 5.7 deg, 18 s walks it
    # 15.9 deg, and at 3.5 m that is 0.76 m of apparent sideways movement.
    #
    # A lateral cross-check cannot tell that from a target that really
    # moved, because they are the same measurement. So the lateral gap is
    # logged with the yaw drift it implies, and the thing that stops the
    # approach is the bearing leaving the cone -- which is the right
    # response either way, since a target outside the cone needs facing
    # again whether it walked or the aircraft turned.
    TARGET_MOVED_M = 0.55       # logged, not enforced
    RANGE_DISAGREES_M = 0.5     # logged, not enforced
    EDGE = 0.01

    accepts_handover = True

    def __init__(self, cfg: Config, target_class: Optional[str] = None,
                 standoff_m: float = 1.5, tolerance_m: float = 0.25,
                 handover: Optional[Dict[str, Any]] = None,
                 **params: Any) -> None:
        super().__init__(cfg, target_class=target_class,
                         standoff_m=standoff_m, tolerance_m=tolerance_m,
                         **params)
        self.handover = handover or {}
        self.target_class = (target_class or "").strip().lower()
        self.standoff = max(self.STANDOFF_MIN_M, float(standoff_m))
        self.tolerance = float(tolerance_m)
        self.phase = ApproachPhase.MEASURE
        self._t_phase = 0.0
        self._ranges: List[float] = []
        self._bearings: List[float] = []
        self._sigmas: List[float] = []
        self._sources: List[str] = []
        self._rejected = 0
        self.lock: Optional[TargetLock] = None
        self._yaw_hist: List[tuple] = []
        self._steady = False
        self._mv: Optional[Move] = None
        self._step_planned = 0.0
        self._last_range: Optional[float] = None
        self._anchor: Optional[Tuple[float, float]] = None
        self._p0: Optional[Tuple[float, float]] = None
        self._t_first: Optional[float] = None
        self.steps = 0
        self.travelled = 0.0
        self.ladder: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def check_preconditions(self, world: WorldState) -> PreconditionResult:
        if not self.target_class:
            return PreconditionResult.failed("no target_class given")
        if self.standoff > self.STANDOFF_MAX_M:
            return PreconditionResult.failed(
                f"standoff {self.standoff:.1f} m is beyond this skill's "
                f"{self.STANDOFF_MAX_M} m")
        return self.base_preconditions(world)

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
        self.metrics["standoff_m"] = self.standoff
        self._enter(ApproachPhase.MEASURE, world)

    def _enter(self, phase: ApproachPhase, world: WorldState) -> None:
        self.phase = phase
        self._t_phase = world.t_mono
        if phase is ApproachPhase.MEASURE:
            self._ranges, self._bearings = [], []
            self._sigmas, self._sources = [], []
            self._rejected = 0

    def _phase_elapsed(self, world: WorldState) -> float:
        return world.t_mono - self._t_phase

    # ------------------------------------------------------------------

    def _scale(self, world: WorldState) -> float:
        """Real metres per EKF metre, this tick. Same ratio `move` uses."""
        p = world.pose
        rf = world.rngfnd_down_m if world.rngfnd_down_trusted else None
        if p is None or rf is None or not p.alt_m or p.alt_m <= 0.2 \
                or rf <= 0.2:
            return 1.0
        return max(self.cfg.skills.move_scale_min,
                   min(self.cfg.skills.move_scale_max, rf / p.alt_m))

    def _moved_since_first(self, world: WorldState
                           ) -> Optional[Tuple[float, float]]:
        """
        How far the aircraft has really travelled since the first look, in
        real metres, as a vector in the EKF's own axes.

        The height ratio corrects a DISPLACEMENT and must never be applied
        to an absolute coordinate. The first version of this did exactly
        that, and on 2026-11-17 the EKF's origin happened to sit at
        (3648, -2060) -- so multiplying the position by a ratio that moved
        from 1.014 to 1.074 between two looks moved the computed position
        by 209 m, and the skill reported that the person had walked 194 m
        and refused to approach them. Near the origin the same bug is
        invisible: at 1 m out it is 6 cm.
        """
        p = world.pose
        if p is None or self._p0 is None:
            return None
        return ((p.x - self._p0[0]) * self._scale(world),
                (p.y - self._p0[1]) * self._scale(world))

    def _target_here(self, world: WorldState, rng: float, bearing: float
                     ) -> Optional[Tuple[float, float]]:
        """
        Where this sighting puts the target, in real metres relative to
        where the aircraft stood at the first look.
        """
        if world.pose is None:
            return None
        if self._p0 is None:
            self._p0 = (world.pose.x, world.pose.y)
        d = self._moved_since_first(world)
        if d is None:
            return None
        a = math.radians(world.pose.yaw_deg + bearing)
        return d[0] + rng * math.cos(a), d[1] + rng * math.sin(a)

    def _predict(self, world: WorldState
                 ) -> Optional[Tuple[float, float]]:
        """Range and bearing a target that had not moved would show now."""
        d = self._moved_since_first(world)
        if d is None or self._anchor is None or world.pose is None:
            return None
        dx, dy = self._anchor[0] - d[0], self._anchor[1] - d[1]
        return (math.hypot(dx, dy),
                wrap_deg(math.degrees(math.atan2(dy, dx))
                         - world.pose.yaw_deg))

    def usable(self, obj: Dict[str, Any]) -> Optional[str]:
        """
        Why this sighting cannot be used for a range, or None if it can.

        The head of the target being cut off by the top of the frame is the
        one that matters: the head range is measured FROM the top edge of
        the box, so a clipped head reads as nearer than it is. Nearer is
        the safe direction to be wrong in, but it is still wrong.
        """
        # The most specific diagnosis first. A clipped head makes the head
        # range unusable, which makes the estimator fall back to the size
        # prior, which is refused below -- but "the top of the target is
        # out of frame" is the reason worth logging, not "range source
        # size_prior".
        box = obj.get("bbox")
        if box and len(box) == 4 and box[1] <= self.EDGE:
            return "the top of the target is out of frame"
        if obj.get("range_m") is None:
            return "no range"
        src = obj.get("range_source")
        src = getattr(src, "value", src)
        if src not in TRUSTED_SOURCES:
            return (f"range source {src}: no head range, so either the "
                    f"camera has no measured tilt, or it has climbed to "
                    f"the target's own height, or the box cannot be the "
                    f"whole target")
        if abs(float(obj.get("bearing_deg") or 0.0)) > self.BEARING_MAX_DEG:
            return (f"{abs(float(obj['bearing_deg'])):.0f} deg off the nose; "
                    f"face it first")
        # A sanity bound only. A loose range is already self-limiting --
        # the floor is held against sigma, so an uncertain measurement
        # stops the aircraft further away rather than closer. What a
        # climb does instead is make the head range undefined, and the
        # estimator refuses to produce one at all below its rise floor:
        # that arrives here as a size_prior source, refused above.
        sig = obj.get("range_sigma_m")
        if sig is not None and float(sig) > self.MAX_SIGMA_M:
            return f"range good only to +-{float(sig):.2f} m"
        return None

    def tick(self, world: WorldState, dt: float) -> SkillStatus:
        self.note_tick()
        if world.pose is None:
            self.request = SetpointRequest.idle("no pose")
            return SkillStatus.RUNNING
        if self.hold is None:
            self.capture_hold(world)
        yaw = world.pose.yaw_rad
        self._yaw_hist.append((world.t_mono, yaw))
        self._yaw_hist = [(t, y) for t, y in self._yaw_hist
                          if world.t_mono - t <= self.STEADY_WINDOW_S]
        span = world.t_mono - self._yaw_hist[0][0]
        spread = max(abs(math.degrees(wrap_pi(y - yaw)))
                     for _, y in self._yaw_hist)
        self._steady = (span >= 0.8 * self.STEADY_WINDOW_S
                        and spread <= self.STEADY_DEG)
        assert self.lock is not None
        obj = self.lock.pick(world)
        if self.phase is ApproachPhase.MEASURE:
            self.track(world)
            return self._tick_measure(world, obj)
        return self._tick_step(world, dt)

    # ------------------------------------------------------------------

    def _tick_measure(self, world: WorldState,
                      obj: Optional[Dict[str, Any]]) -> SkillStatus:
        assert self.lock is not None
        self.request = self.hold_request(
            self.hold_yaw,
            f"measuring the range to {self.target_class}"
            + (f" (track {self.lock.track_id})" if self.lock.track_id
               else ""))
        if self.lock.lost_for(world) > self.LOST_S:
            return self.fail(f"{self.target_class} is not in view",
                             steps=self.steps,
                             travelled_m=round(self.travelled, 2),
                             ladder=self.ladder, lock=self.lock.to_log())
        if self._phase_elapsed(world) > self.MEASURE_LIMIT_S:
            why = ("no steady view of the target" if not self._steady
                   else "no range good enough to act on")
            return self.fail(why, steady=self._steady, steps=self.steps,
                             travelled_m=round(self.travelled, 2),
                             rejected_frames=self._rejected,
                             sources_seen=sorted(set(self._sources)),
                             ladder=self.ladder, lock=self.lock.to_log())
        if not self._steady:
            self._ranges, self._bearings = [], []
            self._sigmas = []
            return SkillStatus.RUNNING
        if obj is not None:
            self._sources.append(str(getattr(obj.get("range_source"), "value",
                                             obj.get("range_source"))))
            why = self.usable(obj)
            if why is not None:
                self._rejected += 1
                self.metrics["last_rejection"] = why
                # The cone is a stop condition, not a filter. Measuring on
                # whatever frames remain under 30 deg while the target
                # slides past it reports a convergence that is really a
                # biased sample: on 2026-11-17 a flight ended claiming
                # 1.52 m with the person's own frames split either side of
                # the limit. If the target has left the cone, say so.
                if "off the nose" in why:
                    return self.fail(
                        f"the {self.target_class} is now "
                        f"{abs(float(obj['bearing_deg'])):.0f} deg off the "
                        f"nose; it needs facing again before any more "
                        f"stepping",
                        bearing_deg=round(float(obj["bearing_deg"]), 1),
                        range_m=obj.get("range_m"), needs_reface=True,
                        track_id=self.lock.track_id, steps=self.steps,
                        travelled_m=round(self.travelled, 2),
                        ladder=self.ladder, lock=self.lock.to_log())
            else:
                self._ranges.append(float(obj["range_m"]))
                self._bearings.append(float(obj["bearing_deg"]))
                self._sigmas.append(float(obj.get("range_sigma_m") or 0.0))
        if len(self._ranges) < self.MEASURE_SAMPLES:
            return SkillStatus.RUNNING

        rng = statistics.median(self._ranges)
        bearing = statistics.median(self._bearings)
        sigma = statistics.median(self._sigmas) if self._sigmas else 0.0
        rung = {"step": self.steps, "range_m": round(rng, 2),
                "sigma_m": round(sigma, 2), "bearing_deg": round(bearing, 1),
                "spread_m": round(
                    statistics.pstdev(self._ranges)
                    if len(self._ranges) > 1 else 0.0, 2)}

        # The cross-check. The aircraft's own displacement says where a
        # target that had not moved would now be; the camera says where it
        # is. The difference is the target's movement, and it is the whole
        # vector, not just the range: on 2026-11-17 a person stepped aside
        # while staying at much the same distance, and a range-only check
        # missed it by 0.10 m while the bearing was 25 deg out.
        here = self._target_here(world, rng, bearing)
        if self._anchor is None:
            self._anchor = here
            self._t_first = world.t_mono
        elif here is not None:
            vx = here[0] - self._anchor[0]
            vy = here[1] - self._anchor[1]
            moved = math.hypot(vx, vy)
            # Split it along and across this look's line of sight. Only
            # the across part is evidence about the target.
            los = math.radians(world.pose.yaw_deg + bearing)
            ux, uy = math.cos(los), math.sin(los)
            along = vx * ux + vy * uy
            across = -vx * uy + vy * ux
            rung["target_moved_m"] = round(moved, 2)
            rung["across_m"] = round(across, 2)
            rung["along_m"] = round(along, 2)
            if abs(along) > self.RANGE_DISAGREES_M:
                rung["range_disagrees_m"] = round(along, 2)
                self.metrics["range_disagrees_m"] = round(along, 2)
            pre = self._predict(world)
            if pre is not None:
                rung["expected_m"], rung["expected_bearing_deg"] = (
                    round(pre[0], 2), round(pre[1], 1))
                rung["miss_m"] = round(rng - pre[0], 2)
                rung["miss_bearing_deg"] = round(
                    wrap_deg(bearing - pre[1]), 1)
            # What yaw drift would account for this lateral gap on its
            # own. On this airframe that is usually the whole of it.
            if rng > 0.1 and self._t_first is not None:
                dps = math.degrees(math.asin(
                    max(-1.0, min(1.0, across / rng))))
                dt = world.t_mono - self._t_first
                if dt > 1.0:
                    rung["implied_yaw_drift_dps"] = round(dps / dt, 2)
                    self.metrics["implied_yaw_drift_dps"] = round(dps / dt, 2)
            if abs(across) > self.TARGET_MOVED_M:
                rung["lateral_gap_m"] = round(across, 2)
                self.metrics["lateral_gap_m"] = round(across, 2)
        self.ladder.append(rung)
        self.metrics["ladder"] = self.ladder
        self._last_range = rng

        # The floor carries the range's uncertainty, so the distance aimed
        # for is never closer than the floor plus what the range might be
        # wrong by -- otherwise a tolerance band drawn around the floor has
        # half of itself inside it. If the measurement says the aircraft may
        # already be inside the floor, the next step is backwards.
        eff = max(self.standoff, self.STANDOFF_MIN_M + sigma)
        too_close = (rng - sigma) < self.STANDOFF_MIN_M
        if eff > self.standoff:
            rung["standoff_raised_to_m"] = round(eff, 2)
        if abs(rng - eff) <= self.tolerance and not too_close:
            return self.succeed(
                f"{self.target_class} is {rng:.2f} m away",
                target_class=self.target_class, range_m=round(rng, 2),
                range_sigma_m=round(sigma, 2),
                standoff_m=self.standoff, aimed_for_m=round(eff, 2),
                error_m=round(rng - eff, 2),
                bearing_deg=round(bearing, 1), steps=self.steps,
                travelled_m=round(self.travelled, 2),
                track_id=self.lock.track_id,
                heading_deg=round(world.pose.yaw_deg, 1),
                ladder=self.ladder, lock=self.lock.to_log(),
                max_excursion_m=round(self.max_excursion, 3))
        if self.steps >= self.MAX_STEPS:
            return self.fail(
                f"still {rng:.2f} m away after {self.steps} steps",
                range_m=round(rng, 2), standoff_m=self.standoff,
                steps=self.steps, travelled_m=round(self.travelled, 2),
                ladder=self.ladder, lock=self.lock.to_log())
        if self.travelled >= self.TOTAL_BUDGET_M:
            return self.fail(
                f"travelled {self.travelled:.2f} m without arriving; the "
                f"budget is {self.TOTAL_BUDGET_M} m",
                range_m=round(rng, 2), steps=self.steps,
                ladder=self.ladder, lock=self.lock.to_log())

        return self._plan_step(world, rng, bearing, sigma, eff)

    # ------------------------------------------------------------------

    def _plan_step(self, world: WorldState, rng: float, bearing: float,
                   sigma: float = 0.0,
                   eff: Optional[float] = None) -> SkillStatus:
        """One capped move along the target's bearing. No yaw."""
        eff = self.standoff if eff is None else eff
        want = rng - eff                      # + closer, - back off
        step = math.copysign(
            min(self.MAX_STEP_M, self.STEP_FRACTION * abs(want),
                max(0.0, self.TOTAL_BUDGET_M - self.travelled)),
            want)
        # Never end up closer than the floor, whatever was asked for -- and
        # the floor is held against the range's own uncertainty, not against
        # the number. A range good to +-0.2 m that is used as if it were
        # exact puts the aircraft 0.2 m inside any floor set from it.
        if step > 0.0:
            step = min(step, max(0.0, rng - sigma - self.STANDOFF_MIN_M))
        if abs(step) < self.MIN_STEP_M:
            return self.succeed(
                f"{self.target_class} is {rng:.2f} m away; the move left to "
                f"make is {abs(step):.2f} m, which is inside this "
                f"aircraft's own scatter or inside the "
                f"{self.STANDOFF_MIN_M} m floor",
                target_class=self.target_class, range_m=round(rng, 2),
                standoff_m=self.standoff, aimed_for_m=round(eff, 2),
                error_m=round(want, 2),
                bearing_deg=round(bearing, 1), steps=self.steps,
                travelled_m=round(self.travelled, 2),
                track_id=self.lock.track_id if self.lock else None,
                ladder=self.ladder, short_of_standoff=True,
                max_excursion_m=round(self.max_excursion, 3))

        b = math.radians(bearing)
        mv = Move(self.cfg, forward_m=step * math.cos(b),
                  right_m=step * math.sin(b),
                  tolerance_m=Move.ARRIVE_M)
        pre = mv.check_preconditions(world)
        if not pre.ok:
            return self.fail(
                f"the next step is not allowed: {pre.reason}",
                range_m=round(rng, 2), step_m=round(step, 2),
                steps=self.steps, ladder=self.ladder)
        mv.start(world)
        self._mv = mv
        self._step_planned = step
        self._enter(ApproachPhase.STEP, world)
        return SkillStatus.RUNNING

    def _tick_step(self, world: WorldState, dt: float) -> SkillStatus:
        assert self._mv is not None
        st = self._mv.tick(world, dt)
        self.request = self._mv.request
        if st is SkillStatus.RUNNING:
            return SkillStatus.RUNNING
        res = self._mv.result(world)
        self.steps += 1
        if st is not SkillStatus.SUCCEEDED:
            return self.fail(
                f"the step did not complete: {res.reason}",
                steps=self.steps, travelled_m=round(self.travelled, 2),
                ladder=self.ladder)
        # What the move believes it really travelled over the floor, which
        # is the EKF displacement times the height ratio it measured.
        real = res.detail.get("moved_real_m")
        self.travelled += abs(float(real if real is not None else
                                    res.detail.get("moved_m") or 0.0))
        self.metrics["steps"] = self.steps
        self.metrics["travelled_m"] = round(self.travelled, 2)
        # The hold is now here, not where the skill started, or the next
        # MEASURE would command the aircraft back.
        self.capture_hold(world)
        self._mv = None
        self._enter(ApproachPhase.MEASURE, world)
        if self.lock is not None:
            self.lock.t_last_seen = world.t_mono
        return SkillStatus.RUNNING
