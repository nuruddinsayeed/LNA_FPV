"""
FaceTarget — turn until an object that is already in view is straight ahead.

A search ends the moment the target is accepted, and the target is then at
the edge of the picture: the first three perception flights all stopped with
the person 38-44 deg off the nose and cut off by the frame. Nothing useful
can be done with a target there. Its range is a guess from a box that does
not contain the whole object, and any motion toward it would be along the
wrong line.

This skill looks, turns, and looks again:

  MEASURE   hold still and take the median bearing over a short window
  REASSERT  position target, as rotate_search does before every yaw
  COMMAND   one CONDITION_YAW by the measured bearing
  SLEWING   quiet until the heading arrives, then MEASURE again

Which object. The search hands over the track id it accepted, and this
skill follows that id and nothing else (see target_lock). It does not pick
the box nearest the nose: on 2026-11-16 that turned the aircraft toward a
second, rejected detection 87 deg away from the person the search had found,
twice.

It never uses a bearing taken while turning. The picture is about a quarter
of a second old when a skill sees it, which at 20 deg/s is five degrees of
error that no calibration can remove. A bearing taken at rest has no such
term, and because every turn is followed by a new measurement, an error in
the camera model or in the EKF's heading only costs another iteration.

The yaw mechanics are rotate_search's, unchanged, because those are the ones
that have flown.
"""

from __future__ import annotations

import math
import statistics
from enum import Enum
from typing import Any, Dict, List, Optional

from ..config import Config
from ..safety.arbiter import SetpointRequest
from ..world.state import WorldState, wrap_pi
from .base import PreconditionResult, SkillStatus, register
from .hover import HoldingSkill
from .rotate_search import RotateSearch
from .target_lock import TargetLock


class FacePhase(str, Enum):
    MEASURE = "measure"
    REASSERT = "reassert"
    COMMAND = "command"
    SLEWING = "slewing"


@register
class FaceTarget(HoldingSkill):
    """Turn in place until an object that is already in view is straight ahead."""

    name = "face_target"
    schema = {
        "type": "object",
        "properties": {
            "target_class": {"type": "string",
                             "description": "one noun, e.g. person. The "
                                            "object must already be in view: "
                                            "use this after rotate_search "
                                            "with the same target_class"},
            "tolerance_deg": {"type": "number", "minimum": 2, "maximum": 20,
                              "description": "how far off the nose still "
                                             "counts as facing it"},
        },
        "required": ["target_class"],
    }

    MIN_FRAMES = 3            # the same bar a search sets for a sighting
    MEASURE_SAMPLES = 5       # bearings in one measurement (median taken)
    LOST_DECIDE_S = 1.0       # out of sight this long: answer with what we
    LOST_DECIDE_MIN = 2       # have, if it already says we are facing it
    LOST_S = 2.5              # no sighting for this long: the target is gone
    MAX_TURNS = 4
    STEADY_DEG = 1.5          # heading spread over the window: not turning
    STEADY_WINDOW_S = 0.5     # two telemetry samples at the link's 4 Hz
    MEASURE_LIMIT_S = 8.0     # a heading that never settles is a failure
    MIN_TURN_DEG = 2.0        # CONDITION_YAW below this is not worth sending
    EDGE = 0.01               # a box this close to the frame edge is cut off

    accepts_handover = True

    def __init__(self, cfg: Config, target_class: Optional[str] = None,
                 tolerance_deg: float = 5.0,
                 handover: Optional[Dict[str, Any]] = None,
                 **params: Any) -> None:
        super().__init__(cfg, target_class=target_class,
                         tolerance_deg=tolerance_deg, **params)
        self.handover = handover or {}
        self.target_class = (target_class or "").strip().lower()
        self.tolerance = float(tolerance_deg)
        self.rate_dps = cfg.skills.yaw_rate_dps
        self.phase = FacePhase.MEASURE
        self._t_phase = 0.0
        self._bearings: List[float] = []
        self._last_obj: Optional[Dict[str, Any]] = None
        self.lock: Optional[TargetLock] = None
        self._settle_until = 0.0
        self._heading: Optional[float] = None      # rad, what we hold
        self._turn_target: Optional[float] = None
        self._turn_dir = 1
        self._turn_step = 0.0
        self._yaw_hist: List[tuple] = []
        self._steady = False
        self._deadline = 0.0
        self.turns = 0
        self.turned_deg = 0.0

    # ------------------------------------------------------------------

    def check_preconditions(self, world: WorldState) -> PreconditionResult:
        if not self.target_class:
            return PreconditionResult.failed("no target_class given")
        return self.base_preconditions(world)

    def on_start(self, world: WorldState) -> None:
        self.capture_hold(world)
        self._heading = self.hold_yaw
        h = self.handover
        self.lock = TargetLock(
            self.target_class,
            track_id=h.get("track_id") if h.get("target_class")
            == self.target_class else None,
            bearing_deg=h.get("bearing_deg"),
            heading_deg=h.get("heading_deg"),
            max_range_sigma_m=RotateSearch.MAX_RANGE_SIGMA_M)
        self.metrics["lock"] = self.lock.to_log()
        self._enter(FacePhase.MEASURE, world)
        if world.pose is not None:
            self.metrics["start_heading_deg"] = round(world.pose.yaw_deg, 1)

    def _enter(self, phase: FacePhase, world: WorldState) -> None:
        self.phase = phase
        self._t_phase = world.t_mono
        if phase is FacePhase.MEASURE:
            self._bearings = []
        if phase is FacePhase.SLEWING:
            self._steady = False
            self._yaw_hist = []

    def _phase_elapsed(self, world: WorldState) -> float:
        return world.t_mono - self._t_phase

    def _cut_off(self, obj: Dict[str, Any]) -> Optional[bool]:
        box = obj.get("bbox")
        if not box or len(box) != 4:
            return None
        return (box[0] <= self.EDGE or box[1] <= self.EDGE
                or box[2] >= 1.0 - self.EDGE or box[3] >= 1.0 - self.EDGE)

    # ------------------------------------------------------------------

    def tick(self, world: WorldState, dt: float) -> SkillStatus:
        self.note_tick()
        self.track(world)
        if world.pose is None:
            self.request = SetpointRequest.idle("no pose")
            return SkillStatus.RUNNING
        if self.hold is None:
            self.capture_hold(world)
            self._heading = self.hold_yaw
        # Not turning = the heading stayed within STEADY_DEG for a window
        # long enough to hold two telemetry samples. A search hands over in
        # the middle of a leg, so this is not true at the start.
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
        if obj is not None:
            self._last_obj = obj
        handler = {
            FacePhase.MEASURE: self._tick_measure,
            FacePhase.REASSERT: self._tick_reassert,
            FacePhase.COMMAND: self._tick_command,
            FacePhase.SLEWING: self._tick_slewing,
        }[self.phase]
        return handler(world, obj)

    def _tick_measure(self, world: WorldState,
                      obj: Optional[Dict[str, Any]]) -> SkillStatus:
        assert self.lock is not None
        self.request = self.hold_request(
            self._heading,
            f"looking for {self.target_class}"
            + (f" (track {self.lock.track_id})" if self.lock.track_id
               else ""))
        lost = self.lock.lost_for(world)
        if lost > self.LOST_S:
            return self.fail(
                f"{self.target_class} is not in view",
                turns=self.turns, turned_deg=round(self.turned_deg, 1),
                samples=len(self._bearings), lock=self.lock.to_log(),
                last_bearing_deg=(self._last_obj or {}).get("bearing_deg"))
        if self._phase_elapsed(world) > self.MEASURE_LIMIT_S:
            return self.fail("no steady view of the target",
                             steady=self._steady, turns=self.turns,
                             samples=len(self._bearings),
                             lock=self.lock.to_log())
        if not self._steady:
            self._bearings = []          # anything taken while moving is void
            return SkillStatus.RUNNING
        if obj is not None:
            self._bearings.append(float(obj["bearing_deg"]))
        enough = len(self._bearings) >= self.MEASURE_SAMPLES
        settled = world.t_mono >= self._settle_until
        if not (enough and settled):
            # The target is going out of sight and what we have already says
            # we are pointing at it: say so rather than throw the samples
            # away. A turn still needs the full count, because committing to
            # a turn toward something we can no longer see is the one
            # decision worth being slow about.
            partial = (lost >= self.LOST_DECIDE_S
                       and len(self._bearings) >= self.LOST_DECIDE_MIN
                       and abs(statistics.median(self._bearings))
                       <= self.tolerance)
            if not partial:
                return SkillStatus.RUNNING

        bearing = statistics.median(self._bearings)
        self.metrics.setdefault("bearings_deg", []).append(round(bearing, 1))
        if abs(bearing) <= self.tolerance:
            seen = obj or self._last_obj or {}
            return self.succeed(
                "facing target", target_class=self.target_class,
                bearing_deg=round(bearing, 1), turns=self.turns,
                turned_deg=round(self.turned_deg, 1),
                samples=len(self._bearings),
                track_id=self.lock.track_id,
                heading_deg=round(world.pose.yaw_deg, 1),
                range_m=seen.get("range_m"),
                range_source=seen.get("range_source"),
                box_cut_off=self._cut_off(seen) if seen else None,
                lock=self.lock.to_log(),
                max_excursion_m=round(self.max_excursion, 3))
        if self.turns >= self.MAX_TURNS:
            return self.fail(
                f"still {bearing:+.0f} deg off after {self.turns} turns",
                bearing_deg=round(bearing, 1), turns=self.turns,
                turned_deg=round(self.turned_deg, 1),
                lock=self.lock.to_log())
        step = max(abs(bearing), self.MIN_TURN_DEG)
        self._turn_dir = 1 if bearing > 0 else -1
        self._turn_target = wrap_pi(world.pose.yaw_rad
                                    + math.radians(self._turn_dir * step))
        self._turn_step = step
        self._enter(FacePhase.REASSERT, world)
        return SkillStatus.RUNNING

    def _tick_reassert(self, world: WorldState,
                       obj: Optional[Dict[str, Any]]) -> SkillStatus:
        if self.cfg.skills.reassert_position_before_yaw:
            self.request = SetpointRequest.position(
                self.hold[0], self.hold[1], self.hold[2], yaw_rad=None,
                reason=f"reassert before turn {self.turns + 1}",
                immediate=True)
        else:
            self.request = SetpointRequest.idle("no reassert")
        if self._phase_elapsed(world) >= 0.3:
            self._enter(FacePhase.COMMAND, world)
        return SkillStatus.RUNNING

    def _tick_command(self, world: WorldState,
                      obj: Optional[Dict[str, Any]]) -> SkillStatus:
        self.request = SetpointRequest.yaw(
            math.degrees(self._turn_target) % 360.0, self.rate_dps,
            self._turn_dir,
            reason=f"turn {self.turns + 1}: {self._turn_dir * self._turn_step:+.0f} deg "
                   f"toward {self.target_class}")
        self._deadline = (world.t_mono + self._turn_step / max(self.rate_dps, 1e-3)
                          + self.cfg.skills.rotation_timeout_pad_s)
        self._enter(FacePhase.SLEWING, world)
        return SkillStatus.RUNNING

    def _tick_slewing(self, world: WorldState,
                      obj: Optional[Dict[str, Any]]) -> SkillStatus:
        self.request = SetpointRequest.idle(
            f"turning toward {self.target_class} - uplink silent by design")
        err = abs(math.degrees(wrap_pi(world.pose.yaw_rad - self._turn_target)))
        # A small turn starts inside the leg tolerance, so "arrived" here
        # only means "close enough to hand over to the hold": the dwell
        # holds the exact heading, and the next measurement is the judge.
        if err <= min(self.cfg.skills.yaw_tolerance_deg,
                      max(2.0, 0.5 * self._turn_step)):
            self._arrive(world)
            return SkillStatus.RUNNING
        if world.t_mono > self._deadline:
            if err <= self.cfg.skills.yaw_tolerance_deg:
                self._arrive(world)
                return SkillStatus.RUNNING
            return self.fail(
                "turn timed out - not tracking the commanded heading",
                yaw_error_deg=round(err, 1), turns=self.turns)
        return SkillStatus.RUNNING

    def _arrive(self, world: WorldState) -> None:
        self.turns += 1
        self.turned_deg += self._turn_dir * self._turn_step
        self._heading = self._turn_target
        self.metrics["turns"] = self.turns
        # Straight back to measuring, but no answer before the aircraft has
        # settled. Bearings seen while it settles still count, as long as
        # the heading is steady: an earlier version threw away every sighting
        # of this window and then failed for want of sightings.
        self._enter(FacePhase.MEASURE, world)
        self._settle_until = world.t_mono + self.cfg.skills.leg_settle_s
        if self.lock is not None:
            # Out of sight for the length of the turn is not being lost.
            self.lock.t_last_seen = world.t_mono
