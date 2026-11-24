"""
RotateSearch — yaw through a total angle in legs, holding position.

This is the original rotate_condition_yaw() turned into a tick-driven state
machine. The two non-obvious constraints it encodes are both load-bearing:

1. DO NOT send a position target during a yaw leg.
   ModeGuided::set_yaw_state() calls auto_yaw.set_mode_to_default() whenever a
   target arrives with the yaw field ignored, which cancels the fixed-yaw
   slew. So the skill goes deliberately IDLE for the duration of each leg.
   That is why the arbiter's watchdog is on skill ticks and not on setpoints —
   a setpoint watchdog would fire on every leg.

2. Legs need a dwell.
   Rotating continuously never gives the position controller a quiet moment to
   pull the aircraft back, and optical flow is at its worst while yawing. The
   dwell between legs is the window where the EKF reconverges and the XY
   correction actually happens. At the default 45 deg legs that is 8
   corrections per revolution instead of 2.

The position re-assert happens BEFORE each CONDITION_YAW, never during one.
Ordering is the whole point.

When perception arrives this becomes the search primitive: the same sweep,
with an extra terminal condition on "target class seen with a stable track".
The hook is deliberately left visible in _target_found().
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Optional

from ..config import Config
from ..safety.arbiter import SetpointRequest
from ..world.state import WorldState, wrap_pi
from .base import PreconditionResult, SkillStatus, register
from .hover import HoldingSkill


class Phase(str, Enum):
    SETTLE = "settle"           # capture the hold target
    REASSERT = "reassert"       # position target, immediate, before the yaw
    COMMAND = "command"         # emit CONDITION_YAW for this leg
    SLEWING = "slewing"         # quiet: no position targets at all
    DWELL = "dwell"             # hold position at the heading just reached
    FINAL = "final"             # settle after the last leg


@register
class RotateSearch(HoldingSkill):
    """Yaw through a total angle in legs while holding position."""

    name = "rotate_search"
    schema = {
        "type": "object",
        "properties": {
            "total_deg": {"type": "number", "minimum": 10, "maximum": 1080},
            "rate_dps": {"type": "number", "minimum": 5, "maximum": 45},
            "direction": {"type": "integer", "enum": [-1, 1],
                          "description": "1 = clockwise"},
            "leg_deg": {"type": "number", "minimum": 15, "maximum": 180,
                        "description": "smaller legs = more XY corrections"},
            "target_class": {"type": "string",
                             "description": "one noun, e.g. person: stop "
                                            "turning as soon as the camera "
                                            "sees and tracks this object"},
        },
    }

    def __init__(self, cfg: Config, total_deg: Optional[float] = None,
                 rate_dps: Optional[float] = None,
                 direction: Optional[int] = None,
                 leg_deg: Optional[float] = None,
                 target_class: Optional[str] = None, **params: Any) -> None:
        super().__init__(cfg, total_deg=total_deg, rate_dps=rate_dps,
                         direction=direction, leg_deg=leg_deg,
                         target_class=target_class, **params)
        s = cfg.skills
        self.total_deg = total_deg if total_deg is not None else s.yaw_total_deg
        self.rate_dps = rate_dps if rate_dps is not None else s.yaw_rate_dps
        self.direction = direction if direction is not None else s.yaw_dir
        self.leg_deg = leg_deg if leg_deg is not None else s.yaw_leg_deg
        self.target_class = target_class

        self.n_legs = max(1, int(math.ceil(self.total_deg / self.leg_deg)))
        self.leg = self.total_deg / self.n_legs

        self.phase = Phase.SETTLE
        self.leg_index = 0
        self.yaw0: Optional[float] = None
        self.leg_target: Optional[float] = None
        self._t_phase = 0.0
        self._leg_deadline = 0.0
        self._accumulated = 0.0
        self._yaw_cmd_sent = False
        # Leg-complete confirmation: successive heading SAMPLES inside the
        # tolerance (see SkillDefaults.yaw_confirm_samples for why).
        self._last_yaw_sample: Optional[float] = None
        self._in_tol_samples = 0
        self._t_in_tol: Optional[float] = None

    # ------------------------------------------------------------------

    def check_preconditions(self, world: WorldState) -> PreconditionResult:
        base = self.base_preconditions(world)
        if not base.ok:
            return base
        if world.pose is None or world.pose.yaw_rad is None:
            return PreconditionResult.failed(
                "no attitude - yaw control would be blind")
        if self.rate_dps > self.cfg.envelope.max_yaw_rate_dps:
            # Not fatal: the arbiter will clamp it. Record the fact.
            self.metrics["requested_rate_dps"] = self.rate_dps
        return PreconditionResult.passed()

    def on_start(self, world: WorldState) -> None:
        self._t_phase = world.t_mono
        self.metrics.update({"n_legs": self.n_legs,
                             "leg_deg": round(self.leg, 1),
                             "legs_completed": 0})

    # ------------------------------------------------------------------

    def tick(self, world: WorldState, dt: float) -> SkillStatus:
        self.note_tick()
        self.track(world)

        if world.pose is None:
            self.request = SetpointRequest.idle("no pose")
            return SkillStatus.RUNNING

        if self.target_class and self._target_found(world):
            got = self.metrics.get("target") or {}
            # The track id is the hand-over to whatever runs next. Without
            # it the following skill has to guess which detection the
            # search meant, and in two flights of 2026-11-16 that guess
            # was a different object 87 deg away.
            return self.succeed("target acquired",
                                target_class=self.target_class,
                                legs_completed=self.leg_index,
                                track_id=got.get("track_id"),
                                bearing_deg=got.get("bearing_deg"),
                                heading_deg=round(world.pose.yaw_deg, 1)
                                if world.pose else None,
                                range_m=got.get("range_m"),
                                range_sigma_m=got.get("range_sigma_m"))

        handler = {
            Phase.SETTLE: self._tick_settle,
            Phase.REASSERT: self._tick_reassert,
            Phase.COMMAND: self._tick_command,
            Phase.SLEWING: self._tick_slewing,
            Phase.DWELL: self._tick_dwell,
            Phase.FINAL: self._tick_final,
        }[self.phase]
        return handler(world)

    # ------------------------------------------------------------------

    def _enter(self, phase: Phase, world: WorldState) -> None:
        self.phase = phase
        self._t_phase = world.t_mono

    def _phase_elapsed(self, world: WorldState) -> float:
        return world.t_mono - self._t_phase

    # ------------------------------------------------------------------

    def _tick_settle(self, world: WorldState) -> SkillStatus:
        if self.hold is None:
            self.capture_hold(world)
            self.yaw0 = world.pose.yaw_rad
            self.metrics["start_heading_deg"] = round(world.pose.yaw_deg, 1)
        self.request = SetpointRequest.idle("preparing first leg")
        self._enter(Phase.REASSERT, world)
        return SkillStatus.RUNNING

    def _tick_reassert(self, world: WorldState) -> SkillStatus:
        """
        Position-only target, sent immediately, BEFORE the CONDITION_YAW of
        this leg. No yaw field: this one is purely the XY/Z correction. The
        0.3 s dwell lets it be acknowledged before the yaw command lands.
        """
        if self.cfg.skills.reassert_position_before_yaw and self.hold:
            self.request = SetpointRequest.position(
                self.hold[0], self.hold[1], self.hold[2], yaw_rad=None,
                reason=f"reassert before leg {self.leg_index + 1}",
                immediate=True)
        else:
            self.request = SetpointRequest.idle("no reassert")

        if self._phase_elapsed(world) >= 0.3:
            self._enter(Phase.COMMAND, world)
        return SkillStatus.RUNNING

    def _tick_command(self, world: WorldState) -> SkillStatus:
        assert self.yaw0 is not None
        self._accumulated = self.leg_index * self.leg
        target = wrap_pi(self.yaw0 + self.direction
                         * math.radians(self._accumulated + self.leg))
        self.leg_target = target

        self.request = SetpointRequest.yaw(
            math.degrees(target) % 360.0, self.rate_dps,
            1 if self.direction > 0 else -1,
            reason=f"leg {self.leg_index + 1}/{self.n_legs}")

        expected = self.leg / max(self.rate_dps, 1e-3)
        self._leg_deadline = (world.t_mono + expected
                              + self.cfg.skills.rotation_timeout_pad_s)
        self._yaw_cmd_sent = True
        self._last_yaw_sample = None
        self._in_tol_samples = 0
        self._t_in_tol = None
        self._enter(Phase.SLEWING, world)
        return SkillStatus.RUNNING

    def _tick_slewing(self, world: WorldState) -> SkillStatus:
        # Deliberately quiet. A position target here cancels the slew.
        self.request = SetpointRequest.idle(
            f"slewing leg {self.leg_index + 1} - uplink silent by design")

        if self.leg_target is None:
            return self.fail("no leg target")

        yaw = world.pose.yaw_rad
        err = abs(math.degrees(wrap_pi(yaw - self.leg_target)))
        inside = err <= self.cfg.skills.yaw_tolerance_deg

        # Count heading SAMPLES, not ticks. The loop runs at 10 Hz and the
        # attitude stream at about 5, so each sample is seen on two or more
        # ticks; a new one is recognised by its value having changed.
        if self._last_yaw_sample is None or yaw != self._last_yaw_sample:
            self._last_yaw_sample = yaw
            if inside:
                self._in_tol_samples += 1
                if self._in_tol_samples == 1:
                    self._t_in_tol = world.t_mono
            else:
                if self._in_tol_samples:
                    # It was inside and the very next sample is not: the
                    # heading did not arrive, a sample said it had.
                    self.metrics["heading_samples_unconfirmed"] = (
                        self.metrics.get("heading_samples_unconfirmed", 0)
                        + 1)
                self._in_tol_samples = 0
                self._t_in_tol = None

        need = max(1, int(self.cfg.skills.yaw_confirm_samples))
        confirmed = self._in_tol_samples >= need
        if (not confirmed and inside and self._t_in_tol is not None
                and world.t_mono - self._t_in_tol
                >= self.cfg.skills.yaw_confirm_fallback_s):
            # The stream has repeated one in-tolerance value for long
            # enough that waiting for a second is pointless.
            confirmed = True

        if confirmed:
            self.leg_index += 1
            self.metrics["legs_completed"] = self.leg_index
            if self.leg_index >= self.n_legs:
                self._enter(Phase.FINAL, world)
            else:
                self._enter(Phase.DWELL, world)
            return SkillStatus.RUNNING

        if world.t_mono > self._leg_deadline:
            return self.fail(
                f"yaw leg {self.leg_index + 1} timed out - not tracking the "
                f"commanded heading",
                leg=self.leg_index + 1, yaw_error_deg=round(err, 1),
                hint="check ATC_SLEW_YAW and yaw authority")
        return SkillStatus.RUNNING

    def _tick_dwell(self, world: WorldState) -> SkillStatus:
        """
        Hold position at the heading just reached and let the controller
        actually claw back whatever the rotation cost us.
        """
        self.request = self.hold_request(
            self.leg_target, f"dwell after leg {self.leg_index}")
        if self._phase_elapsed(world) >= self.cfg.skills.leg_settle_s:
            if world.pose is not None and self.hold is not None:
                d = world.pose.horizontal_distance_to(self.hold[0],
                                                      self.hold[1])
                self.metrics.setdefault("leg_settle_excursions_m", []).append(
                    round(d, 3))
            self._enter(Phase.REASSERT, world)
        return SkillStatus.RUNNING

    def _tick_final(self, world: WorldState) -> SkillStatus:
        self.request = self.hold_request(self.leg_target, "final settle")
        if self._phase_elapsed(world) >= self.cfg.skills.yaw_settle_s:
            flow = self.worst_flow
            detail = {"legs_completed": self.leg_index,
                      "max_excursion_m": round(self.max_excursion, 3)}
            if flow is not None:
                detail["worst_flow_quality"] = flow
                if flow < self.cfg.skills.flow_quality_warn:
                    # Advisory, never an abort: a bad flow number is a reason
                    # to distrust position, and aborting into a position-hold
                    # landing would use that same bad estimate.
                    detail["flow_warning"] = (
                        "low flow quality means the EKF was partly guessing; "
                        "if the aircraft visibly moved further than "
                        "max_excursion_m, that gap IS the flow failing to "
                        "observe the motion")
            return self.succeed("rotation complete", **detail)
        return SkillStatus.RUNNING

    # ------------------------------------------------------------------

    # What counts as "found". Deliberately strict, because the reference
    # implementations stop on a single frame's detection and then fly at a
    # bearing derived from that one bounding box.
    MIN_TRACK_AGE_S = 0.5
    MIN_FRAMES = 3
    MAX_RANGE_SIGMA_M = 0.6

    def _target_found(self, world: WorldState) -> bool:
        """
        Perception hook.

        Requires a persistent track_id, a track old enough and seen in enough
        frames to not be a flicker, and a range whose stated uncertainty is
        small enough to act on. A bearing with no range is recorded as a
        sighting but does NOT stop the sweep: without a range there is nothing
        a follow or approach skill could do with it, so continuing to sweep is
        the honest response.

        Returns False until perception fills world.objects, so the sweep runs
        to completion when no detector is attached.
        """
        if not world.objects:
            return False
        for obj in world.objects:
            if obj.get("label") != self.target_class:
                continue
            if obj.get("track_id") is None:
                continue
            if obj.get("age_s", 0.0) < self.MIN_TRACK_AGE_S:
                continue
            if obj.get("frames_seen", 0) < self.MIN_FRAMES:
                continue

            # A sighting worth recording even if unusable, so a failed search
            # can be distinguished from one that saw the target but could not
            # range it — two different failures with two different fixes.
            self.metrics.setdefault("sightings", []).append(obj)

            if obj.get("range_m") is None:
                self.metrics["range_missing"] = True
                continue
            sigma = obj.get("range_sigma_m")
            if sigma is None or sigma > self.MAX_RANGE_SIGMA_M:
                self.metrics["range_too_uncertain"] = sigma
                continue

            self.metrics["target"] = obj
            return True
        return False
