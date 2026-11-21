"""
ClimbTo — move to a target altitude, holding XY.

Note what is NOT here: arming and MAV_CMD_NAV_TAKEOFF. Those live in the
launch sequence (executor.LaunchSequence), because on a flow-only airframe
the aircraft is armed in ALT_HOLD and climbed by the pilot before GUIDED will
accept anything. Keeping them out of the skill layer preserves the invariant
that a skill only ever emits setpoints, which is what lets skills be replayed
offline against recorded WorldStates with no vehicle attached.

Use this for altitude changes once airborne and in GUIDED.
"""

from __future__ import annotations

from typing import Any, Optional

from ..config import Config
from ..safety.arbiter import SetpointRequest
from ..world.state import WorldState
from .base import PreconditionResult, SkillStatus, register
from .hover import HoldingSkill


@register
class ClimbTo(HoldingSkill):
    """Change altitude to a target while holding the current XY position."""

    name = "climb_to"
    schema = {
        "type": "object",
        "properties": {
            "altitude_m": {"type": "number", "minimum": 0.3, "maximum": 2.0,
                           "description": "target altitude above the origin"},
            "tolerance_m": {"type": "number", "minimum": 0.05,
                            "maximum": 0.5},
            "timeout_s": {"type": "number", "minimum": 2, "maximum": 60},
        },
        "required": ["altitude_m"],
    }

    def __init__(self, cfg: Config, altitude_m: Optional[float] = None,
                 tolerance_m: float = 0.12, timeout_s: float = 30.0,
                 **params: Any) -> None:
        super().__init__(cfg, altitude_m=altitude_m, tolerance_m=tolerance_m,
                         timeout_s=timeout_s, **params)
        self.target_alt = (altitude_m if altitude_m is not None
                           else cfg.skills.takeoff_height_m)
        self.tolerance = tolerance_m
        self.timeout = timeout_s
        self._best_alt: Optional[float] = None

    def check_preconditions(self, world: WorldState) -> PreconditionResult:
        base = self.base_preconditions(world)
        if not base.ok:
            return base
        env = self.cfg.envelope
        if not (env.alt_min_m <= self.target_alt <= env.alt_max_m):
            # The arbiter would clamp this anyway; failing the precondition
            # means the caller learns why instead of silently getting a
            # different altitude.
            return PreconditionResult.failed(
                "target altitude outside the envelope",
                requested_m=self.target_alt,
                allowed=[env.alt_min_m, env.alt_max_m])
        return PreconditionResult.passed()

    def on_start(self, world: WorldState) -> None:
        self.capture_hold(world)
        self.metrics["target_alt_m"] = self.target_alt

    def tick(self, world: WorldState, dt: float) -> SkillStatus:
        self.note_tick()
        self.track(world)

        alt = world.altitude_m
        if alt is None or self.hold is None:
            self.request = SetpointRequest.idle("altitude unknown")
            return SkillStatus.RUNNING

        self._best_alt = alt if self._best_alt is None else max(self._best_alt,
                                                                alt)
        err = self.target_alt - alt

        if abs(err) <= self.tolerance:
            return self.succeed("altitude reached",
                                altitude_m=round(alt, 3),
                                source=world.altitude_source)

        if self.elapsed > self.timeout:
            return self.fail("climb timed out",
                             target_m=self.target_alt,
                             reached_m=round(self._best_alt or alt, 3),
                             source=world.altitude_source)

        self.request = SetpointRequest.position(
            self.hold[0], self.hold[1], -self.target_alt,
            yaw_rad=self.hold_yaw,
            reason=f"climb to {self.target_alt:.2f} m (err {err:+.2f})")
        return SkillStatus.RUNNING
