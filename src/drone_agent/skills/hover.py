"""
Hover — hold a captured position and heading for a fixed duration.

Also the base class for anything that holds station, because capturing the
hold target correctly is fiddly and worth doing in one place: average several
LOCAL_POSITION_NED samples rather than trusting one, and average yaw as a unit
vector rather than as a number (the mean of 359 deg and 1 deg is 0, not 180).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from ..config import Config
from ..safety.arbiter import SetpointRequest
from ..world.state import WorldState
from .base import PreconditionResult, Skill, SkillStatus, register


class HoldingSkill(Skill):
    """Shared hold-target capture and station-keeping."""

    def __init__(self, cfg: Config, **params: Any) -> None:
        super().__init__(cfg, **params)
        self.hold: Optional[Tuple[float, float, float]] = None
        self.hold_yaw: Optional[float] = None
        self.max_excursion = 0.0
        self.worst_flow: Optional[int] = None

    def capture_hold(self, world: WorldState) -> None:
        if world.pose is None:
            return
        self.hold = (world.pose.x, world.pose.y, world.pose.z)
        self.hold_yaw = world.pose.yaw_rad

    def track(self, world: WorldState) -> None:
        if world.pose is not None and self.hold is not None:
            d = world.pose.horizontal_distance_to(self.hold[0], self.hold[1])
            self.max_excursion = max(self.max_excursion, d)
            self.metrics["max_excursion_m"] = round(self.max_excursion, 3)
        if world.flow_quality is not None:
            self.worst_flow = (world.flow_quality if self.worst_flow is None
                               else min(self.worst_flow, world.flow_quality))
            self.metrics["worst_flow_quality"] = self.worst_flow

    def hold_request(self, yaw_rad: Optional[float],
                     reason: str) -> SetpointRequest:
        assert self.hold is not None
        return SetpointRequest.position(self.hold[0], self.hold[1],
                                        self.hold[2], yaw_rad, reason)

    def base_preconditions(self, world: WorldState) -> PreconditionResult:
        if not world.armed:
            return PreconditionResult.failed("not armed")
        if world.pose is None:
            return PreconditionResult.failed("no position estimate")
        if not world.pose_fresh:
            return PreconditionResult.failed(
                "position estimate stale", age_s=round(world.pose_age_s, 2))
        if world.mode != "GUIDED":
            return PreconditionResult.failed("not in GUIDED",
                                             mode=world.mode)
        alt = world.altitude_m
        if alt is None:
            return PreconditionResult.failed("altitude unknown")
        return PreconditionResult.passed()


@register
class Hover(HoldingSkill):
    """Hold the current position and heading for a fixed duration."""

    name = "hover"
    schema = {
        "type": "object",
        "properties": {
            "duration_s": {"type": "number", "minimum": 0.5, "maximum": 120,
                           "description": "how long to hold station"},
            "settle_s": {"type": "number", "minimum": 0.0, "maximum": 30,
                         "description": "settle before capturing the hold "
                                        "target"},
        },
        "required": ["duration_s"],
    }

    def __init__(self, cfg: Config, duration_s: Optional[float] = None,
                 settle_s: float = 0.0, **params: Any) -> None:
        super().__init__(cfg, duration_s=duration_s, settle_s=settle_s,
                         **params)
        self.duration = (duration_s if duration_s is not None
                         else cfg.skills.hover_s)
        self.settle = settle_s
        self._settle_done = False
        self._pos_samples: list = []
        self._yaw_sin = 0.0
        self._yaw_cos = 0.0
        self._yaw_n = 0
        self._t_hold_start: Optional[float] = None

    def check_preconditions(self, world: WorldState) -> PreconditionResult:
        return self.base_preconditions(world)

    def on_start(self, world: WorldState) -> None:
        if self.settle <= 0:
            self._finish_capture(world)

    # ------------------------------------------------------------------

    def tick(self, world: WorldState, dt: float) -> SkillStatus:
        self.note_tick()
        self.track(world)

        if world.pose is None:
            self.request = SetpointRequest.idle("no pose")
            return SkillStatus.RUNNING

        # --- settle, then capture the hold target ------------------------
        if not self._settle_done:
            self.request = SetpointRequest.idle("settling")
            self._accumulate(world)
            if self.elapsed >= self.settle:
                self._finish_capture(world)
            return SkillStatus.RUNNING

        if self.hold is None:
            return self.fail("hold target never captured")

        if self._t_hold_start is None:
            self._t_hold_start = world.t_mono

        held = world.t_mono - self._t_hold_start
        if held >= self.duration:
            # Explicit terminal condition, evaluated in code.
            return self.succeed(
                "hold duration elapsed", held_s=round(held, 2),
                max_excursion_m=round(self.max_excursion, 3))

        self.request = self.hold_request(
            self.hold_yaw, f"hover {held:.1f}/{self.duration:.0f}s")
        return SkillStatus.RUNNING

    # ------------------------------------------------------------------

    def _accumulate(self, world: WorldState) -> None:
        p = world.pose
        if p is None:
            return
        sample = (p.x, p.y, p.z)
        if not self._pos_samples or self._pos_samples[-1] != sample:
            self._pos_samples.append(sample)
            # Average yaw as a unit vector: the mean of 359 and 1 is 0.
            self._yaw_sin += math.sin(p.yaw_rad)
            self._yaw_cos += math.cos(p.yaw_rad)
            self._yaw_n += 1

    def _finish_capture(self, world: WorldState) -> None:
        self._settle_done = True
        if self._yaw_n >= 3 and self._pos_samples:
            n = len(self._pos_samples)
            self.hold = (sum(s[0] for s in self._pos_samples) / n,
                         sum(s[1] for s in self._pos_samples) / n,
                         sum(s[2] for s in self._pos_samples) / n)
            self.hold_yaw = math.atan2(self._yaw_sin / self._yaw_n,
                                       self._yaw_cos / self._yaw_n)
            self.metrics["hold_samples"] = n
        else:
            self.capture_hold(world)
            self.metrics["hold_samples"] = 1
        if self.hold is not None:
            self.metrics["hold_target"] = {
                "x": round(self.hold[0], 3), "y": round(self.hold[1], 3),
                "z": round(self.hold[2], 3),
                "yaw_deg": round(math.degrees(self.hold_yaw or 0.0), 1)}
