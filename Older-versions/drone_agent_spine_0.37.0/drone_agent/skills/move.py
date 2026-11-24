"""
Move — a short, straight displacement relative to the nose, holding heading
and height. The first skill that translates the aircraft.

Position here is optical-flow dead reckoning with no outside reference, so
this is deliberately small: at most one metre per call, and the target is
approached in sub-steps instead of in one jump.

Why sub-steps. The autopilot flies to a position target at whatever speed
its own parameters allow, and this stack does not set those. A target one
metre away may be taken at several metres per second. A target 0.25 m away
cannot be: the aircraft has to stop again before it has gone anywhere. One
sub-step is sent, the aircraft is given time to get there, then the next.
The link carries one position target per sub-step, well inside its budget.

What "one metre" means. Real metres over the floor, not the EKF's. Flow
velocity is scaled by the height the EKF believes, and that height reads
below the rangefinder, so an uncorrected move travels further than it is
told. Six tape-measured flights (2026-11-16): commanded 0.5 m four times
and 1.0 m twice, really went 0.51 to 0.71 m and 1.25 to 1.31 m, 24% long on
average, while the rangefinder read 1.20 times the EKF height in the same
flights. So the EKF-frame goal is the asked-for distance DIVIDED by that
ratio, measured at the start of each move. The residual over those six
flights would have been 8% +- 10%, and the scatter, not the bias, is what
is left. `skills.move_scale_by_height` turns the correction off.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from ..config import Config
from ..safety.arbiter import SetpointRequest
from ..world.state import WorldState
from .base import PreconditionResult, SkillStatus, register
from .hover import HoldingSkill


@register
class Move(HoldingSkill):
    """Move a short distance in a straight line relative to the nose, keeping heading and height."""

    name = "move"
    MAX_M = 1.0
    schema = {
        "type": "object",
        "properties": {
            "forward_m": {"type": "number", "minimum": -MAX_M,
                          "maximum": MAX_M,
                          "description": "metres toward the nose; negative "
                                         "is backward"},
            "right_m": {"type": "number", "minimum": -MAX_M, "maximum": MAX_M,
                        "description": "metres to the right; negative is "
                                       "left"},
        },
    }

    SUBSTEP_M = 0.25          # one position target; the aircraft must stop
    SUBSTEP_S = 1.2           # time given to each sub-step
    ARRIVE_M = 0.20           # the hold's own p95 is 0.15-0.18 m
    SETTLE_S = 1.0            # inside ARRIVE_M this long = arrived
    TIMEOUT_PAD_S = 8.0

    def __init__(self, cfg: Config, forward_m: float = 0.0,
                 right_m: float = 0.0, tolerance_m: Optional[float] = None,
                 **params: Any) -> None:
        super().__init__(cfg, forward_m=forward_m, right_m=right_m, **params)
        self.arrive_m = float(tolerance_m or self.ARRIVE_M)
        self.forward, self.right = float(forward_m), float(right_m)
        self.dist = math.hypot(self.forward, self.right)
        self.scale = 1.0          # real metres -> EKF metres, set at start
        self.p0: Optional[Tuple[float, float, float]] = None
        self.goal: Optional[Tuple[float, float]] = None
        self.n_sub = max(1, math.ceil(self.dist / self.SUBSTEP_M - 1e-9))
        self._sub_sent = -1
        self._t0: Optional[float] = None
        self._t_inside: Optional[float] = None

    def check_preconditions(self, world: WorldState) -> PreconditionResult:
        if self.dist < 0.05:
            return PreconditionResult.failed(
                "nothing to do: give forward_m and/or right_m")
        if self.dist > self.MAX_M + 1e-6:
            return PreconditionResult.failed(
                f"{self.dist:.2f} m in one move; the limit is {self.MAX_M} m")
        base = self.base_preconditions(world)
        if not base.ok:
            return base
        # Refuse here, with a reason, what the arbiter would otherwise clamp
        # silently into a shorter move. Checked UNSCALED, which is the goal
        # further from the origin of the two.
        origin = world.origin
        if origin is not None:
            gx, gy = self._goal_from(world)
            far = math.hypot(gx - origin[0], gy - origin[1])
            if far > self.cfg.envelope.fence_radius_m:
                return PreconditionResult.failed(
                    f"that would end {far:.1f} m from where the flight "
                    f"began; the limit is "
                    f"{self.cfg.envelope.fence_radius_m} m")
        return PreconditionResult.passed()

    def height_scale(self, world: WorldState) -> float:
        """
        Real metres per EKF metre, from this tick's two heights. 1.0 when
        the correction is off or either height is missing or implausible;
        the goal is then the EKF-frame distance, as before 0.20.0.
        """
        if not self.cfg.skills.move_scale_by_height:
            return 1.0
        rf = world.rngfnd_down_m if world.rngfnd_down_trusted else None
        ekf = world.pose.alt_m if world.pose is not None else None
        if rf is None or ekf is None or ekf < 0.2 or rf < 0.2:
            return 1.0
        return max(self.cfg.skills.move_scale_min,
                   min(self.cfg.skills.move_scale_max, rf / ekf))

    def _goal_from(self, world: WorldState, scale: float = 1.0
                   ) -> Tuple[float, float]:
        # NED: x north, y east, yaw clockwise from north. The goal is in the
        # EKF frame, so the real distance asked for is divided by the scale.
        p = world.pose
        c, s = math.cos(p.yaw_rad), math.sin(p.yaw_rad)
        fwd, rgt = self.forward / scale, self.right / scale
        return (p.x + fwd * c - rgt * s, p.y + fwd * s + rgt * c)

    def on_start(self, world: WorldState) -> None:
        if world.pose is None:
            return
        self.capture_hold(world)
        self.p0 = self.hold
        self.scale = self.height_scale(world)
        self.goal = self._goal_from(world, self.scale)
        self._t0 = world.t_mono
        self.metrics["start_heading_deg"] = round(world.pose.yaw_deg, 1)
        self.metrics["height_scale"] = round(self.scale, 3)

    def tick(self, world: WorldState, dt: float) -> SkillStatus:
        self.note_tick()
        if world.pose is None:
            self.request = SetpointRequest.idle("no pose")
            return SkillStatus.RUNNING
        if self.goal is None:
            self.on_start(world)
        assert self.p0 is not None and self.goal is not None
        self.track(world)

        elapsed = world.t_mono - self._t0
        sub = min(self.n_sub, int(elapsed // self.SUBSTEP_S) + 1)
        frac = sub / self.n_sub
        tx = self.p0[0] + frac * (self.goal[0] - self.p0[0])
        ty = self.p0[1] + frac * (self.goal[1] - self.p0[1])
        new = sub != self._sub_sent
        self._sub_sent = sub
        self.hold = (tx, ty, self.p0[2])
        self.request = SetpointRequest.position(
            tx, ty, self.p0[2], self.hold_yaw,
            reason=f"move {sub}/{self.n_sub}: "
                   f"{frac * self.dist:.2f} of {self.dist:.2f} m",
            immediate=new)

        left = world.pose.horizontal_distance_to(self.goal[0], self.goal[1])
        moved = world.pose.horizontal_distance_to(self.p0[0], self.p0[1])
        self.metrics["moved_m"] = round(moved, 3)
        if sub == self.n_sub and left <= self.arrive_m:
            if self._t_inside is None:
                self._t_inside = world.t_mono
            if world.t_mono - self._t_inside >= self.SETTLE_S:
                return self.succeed(
                    "arrived", moved_m=round(moved, 2),
                    moved_real_m=round(moved * self.scale, 2),
                    error_m=round(left, 2), forward_m=self.forward,
                    right_m=self.right, height_scale=round(self.scale, 3),
                    sub_steps=self.n_sub, took_s=round(elapsed, 1))
        else:
            self._t_inside = None
        if elapsed > self.n_sub * self.SUBSTEP_S + self.TIMEOUT_PAD_S:
            return self.fail(
                f"did not arrive: still {left:.2f} m short",
                moved_m=round(moved, 2), error_m=round(left, 2))
        return SkillStatus.RUNNING
