"""
The arbiter. Every actuation the aircraft receives passes through here.

It holds the only actuation token the vehicle will accept, so a skill cannot
reach past it even by accident. Skills return a SetpointRequest from tick();
the executor hands that to submit(); submit() clamps, blocks or passes it and
returns a decision that gets logged.

Why the requested and applied setpoints are logged separately
-------------------------------------------------------------
The gap between request and applied IS the safety-intervention metric. Count
the ticks where they differ, group by the action that caused the difference,
and you have a defensible number for how often the language model (or a
controller bug) asked for something the envelope refused. That number is hard
to produce after the fact, so it is produced at the point of decision.

What the arbiter does NOT do
----------------------------
It does not plan, and it does not decide the mission is over. It clamps,
blocks, and raises an abort. Landing is the landing state machine's job; the
arbiter's abort is what sends control there.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

from ..config import Config
from ..world.state import WorldState


class RequestKind(str, Enum):
    IDLE = "idle"               # deliberately quiet: send nothing this tick
    POSITION = "position"       # SET_POSITION_TARGET_LOCAL_NED
    YAW = "yaw"                 # MAV_CMD_CONDITION_YAW


@dataclass
class SetpointRequest:
    """What a skill wants. Never what the vehicle gets — that is the
    decision."""

    kind: RequestKind = RequestKind.IDLE
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None           # NED down, negative = up
    yaw_rad: Optional[float] = None
    heading_deg: Optional[float] = None
    yaw_rate_dps: Optional[float] = None
    yaw_dir: int = 0
    reason: str = ""

    # Bypass the uplink rate limiter — never the envelope. Needed for the
    # position re-assert that must land immediately BEFORE a CONDITION_YAW:
    # the ordering is the whole point, and waiting for the rate limiter would
    # put it inside the leg, where it cancels the slew.
    immediate: bool = False

    @staticmethod
    def idle(reason: str = "") -> "SetpointRequest":
        return SetpointRequest(kind=RequestKind.IDLE, reason=reason)

    @staticmethod
    def position(x: float, y: float, z: float,
                 yaw_rad: Optional[float] = None,
                 reason: str = "",
                 immediate: bool = False) -> "SetpointRequest":
        return SetpointRequest(kind=RequestKind.POSITION, x=x, y=y, z=z,
                               yaw_rad=yaw_rad, reason=reason,
                               immediate=immediate)

    @staticmethod
    def yaw(heading_deg: float, rate_dps: float, direction: int,
            reason: str = "") -> "SetpointRequest":
        return SetpointRequest(kind=RequestKind.YAW, heading_deg=heading_deg,
                               yaw_rate_dps=rate_dps, yaw_dir=direction,
                               reason=reason)

    def to_log(self) -> Optional[dict]:
        if self.kind is RequestKind.IDLE:
            return {"kind": "idle", "reason": self.reason}
        if self.kind is RequestKind.POSITION:
            return {"kind": "position",
                    "x": _r(self.x), "y": _r(self.y), "z": _r(self.z),
                    "yaw_deg": (_r(math.degrees(self.yaw_rad))
                                if self.yaw_rad is not None else None),
                    "reason": self.reason}
        return {"kind": "yaw", "heading_deg": _r(self.heading_deg),
                "rate_dps": self.yaw_rate_dps, "dir": self.yaw_dir,
                "reason": self.reason}


@dataclass
class ArbiterDecision:
    applied: SetpointRequest
    actions: List[str] = field(default_factory=list)   # what it changed
    blocked: bool = False
    abort_reason: Optional[str] = None

    @property
    def intervened(self) -> bool:
        return bool(self.actions) or self.blocked


class AbortRequested(Exception):
    """The arbiter has decided this flight must end. The executor catches
    this and hands the aircraft to the landing state machine."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 3)



def _dist_to_segment(p: Tuple[float, float], a: Tuple[float, float],
                     b: Tuple[float, float]) -> float:
    """Horizontal distance from p to the segment a-b (a point if a == b)."""
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * vx
                                                 + (py - ay) * vy) / L2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


class Arbiter:
    """
    Construct exactly one, with the vehicle. It mints the actuation token in
    its constructor, which is what makes every other path to the vehicle fail.
    """

    def __init__(self, vehicle: Any, cfg: Config) -> None:
        self.vehicle = vehicle
        self.cfg = cfg
        self.env = cfg.envelope
        self._token = vehicle.mint_actuation_token()

        self.origin: Optional[Tuple[float, float, float]] = None
        self.estop_latched = False
        self.estop_reason: Optional[str] = None

        self._t_last_tick = time.monotonic()
        self._now = time.monotonic()
        self._t_pose_stale_since: Optional[float] = None
        self._t_last_position_sent = 0.0
        # Where the aircraft was last TOLD to be. The drift check measures
        # from here, not from the origin: see "drift from the hold point".
        # The commanded path: from the previous position target to the
        # current one. Commanded point to commanded point, so it does not
        # move with the aircraft -- a path that restarted wherever the
        # aircraft had got to would hide every slow drift.
        self.hold_from_xy: Optional[Tuple[float, float]] = None
        self.hold_xy: Optional[Tuple[float, float]] = None
        self.peak_drift_m = 0.0
        self._gross_run = 0
        self._breach_run = 0
        # Peak excursion of the AIRCRAFT past the envelope, in metres. The
        # headline number from any clamp test: the setpoint log cannot show it.
        self.peak_alt_breach_m = 0.0        # best-estimate frame
        self.peak_alt_breach_ekf_m = 0.0    # commanded (LOCAL_NED) frame
        self.peak_fence_breach_m = 0.0

        # Counters — these are the evaluation metrics.
        self.counters = {
            "submitted": 0, "intervened": 0, "blocked": 0,
            "clamp_alt": 0, "clamp_fence": 0, "clamp_step": 0,
            "clamp_yaw_rate": 0, "block_gross": 0, "abort_gross": 0,
            "breach_ticks": 0,
            "block_estop": 0, "block_stale": 0,
            "block_disarmed": 0, "sent_position": 0, "sent_yaw": 0,
            "suppressed_rate": 0,
        }

    # ───────────────────────── lifecycle ──────────────────────

    def set_origin(self, x: float, y: float, z: float) -> None:
        """Captured at handover. The geofence is measured from here."""
        self.origin = (x, y, z)
        self.vehicle.origin = self.origin
        if self.hold_xy is None:
            self.hold_xy = (x, y)
            self.hold_from_xy = (x, y)

    def estop(self, reason: str) -> None:
        """Latching. Once tripped, only a manual reset clears it — an
        automatic recovery from an emergency stop is not an emergency stop."""
        if not self.estop_latched:
            self.estop_latched = True
            self.estop_reason = reason

    def reset_estop(self) -> None:
        self.estop_latched = False
        self.estop_reason = None

    def note_tick(self, now: Optional[float] = None) -> None:
        self._t_last_tick = time.monotonic() if now is None else now

    # ─────────────────────── the gate ─────────────────────────

    def submit(self, req: SetpointRequest,
               world: WorldState) -> ArbiterDecision:
        """Clamp, block or pass. The only path to the vehicle."""
        self.counters["submitted"] += 1
        # Every timer in the arbiter runs on the world's clock. In flight
        # that is time.monotonic(); in replay it is the recorded sample time,
        # so the rate limiter and the staleness grace window reproduce what
        # actually happened instead of collapsing into the replay's runtime.
        self._now = world.t_mono
        self.note_tick(self._now)

        decision = self._evaluate(req, world)

        if decision.abort_reason:
            self.counters["blocked"] += 1
            if "gross violation" in decision.abort_reason:
                self.counters["abort_gross"] += 1
            raise AbortRequested(decision.abort_reason)

        if decision.intervened:
            self.counters["intervened"] += 1
        if decision.blocked:
            self.counters["blocked"] += 1
            return decision

        self._actuate(decision.applied)
        return decision

    # ------------------------------------------------------------------

    def _evaluate(self, req: SetpointRequest,
                  world: WorldState) -> ArbiterDecision:
        actions: List[str] = []

        # --- hard aborts, checked before anything else -------------------
        if self.estop_latched:
            self.counters["block_estop"] += 1
            return ArbiterDecision(SetpointRequest.idle("estop"),
                                   ["block:estop"], blocked=True,
                                   abort_reason=f"e-stop: {self.estop_reason}")

        if not world.armed:
            self.counters["block_disarmed"] += 1
            return ArbiterDecision(SetpointRequest.idle("disarmed"),
                                   ["block:disarmed"], blocked=True)

        if not world.link_up and math.isfinite(world.link_age_s):
            return ArbiterDecision(
                SetpointRequest.idle("link down"), ["block:link"],
                blocked=True,
                abort_reason=f"no heartbeat for {world.link_age_s:.1f}s")

        # --- staleness: an old estimate is not an estimate ---------------
        stale = world.pose is None or world.pose_age_s > self.env.pose_max_age_s
        now = world.t_mono
        if stale:
            if self._t_pose_stale_since is None:
                self._t_pose_stale_since = now
            held = now - self._t_pose_stale_since
            self.counters["block_stale"] += 1
            if held > self.env.pose_stale_hold_s:
                return ArbiterDecision(
                    SetpointRequest.idle("pose stale"), ["block:stale"],
                    blocked=True,
                    abort_reason=f"position estimate stale for {held:.1f}s")
            # Inside the grace window: hold, do not command anything new.
            return ArbiterDecision(SetpointRequest.idle("pose stale (hold)"),
                                   ["block:stale_hold"], blocked=True)
        self._t_pose_stale_since = None

        # --- drift from the hold point -----------------------------------
        # Measured from the last position target actually transmitted, not
        # from the origin. Until 0.38 it was the origin, which made it a
        # 1.5 m geofence: tighter than the 4 m fence, and tighter than
        # approach's own 3 m travel budget. On 2026-11-17 an approach from
        # 3.65 m had flown 1.49 m exactly as commanded when it fired
        # "drifted 1.50 m from the hold point". Distance from the origin is
        # the state fence's job (`_state_breach`, fence_radius_m). Across the
        # 20 flights of that evening the worst distance between the aircraft
        # and its commanded point was 0.46 m, during 0.25-0.33 m move steps.
        #
        # From the PATH, not the point: the envelope allows a 2 m step, and
        # right after one goes out the aircraft is legitimately 2 m from its
        # target. So the distance is to the segment from the previous target
        # to the current one: flying along it costs nothing; sliding off it,
        # overshooting, or backing away all count. A re-sent identical
        # target leaves the segment alone.
        if (self.env.drift_abort_enable and self.hold_xy is not None
                and world.pose is not None):
            d = _dist_to_segment((world.pose.x, world.pose.y),
                                 self.hold_from_xy or self.hold_xy,
                                 self.hold_xy)
            if d > self.peak_drift_m:
                self.peak_drift_m = d
            if d > self.env.drift_abort_m:
                return ArbiterDecision(
                    SetpointRequest.idle("drift"), ["block:drift"],
                    blocked=True,
                    abort_reason=(f"drifted {d:.2f} m off the commanded "
                                  f"path (limit {self.env.drift_abort_m} m)"))

        # --- state-based envelope ----------------------------------------
        # Where the aircraft ACTUALLY is, not where it was told to go. A
        # clamped setpoint does not bound the vehicle: its own controller
        # overshoots. Transient overshoot is recorded; a sustained excursion
        # aborts, because it means the aircraft is not where the arbiter
        # believes it is.
        breach = self._state_breach(world)
        if breach is not None:
            self._breach_run += 1
            self.counters["breach_ticks"] += 1
            actions.append(f"breach:{breach}")
            if (self.env.breach_abort_after > 0
                    and self._breach_run >= self.env.breach_abort_after):
                return ArbiterDecision(
                    SetpointRequest.idle("envelope breached"),
                    actions, blocked=True,
                    abort_reason=(f"aircraft outside the envelope for "
                                  f"{self._breach_run} ticks: {breach}"))
        else:
            self._breach_run = 0

        # --- watchdog on skill ticks -------------------------------------
        # Deliberately not a setpoint watchdog: a rotate leg goes quiet on the
        # uplink for ~2.25 s because a position target mid-leg cancels the
        # CONDITION_YAW slew.
        if now - self._t_last_tick > self.env.tick_watchdog_s:
            actions.append("watchdog:late_tick")

        if req.kind is RequestKind.IDLE:
            return ArbiterDecision(req, actions)

        if req.kind is RequestKind.YAW:
            return self._evaluate_yaw(req, actions)

        return self._evaluate_position(req, world, actions)

    # ------------------------------------------------------------------

    def _evaluate_yaw(self, req: SetpointRequest,
                      actions: List[str]) -> ArbiterDecision:
        rate = req.yaw_rate_dps or 0.0
        if abs(rate) > self.env.max_yaw_rate_dps:
            self.counters["clamp_yaw_rate"] += 1
            actions.append(f"clamp:yaw_rate {rate:.0f}->"
                           f"{self.env.max_yaw_rate_dps:.0f}")
            rate = math.copysign(self.env.max_yaw_rate_dps, rate)
        applied = SetpointRequest.yaw(req.heading_deg % 360.0, rate,
                                      req.yaw_dir, req.reason)
        return ArbiterDecision(applied, actions)

    def _evaluate_position(self, req: SetpointRequest, world: WorldState,
                           actions: List[str]) -> ArbiterDecision:
        x, y, z = req.x, req.y, req.z
        pose = world.pose
        assert pose is not None     # guaranteed by the staleness check

        # --- gross violation: reject rather than clamp --------------------
        # A clamp is the right response to a request that is slightly out of
        # bounds — a controller overshooting, a goal just past the fence. It
        # is the WRONG response to nonsense: clamping "fly to 10 m" yields
        # "fly to the ceiling", which is a legal command the caller never
        # asked for and nobody wants executed in a room. A hallucinated
        # planner parameter looks exactly like this.
        #
        # So: past a multiple of the envelope, the request is treated as
        # evidence that the caller is broken, and nothing is sent.
        gross = self._gross_violation(x, y, z, pose)
        if gross is not None:
            self.counters["block_gross"] += 1
            self._gross_run += 1
            # One gross violation is a glitch: block the tick and carry on.
            # A run of them means the caller is not going to recover, and
            # continuing just means more nonsense — so escalate to an abort,
            # which lands the aircraft. A planner emitting hallucinated
            # parameters repeatedly is exactly this case; the planner's own
            # retry budget sits above, and this is the backstop when it is
            # exhausted or absent.
            if (self.env.gross_violation_abort_after > 0
                    and self._gross_run >= self.env.gross_violation_abort_after):
                return ArbiterDecision(
                    SetpointRequest.idle("repeated gross violations"),
                    [f"block:gross {gross}"], blocked=True,
                    abort_reason=(
                        f"{self._gross_run} consecutive gross violations, "
                        f"latest: {gross}"))  # counted below on the raise
            return ArbiterDecision(
                SetpointRequest.idle(f"gross violation: {gross}"),
                [f"block:gross {gross}"], blocked=True)
        # A request that is merely clampable means the caller is still in
        # touch with reality, so the run resets here rather than on any tick.
        self._gross_run = 0

        # --- altitude envelope -------------------------------------------
        target_alt = -z
        if target_alt < self.env.alt_min_m:
            actions.append(f"clamp:alt {target_alt:.2f}->"
                           f"{self.env.alt_min_m:.2f}")
            self.counters["clamp_alt"] += 1
            target_alt = self.env.alt_min_m
        elif target_alt > self.env.alt_max_m:
            actions.append(f"clamp:alt {target_alt:.2f}->"
                           f"{self.env.alt_max_m:.2f}")
            self.counters["clamp_alt"] += 1
            target_alt = self.env.alt_max_m
        z = -target_alt

        # --- geofence, measured from the captured origin ------------------
        if self.origin is not None:
            ox, oy = self.origin[0], self.origin[1]
            d = math.hypot(x - ox, y - oy)
            if d > self.env.fence_radius_m:
                scale = self.env.fence_radius_m / d
                nx = ox + (x - ox) * scale
                ny = oy + (y - oy) * scale
                actions.append(f"clamp:fence {d:.2f}->"
                               f"{self.env.fence_radius_m:.2f}m")
                self.counters["clamp_fence"] += 1
                x, y = nx, ny

        # --- step limit: no teleporting from where we actually are ---------
        step = math.hypot(x - pose.x, y - pose.y)
        if step > self.env.max_step_m:
            scale = self.env.max_step_m / step
            x = pose.x + (x - pose.x) * scale
            y = pose.y + (y - pose.y) * scale
            actions.append(f"clamp:step {step:.2f}->"
                           f"{self.env.max_step_m:.2f}m")
            self.counters["clamp_step"] += 1

        applied = SetpointRequest.position(x, y, z, req.yaw_rad, req.reason,
                                           immediate=req.immediate)
        return ArbiterDecision(applied, actions)

    # ------------------------------------------------------------------

    def _state_breach(self, world: WorldState) -> Optional[str]:
        """
        Is the AIRCRAFT outside the envelope right now? Returns a description
        if so. Tracks peak excursion either way, because "how far past the
        limit did the vehicle actually get" is the number that matters and it
        cannot be recovered from setpoint logs.
        """
        alt = world.altitude_m
        if alt is None or world.pose is None:
            return None
        m = self.env.breach_margin_m

        # Two altitudes, and the gap between them matters. The arbiter clamps
        # a LOCAL_NED z, so the command lives in the EKF frame; with
        # EK3_SRC1_POSZ on baro that frame is not height above the floor. The
        # rangefinder is. On 2026-11-13 they differed by ~0.4 m mid-climb, so
        # "how far past the ceiling did it go" has two answers:
        #
        #   altitude_m  the system's own best estimate (rangefinder while
        #               trusted, EKF otherwise) — the physical number, and
        #               what the breach decision uses
        #   pose.alt_m  the frame the command was given in — measures how
        #               well the vehicle tracked the clamped setpoint
        #
        # Both are recorded; only the first can abort.
        over = alt - self.env.alt_max_m
        if over > self.peak_alt_breach_m:
            self.peak_alt_breach_m = over
        over_ekf = world.pose.alt_m - self.env.alt_max_m
        if over_ekf > self.peak_alt_breach_ekf_m:
            self.peak_alt_breach_ekf_m = over_ekf
        under = self.env.alt_min_m - alt

        if self.origin is not None:
            d = world.pose.horizontal_distance_to(self.origin[0],
                                                  self.origin[1])
            out = d - self.env.fence_radius_m
            if out > self.peak_fence_breach_m:
                self.peak_fence_breach_m = out
            if out > m:
                return f"fence by {out:.2f}m"

        if over > m:
            return f"alt by {over:.2f}m"
        # Below the floor is normal during takeoff and landing, so only count
        # it while the skill believes it is holding station.
        if under > m and alt > 0.10:
            return f"alt below floor by {under:.2f}m"
        return None

    def _gross_violation(self, x: float, y: float, z: float,
                         pose: Any) -> Optional[str]:
        """Return a description if this request is nonsense, else None."""
        k = self.env.gross_violation_factor
        if k <= 0:
            return None

        target_alt = -z
        if target_alt > self.env.alt_max_m * k:
            return (f"alt {target_alt:.1f}m > "
                    f"{self.env.alt_max_m * k:.1f}m")
        if target_alt < -self.env.alt_max_m * k:
            return f"alt {target_alt:.1f}m below ground"

        if self.origin is not None:
            d = math.hypot(x - self.origin[0], y - self.origin[1])
            if d > self.env.fence_radius_m * k:
                return (f"range {d:.1f}m > "
                        f"{self.env.fence_radius_m * k:.1f}m")

        step = math.hypot(x - pose.x, y - pose.y)
        if step > self.env.max_step_m * k:
            return f"step {step:.1f}m > {self.env.max_step_m * k:.1f}m"

        if not all(math.isfinite(v) for v in (x, y, z)):
            return "non-finite coordinate"
        return None

    def _actuate(self, applied: SetpointRequest) -> None:
        if applied.kind is RequestKind.IDLE:
            return

        if applied.kind is RequestKind.YAW:
            self.counters["sent_yaw"] += 1
            self.vehicle.send_condition_yaw(
                self._token, applied.heading_deg, applied.yaw_rate_dps,
                applied.yaw_dir)
            return

        # Rate-limit position targets. GUIDED holds the last target, so
        # re-sending at tick rate buys nothing and costs uplink we do not have.
        now = self._now
        if (not applied.immediate
                and now - self._t_last_position_sent
                < self.cfg.skills.setpoint_period_s):
            self.counters["suppressed_rate"] += 1
            return
        self._t_last_position_sent = now
        self.counters["sent_position"] += 1
        new = (applied.x, applied.y)
        if (self.hold_xy is None
                or math.hypot(new[0] - self.hold_xy[0],
                              new[1] - self.hold_xy[1]) > 0.01):
            self.hold_from_xy = self.hold_xy or new
            self.hold_xy = new
        self.vehicle.send_position_target(
            self._token, applied.x, applied.y, applied.z, applied.yaw_rad)

    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {"estop": self.estop_latched,
                "peak_alt_breach_m": round(self.peak_alt_breach_m, 3),
                "peak_alt_breach_ekf_m": round(self.peak_alt_breach_ekf_m, 3),
                "peak_fence_breach_m": round(self.peak_fence_breach_m, 3),
                "peak_drift_m": round(self.peak_drift_m, 3),
                "counters": dict(self.counters)}

    def summary_lines(self) -> List[str]:
        c = self.counters
        lines = [
            f"Arbiter: {c['submitted']} requests, {c['intervened']} "
            f"interventions, {c['blocked']} blocked",
            f"  sent: {c['sent_position']} position, {c['sent_yaw']} yaw "
            f"({c['suppressed_rate']} suppressed by the rate limiter)",
        ]
        if self.peak_alt_breach_m > 0 or self.peak_fence_breach_m > 0:
            lines.append(
                f"  aircraft peak excursion past the envelope: "
                f"alt +{self.peak_alt_breach_m:.2f} m "
                f"(+{self.peak_alt_breach_ekf_m:.2f} m in the commanded "
                f"frame), fence +{self.peak_fence_breach_m:.2f} m "
                f"({c['breach_ticks']} ticks beyond the "
                f"{self.env.breach_margin_m:.2f} m margin)")
        clamps = {k: v for k, v in c.items()
                  if k.startswith(("clamp_", "block_")) and v}
        if clamps:
            lines.append("  " + ", ".join(f"{k}={v}"
                                          for k, v in sorted(clamps.items())))
        return lines
