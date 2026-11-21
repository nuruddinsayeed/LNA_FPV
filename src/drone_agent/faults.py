"""
Fault injection — deliberately ask for something the envelope must refuse.

Lives in the package rather than in the CLI because two things need it: the
live runner (run_mission.py) and the replay harness. Replay is the one you
should reach for first.

Where to inject, and why it matters
-----------------------------------
An injection only fires once a skill is ticking, which means the aircraft is
already airborne. So there are three places to do this, in increasing order
of risk:

  1. offline_check.py       pure logic, FakeVehicle, no autopilot at all.
                            Proves the arbiter's arithmetic. Already passing.

  2. replay / SITL          replay re-fires the injection against a RECORDED
                            real flight: real pose, real noise, real timing,
                            real arbiter, no aircraft. SITL runs the real
                            autopilot with simulated physics, so you also see
                            how the vehicle responds to a clamped command.
                            Between them these cover everything that matters.

  3. the actual aircraft    only worth doing once 1 and 2 pass, with the
                            envelope narrowed to the room and a pilot on the
                            sticks.

Props off is NOT on that list. With no thrust the takeoff check times out and
the skill never starts, so the injection never fires — a props-off run tests
connection, streams, arming and rangefinder routing, and nothing in flight.

The immediate flag
------------------
Injected requests set immediate=True, which bypasses the arbiter's uplink
rate limiter and nothing else. Every envelope check still applies — that is
the thing under test. Without it a clamped command can be computed, logged,
and then dropped before it reaches the vehicle, and the log shows an
intervention that proves nothing. Three real flights on 2026-11-11 did
exactly that: the injection landed one tick before a scheduled send, and
sent_position never incremented.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

from .safety.arbiter import SetpointRequest

# Each builder returns the bad request, given the current pose and the
# envelope. Keep them small and obvious — this is test apparatus, and an
# injection that is hard to reason about is worse than no injection.
FAULTS: Dict[str, str] = {
    "altitude": "10 m commanded altitude — a gross violation, expect REJECT",
    "step": "50 m horizontal jump — a gross violation, expect REJECT",
    "fence": ("outside the geofence — expect CLAMP to the fence edge; "
              "needs --fence below the step limit to isolate"),
    "clamp_alt": "1.4x the ceiling — expect CLAMP to alt_max",
    "clamp_step": "1.3x the step limit — expect CLAMP to max_step",
    "nan": "a non-finite coordinate — expect REJECT",
}


def build_fault(kind: str, pose: Any, env: Any,
                origin: Any = None,
                immediate: bool = True) -> SetpointRequest:
    req = _build_fault(kind, pose, env, origin)
    req.immediate = immediate
    return req


def _build_fault(kind: str, pose: Any, env: Any,
                 origin: Any = None) -> SetpointRequest:
    if kind == "altitude":
        return SetpointRequest.position(
            pose.x, pose.y, -10.0,
            reason="INJECTED: 10 m altitude", immediate=True)
    if kind == "step":
        return SetpointRequest.position(
            pose.x + 50.0, pose.y, pose.z,
            reason="INJECTED: 50 m step", immediate=True)
    if kind == "nan":
        return SetpointRequest.position(
            float("nan"), pose.y, pose.z,
            reason="INJECTED: non-finite coordinate", immediate=True)
    if kind == "clamp_alt":
        return SetpointRequest.position(
            pose.x, pose.y, -(env.alt_max_m * 1.4),
            reason="INJECTED: altitude just past the ceiling", immediate=True)
    if kind == "clamp_step":
        return SetpointRequest.position(
            pose.x + env.max_step_m * 1.3, pose.y, pose.z,
            reason="INJECTED: step just past the limit", immediate=True)
    if kind == "fence":
        # Aim just outside the fence, measured from the ORIGIN — the fence is
        # defined there, not at the current pose. Aiming from the pose (an
        # earlier bug) produced a target inside the fence and no clamp at all.
        #
        # Isolating the fence clamp needs fence_radius < max_step, otherwise
        # the step limit is the tighter constraint near the origin and both
        # clamps fire. Narrow it with --fence to see the fence clamp alone.
        ox, oy = (origin[0], origin[1]) if origin else (pose.x, pose.y)
        dx, dy = pose.x - ox, pose.y - oy
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            dx, dy, norm = 1.0, 0.0, 1.0        # at the origin: head +x
        reach = env.fence_radius_m * 1.3
        return SetpointRequest.position(
            ox + dx / norm * reach, oy + dy / norm * reach, pose.z,
            reason=f"INJECTED: {reach:.2f} m from origin, outside the "
                   f"{env.fence_radius_m:.2f} m fence", immediate=True)
    raise KeyError(f"unknown fault {kind!r}; have {sorted(FAULTS)}")


class BadSetpointInjector:
    """
    Transparent wrapper around a Skill. On the chosen tick it replaces
    whatever the skill asked for with a bad request, then gets out of the way.

    Everything not overridden here delegates to the wrapped skill, so the
    executor and the replay runner cannot tell the difference.
    """

    def __init__(self, skill: Any, at_tick: int = 20,
                 kind: str = "altitude",
                 on_fire: Optional[Callable[[str], None]] = None,
                 immediate: bool = True, repeat: int = 1) -> None:
        if kind not in FAULTS:
            raise KeyError(f"unknown fault {kind!r}; have {sorted(FAULTS)}")
        self._skill = skill
        self._at = at_tick
        self._kind = kind
        # immediate=False reproduces the hollow test: the clamped command may
        # be swallowed by the rate limiter before it reaches the vehicle.
        # Useful for demonstrating why the flag matters; never what you want
        # when actually testing the envelope.
        self._immediate = immediate
        # repeat > 1 fires on consecutive ticks, which is what exercises the
        # arbiter's escalation from "block one tick" to "the caller is broken,
        # land". A single fault can never trip it by design.
        self._repeat = max(1, repeat)
        self._on_fire = on_fire
        self._n = 0
        self.fired = False
        self.fired_at_tick: Optional[int] = None
        self.fire_count = 0

    # Delegate everything the wrapper does not define itself.
    def __getattr__(self, item: str) -> Any:
        return getattr(self._skill, item)

    @property
    def request(self) -> Any:
        return self._skill.request

    @request.setter
    def request(self, v: Any) -> None:
        self._skill.request = v

    def tick(self, world: Any, dt: float) -> Any:
        status = self._skill.tick(world, dt)
        self._n += 1
        if (self._at <= self._n < self._at + self._repeat
                and world.pose is not None):
            bad = build_fault(self._kind, world.pose,
                              self._skill.cfg.envelope,
                              origin=getattr(world, "origin", None),
                              immediate=self._immediate)
            if not self.fired:
                self.fired = True
                self.fired_at_tick = self._n
            self.fire_count = getattr(self, "fire_count", 0) + 1
            if self._on_fire:
                self._on_fire(bad.reason)
            self._skill.request = bad
        return status
