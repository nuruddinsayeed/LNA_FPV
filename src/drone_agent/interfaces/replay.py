"""
ReplayVehicle — fly a recorded flight again, offline, with no aircraft.

The point
---------
Ten flights per condition is achievable. Eight hundred planner evaluations is
not, unless you replay. Record once, replay many: the same skills and the same
arbiter see the same WorldState stream they saw in the air, and you can run
that as often as you like, change one thing at a time, and attribute the
difference to that thing.

It also separates two kinds of variance you otherwise cannot tell apart:

  perception variance  differences between conditions, measured from flight
  decision variance    differences between repeated runs on identical input

The second is free. The first costs flight time.

Two modes
---------
  OPEN LOOP (default)    pose comes straight from the log. Commands are
                         recorded but do not move anything. This is the right
                         mode for asking "what would the arbiter / planner
                         have decided, given exactly what it saw?" — the
                         answer is reproducible to the bit.

  CLOSED LOOP            pose is advanced by a first-order model that chases
                         the commanded target, seeded from the log. Use this
                         to ask "would a different controller have flown it
                         better?", knowing the dynamics are approximate.

Degradations
------------
Transforms applied to the replayed stream before anything sees it: range
error, pose noise, dropout, link loss, flow collapse. Injecting a known
magnitude and measuring what changes downstream is how you build a
dose-response curve — how much range error does it take before the planner
picks a different tool — for no flight time at all.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from ..world.state import Pose, WorldState, from_log


class BypassError(RuntimeError):
    pass


# ───────────────────────── degradations ──────────────────────────

@dataclass
class Degradation:
    """
    A named transform on the replayed stream. Everything is off by default, so
    a plain replay is a faithful one.
    """

    name: str = "none"
    seed: int = 0

    range_bias_m: float = 0.0           # constant offset on the down RF
    range_noise_m: float = 0.0          # gaussian, per sample
    range_dropout_p: float = 0.0        # chance a reading goes missing
    range_distrust: bool = False        # force the RF to read as untrusted

    pose_noise_m: float = 0.0           # gaussian on x/y
    pose_drift_mps: float = 0.0         # linear drift, m per second of replay
    pose_dropout_p: float = 0.0         # chance the pose goes missing
    pose_extra_age_s: float = 0.0       # add latency to pose_age_s

    flow_quality_scale: float = 1.0     # multiply flow quality
    link_drop_from_s: Optional[float] = None    # link down after this offset

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._t0: Optional[float] = None

    def apply(self, w: WorldState) -> WorldState:
        if self.name == "none":
            return w
        if self._t0 is None:
            self._t0 = w.t_mono
        dt = w.t_mono - self._t0
        r = self._rng

        if w.rngfnd_down_m is not None:
            v = w.rngfnd_down_m + self.range_bias_m
            if self.range_noise_m:
                v += r.gauss(0.0, self.range_noise_m)
            w.rngfnd_down_m = v
            w.rngfnd_down_raw_m = v
        if self.range_dropout_p and r.random() < self.range_dropout_p:
            w.rngfnd_down_m = None
            w.rngfnd_down_trusted = False
        if self.range_distrust:
            w.rngfnd_down_trusted = False

        if w.pose is not None:
            x, y = w.pose.x, w.pose.y
            if self.pose_noise_m:
                x += r.gauss(0.0, self.pose_noise_m)
                y += r.gauss(0.0, self.pose_noise_m)
            if self.pose_drift_mps:
                x += self.pose_drift_mps * dt
            w.pose = Pose(x, y, w.pose.z, w.pose.yaw_rad, w.pose.vz,
                          w.pose.pitch_deg, w.pose.roll_deg)
        if self.pose_dropout_p and r.random() < self.pose_dropout_p:
            w.pose = None
            w.pose_age_s = math.inf
        elif self.pose_extra_age_s:
            w.pose_age_s = w.pose_age_s + self.pose_extra_age_s

        if self.flow_quality_scale != 1.0 and w.flow_quality is not None:
            w.flow_quality = int(w.flow_quality * self.flow_quality_scale)

        if self.link_drop_from_s is not None and dt >= self.link_drop_from_s:
            w.link_up = False
            w.link_age_s = dt - self.link_drop_from_s + 5.0
        return w


# Ready-made sweeps for the dose-response experiment.
def range_error_sweep(errors_m: List[float], seed: int = 0) -> List[Degradation]:
    return [Degradation(name=f"range_bias_{e:+.2f}m", seed=seed,
                        range_bias_m=e) for e in errors_m]


def pose_noise_sweep(sigmas_m: List[float], seed: int = 0) -> List[Degradation]:
    return [Degradation(name=f"pose_noise_{s:.2f}m", seed=seed,
                        pose_noise_m=s) for s in sigmas_m]


# ───────────────────────── replay vehicle ────────────────────────

class ReplayVehicle:
    """
    Satisfies the actuation surface the Arbiter uses, backed by a recorded
    tick stream instead of a radio.
    """

    def __init__(self, states: List[WorldState], closed_loop: bool = False,
                 loop: bool = False) -> None:
        if not states:
            raise ValueError("no tick records in that log")
        self.states = states
        self.closed_loop = closed_loop
        self.loop = loop

        self._i = 0
        self._token: Optional[object] = None
        self._minted = False
        self.origin = None

        self.commands: List[Dict[str, Any]] = []
        self.exhausted = False

        s0 = states[0]
        self._sim_pose = s0.pose
        self._target = (s0.pose.x, s0.pose.y, s0.pose.z) if s0.pose else None
        self._yaw_target: Optional[float] = None
        self._yaw_rate = 0.0

    # ---- actuation surface ------------------------------------------

    def mint_actuation_token(self) -> object:
        if self._minted:
            raise BypassError("an actuation token has already been minted")
        self._minted = True
        self._token = object()
        return self._token

    def _check(self, token: Any, what: str) -> None:
        if self._token is None or token is not self._token:
            raise BypassError(f"{what} without the arbiter's actuation token")

    def send_position_target(self, token: Any, x: float, y: float, z: float,
                             yaw_rad: Optional[float] = None) -> None:
        self._check(token, "send_position_target")
        self.commands.append({"t": self.now(), "kind": "position",
                              "x": x, "y": y, "z": z,
                              "yaw_deg": (None if yaw_rad is None
                                          else math.degrees(yaw_rad))})
        self._target = (x, y, z)
        if yaw_rad is not None:
            self._yaw_target, self._yaw_rate = yaw_rad, 60.0

    def send_condition_yaw(self, token: Any, heading_deg: float,
                           rate_dps: float, direction: int) -> int:
        self._check(token, "send_condition_yaw")
        self.commands.append({"t": self.now(), "kind": "yaw",
                              "heading_deg": heading_deg,
                              "rate_dps": rate_dps, "dir": direction})
        self._yaw_target = math.radians(heading_deg)
        self._yaw_rate = rate_dps
        return 0

    # ---- stream ------------------------------------------------------

    def now(self) -> float:
        i = min(self._i, len(self.states) - 1)
        return self.states[i].t_mono

    def poll(self) -> WorldState:
        if self._i >= len(self.states):
            if self.loop:
                self._i = 0
            else:
                self.exhausted = True
                return self.states[-1]
        w = self.states[self._i]
        self._i += 1

        if self.closed_loop:
            w = self._advance(w)
        return w

    def get_state(self) -> WorldState:
        return self.states[min(self._i, len(self.states) - 1)]

    def _advance(self, w: WorldState) -> WorldState:
        """First-order chase. Approximate dynamics — say so in any result
        that depends on them."""
        if self._sim_pose is None or self._target is None:
            return w
        prev = self.states[max(0, self._i - 2)]
        dt = max(1e-3, w.t_mono - prev.t_mono)
        k = min(1.0, dt * 2.0)
        x = self._sim_pose.x + (self._target[0] - self._sim_pose.x) * k
        y = self._sim_pose.y + (self._target[1] - self._sim_pose.y) * k
        z = self._sim_pose.z + (self._target[2] - self._sim_pose.z) * k
        yaw = self._sim_pose.yaw_rad
        if self._yaw_target is not None:
            err = (self._yaw_target - yaw + math.pi) % (2 * math.pi) - math.pi
            yaw += math.copysign(min(abs(err), math.radians(self._yaw_rate) * dt),
                                 err)
            yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        self._sim_pose = Pose(x, y, z, yaw, w.pose.vz if w.pose else 0.0,
                              w.pose.pitch_deg if w.pose else 0.0,
                              w.pose.roll_deg if w.pose else 0.0)
        w.pose = self._sim_pose
        w.rngfnd_down_m = -z if w.rngfnd_down_trusted else w.rngfnd_down_m
        return w


# ─────────────────────────── loading ─────────────────────────────

def load_records(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue        # a truncated last line is normal after a crash
    return out


def load_states(path: str, skill: Optional[str] = None,
                degradation: Optional[Degradation] = None) -> List[WorldState]:
    """Rebuild the WorldState stream from a flight log."""
    states: List[WorldState] = []
    for rec in load_records(path):
        if rec.get("type") != "tick":
            continue
        if skill and rec.get("skill") != skill:
            continue
        if not skill and rec.get("skill") == "land":
            continue        # 0.38: the recorded descent is not a skill's input
        w = from_log(rec)
        if degradation is not None:
            w = degradation.apply(w)
        states.append(w)
    return states


def load_meta(path: str) -> Dict[str, Any]:
    for rec in load_records(path):
        if rec.get("type") == "meta":
            return rec
    return {}


# ─────────────────────────── the runner ──────────────────────────

@dataclass
class ReplayOutcome:
    status: str
    reason: str = ""
    elapsed_s: float = 0.0
    ticks: int = 0
    interventions: int = 0
    blocked: int = 0
    arbiter_actions: Dict[str, int] = field(default_factory=dict)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    degradation: str = "none"
    exhausted: bool = False
    inject: str = "none"
    inject_tick: Optional[int] = None
    inject_actions: List[str] = field(default_factory=list)
    inject_transmitted: Optional[bool] = None

    def to_row(self) -> Dict[str, Any]:
        row = {"degradation": self.degradation, "status": self.status,
               "reason": self.reason, "ticks": self.ticks,
               "elapsed_s": round(self.elapsed_s, 2),
               "interventions": self.interventions, "blocked": self.blocked,
               "commands": len(self.commands),
               "exhausted": self.exhausted}
        if self.inject != "none":
            row.update({
                "inject": self.inject,
                "inject_tick": self.inject_tick,
                "inject_actions": ";".join(self.inject_actions) or "NONE",
                "inject_transmitted": self.inject_transmitted,
            })
        row.update({f"arb.{k}": v for k, v in self.arbiter_actions.items()})
        return row


def run_skill_on_states(skill_name: str, cfg: Any, states: List[WorldState],
                        skill_params: Optional[Dict[str, Any]] = None,
                        closed_loop: bool = False,
                        degradation_name: str = "none",
                        inject: Optional[str] = None,
                        inject_at_tick: int = 20,
                        inject_immediate: bool = True,
                        inject_repeat: int = 1,
                        max_ticks: int = 20000) -> ReplayOutcome:
    """
    Run one skill against a recorded stream, through the real arbiter.

    `inject` fires a fault setpoint on `inject_at_tick`, against real
    recorded pose data. This is where setpoint faults belong: the aircraft
    is not involved, so the test costs nothing and can be swept.

    Imported lazily so this module stays importable without the skill
    registry (and so a circular import cannot bite).
    """
    from collections import Counter

    from ..faults import BadSetpointInjector
    from ..safety.arbiter import AbortRequested
    from ..safety.arbiter import Arbiter
    from ..skills import build

    vehicle = ReplayVehicle(states, closed_loop=closed_loop)
    arbiter = Arbiter(vehicle, cfg)
    skill = build(skill_name, cfg, **(skill_params or {}))
    injector: Optional[BadSetpointInjector] = None
    if inject:
        injector = BadSetpointInjector(skill, inject_at_tick, inject,
                                       immediate=inject_immediate,
                                       repeat=inject_repeat)
        skill = injector

    # Peek at the first state rather than poll()ing it: consuming a tick here
    # would make a faithful replay one sample short of its own log, which
    # shows up as a spurious stream_exhausted at the end of a long skill.
    world = vehicle.get_state()
    if world.pose is not None:
        arbiter.set_origin(world.pose.x, world.pose.y, world.pose.z)

    pre = skill.check_preconditions(world)
    if not pre.ok:
        return ReplayOutcome(status="precondition_failed", reason=pre.reason,
                             degradation=degradation_name)

    skill.start(world)
    t0 = world.t_mono
    t_prev = t0
    actions: "Counter[str]" = Counter()
    interventions = blocked = ticks = 0
    inject_info: Dict[str, Any] = {}

    for _ in range(max_ticks):
        world = vehicle.poll()
        if vehicle.exhausted:
            break
        dt = max(1e-3, world.t_mono - t_prev)
        t_prev = world.t_mono
        ticks += 1

        sent_before = arbiter.counters["sent_position"]
        just_fired = injector is not None and not injector.fired
        status = skill.tick(world, dt)
        fired_now = (injector is not None and injector.fired and just_fired)

        try:
            decision = arbiter.submit(skill.request, world)
        except AbortRequested as e:
            if fired_now:
                inject_info.update(tick=ticks, actions=["ABORT"],
                                   transmitted=False)
            return ReplayOutcome(
                status="aborted", reason=e.reason,
                elapsed_s=world.t_mono - t0, ticks=ticks,
                interventions=interventions, blocked=blocked,
                arbiter_actions=dict(actions), commands=vehicle.commands,
                metrics=dict(skill.metrics), degradation=degradation_name,
                **_inject_fields(inject, inject_info))

        if fired_now:
            inject_info.update(
                tick=ticks, actions=list(decision.actions),
                transmitted=(not decision.blocked
                             and arbiter.counters["sent_position"]
                             > sent_before))

        if decision.intervened:
            interventions += 1
        if decision.blocked:
            blocked += 1
        for a in decision.actions:
            actions[a.split()[0]] += 1

        if status.value in ("succeeded", "failed", "aborted"):
            res = skill.result(world)
            return ReplayOutcome(
                status=res.status.value, reason=res.reason,
                elapsed_s=world.t_mono - t0, ticks=ticks,
                interventions=interventions, blocked=blocked,
                arbiter_actions=dict(actions), commands=vehicle.commands,
                metrics=dict(res.metrics), degradation=degradation_name,
                **_inject_fields(inject, inject_info))

    return ReplayOutcome(
        status="stream_exhausted", reason="log ran out before the skill ended",
        elapsed_s=t_prev - t0, ticks=ticks, interventions=interventions,
        blocked=blocked, arbiter_actions=dict(actions),
        commands=vehicle.commands, metrics=dict(skill.metrics),
        degradation=degradation_name, exhausted=True)


def _inject_fields(inject: Optional[str],
                   info: Dict[str, Any]) -> Dict[str, Any]:
    if not inject:
        return {}
    return {"inject": inject,
            "inject_tick": info.get("tick"),
            "inject_actions": info.get("actions") or [],
            "inject_transmitted": info.get("transmitted")}
