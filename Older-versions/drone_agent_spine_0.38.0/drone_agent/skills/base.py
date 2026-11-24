"""
Skill base class.

A skill is a deterministic state machine with an explicit convergence
criterion and an explicit terminal condition. That is not incidental — it is
the layer whose absence sank three of the four models in Baig et al. (ICAIC
2026), where the LLM emitted rotate/hover/descend/land directly: magistral-24b
descended correctly onto the target but never emitted the terminating action
and timed out on all ten runs; gemma3 could not tell that a centering step had
converged; llama4 never converged at all. None of those are reasoning
failures. They are missing exit conditions.

So: tick() returns a status, and SUCCEEDED means a criterion in code was met.
No language model is ever asked whether it has arrived.

Contract
--------
  check_preconditions(world) -> PreconditionResult
      Cheap, pure, no actuation. Run once before the skill starts.

  tick(world, dt) -> SkillStatus
      Called at tick_hz. Must not block for long and must not touch the
      vehicle; to command anything, set self.request.

  self.request
      The SetpointRequest the executor hands to the arbiter after tick().
      Default SetpointRequest.idle() means "send nothing this tick", which is
      a real and necessary state: during a CONDITION_YAW leg a position target
      cancels the slew.

  on_abort(world, reason)
      Must leave the vehicle in a state the landing controller can take over.

  result(...)
      The structured outcome. This is what the planner will consume and what
      the evaluation parses; it is not a return code.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ..config import Config
from ..safety.arbiter import SetpointRequest
from ..world.state import WorldState


class SkillStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class PreconditionResult:
    ok: bool
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def passed() -> "PreconditionResult":
        return PreconditionResult(True)

    @staticmethod
    def failed(reason: str, **detail: Any) -> "PreconditionResult":
        return PreconditionResult(False, reason, detail)


@dataclass
class SkillResult:
    """
    The structured result. Every field here exists because something
    downstream needs it:

      status/reason  -> the planner decides whether to replan
      detail         -> the failure taxonomy for the evaluation chapter
      final_pose     -> where the aircraft actually ended up
      elapsed_s      -> latency metrics
      metrics        -> skill-specific numbers (max excursion, worst flow, ...)
    """

    skill: str
    status: SkillStatus
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    final_pose: Optional[Dict[str, float]] = None
    elapsed_s: float = 0.0
    ticks: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is SkillStatus.SUCCEEDED

    def to_log(self) -> Dict[str, Any]:
        return {
            "skill": self.skill,
            "status": self.status.value,
            "reason": self.reason,
            "detail": self.detail,
            "final_pose": self.final_pose,
            "elapsed_s": round(self.elapsed_s, 3),
            "ticks": self.ticks,
            "metrics": self.metrics,
        }

    def to_planner_view(self) -> Dict[str, Any]:
        """What the LLM gets back after a tool call. Deliberately small."""
        out: Dict[str, Any] = {
            "skill": self.skill,
            "status": self.status.value,
            "elapsed_s": round(self.elapsed_s, 1),
        }
        if self.reason:
            out["reason"] = self.reason
        if self.detail:
            out["detail"] = self.detail
        if self.final_pose:
            out["final_pose"] = self.final_pose
        return out


class Skill(ABC):
    """Base for every deterministic behaviour."""

    name: str = "skill"
    schema: Dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, cfg: Config, **params: Any) -> None:
        self.cfg = cfg
        self.params = params
        self.status = SkillStatus.PENDING
        self.request: SetpointRequest = SetpointRequest.idle()
        self.metrics: Dict[str, Any] = {}
        self._t_start: Optional[float] = None
        self._ticks = 0
        self._reason = ""
        self._detail: Dict[str, Any] = {}

    # ------------------------------------------------------------------

    def start(self, world: WorldState) -> None:
        self._t_start = time.monotonic()
        self.status = SkillStatus.RUNNING
        self.on_start(world)

    def on_start(self, world: WorldState) -> None:
        """Hook. Capture whatever the skill needs from the first state."""

    @abstractmethod
    def check_preconditions(self, world: WorldState) -> PreconditionResult:
        ...

    @abstractmethod
    def tick(self, world: WorldState, dt: float) -> SkillStatus:
        ...

    def on_abort(self, world: WorldState, reason: str) -> None:
        """Leave the vehicle somewhere the landing controller can take over.
        Default: stop asking for anything."""
        self.request = SetpointRequest.idle(f"aborted: {reason}")

    # ------------------------------------------------------------------

    def note_tick(self) -> None:
        self._ticks += 1

    def succeed(self, reason: str = "", **detail: Any) -> SkillStatus:
        self._reason, self._detail = reason, detail
        self.status = SkillStatus.SUCCEEDED
        self.request = SetpointRequest.idle("done")
        return self.status

    def fail(self, reason: str, **detail: Any) -> SkillStatus:
        self._reason, self._detail = reason, detail
        self.status = SkillStatus.FAILED
        self.request = SetpointRequest.idle(f"failed: {reason}")
        return self.status

    def abort(self, reason: str, **detail: Any) -> SkillStatus:
        self._reason, self._detail = reason, detail
        self.status = SkillStatus.ABORTED
        self.request = SetpointRequest.idle(f"aborted: {reason}")
        return self.status

    @property
    def elapsed(self) -> float:
        return 0.0 if self._t_start is None else time.monotonic() - self._t_start

    def result(self, world: Optional[WorldState] = None) -> SkillResult:
        pose = None
        if world is not None and world.pose is not None:
            pose = {"x": round(world.pose.x, 3), "y": round(world.pose.y, 3),
                    "z": round(world.pose.z, 3),
                    "yaw_deg": round(world.pose.yaw_deg, 1)}
        return SkillResult(
            skill=self.name, status=self.status, reason=self._reason,
            detail=dict(self._detail), final_pose=pose,
            elapsed_s=self.elapsed, ticks=self._ticks,
            metrics=dict(self.metrics))


# ─────────────────────────── registry ────────────────────────────

SKILL_REGISTRY: Dict[str, type] = {}


def register(cls: type) -> type:
    """Decorator. The registry is also the tool catalogue the planner will be
    handed — name plus JSON Schema, nothing else — so there is exactly one
    place where a skill becomes callable."""
    SKILL_REGISTRY[cls.name] = cls
    return cls


def build(name: str, cfg: Config, **params: Any) -> Skill:
    if name not in SKILL_REGISTRY:
        raise KeyError(f"unknown skill {name!r}; "
                       f"have {sorted(SKILL_REGISTRY)}")
    return SKILL_REGISTRY[name](cfg, **params)


def catalogue() -> Dict[str, Dict[str, Any]]:
    """The tool catalogue, for the planner. Not used yet."""
    return {name: {"name": name, "schema": cls.schema,
                   "description": (cls.__doc__ or "").strip().split("\n")[0]}
            for name, cls in SKILL_REGISTRY.items()}
