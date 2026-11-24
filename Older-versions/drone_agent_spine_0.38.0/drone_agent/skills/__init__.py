"""Importing this package registers every skill in SKILL_REGISTRY."""

from .base import (PreconditionResult, Skill, SkillResult, SkillStatus,  # noqa
                   SKILL_REGISTRY, build, catalogue, register)
from .hover import Hover, HoldingSkill          # noqa: F401
from .rotate_search import RotateSearch         # noqa: F401
from .climb import ClimbTo                      # noqa: F401
from .face_target import FaceTarget            # noqa: F401
from .move import Move                          # noqa: F401
from .measure_range import MeasureRange        # noqa: F401
from .approach import Approach                  # noqa: F401

__all__ = ["Skill", "SkillResult", "SkillStatus", "PreconditionResult",
           "SKILL_REGISTRY", "build", "catalogue", "register",
           "Hover", "HoldingSkill", "RotateSearch", "ClimbTo", "FaceTarget", "Move", "MeasureRange"]
