"""
What happened to a mission, in one word, decided by rule.

The planner has a graded evaluation: 45 instructions, rubrics checked in
code, a pass rate per category. The flights have had nothing of the kind.
Every flight so far has been read by hand, which is fine for three flights
and useless for thirty-three, and a thesis that reports an end-to-end
success rate cannot get it that way.

The difficulty is that "did the mission succeed" is not the same question
as "did every skill return SUCCEEDED", and the difference matters in both
directions:

  * A flight that ends with `approach` refusing because the person walked
    away is not a failure of the aircraft. The cross-check fired, the
    aircraft stopped, nobody was flown into. Counting that as a failure
    would mean the safest possible behaviour lowers the score.
  * A flight where the arbiter aborted on real height is also not a
    software fault -- the guard did its job, 12 times out of 12 -- but it
    IS a mission that did not do what was asked, and hiding it inside
    "stopped safely" would flatter the aircraft.

So outcomes have three kinds, not two: the mission finished, the mission
was stopped by a guard that was right to stop it, or something went wrong.
All three are reported, and the failure taxonomy is declared here in code
rather than assembled by eye afterwards, so the same log always lands in
the same bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Kind(str, Enum):
    COMPLETED = "completed"       # every stage the mission asked for ran
    STOPPED = "stopped safely"    # a guard fired and was right to
    FAILED = "failed"             # something did not work
    UNFLOWN = "never flew"        # it did not get off the ground


# Each rule is (kind, label, the test). The first that matches decides, so
# the order is the priority: a guard firing outranks the skill it
# interrupted, because the guard is why the skill stopped.
#
# Every label here comes from a reason string this project has actually
# logged. Anything unmatched is reported verbatim under "unclassified",
# which is the signal to add a rule rather than to widen one.
RULES: List[Tuple[Kind, str, str]] = [
    (Kind.STOPPED, "envelope: real height",        "alt by"),
    (Kind.STOPPED, "envelope: real height",        "rangefinder above"),
    (Kind.STOPPED, "envelope: too low",            "rangefinder below"),
    (Kind.STOPPED, "envelope: fence",              "from where the flight began"),
    # 0.38 measures drift from the commanded path; a real slide off it.
    (Kind.STOPPED, "drifted off the commanded path", "off the commanded path"),
    # Up to 0.37 the same check measured from the takeoff point, a 1.5 m
    # geofence inside approach's 3 m budget: on 2026-11-17 it aborted an
    # approach that had flown 1.49 m exactly as commanded. A stack fault.
    (Kind.FAILED,  "drift check measured from takeoff (stack fault, <=0.37)",
     "from the hold point"),
    (Kind.STOPPED, "target moved",                 "has moved"),
    (Kind.STOPPED, "target left the cone",         "off the nose"),
    (Kind.STOPPED, "range not good enough to act on", "range good only to"),
    (Kind.STOPPED, "no range source it would act on", "range source"),
    (Kind.FAILED,  "target lost",                  "is not in view"),
    (Kind.FAILED,  "target lost",                  "not in view"),
    (Kind.FAILED,  "no steady view",               "no steady view"),
    (Kind.FAILED,  "nothing found",                "found nothing"),
    (Kind.FAILED,  "nothing found",                "no person"),
    (Kind.FAILED,  "did not arrive",               "did not arrive"),
    (Kind.FAILED,  "did not converge",             "still "),
    (Kind.FAILED,  "step refused",                 "not allowed"),
    (Kind.FAILED,  "range and movement disagree",  "disagree"),
    (Kind.FAILED,  "mode left GUIDED",             "mode"),
    (Kind.FAILED,  "link",                         "link"),
    (Kind.FAILED,  "pose went stale",              "pose"),
]


@dataclass
class Stage:
    """One skill in the mission, and how it ended."""

    skill: str
    status: str
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"


@dataclass
class Outcome:
    """A whole flight's verdict."""

    name: str
    mission: List[str] = field(default_factory=list)   # what was asked for
    stages: List[Stage] = field(default_factory=list)  # what ran
    instruction: str = ""
    kind: Kind = Kind.FAILED
    label: str = ""
    reason: str = ""
    reached: int = 0        # how many stages of the mission actually ran
    # For a flight started from words there are two questions, and they
    # have to be kept apart: was the plan right for the instruction, and
    # did the aircraft carry out the plan. End to end needs both.
    plan_ok: Optional[bool] = None
    plan_why: List[str] = field(default_factory=list)
    planned: List[str] = field(default_factory=list)

    @property
    def plan_refused(self) -> bool:
        """The planner declined: a plan with no steps in it."""
        return self.plan_ok is not None and not self.planned

    @property
    def end_to_end(self) -> Optional[bool]:
        """
        Words in, right outcome out. None where the question does not
        apply, so those flights can be left out of the denominator rather
        than counted as misses.

        A correctly refused instruction IS a success: the system was asked
        for something it should not do, said so, and nothing took off.
        Counting the absent flight as a failure would score a refusal the
        same as a crash. A flight that never left the ground for any other
        reason -- a dry run on the bench -- is neither, and returns None.
        """
        if not self.instruction or self.plan_ok is None:
            return None
        if not self.plan_ok:
            return False
        if self.plan_refused:
            return self.kind is Kind.UNFLOWN
        if self.kind is Kind.UNFLOWN:
            return None            # planned but never flown: not an answer
        return self.completed

    @property
    def completed(self) -> bool:
        return self.kind is Kind.COMPLETED

    def to_log(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind.value,
                "label": self.label, "mission": self.mission,
                "reached": self.reached, "instruction": self.instruction,
                "planned": self.planned, "plan_ok": self.plan_ok,
                "plan_why": self.plan_why[:2],
                "end_to_end": self.end_to_end,
                "plan_refused": self.plan_refused,
                "stages": [{"skill": s.skill, "status": s.status}
                           for s in self.stages],
                "reason": self.reason[:160]}


def classify(reason: str) -> Optional[Tuple[Kind, str]]:
    """The first rule whose text appears in the reason."""
    low = (reason or "").lower()
    for kind, label, needle in RULES:
        if needle in low:
            return kind, label
    return None


def mission_from_argv(argv: List[str]) -> List[str]:
    """
    The stages the operator asked for, read off the command line. A flight
    is judged against what it was told to do, not against what it managed:
    a search that found nobody and therefore never ran `approach` has not
    completed a search-face-approach mission.
    """
    if not argv:
        return []
    if "--instruction" in argv:
        return ["(planned)"]
    stages: List[str] = []
    for a in argv[1:]:
        if not a.startswith("-"):
            stages.append(a)
            break
    if "--then-face" in argv:
        stages.append("face_target")
    if "--then-measure" in argv:
        stages.append("measure_range")
    if "--then-approach" in argv:
        stages.append("approach")
    return stages


def verdict(name: str, argv: List[str], stages: List[Stage],
            instruction: str = "", airborne: bool = True,
            planned: Optional[List[str]] = None) -> Outcome:
    """
    Decide a flight. The rules, in order:

      1. Nothing left the ground -> never flew.
      2. Any stage that did not succeed -> that stage's reason decides,
         through RULES.
      3. Every asked-for stage ran and succeeded -> completed.
      4. Every stage that ran succeeded but the mission asked for more ->
         failed, because the mission did not happen.
    """
    # A flight started from words is judged against ITS OWN plan: the
    # question "did the aircraft do what it was told" becomes "did it
    # carry out the plan it made". Whether that plan was right for the
    # instruction is a separate question, graded by the rubrics, and
    # conflating the two would hide which half went wrong.
    mission = list(planned) if planned else mission_from_argv(argv)
    out = Outcome(name=name, mission=mission, stages=stages,
                  instruction=instruction, planned=list(planned or []),
                  reached=sum(1 for s in stages if s.ok))
    if not airborne:
        out.kind, out.label = Kind.UNFLOWN, "never left the ground"
        return out
    bad = next((s for s in stages if not s.ok), None)
    if bad is not None:
        out.reason = bad.reason
        got = classify(bad.reason)
        if got is None:
            out.kind, out.label = Kind.FAILED, f"unclassified ({bad.skill})"
        else:
            out.kind, out.label = got
            out.label = f"{out.label} ({bad.skill})"
        return out
    want = [m for m in out.mission if m != "(planned)"]
    ran = [s.skill for s in stages]
    missing = [m for m in want if m not in ran]
    if missing:
        out.kind = Kind.FAILED
        out.label = f"never reached {missing[0]}"
        out.reason = (f"asked for {' -> '.join(want)}, ran "
                      f"{' -> '.join(ran) or 'nothing'}")
        return out
    if not stages:
        out.kind, out.label = Kind.FAILED, "no skill ran at all"
        return out
    out.kind, out.label = Kind.COMPLETED, "completed"
    return out


def attrition(outcomes: List[Outcome], stages: List[str]
              ) -> List[Tuple[str, int, int]]:
    """
    How many flights reached each stage and how many got through it. This
    is the shape of an end-to-end result: a single success rate hides
    whether the misses are all in one place.
    """
    out = []
    for i, stage in enumerate(stages):
        got = [o for o in outcomes if any(s.skill == stage for s in o.stages)]
        through = [o for o in got
                   if any(s.skill == stage and s.ok for s in o.stages)]
        out.append((stage, len(got), len(through)))
    return out
