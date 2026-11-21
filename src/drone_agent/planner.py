"""
The planner: one instruction in plain language -> a short list of skill calls.

The language model never touches the vehicle. It is asked once, on the
ground, before anything is armed. What it returns is treated as untrusted
text: it is parsed, checked against each skill's schema and the flight
envelope, and only then shown to the operator. A reply that does not pass is
sent back with the reasons, at most twice; after that there is no plan and
no flight. In the air the plan is fixed, and the skills, the arbiter and the
landing logic are exactly the ones that fly without a planner.

Transport is the OpenAI-compatible chat API (POST {url}/chat/completions),
which vLLM, Ollama, LiteLLM and Open WebUI all speak. Standard library only.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MAX_STEPS = 6
MAX_PLAN_S = 90.0          # one battery; a 360 deg sweep alone takes ~40 s
MAX_REPAIRS = 2
# Measured on the 2.2in airframe: 360 deg in 8 legs at 20 deg/s takes
# 38.7-40.5 s, which is 18 s of turning and about 2.75 s per leg of settling.
DEFAULT_RATE_DPS = 20.0
DEFAULT_LEG_DEG = 45.0
LEG_OVERHEAD_S = 2.75
CLIMB_S = 8.0
MAX_PLAN_MOVE_M = 2.0      # all moves of one plan together
RANGE_S = 25.0             # two looks and two sideways steps
FACE_S = 15.0              # up to four look-turn-look rounds; usually two
# measure, step, measure again, about 3.8 s a round. A typical figure, not a
# worst case: 3-5 rounds under the 1.7 m floor, 6-10 for 1.5 m at the
# 1.0 m floor from 3.5-4 m (0.39.1); the worst is 12 rounds, about 46 s.
# Kept at 30 so that search + face + approach, the plan the thesis flies,
# estimates as it did when the planner benchmarks were run: changing it
# would move that plan over MAX_PLAN_S and change what the validator
# accepts, which is a planner change and not part of this fix.
APPROACH_S = 30.0
# A reasoning model spends tokens thinking before it answers. With the first
# limit of 800, qwen3:8b ran out while still thinking and returned nothing
# for the longer instructions.
DEFAULT_MAX_TOKENS = 4096


class LLMError(Exception):
    """The model could not be reached or did not answer."""


# ───────────────────────────── plan ──────────────────────────────

@dataclass
class Step:
    skill: str
    params: Dict[str, Any]
    why: str = ""

    def to_log(self) -> Dict[str, Any]:
        return {"skill": self.skill, "params": self.params, "why": self.why}


@dataclass
class PlanResult:
    instruction: str
    ok: bool = False
    steps: List[Step] = field(default_factory=list)
    understood: str = ""
    cannot_do: str = ""
    errors: List[str] = field(default_factory=list)
    attempts: int = 0
    latency_s: float = 0.0
    model: str = ""
    raw: List[str] = field(default_factory=list)
    finish: List[Optional[str]] = field(default_factory=list)
    cut_off: bool = False

    @property
    def flyable(self) -> bool:
        return self.ok and bool(self.steps)

    def signature(self) -> str:
        """Same plan, same string: for counting how often a model repeats
        itself on one instruction."""
        if not self.ok:
            return "INVALID"
        if not self.steps:
            return "REFUSED"
        return json.dumps([[s.skill, s.params] for s in self.steps],
                          sort_keys=True)

    def to_log(self) -> Dict[str, Any]:
        return {"instruction": self.instruction, "ok": self.ok,
                "steps": [s.to_log() for s in self.steps],
                "understood": self.understood, "cannot_do": self.cannot_do,
                "errors": self.errors, "attempts": self.attempts,
                "latency_s": round(self.latency_s, 3), "model": self.model,
                "est_duration_s": round(estimate_s(self.steps), 1),
                "cut_off": self.cut_off, "finish": self.finish,
                "raw": self.raw}


def estimate_s(steps: List[Step]) -> float:
    total = 0.0
    for s in steps:
        p = s.params
        if s.skill == "hover":
            total += float(p.get("duration_s", 0)) + float(p.get("settle_s", 0))
        elif s.skill == "rotate_search":
            deg = float(p.get("total_deg") or 360.0)
            rate = float(p.get("rate_dps") or DEFAULT_RATE_DPS)
            leg = float(p.get("leg_deg") or DEFAULT_LEG_DEG)
            total += deg / rate + (deg / leg) * LEG_OVERHEAD_S
        elif s.skill == "face_target":
            total += FACE_S
        elif s.skill == "measure_range":
            total += RANGE_S
        elif s.skill == "approach":
            total += APPROACH_S
        elif s.skill == "move":
            d = math.hypot(float(p.get("forward_m") or 0.0),
                           float(p.get("right_m") or 0.0))
            total += math.ceil(d / 0.25) * 1.2 + 3.0
        else:
            total += CLIMB_S
    return total


# ─────────────────────────── validation ──────────────────────────

def _check_value(name: str, v: Any, spec: Dict[str, Any]) -> Optional[str]:
    t = spec.get("type")
    if t == "number":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return f"{name} must be a number, got {v!r}"
    elif t == "integer":
        if isinstance(v, bool) or not isinstance(v, int):
            return f"{name} must be a whole number, got {v!r}"
    elif t == "string":
        if not isinstance(v, str) or not v.strip():
            return f"{name} must be a non-empty string, got {v!r}"
    if "enum" in spec and v not in spec["enum"]:
        return f"{name} must be one of {spec['enum']}, got {v!r}"
    if t in ("number", "integer"):
        if "minimum" in spec and v < spec["minimum"]:
            return f"{name}={v} is below the minimum {spec['minimum']}"
        if "maximum" in spec and v > spec["maximum"]:
            return f"{name}={v} is above the maximum {spec['maximum']}"
    return None


# Skills that act on a target which must already be in view and near the
# nose. The prompt has always said so; until 0.38 nothing checked it, and on
# 2026-11-17 three German plans went search -> approach, passed validation,
# were approved, and failed in flight within two seconds: a search stops with
# the target 28-49 deg off the nose and approach refuses past 30.
NEEDS_FACED = ("approach", "measure_range")
# Between a face_target and the skill that relies on it, only steps that keep
# the heading and leave the target where face_target put it.
KEEPS_FACING = ("hover", "climb_to", "measure_range", "approach")


def check_order(steps: List[Step]) -> List[str]:
    """
    The ordering rules the prompt states, enforced. Each error names the step
    and says what to insert, so the model can repair the plan.
    """
    errs: List[str] = []
    for i, s in enumerate(steps):
        tc = s.params.get("target_class")
        if s.skill == "face_target":
            if not any(p.skill == "rotate_search"
                       and p.params.get("target_class") == tc
                       for p in steps[:i]):
                errs.append(
                    f"step {i + 1}: face_target needs a rotate_search with "
                    f"target_class {tc!r} before it, to find the {tc} first")
        elif s.skill in NEEDS_FACED:
            ok = False
            for p in reversed(steps[:i]):
                if p.skill == "face_target":
                    ok = p.params.get("target_class") == tc
                    break
                if p.skill not in KEEPS_FACING:
                    break
            if not ok:
                errs.append(
                    f"step {i + 1}: {s.skill} only works on a {tc} that is "
                    f"already roughly ahead, and a search stops with it "
                    f"30-50 deg off the nose; put a face_target with "
                    f"target_class {tc!r} directly before it")
    return errs


def validate_plan(obj: Any, catalogue: Dict[str, Any], alt_max_m: float,
                  classes: Optional[List[str]] = None
                  ) -> Tuple[List[Step], List[str]]:
    """
    Everything the model said, checked. Returns (steps, errors); the plan is
    usable only when errors is empty. Nothing is corrected silently: a value
    out of range is an error for the model to fix, not something to clamp,
    because a clamped plan is no longer the plan the operator was shown.
    """
    errs: List[str] = []
    if not isinstance(obj, dict):
        return [], ["the reply must be one JSON object"]
    raw_steps = obj.get("steps")
    if not isinstance(raw_steps, list):
        return [], ['"steps" must be a list']
    cannot = obj.get("cannot_do") or ""
    if not isinstance(cannot, str):
        errs.append('"cannot_do" must be a string')
        cannot = ""
    if not raw_steps and not cannot.strip():
        errs.append('no steps and no "cannot_do": give one or the other')
    if raw_steps and cannot.strip():
        errs.append('both steps and "cannot_do" were given: if any part of '
                    'the instruction cannot be done, return no steps')
    if len(raw_steps) > MAX_STEPS:
        errs.append(f"{len(raw_steps)} steps; the limit is {MAX_STEPS}")

    steps: List[Step] = []
    for i, rs in enumerate(raw_steps, 1):
        if not isinstance(rs, dict):
            errs.append(f"step {i} must be an object")
            continue
        name = rs.get("skill")
        if name not in catalogue:
            errs.append(f"step {i}: unknown skill {name!r}; the skills are "
                        f"{sorted(catalogue)}")
            continue
        params = rs.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            errs.append(f"step {i}: params must be an object")
            continue
        schema = catalogue[name]["schema"]
        props = schema.get("properties", {})
        for k in params:
            if k not in props:
                errs.append(f"step {i}: {name} has no parameter {k!r}; it "
                            f"has {sorted(props)}")
        for k in schema.get("required", []):
            if k not in params:
                errs.append(f"step {i}: {name} needs {k}")
        clean: Dict[str, Any] = {}
        for k, v in params.items():
            if k not in props:
                continue
            bad = _check_value(k, v, props[k])
            if bad:
                errs.append(f"step {i}: {bad}")
                continue
            clean[k] = v.strip().lower() if isinstance(v, str) else v
        if "altitude_m" in clean and clean["altitude_m"] > alt_max_m:
            errs.append(f"step {i}: altitude_m={clean['altitude_m']} is above "
                        f"this room's limit of {alt_max_m} m")
        tc = clean.get("target_class")
        if tc and classes and tc not in classes:
            errs.append(f"step {i}: the camera is not looking for {tc!r}; "
                        f"it can look for {classes}")
        if name == "move":
            d = math.hypot(float(clean.get("forward_m") or 0.0),
                           float(clean.get("right_m") or 0.0))
            if d < 0.05:
                errs.append(f"step {i}: move needs forward_m and/or right_m")
            elif d > 1.0 + 1e-6:
                errs.append(f"step {i}: {d:.2f} m in one move; the limit is "
                            f"1.0 m along the diagonal")
        why = rs.get("why", "")
        steps.append(Step(name, clean, why if isinstance(why, str) else ""))

    errs.extend(check_order(steps))

    moved = sum(math.hypot(float(s.params.get("forward_m") or 0.0),
                           float(s.params.get("right_m") or 0.0))
                for s in steps if s.skill == "move")
    if moved > MAX_PLAN_MOVE_M + 1e-6:
        errs.append(f"the moves add up to {moved:.1f} m; one plan may move "
                    f"{MAX_PLAN_MOVE_M:.0f} m in total")
    if not errs and steps:
        est = estimate_s(steps)
        if est > MAX_PLAN_S:
            errs.append(f"the plan would take about {est:.0f} s; the limit "
                        f"is {MAX_PLAN_S:.0f} s for one battery")
    return steps, errs


def extract_json(text: str) -> Any:
    """
    The JSON object in a model's reply. Models wrap it: reasoning in
    <think> tags, a code fence, a sentence before or after. The first
    balanced {...} outside any <think> block is taken; anything that does
    not parse is an error, never a guess.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S)   # unclosed opener
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object in the reply")


# ───────────────────────────── prompt ────────────────────────────

def _skill_lines(catalogue: Dict[str, Any], alt_max_m: float) -> str:
    out: List[str] = []
    for name, entry in catalogue.items():
        out.append(f"- {name}: {entry.get('description', '')}")
        schema = entry["schema"]
        req = set(schema.get("required", []))
        for k, spec in schema.get("properties", {}).items():
            bits = [spec.get("type", "")]
            if "enum" in spec:
                bits.append(f"one of {spec['enum']}")
            hi = spec.get("maximum")
            if k == "altitude_m" and hi is not None:
                hi = min(hi, alt_max_m)
            if "minimum" in spec or hi is not None:
                bits.append(f"{spec.get('minimum', '')}..{hi if hi is not None else ''}")
            bits.append("REQUIRED" if k in req else "optional")
            if spec.get("description"):
                bits.append(spec["description"])
            out.append(f"    {k}: {', '.join(b for b in bits if b)}")
    return "\n".join(out)


def _standoff_floor(catalogue: Dict[str, Any]) -> Any:
    """approach's own schema minimum, so the prompt cannot drift from it
    again: until 0.38 the prompt said 1.2 m while the schema said 1.7, and the
    model planned 1.5 m, was refused, and spent a repair round on it."""
    spec = catalogue.get("approach", {}).get("schema", {}).get(
        "properties", {}).get("standoff_m", {})
    return spec.get("minimum", 1.0)


def system_prompt(catalogue: Dict[str, Any], alt_max_m: float,
                  classes: Optional[List[str]] = None) -> str:
    cam = (f"The camera can look for: {', '.join(classes)}. Nothing else."
           if classes else
           "The camera can look for an object named by one short English "
           "noun, for example person or chair.")
    return f"""You plan flights for a small indoor quadcopter. You do not fly it. You choose steps from a fixed list of skills; a separate flight program runs them and enforces every safety limit.

Before your first step the drone takes off by itself to about 0.8 m. After your last step it lands by itself. Never add steps for takeoff or landing.

SKILLS
{_skill_lines(catalogue, alt_max_m)}

RULES
- Use only the skills and parameters listed. Leave out optional parameters you do not need.
- At most {MAX_STEPS} steps. The whole plan must take under {MAX_PLAN_S:.0f} seconds. A full 360 degree turn takes about 40 seconds at the default rate_dps of {DEFAULT_RATE_DPS:.0f}, and about 60 seconds at 10; slower than that does not fit.
- Maximum altitude is {alt_max_m} m.
- {cam}
- rotate_search with target_class turns until it sees that object and then stops. If it finishes the turn without seeing it, the mission ends there and the drone lands; later steps do not run.
- measure_range says how far away an object is. It only works on an object that is in view and roughly ahead, so put it after a face_target for the same object. It steps sideways and back by itself.
- face_target only works on an object that is in view. Put it directly after a rotate_search with the same target_class.
- approach closes on an object until it is standoff_m away, measuring the distance again between every step, and stops no closer than {_standoff_floor(catalogue)} m. It only works on an object that is in view and roughly ahead, so put it after a face_target for the same object. It does not need a measure_range first: it measures for itself. Prefer it over move for getting closer to something.
- move goes at most 1 m per step and all moves of a plan together at most {MAX_PLAN_MOVE_M:.0f} m. Distances are relative to the nose. The drone does not know where rooms, doors or furniture are, and it cannot judge a distance by eye: to move a measured distance toward something, measure_range it first.
- The drone cannot go to a named place or room, cannot follow something that is moving, cannot carry anything and cannot speak. It can approach an object it can see, with approach. It can hold position, turn, change height, move a short measured distance, approach something it can see, and look.
- If any part of the instruction needs something the skills cannot do, return no steps and say in "cannot_do" what cannot be done. Do not do only part of an instruction.

REPLY
Reply with one JSON object and nothing else:
{{"understood": "<one sentence: what you were asked>", "steps": [{{"skill": "<name>", "params": {{}}, "why": "<a few words>"}}], "cannot_do": ""}}

EXAMPLES
Instruction: look around for a person and stay there for five seconds when you see one
{{"understood": "Search for a person, then hold for 5 s.", "steps": [{{"skill": "rotate_search", "params": {{"total_deg": 360, "target_class": "person"}}, "why": "turn until a person is seen"}}, {{"skill": "hover", "params": {{"duration_s": 5}}, "why": "stay after finding"}}], "cannot_do": ""}}
Instruction: fly to the kitchen
{{"understood": "Go to another room.", "steps": [], "cannot_do": "The drone cannot go to a named place; it can only move a short measured distance."}}"""


# ───────────────────────────── models ────────────────────────────

class OpenAICompatLLM:
    """POST {base_url}/chat/completions. base_url is everything up to and
    including /v1 (or /api for Open WebUI)."""

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout_s: float = 120.0, json_mode: bool = False,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 no_think: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key = model, api_key
        self.timeout_s, self.json_mode = timeout_s, json_mode
        self.max_tokens, self.no_think = max_tokens, no_think
        self.last_finish: Optional[str] = None
        self.last_usage: Dict[str, Any] = {}

    def _request(self, path: str, body: Optional[Dict[str, Any]]) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            hint = {401: " (the API key is missing or wrong)",
                    403: " (the API key is not allowed to do this)",
                    404: " (wrong address or model name; the address must "
                         "end where /chat/completions begins)"}.get(e.code, "")
            raise LLMError(f"{url} answered HTTP {e.code}{hint}: {detail}")
        except urllib.error.URLError as e:
            raise LLMError(f"could not reach {url}: {e.reason}. Is the VPN "
                           f"up, and is the address right?")
        except (TimeoutError, OSError) as e:
            raise LLMError(f"{url} did not answer within "
                           f"{self.timeout_s:g} s ({e!r})")
        except json.JSONDecodeError:
            raise LLMError(f"{url} answered, but not with JSON: this is "
                           f"probably a web page, not the API address")

    def list_models(self) -> List[str]:
        got = self._request("/models", None)
        rows = got.get("data", got.get("models", [])) if isinstance(
            got, dict) else got
        return [r.get("id") or r.get("name") or str(r) if isinstance(r, dict)
                else str(r) for r in rows]

    def chat(self, messages: List[Dict[str, str]]) -> str:
        body: Dict[str, Any] = {"model": self.model, "messages": messages,
                                "temperature": 0,
                                "max_tokens": self.max_tokens}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.no_think:
            # Ollama: "none" turns thinking off for models that can (qwen3).
            body["reasoning_effort"] = "none"
        got = self._request("/chat/completions", body)
        try:
            self.last_finish = got["choices"][0].get("finish_reason")
            self.last_usage = got.get("usage") or {}
            return got["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected reply shape: {str(got)[:300]}")


class ScriptedLLM:
    """Replies from a list, for tests and for trying the tools offline."""

    model = "scripted"

    def __init__(self, replies: List[str]) -> None:
        self.replies, self.calls = list(replies), []

    def chat(self, messages: List[Dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.replies:
            raise LLMError("the scripted model has no more replies")
        return self.replies.pop(0)


# ──────────────────────────── planner ────────────────────────────

class Planner:
    def __init__(self, llm: Any, catalogue: Dict[str, Any], alt_max_m: float,
                 classes: Optional[List[str]] = None) -> None:
        self.llm, self.catalogue = llm, catalogue
        self.alt_max_m, self.classes = alt_max_m, classes
        self.system = system_prompt(catalogue, alt_max_m, classes)

    def plan(self, instruction: str) -> PlanResult:
        res = PlanResult(instruction=instruction,
                         model=getattr(self.llm, "model", "?"))
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": f"Instruction: {instruction}"}]
        t0 = time.monotonic()
        for _ in range(1 + MAX_REPAIRS):
            res.attempts += 1
            reply = self.llm.chat(messages)      # LLMError goes to the caller
            res.raw.append(reply)
            finish = getattr(self.llm, "last_finish", None)
            res.finish.append(finish)
            try:
                obj = extract_json(reply)
                steps, errs = validate_plan(obj, self.catalogue,
                                            self.alt_max_m, self.classes)
            except ValueError as e:
                obj, steps, errs = None, [], [str(e)]
            if errs and finish == "length":
                # Asking again with the same limit cuts it off again, and
                # costs the same wait. Stop and say what to change.
                res.cut_off = True
                res.errors = [
                    f"the model's reply was cut off at the token limit "
                    f"({getattr(self.llm, 'max_tokens', '?')}) "
                    + ("while it was still thinking, before it wrote any "
                       "answer" if not reply.strip() else
                       "in the middle of its answer")
                    + ". Raise the limit (--max-tokens) or turn thinking "
                      "off (--no-think)."]
                break
            res.errors = errs
            if not errs:
                res.ok, res.steps = True, steps
                res.understood = str(obj.get("understood", ""))
                res.cannot_do = str(obj.get("cannot_do") or "").strip()
                break
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content":
                          "That reply cannot be used:\n- " + "\n- ".join(errs)
                          + "\nReply again with one corrected JSON object "
                            "and nothing else."}]
        res.latency_s = time.monotonic() - t0
        return res


def format_plan(res: PlanResult) -> str:
    """The plan as the operator reads it before deciding to fly."""
    lines = [f"  instruction   {res.instruction}",
             f"  model         {res.model}, {res.latency_s:.1f} s, "
             f"{res.attempts} attempt(s)"]
    if res.understood:
        lines.append(f"  understood    {res.understood}")
    if not res.ok:
        lines.append("  NO PLAN: the model's reply did not pass the checks")
        lines += [f"    - {e}" for e in res.errors]
    elif not res.steps:
        lines.append(f"  REFUSED       {res.cannot_do}")
    else:
        lines.append("  plan          takeoff (automatic)")
        for i, s in enumerate(res.steps, 1):
            ps = ", ".join(f"{k}={v}" for k, v in s.params.items())
            lines.append(f"                {i}. {s.skill}({ps})"
                         + (f"   - {s.why}" if s.why else ""))
        lines.append("                land (automatic)")
        lines.append(f"  about         {estimate_s(res.steps):.0f} s in the air")
    return "\n".join(lines)
