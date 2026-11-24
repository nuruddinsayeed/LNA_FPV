#!/usr/bin/env python3
"""
The planner on the ground. Sends an instruction to the language model and
shows the plan it would fly. Nothing here opens a MAVLink connection.

    # 1. is the model reachable, and what is it called?
    python tools/plan_mission.py models --llm-url http://HOST:PORT/v1

    # 2. one instruction
    python tools/plan_mission.py plan "look around for a person" \\
        --llm-url http://HOST:PORT/v1 --llm-model NAME

    # 3. many instructions, each several times: the numbers for the thesis
    python tools/plan_mission.py bench instructions.txt --repeat 5 ...

The address is everything before /chat/completions: usually it ends in /v1
(vLLM, Ollama, LiteLLM), or in /api for Open WebUI. Set it once instead of
typing it each time:

    export DRONE_LLM_URL=http://HOST:PORT/v1
    export DRONE_LLM_MODEL=NAME
    export DRONE_LLM_KEY=...        # only if the server wants one

`bench` reads one instruction per line. A line starting with `!` is one the
drone must REFUSE (it asks for something no skill can do); `#` is a comment.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_agent import __version__
from drone_agent.config import Config
from drone_agent.planner import (DEFAULT_MAX_TOKENS, LLMError,
                                 OpenAICompatLLM, Planner, format_plan)
from drone_agent.grade import Case, grade, load_cases
from drone_agent.planner import Step
from drone_agent.skills import catalogue


def make_llm(args: argparse.Namespace, need_model: bool = True) -> Any:
    if not args.llm_url:
        sys.exit("No model address. Give --llm-url http://HOST:PORT/v1 or "
                 "set DRONE_LLM_URL.")
    if need_model and not args.llm_model:
        sys.exit("No model name. Give --llm-model NAME or set "
                 "DRONE_LLM_MODEL; `models` lists what the server has.")
    return OpenAICompatLLM(args.llm_url, args.llm_model or "",
                           api_key=os.environ.get("DRONE_LLM_KEY", ""),
                           timeout_s=args.timeout, json_mode=args.json_mode,
                           max_tokens=args.max_tokens, no_think=args.no_think)


def make_planner(args: argparse.Namespace) -> Planner:
    classes = ([c.strip().lower() for c in args.classes.split(",")
                if c.strip()] if args.classes else None)
    alt = args.alt_max or Config().envelope.alt_max_m
    return Planner(make_llm(args), catalogue(), alt, classes)


def run_models(args: argparse.Namespace) -> int:
    llm = make_llm(args, need_model=False)
    try:
        names = llm.list_models()
    except LLMError as e:
        print(f"  {e}")
        return 1
    print(f"  {args.llm_url} is reachable and offers {len(names)} model(s):")
    for n in names:
        print(f"    {n}")
    return 0


def run_plan(args: argparse.Namespace) -> int:
    planner = make_planner(args)
    if args.show_prompt:
        print(planner.system, "\n")
    sigs: List[str] = []
    for k in range(args.repeat):
        try:
            res = planner.plan(args.text)
        except LLMError as e:
            print(f"  {e}")
            return 1
        sigs.append(res.signature())
        print(format_plan(res))
        if args.show_raw or not res.ok:
            for i, r in enumerate(res.raw, 1):
                print(f"  --- reply {i} ---\n{r}")
        print()
    if args.repeat > 1:
        top = Counter(sigs).most_common(1)[0][1]
        print(f"  {args.repeat} runs: {len(set(sigs))} different result(s), "
              f"the most common one {top} time(s)")
    print("  Nothing was sent to any vehicle.")
    return 0


def run_bench(args: argparse.Namespace) -> int:
    lines = [l.strip() for l in Path(args.text).read_text().splitlines()]
    cases = [(l.lstrip("! ").strip(), l.startswith("!")) for l in lines
             if l and not l.startswith("#")]
    if not cases:
        sys.exit(f"{args.text} has no instructions in it")
    planner = make_planner(args)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir) / f"plans-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    lat: List[float] = []
    n = valid = right = first_try = stable = cut = 0
    print(f"  {len(cases)} instruction(s) x {args.repeat}, model "
          f"{args.llm_model}\n")
    print(f"  {'valid':>5} {'right':>5} {'same':>5} {'s':>5}  instruction")
    with open(out, "w") as fh:
        fh.write(json.dumps({"type": "meta", "version": __version__,
                             "model": args.llm_model, "url": args.llm_url,
                             "repeat": args.repeat, "classes": args.classes,
                             "max_tokens": args.max_tokens,
                             "no_think": args.no_think,
                             "system_prompt": planner.system}) + "\n")
        for text, must_refuse in cases:
            sigs, ok_n, right_n, ls = [], 0, 0, []
            for k in range(args.repeat):
                try:
                    res = planner.plan(text)
                except LLMError as e:
                    print(f"  {e}")
                    return 1
                rec = res.to_log()
                rec.update({"type": "plan", "run": k,
                            "must_refuse": must_refuse})
                fh.write(json.dumps(rec) + "\n")
                sigs.append(res.signature())
                ok_n += res.ok
                # "right" is only what can be judged without a human: a
                # request the drone cannot do was refused, one it can do
                # produced a flyable plan. Whether that plan is the RIGHT
                # plan is for the operator to read in the log.
                right_n += res.ok and (res.flyable != must_refuse)
                first_try += res.ok and res.attempts == 1
                cut += res.cut_off
                ls.append(res.latency_s)
            same = Counter(sigs).most_common(1)[0][1]
            n += args.repeat
            valid += ok_n
            right += right_n
            stable += same
            lat += ls
            print(f"  {ok_n:>3}/{args.repeat:<1} {right_n:>3}/{args.repeat:<1} "
                  f"{same:>3}/{args.repeat:<1} {st.median(ls):>5.1f}  "
                  f"{'! ' if must_refuse else ''}{text[:70]}")
    lat.sort()
    print(f"\n  valid plan or refusal      {valid}/{n} "
          f"({100.0 * valid / n:.0f}%), {first_try} on the first attempt")
    if cut:
        print(f"  cut off at the token limit   {cut}/{n}: raise --max-tokens "
              f"or use --no-think")
    print(f"  refused what it must, planned what it can   {right}/{n} "
          f"({100.0 * right / n:.0f}%)")
    print(f"  same result on repeats     {stable}/{n} "
          f"({100.0 * stable / n:.0f}%)")
    print(f"  time per plan              median {st.median(lat):.1f} s, "
          f"worst {lat[-1]:.1f} s")
    print(f"\n  log   {out}")
    print("  Nothing was sent to any vehicle.")
    return 0


def run_regrade(args: argparse.Namespace) -> int:
    """
    Re-score a `grade` run that already happened, against the rubrics as
    they stand now. No model is called and nothing is flown.

    This exists because the first run of the evaluation set found four
    rubrics wrong rather than four model failures -- the model had clamped
    to a skill's own schema limit and been marked down for it. Correcting a
    rubric must not cost a fresh 225-plan run on the aircraft's operator,
    and a score that moved because the rubric moved has to be reproducible
    from the same replies. The plans are in the log; only the marking
    changes.
    """
    cases = {c.id or c.text: c for c in load_cases(args.text)}
    logs = [args.text] if not args.logs else args.logs
    if not args.logs:
        print("  give the eval set and then the graded-*.jsonl log(s)")
        return 2
    rc = 0
    for path in logs:
        # One report per log. Two runs merged under one heading would be
        # labelled with whichever meta line came last, which is how a
        # thinking run and a no-think run get reported as one number.
        rc |= _regrade_one(path, cases)
    return rc


def _regrade_one(path: str, cases: Dict[str, Any]) -> int:
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    changed: List[Tuple[str, bool, bool]] = []
    for line in open(path):
        rec = json.loads(line)
        if rec.get("type") == "meta":
            meta = rec
            continue
        if rec.get("type") != "graded":
            continue
        case = cases.get(rec.get("case") or rec.get("instruction"))
        if case is None:
            print(f"  no rubric any more for {rec.get('case')!r}; skipping")
            continue
        steps = [Step(skill=d["skill"], params=d.get("params") or {},
                      why=d.get("why", ""))
                 for d in (rec.get("steps") or [])]
        passed, why = grade(case, steps, bool(rec.get("ok")),
                            rec.get("cannot_do") or "")
        if passed != rec.get("passed"):
            changed.append((case.id, bool(rec.get("passed")), passed))
        rows.append({"case": case.id, "category": case.category,
                     "passed": passed, "was": bool(rec.get("passed")),
                     "ok": bool(rec.get("ok")),
                     "latency_s": rec.get("latency_s") or 0.0,
                     "refuse": case.refuse, "why_not": why})
    if not rows:
        print(f"\n  {Path(path).name}: nothing to re-score")
        return 1
    tag = "no thinking" if meta.get("no_think") else "thinking"
    print(f"\n{Path(path).name}")
    print(f"  {meta.get('model')} ({tag}), {len(rows)} plans re-scored "
          f"against the rubrics as they now stand")
    was = sum(r["was"] for r in rows)
    now = sum(r["passed"] for r in rows)
    print(f"  as marked at the time   {was}/{len(rows)} "
          f"({100.0 * was / len(rows):.1f}%)")
    print(f"  as marked now           {now}/{len(rows)} "
          f"({100.0 * now / len(rows):.1f}%)")
    able = [r for r in rows if not r["refuse"]]
    deny = [r for r in rows if r["refuse"]]
    if able and deny:
        print(f"    of the {len(able)} it can do      "
              f"{sum(r['passed'] for r in able)}/{len(able)}")
        print(f"    of the {len(deny)} it must refuse "
              f"{sum(r['passed'] for r in deny)}/{len(deny)}")
    lat = sorted(r["latency_s"] for r in rows)
    print(f"  latency median {st.median(lat):.1f} s, p90 "
          f"{lat[int(0.9 * len(lat))]:.1f} s, worst {lat[-1]:.1f} s")
    if changed:
        print(f"\n  {len(changed)} marks moved:")
        for (cid, before, after), k in Counter(changed).most_common():
            print(f"    {k:3d}x {cid:18} "
                  f"{'fail -> pass' if after else 'pass -> fail'}")
    cats = sorted({r["category"] for r in rows})
    width = max(len(c) for c in cats)
    print()
    for c in cats:
        got = [r for r in rows if r["category"] == c]
        print(f"    {c:{width}}  {sum(r['passed'] for r in got):>3}/"
              f"{len(got):<3} "
              f"({100.0 * sum(r['passed'] for r in got) / len(got):>3.0f}%)")
    bad = [r for r in rows if not r["passed"]]
    if bad:
        print(f"\n  what is still wrong:")
        for (cid, w), k in Counter(
                (r["case"], r["why_not"][0] if r["why_not"] else "?")
                for r in bad).most_common(12):
            print(f"    {k}x {cid:18} {w[:72]}")
    else:
        print("\n  nothing failed. A set that nothing fails has stopped "
              "measuring: add harder cases.")
    return 0


def run_grade(args: argparse.Namespace) -> int:
    """
    The graded evaluation: not "was it valid JSON" but "did it plan the
    mission the instruction asked for", checked against a rubric in code so
    the number is reproducible rather than marked by hand.
    """
    cases = load_cases(args.text)
    planner = make_planner(args)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir) / f"graded-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    rows: List[Dict[str, Any]] = []
    print(f"  {len(cases)} instruction(s) x {args.repeat}, model "
          f"{args.llm_model}"
          + (", no thinking" if args.no_think else "") + "\n")
    print(f"  {'valid':>5} {'right':>5} {'same':>5} {'s':>5}  instruction")
    with open(out, "w") as fh:
        fh.write(json.dumps({
            "type": "meta", "version": __version__, "model": args.llm_model,
            "url": args.llm_url, "repeat": args.repeat,
            "classes": args.classes, "max_tokens": args.max_tokens,
            "no_think": args.no_think, "eval_set": str(args.text),
            "cases": len(cases), "system_prompt": planner.system}) + "\n")
        for case in cases:
            sigs, ok_n, pass_n, ls, whys = [], 0, 0, [], []
            for k in range(args.repeat):
                try:
                    res = planner.plan(case.text)
                except LLMError as e:
                    print(f"  {e}")
                    return 1
                passed, why = grade(case, res.steps, res.ok, res.cannot_do)
                rec = res.to_log()
                rec.update({"type": "graded", "run": k, "case": case.id,
                            "category": case.category,
                            "must_refuse": case.refuse,
                            "passed": passed, "why_not": why})
                fh.write(json.dumps(rec) + "\n")
                sigs.append(res.signature())
                ok_n += res.ok
                pass_n += passed
                ls.append(res.latency_s)
                if not passed:
                    whys += why
                rows.append({"case": case.id, "category": case.category,
                             "passed": passed, "ok": res.ok,
                             "latency_s": res.latency_s,
                             "cut_off": res.cut_off,
                             "attempts": res.attempts,
                             "refuse": case.refuse})
            same = Counter(sigs).most_common(1)[0][1]
            mark = "  " if pass_n == args.repeat else ("!!" if pass_n == 0
                                                       else " ~")
            print(f"{mark}{ok_n:>3}/{args.repeat:<1} {pass_n:>3}/"
                  f"{args.repeat:<1} {same:>3}/{args.repeat:<1} "
                  f"{st.median(ls):>5.1f}  "
                  f"{'! ' if case.refuse else ''}{case.text[:64]}")
            for w in Counter(whys).most_common(2):
                print(f"            {w[0][:88]}")

    n = len(rows)
    valid = sum(r["ok"] for r in rows)
    good = sum(r["passed"] for r in rows)
    lat = sorted(r["latency_s"] for r in rows)
    print(f"\n  valid plan or refusal          {valid}/{n} "
          f"({100.0 * valid / n:.0f}%)")
    print(f"  planned what was asked         {good}/{n} "
          f"({100.0 * good / n:.0f}%)   <- the number that matters")
    cats = sorted({r["category"] for r in rows})
    width = max(len(c) for c in cats)
    print()
    for c in cats:
        got = [r for r in rows if r["category"] == c]
        p = sum(r["passed"] for r in got)
        print(f"    {c:{width}}  {p:>3}/{len(got):<3} "
              f"({100.0 * p / len(got):>3.0f}%)")
    able = [r for r in rows if not r["refuse"]]
    deny = [r for r in rows if r["refuse"]]
    if able and deny:
        print(f"\n    of the {len(able)} it can do, it planned "
              f"{sum(r['passed'] for r in able)} correctly; of the "
              f"{len(deny)} it cannot,")
        print(f"    it refused {sum(r['passed'] for r in deny)}.")
    cut = sum(r["cut_off"] for r in rows)
    if cut:
        print(f"\n  cut off at the token limit     {cut}/{n}")
    print(f"  time per plan                  median {st.median(lat):.1f} s, "
          f"worst {lat[-1]:.1f} s")
    print(f"\n  log   {out}")
    print("  Nothing was sent to any vehicle.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode",
                    choices=["models", "plan", "bench", "grade", "regrade"])
    ap.add_argument("text", nargs="?",
                    help="plan: the instruction. bench/grade/regrade: the "
                         "file of them")
    ap.add_argument("logs", nargs="*",
                    help="regrade: graded-*.jsonl logs to re-score against "
                         "the rubrics as they now stand. No model is called")
    ap.add_argument("--llm-url", default=os.environ.get("DRONE_LLM_URL"))
    ap.add_argument("--llm-model", default=os.environ.get("DRONE_LLM_MODEL"))
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="limit on the model's reply, thinking included")
    ap.add_argument("--no-think", action="store_true",
                    help="ask the server to turn the model's thinking off "
                         "(Ollama: qwen3 and similar). Much faster; compare "
                         "the bench with and without it")
    ap.add_argument("--json-mode", action="store_true",
                    help="ask the server to force JSON output (not every "
                         "server supports it)")
    ap.add_argument("--classes", help="what the camera will look for, e.g. "
                                      "person,chair")
    ap.add_argument("--alt-max", type=float)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--show-raw", action="store_true")
    ap.add_argument("--out-dir", default="logs")
    ap.add_argument("--version", action="version",
                    version=f"drone_agent {__version__}")
    args = ap.parse_args()
    print(f"drone_agent {__version__}")
    if args.mode == "models":
        return run_models(args)
    if not args.text:
        ap.error(f"{args.mode} needs "
                 f"{'an instruction' if args.mode == 'plan' else 'a file'}")
    if args.mode == "plan":
        return run_plan(args)
    if args.mode == "regrade":
        return run_regrade(args)
    return run_grade(args) if args.mode == "grade" else run_bench(args)


if __name__ == "__main__":
    sys.exit(main())
