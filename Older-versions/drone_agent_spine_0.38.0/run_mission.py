#!/usr/bin/env python3
"""
Run one skill (or a short sequence) and log everything.

    python run_mission.py --list
    python run_mission.py hover --duration 10
    python run_mission.py rotate_search --total-deg 360 --leg-deg 45
    python run_mission.py rotate_search --sweep 360          # the smoke test

Default launch is GUIDED + MAV_CMD_NAV_TAKEOFF: the script climbs by itself.
Only pass --handover when GUIDED refuses to pre-arm ("Need Position
Estimate"); that path needs a pilot on the throttle, because ArduPilot
auto-disarms after DISARM_DELAY if the throttle stays at minimum.

    python run_mission.py hover --duration 10 --takeoff-height 0.8
    python run_mission.py hover --duration 10 --handover

Bench tests for the safety layer, props OFF:
    python run_mission.py hover --duration 10 --inject-bad-setpoint
    python run_mission.py hover --duration 20 --inject-estop-at 8

Every abort ends in a landing. There is no path from an abort to a disarm.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

from drone_agent import __version__
from drone_agent.config import Config
from drone_agent.executor import Executor
from drone_agent.interfaces.vehicle import (FlightError, MavlinkVehicle,
                                            PilotTakeover)
from drone_agent.perception.geometry import CameraIntrinsics
from drone_agent.recording.recorder import FlightRecorder
from drone_agent.faults import FAULTS, BadSetpointInjector
from drone_agent.skills import SKILL_REGISTRY, build, catalogue

_EXECUTOR: Optional[Executor] = None
_RECORDER: Optional[FlightRecorder] = None
_PERCEPTION: Optional[Any] = None


# ───────────────────────────── signals ───────────────────────────

def _shutdown(sig: int, frame: Any) -> None:
    print("\nInterrupted.")
    if _EXECUTOR is not None and _EXECUTOR.vehicle.is_armed():
        _EXECUTOR.land("operator interrupt (Ctrl-C)")
    if _RECORDER is not None:
        _RECORDER.event("interrupt", {"signal": sig})
        _RECORDER.close()
    sys.exit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ────────────────────────────── cli ──────────────────────────────

def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.connection:
        cfg.link.connection = args.connection
    if args.airframe is not None:
        cfg.airframe = args.airframe
    if args.alt_max is not None:
        cfg.envelope.alt_max_m = args.alt_max
    if args.fence is not None:
        cfg.envelope.fence_radius_m = args.fence
    if args.max_step is not None:
        cfg.envelope.max_step_m = args.max_step
    if args.forward_rf:
        cfg.forward_rf.enabled = True
    if args.no_preflight:
        cfg.preflight.enabled = False
    cfg.preflight.handover_enabled = bool(args.handover)
    if args.no_confirm:
        cfg.preflight.handover_confirm = False
    if args.takeoff_height is not None:
        cfg.skills.takeoff_height_m = args.takeoff_height
    return cfg


def skill_params(args: argparse.Namespace) -> Dict[str, Any]:
    p: Dict[str, Any] = {}
    if args.skill == "hover":
        p["duration_s"] = args.duration
        p["settle_s"] = args.settle
    elif args.skill == "rotate_search":
        p["total_deg"] = args.sweep if args.sweep else args.total_deg
        p["rate_dps"] = args.rate_dps
        p["leg_deg"] = args.leg_deg
        p["direction"] = args.direction
        if args.target_class:
            p["target_class"] = args.target_class
    elif args.skill == "climb_to":
        p["altitude_m"] = args.altitude
    elif args.skill == "face_target":
        p["target_class"] = args.target_class
    elif args.skill == "measure_range":
        p["target_class"] = args.target_class
        p["baseline_m"] = args.baseline
    elif args.skill == "move":
        p["forward_m"] = args.forward
        p["right_m"] = args.right
    return {k: v for k, v in p.items() if v is not None}


def main() -> int:
    global _EXECUTOR, _RECORDER

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("skill", nargs="?", help="skill to run")
    ap.add_argument("--list", action="store_true",
                    help="list skills and their schemas, then exit")

    ap.add_argument("--connection", help="override the MAVLink connection")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--label", default="", help="label for this flight")
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--handover", action="store_true",
                    help="arm in ALT_HOLD, let the PILOT climb, then hand "
                         "over to GUIDED. Only needed when GUIDED refuses to "
                         "pre-arm ('Need Position Estimate'). Requires a "
                         "pilot on the throttle: ArduPilot auto-disarms after "
                         "DISARM_DELAY if the throttle stays at minimum.")
    ap.add_argument("--no-confirm", action="store_true",
                    help="skip the Enter prompt before arming in --handover")
    ap.add_argument("--auto-takeoff", action="store_true",
                    help=argparse.SUPPRESS)      # default; kept for muscle memory
    ap.add_argument("--takeoff-height", type=float,
                    help="auto-takeoff target altitude, m")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--hold-after", type=float, default=0.0,
                    help="hover this long after the skill before landing")
    ap.add_argument("--no-land", action="store_true",
                    help="leave the aircraft airborne (pilot lands it)")

    ap.add_argument("--airframe", default=None,
                    help="WHICH AIRCRAFT this flight is on, e.g. "
                         "'2.2in-cinewhoop' or '3.5in-cinewhoop-mtf15'. "
                         "Recorded in the log's meta. Two airframes share "
                         "this stack and their hover throttle, climb "
                         "authority and sensor count all differ, so a "
                         "figure measured on one does not transfer.")
    ap.add_argument("--alt-max", type=float, help="envelope ceiling, m")
    ap.add_argument("--fence", type=float, help="geofence radius, m")
    ap.add_argument("--max-step", type=float,
                    help="largest single setpoint jump, m — narrow this "
                         "before flying a clamp_step test indoors")
    ap.add_argument("--forward-rf", action="store_true",
                    help="enable the forward MT-15")

    ap.add_argument("--duration", type=float, help="hover duration, s")
    ap.add_argument("--settle", type=float, default=0.0)
    ap.add_argument("--altitude", type=float, help="climb_to target, m")
    ap.add_argument("--sweep", type=float, help="rotate_search total, deg")
    ap.add_argument("--total-deg", type=float)
    ap.add_argument("--rate-dps", type=float)
    ap.add_argument("--leg-deg", type=float)
    ap.add_argument("--direction", type=int, choices=[-1, 1])
    ap.add_argument("--target-class")
    ap.add_argument("--target-height", type=float,
                    help="how tall the target really is, in metres. The "
                         "head-height range is proportional to this MINUS "
                         "the camera's height, so with the camera at 1.1 m "
                         "the 0.11 m spread on 'a person' is a fifth of the "
                         "answer and a tape-measured height is a few "
                         "centimetres of it. Give it if you have it.")

    ap.add_argument("--perception", choices=["none", "yoloe"], default="none",
                    help="attach a detector; world.objects is then populated "
                         "from its latest FRESH snapshot and [] when stale")
    ap.add_argument("--camera", default="0",
                    help="capture source index or path")
    ap.add_argument("--model", default="yoloe-11s-seg.pt",
                    help="YOLOE weights. The small model ran at 20 scenes "
                         "per second on the ground-station Mac (CPU, 43 ms "
                         "an inference); larger ones are slower")
    ap.add_argument("--device", help="cpu, mps, ...; default lets the "
                                     "library choose")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", help="comma-separated open-vocabulary class "
                                      "list, e.g. person,drone,chair")
    ap.add_argument("--intrinsics", help="camera calibration JSON; without "
                                         "it an FOV is ASSUMED and bearings "
                                         "are wrong toward the frame edges")
    ap.add_argument("--hfov", type=float, default=150.0,
                    help="fallback horizontal FOV when no --intrinsics")

    ap.add_argument("--inject", choices=sorted(FAULTS),
                    help="inject a bad setpoint mid-skill. Test this in "
                         "replay or SITL first: an injection only fires once "
                         "airborne, so it cannot be bench-tested with the "
                         "props off (the takeoff check times out and the "
                         "skill never starts).")
    ap.add_argument("--inject-at-tick", type=int, default=20)
    ap.add_argument("--inject-repeat", type=int, default=1,
                    help="fire on this many consecutive ticks")
    ap.add_argument("--inject-bad-setpoint", action="store_true",
                    help=argparse.SUPPRESS)      # = --inject altitude
    ap.add_argument("--inject-bad-step", action="store_true",
                    help=argparse.SUPPRESS)      # = --inject step
    ap.add_argument("--inject-estop-at", type=float,
                    help="bench test: latch the e-stop after N seconds")

    ap.add_argument("--baseline", type=float, default=None,
                    help="measure_range: how far to step sideways (m)")
    ap.add_argument("--forward", type=float, default=0.0,
                    help="move: metres toward the nose (negative: back)")
    ap.add_argument("--right", type=float, default=0.0,
                    help="move: metres to the right (negative: left)")
    ap.add_argument("--then-measure", action="store_true",
                    help="with --then-face: also measure the range to the "
                         "target by stepping sideways")
    ap.add_argument("--then-approach", action="store_true",
                    help="with --then-face: close on the target until it is "
                         "--standoff away, measuring the range again between "
                         "every step. Needs a camera tilt in the "
                         "calibration: it refuses to act on a size-prior "
                         "range, which read 25-31%% long against a tape")
    ap.add_argument("--standoff", type=float, default=1.5,
                    help="approach: where to stop, in metres from the "
                         "target. Never closer than 1.2 m")
    ap.add_argument("--then-face", action="store_true",
                    help="rotate_search with --target-class: when the "
                         "target is found, turn to face it (face_target) "
                         "before landing")
    ap.add_argument("--instruction",
                    help="plain-language instruction. A language model turns "
                         "it into a list of skills ON THE GROUND; the plan "
                         "is checked, shown, and flown only after you "
                         "confirm. Use instead of naming a skill.")
    ap.add_argument("--llm-url", default=os.environ.get("DRONE_LLM_URL"),
                    help="model address up to /v1 (or DRONE_LLM_URL)")
    ap.add_argument("--llm-model", default=os.environ.get("DRONE_LLM_MODEL"))
    ap.add_argument("--llm-timeout", type=float, default=120.0)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="limit on the model's reply, thinking included")
    ap.add_argument("--no-think", action="store_true",
                    help="ask the server to turn the model's thinking off")
    ap.add_argument("--json-mode", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="fly the plan without asking (unattended tests)")

    ap.add_argument("--version", action="version",
                    version=f"drone_agent {__version__}")
    args = ap.parse_args()

    if args.list:
        print(json.dumps(catalogue(), indent=2))
        return 0
    if args.instruction:
        if args.skill:
            ap.error("give a skill OR --instruction, not both")
        if not args.llm_url or not args.llm_model:
            ap.error("--instruction needs --llm-url and --llm-model (or "
                     "DRONE_LLM_URL and DRONE_LLM_MODEL)")
        args.skill = "plan"
    elif not args.skill:
        ap.error("a skill is required (or --instruction, or --list)")
    elif args.skill not in SKILL_REGISTRY:
        ap.error(f"unknown skill {args.skill!r}; "
                 f"have {sorted(SKILL_REGISTRY)}")
    if args.skill == "move":
        import math
        d = math.hypot(args.forward, args.right)
        if not 0.05 <= d <= 1.0:
            ap.error(f"move needs --forward and/or --right adding up to "
                     f"between 0.05 and 1.0 m (got {d:.2f} m). Nothing was "
                     f"started.")

    cfg = build_config(args)
    verbose = not args.quiet

    recorder = FlightRecorder(log_dir=args.log_dir,
                              session_label=args.label or args.skill)
    _RECORDER = recorder
    # The calibration in force goes in the log. Not recording it cost a
    # whole evening on 2026-11-17: four flights read the range 39% short,
    # and whether the stored camera tilt had been re-measured between
    # sessions could not be established from the logs at all.
    cal: Dict[str, Any] = {"intrinsics_path": args.intrinsics}
    try:
        _i = CameraIntrinsics.load(args.intrinsics) if args.intrinsics \
            else None
        if _i is not None:
            cal.update({"model": _i.model, "fx": _i.fx, "fy": _i.fy,
                        "cx": _i.cx, "cy": _i.cy,
                        "tilt_up_deg": _i.tilt_up_deg,
                        "width_px": _i.width_px, "height_px": _i.height_px,
                        "note": _i.note})
    except Exception as e:                       # noqa: BLE001
        cal["error"] = f"{type(e).__name__}: {e}"
    recorder.meta(cfg.to_dict(), label=args.label,
                  extra={"argv": sys.argv, "skill": args.skill,
                         "skill_params": skill_params(args),
                         "instruction": args.instruction,
                         "target_height_m": getattr(args, "target_height",
                                                    None),
                         "calibration": cal})
    print(f"drone_agent {__version__}  ->  {recorder.path}")

    # The plan is made HERE, on the ground, before the camera and before
    # the vehicle. The model is asked once; nothing asks it again in the air.
    steps: List[tuple] = [(args.skill, skill_params(args))]
    if (args.then_measure or args.then_approach) and not args.then_face:
        print("--then-measure and --then-approach need --then-face")
        recorder.close()
        return 2
    if args.then_face:
        if args.skill != "rotate_search" or not args.target_class:
            print("--then-face needs rotate_search and --target-class")
            recorder.close()
            return 2
        steps.append(("face_target", {"target_class": args.target_class}))
        if args.then_measure:
            mp: Dict[str, Any] = {"target_class": args.target_class}
            if args.baseline is not None:
                mp["baseline_m"] = args.baseline
            steps.append(("measure_range", mp))
        if args.then_approach:
            steps.append(("approach",
                          {"target_class": args.target_class,
                           "standoff_m": args.standoff}))
    if args.instruction:
        try:
            steps = plan_on_ground(args, cfg, recorder)
        except NoPlan as e:
            print(f"\nNOT FLYING: {e}")
            recorder.close()
            return 4

    # Perception starts HERE, on the ground, before anything is armed.
    # Loading a detector takes seconds (the first time, a download), and it
    # used to happen after takeoff, with the aircraft hovering and nothing
    # servicing it. And a flight that was asked for perception and cannot
    # have it does not go: a search that finds nothing because the detector
    # never started looks, in the log, exactly like a search that failed.
    runner = None
    if args.perception != "none":
        try:
            runner = start_perception(args, recorder)
        except PerceptionUnavailable as e:
            print(f"\nNOT FLYING: perception was requested and is not "
                  f"available.\n  {e}")
            recorder.event("perception_unavailable", {"detail": str(e)})
            recorder.close()
            return 3
    elif args.target_class:
        print("  note: --target-class without --perception yoloe. Nothing "
              "can be found; the sweep will run to its end.")

    vehicle = MavlinkVehicle(cfg, verbose=verbose)
    executor = Executor(vehicle, cfg, recorder, verbose=verbose)
    _EXECUTOR = executor
    if runner is not None:
        vehicle.attach_perception(runner)

    outcome = "unknown"
    results: List[Dict[str, Any]] = []

    try:
        executor.preflight()
        executor.launch(handover=bool(args.handover),
                        takeoff_height_m=args.takeoff_height)

        kind = args.inject
        if kind is None and args.inject_bad_setpoint:
            kind = "altitude"
        elif kind is None and args.inject_bad_step:
            kind = "step"
        wrap = None
        if kind:
            wrap = lambda sk: BadSetpointInjector(     # noqa: E731
                sk, args.inject_at_tick, kind,
                on_fire=lambda why: print(f"  [INJECT] {why}"),
                repeat=args.inject_repeat)
            print(f"  [INJECT] {kind} armed for tick "
                  f"{args.inject_at_tick}: {FAULTS[kind]}")

        if args.inject_estop_at:
            _arm_estop_timer(executor, args.inject_estop_at, recorder)

        # `results` is filled in place, so the steps that did finish are
        # in the summary even when a later one raises.
        _, outcome = run_steps(executor, cfg, steps, recorder, wrap, results)

        if args.hold_after > 0 and outcome == "complete":
            hold = build("hover", cfg, duration_s=args.hold_after)
            results.append(executor.run_skill(hold).to_log())

        if not args.no_land:
            executor.land(f"mission {outcome}")

    except PilotTakeover as e:
        print("PILOT TAKEOVER:", e)
        outcome = "pilot_takeover"
        recorder.event("pilot_takeover", {"detail": str(e)})
        return 2
    except FlightError as e:
        # Every abort - link loss included - lands.
        print("FLIGHT ABORT:", e)
        outcome = "flight_error"
        recorder.event("flight_error", {"detail": str(e)})
        if executor.landing.airborne or vehicle.is_armed():
            executor.land(f"abort: {e}")
        return 1
    except Exception as e:
        print("UNEXPECTED ERROR:", e)
        outcome = "unexpected_error"
        recorder.event("unexpected_error", {"detail": repr(e)})
        try:
            if executor.landing.airborne or vehicle.is_armed():
                executor.land(f"unexpected error: {e}")
        except Exception:
            pass
        raise
    finally:
        if _PERCEPTION is not None:
            _PERCEPTION.stop()
            det = _PERCEPTION.detector
            health = dict(_PERCEPTION.health())
            if hasattr(det, "health"):
                health.update(det.health())
            recorder.event("perception_health", health)
            try:
                det.close()
            except Exception:
                pass
        summary = executor.summary()
        summary.update({"outcome": outcome, "skills": results})
        recorder.summary(summary)
        for line in executor.arbiter.summary_lines():
            print(line)
        print(f"Uplink used ~{summary['uplink_bps_avg']:.0f} B/s average.")
        print(f"Log: {recorder.path}")
        recorder.close()

    return 0


class NoPlan(Exception):
    """There is no plan to fly: the model was unreachable, its reply did not
    pass the checks, it refused, or the operator said no."""


def plan_on_ground(args: argparse.Namespace, cfg: Config,
                   recorder: FlightRecorder, llm: Optional[Any] = None,
                   ask: Any = input) -> List[tuple]:
    """
    Instruction -> checked plan -> the operator's yes. Returns the steps as
    (skill, params) or raises NoPlan. Nothing here touches the vehicle, and
    the whole exchange with the model goes into the log.
    """
    from drone_agent.planner import (LLMError, OpenAICompatLLM, Planner,
                                     format_plan)
    classes = ([c.strip().lower() for c in args.classes.split(",")
                if c.strip()] if args.classes else None)
    if llm is None:
        llm = OpenAICompatLLM(args.llm_url, args.llm_model,
                              api_key=os.environ.get("DRONE_LLM_KEY", ""),
                              timeout_s=args.llm_timeout,
                              json_mode=args.json_mode,
                              max_tokens=args.max_tokens,
                              no_think=args.no_think)
    planner = Planner(llm, catalogue(), cfg.envelope.alt_max_m, classes)
    print(f"  [PLAN] asking {getattr(llm, 'model', '?')} ...")
    try:
        res = planner.plan(args.instruction)
    except LLMError as e:
        recorder.event("plan_failed", {"instruction": args.instruction,
                                       "detail": str(e)})
        raise NoPlan(f"the language model could not be asked.\n  {e}")
    recorder.event("plan", res.to_log())
    print(format_plan(res))
    if not res.ok:
        raise NoPlan("the model did not produce a plan that passes the "
                     "checks.")
    if not res.steps:
        raise NoPlan("the model says this cannot be done with the skills "
                     "the drone has.")
    targets = sorted({s.params["target_class"] for s in res.steps
                      if s.params.get("target_class")})
    if targets and args.perception == "none":
        raise NoPlan(f"the plan looks for {targets} and the camera is off. "
                     f"Add --perception yoloe --camera N --intrinsics FILE.")
    if targets and not args.classes:
        args.classes = ",".join(targets)   # the detector looks for these
    if not args.yes:
        try:
            answer = ask("\n  Fly this plan? Type y and Enter (anything "
                         "else cancels): ")
        except EOFError:
            answer = ""
        recorder.event("plan_confirmation", {"answer": answer.strip()})
        if answer.strip().lower() not in ("y", "yes"):
            raise NoPlan("cancelled by the operator.")
    return [(s.skill, dict(s.params)) for s in res.steps]


def run_steps(executor: Any, cfg: Config, steps: List[tuple],
              recorder: FlightRecorder, wrap: Optional[Any] = None,
              results: Optional[List[Dict[str, Any]]] = None) -> tuple:
    """
    Run the steps in order. Stops at the first one that does not succeed,
    and at a search that ends without its target when more steps follow:
    what comes after a search was planned on the assumption that it found
    something. Returns (results, outcome); the caller lands.
    """
    results = [] if results is None else results
    outcome = "complete"
    handover: Dict[str, Any] = {}
    for i, (name, params) in enumerate(steps):
        extra = {}
        if getattr(SKILL_REGISTRY[name], "accepts_handover", False):
            # What the previous step found, so that a chain of skills is
            # about ONE object. The planner cannot set this; it comes only
            # from the step before.
            extra["handover"] = handover
        skill = build(name, cfg, **params, **extra)
        if wrap is not None and i == 0:
            skill = wrap(skill)
        if len(steps) > 1:
            print(f"  [PLAN] step {i + 1} of {len(steps)}: {name} {params}")
        result = executor.run_skill(skill)
        results.append(result.to_log())
        got = getattr(result, "detail", None) or {}
        if result.ok and got.get("track_id") is not None:
            handover = dict(got)
        if not result.ok:
            outcome = f"skill_{result.status.value}"
            break
        # Only a SEARCH that found nothing ends the plan. face_target also
        # has a target_class, and its success is worded differently.
        if (name == "rotate_search" and params.get("target_class")
                and i < len(steps) - 1
                and result.reason != "target acquired"):
            outcome = "target_not_found"
            print(f"  [PLAN] {params['target_class']} was not found: the "
                  f"remaining {len(steps) - 1 - i} step(s) are skipped")
            recorder.event("plan_stopped",
                           {"after_step": i + 1, "of": len(steps),
                            "reason": "target not found",
                            "target_class": params["target_class"]})
            break
    return results, outcome


class PerceptionUnavailable(Exception):
    """Perception was asked for and cannot be provided. Raised on the
    ground, before the vehicle is connected, so the answer is 'do not fly'
    and never 'fly without it'."""


def start_perception(args: argparse.Namespace, recorder: FlightRecorder,
                     detector_factory: Optional[Any] = None,
                     ready_timeout_s: float = 20.0) -> Any:
    """
    Build the detector and the runner, and prove frames are flowing.

    Nothing here touches the vehicle. It is called before the MAVLink
    connection exists, and it either returns a runner that has already
    published scenes or raises PerceptionUnavailable.

    The capture is opened at the size the camera was CALIBRATED at. Left to
    itself the capture card hands back 1920x1080, a 16:9 frame around a 4:3
    picture, and a calibration made at 720x576 then places every bearing
    wrong. The ground tool asked for the calibrated size from the start;
    this path did not.
    """
    global _PERCEPTION
    from drone_agent.perception import (CameraIntrinsics, PerceptionRunner,
                                        RangeEstimator)
    if args.intrinsics:
        try:
            intr = CameraIntrinsics.load(args.intrinsics)
            boresight = CameraIntrinsics.load_boresight(args.intrinsics)
        except Exception as e:
            raise PerceptionUnavailable(
                f"could not read {args.intrinsics}: {e!r}")
        print(f"  [PERCEPTION] intrinsics from {args.intrinsics} "
              f"({intr.model}, {intr.width_px}x{intr.height_px}, hfov "
              f"{intr.hfov_deg:.1f} deg, reproj {intr.reproj_err_px} px)")
        if boresight.calibrated:
            print(f"  [PERCEPTION] boresight ({boresight.u:.3f}, "
                  f"{boresight.v:.3f}), beam {boresight.beam_deg:g} deg")
        else:
            print("  [PERCEPTION] no boresight calibration: the forward "
                  "lidar range will not be matched to any detection")
        want = (intr.width_px, intr.height_px)
    else:
        intr = CameraIntrinsics.from_hfov(args.hfov)
        boresight = CameraIntrinsics.load_boresight("")
        want = (None, None)
        print(f"  [PERCEPTION] WARNING: no --intrinsics, assuming "
              f"{args.hfov:g} deg HFOV. Bearings toward the frame edges will "
              f"be wrong; calibrate through the full analog path.")

    classes = ([c.strip() for c in args.classes.split(",") if c.strip()]
               if args.classes else None)
    if args.target_class and classes and args.target_class not in classes:
        raise PerceptionUnavailable(
            f"--target-class {args.target_class!r} is not in --classes "
            f"{classes}: the detector would never be asked to look for it")
    if args.target_class and not classes:
        classes = [args.target_class]

    try:
        if detector_factory is not None:
            det = detector_factory(classes, want)
        else:
            from drone_agent.perception.yoloe import YoloeDetector
            src: Any = args.camera
            try:
                src = int(args.camera)
            except ValueError:
                pass
            print(f"  [PERCEPTION] loading {args.model} ...")
            det = YoloeDetector(model_path=args.model, source=src,
                                classes=classes, imgsz=args.imgsz,
                                device=args.device,
                                width=want[0], height=want[1])
    except Exception as e:
        raise PerceptionUnavailable(f"the detector did not start: {e!r}")

    known = {}
    th = getattr(args, "target_height", None)
    tc = getattr(args, "target_class", None)
    if th and tc:
        known[tc] = float(th)
    runner = PerceptionRunner(
        det, intrinsics=intr,
        estimator=RangeEstimator(intrinsics=intr, boresight=boresight,
                                 known_heights=known),
        labels=classes)
    if abs(intr.tilt_up_deg) < 1e-9:
        print("  [PERCEPTION] no camera tilt in the calibration: ranges come "
              "from the size of the box,")
        print("               which read +25% to +31% against a tape. "
              "`calibrate_camera.py tilt` fixes it.")
    else:
        src = f"a measured {th} m" if known else "the size prior's"
        print(f"  [PERCEPTION] head-height range on, camera "
              f"{intr.tilt_up_deg:+.2f} deg up, {src} target height")
    if runner.capture_warning:
        try:
            det.close()
        except Exception:
            pass
        raise PerceptionUnavailable(runner.capture_warning)

    # Frames must actually be arriving before anything is armed.
    runner.start()
    t0 = time.monotonic()
    first = None
    while time.monotonic() - t0 < ready_timeout_s:
        scene = runner.latest()
        if scene.frame_id >= 1 and first is None:
            first = (time.monotonic(), scene.frame_id)
        if first and scene.frame_id - first[1] >= 20:
            break
        if first and time.monotonic() - first[0] > 3.0:
            break
        time.sleep(0.05)
    scene = runner.latest()
    if first is None or scene.frame_id < 3 or runner.fresh_scene() is None:
        why = runner.last_error or "no frames were read from the camera"
        runner.stop()
        try:
            det.close()
        except Exception:
            pass
        raise PerceptionUnavailable(
            f"the detector started but published nothing usable in "
            f"{ready_timeout_s:g} s ({why})")
    rate = ((scene.frame_id - first[1])
            / max(1e-3, time.monotonic() - first[0]))

    _PERCEPTION = runner
    health = det.health() if hasattr(det, "health") else {}
    recorder.event("perception_started",
                   {"detector": det.name, "classes": classes,
                    "model": getattr(args, "model", None),
                    "device_used": health.get("device_used"),
                    "capture_size": (health.get("capture_size")
                                     or getattr(det, "capture_size", None)),
                    "scene_rate_hz": round(rate, 1),
                    "inference_ms": scene.inference_ms,
                    "max_age_s": runner.max_age_s,
                    "intrinsics_calibrated": intr.is_calibrated,
                    "hfov_deg": round(intr.hfov_deg, 2),
                    "reproj_err_px": intr.reproj_err_px,
                    "boresight_calibrated": boresight.calibrated})
    print(f"  [PERCEPTION] {det.name} running on "
          f"{health.get('device_used') or '?'}: {rate:.0f} scenes/s, "
          f"classes={classes}")
    return runner


def _arm_estop_timer(executor: Executor, after_s: float,
                     recorder: FlightRecorder) -> None:
    import threading

    def fire() -> None:
        time.sleep(after_s)
        print(f"  [INJECT] e-stop after {after_s:.1f}s")
        recorder.event("injected_estop", {"after_s": after_s})
        executor.arbiter.estop("injected bench test")

    threading.Thread(target=fire, daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
