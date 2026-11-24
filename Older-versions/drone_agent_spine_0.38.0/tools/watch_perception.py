#!/usr/bin/env python3
"""
Perception on the ground: the detector, the calibrated geometry and the
staleness gate, run exactly as they would run in flight, with no vehicle.

    # what does the stack see, how fast, and how stale?
    python tools/watch_perception.py watch --camera 1 --intrinsics camera.json \\
        --classes person,chair,bottle --duration 30

    # how old is a frame by the time it is read? (point the drone's camera
    # at this computer's screen)
    python tools/watch_perception.py latency --camera 1

    # no camera and no torch: a synthetic detector, to see the output
    python tools/watch_perception.py watch --detector synthetic --duration 5

This tool never opens a MAVLink connection and cannot arm or command
anything. It exists because the only other way to run the perception code is
`run_mission.py --perception yoloe`, which flies.

What `watch` answers, in the order the answers matter:

  1. Does the detector run here at all, on which device, at what rate?
  2. How old is the scene when a 10 Hz control loop reads it? The loop only
     accepts a scene younger than max_age_s, so this decides whether the
     skills ever see anything.
  3. Is a track stable -- one id for one object -- or does it flicker?
  4. Is the bearing right? Put an object at a tape-measured angle and pass
     --expect chair:30.
  5. Would a search actually STOP on this object? rotate_search only counts
     a target as found with a track id, a minimum age, and a range whose
     stated uncertainty is small enough. On an airframe with no forward
     rangefinder the range comes from the object's assumed size, so this is
     not a formality.

What `latency` answers: the capture timestamp is taken when read() returns,
but the picture is older than that by the whole analog chain -- camera,
video transmitter, goggles, capture card. The flash test puts a number on
it. It includes this computer's display lag, so it is an upper bound.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drone_agent import __version__                                # noqa: E402
from drone_agent.perception import (CameraIntrinsics, Detection,   # noqa: E402
                                    PerceptionRunner, RangeEstimator)


# ───────────────────────── a detector with no camera ─────────────────────

class SyntheticDetector:
    """
    One "person" walking slowly across the frame, with real timestamps.

    For trying the tool, and for testing it. MockDetector will not do here:
    it stamps frames with a counter, so every scene looks minutes old to a
    staleness gate that compares against the real clock.
    """

    name = "synthetic"

    def __init__(self, period_s: float = 0.05, inference_ms: float = 20.0,
                 size: Tuple[int, int] = (720, 576)) -> None:
        self.period_s = period_s
        self.inference_ms = inference_ms
        self.capture_size = size
        self._n = 0
        self.last_frame = None

    def detect(self) -> tuple:
        time.sleep(self.period_s)
        t_cap = time.monotonic()
        self._n += 1
        u = 0.25 + 0.5 * (0.5 + 0.5 * math.sin(self._n / 40.0))
        box = (u - 0.06, 0.30, u + 0.06, 0.78)
        dets = [Detection("person", box, 0.82, track_id=1)]
        if self._n % 7 == 0:        # an untracked flicker, as real ones do
            dets.append(Detection("chair", (0.70, 0.55, 0.82, 0.80), 0.41,
                                  track_id=None))
        return t_cap, dets, self.inference_ms

    def close(self) -> None:
        pass

    def health(self) -> dict:
        return {"frames": self._n, "device_used": "none (synthetic)"}


# ───────────────────────────── the found rule ────────────────────────────

def found_rule() -> Dict[str, float]:
    """The thresholds rotate_search uses, read from the skill itself so this
    tool cannot drift away from what the flight code does."""
    try:
        from drone_agent.skills.rotate_search import RotateSearch as R
        return {"min_age_s": R.MIN_TRACK_AGE_S, "min_frames": R.MIN_FRAMES,
                "max_sigma_m": R.MAX_RANGE_SIGMA_M}
    except Exception:
        return {"min_age_s": 0.5, "min_frames": 3, "max_sigma_m": 0.6}


def why_not_found(obj: Dict[str, Any], rule: Dict[str, float]
                  ) -> Optional[str]:
    """None if rotate_search would stop on this observation, else the first
    reason it would not. Same order of tests as the skill."""
    if obj.get("track_id") is None:
        return "no track id"
    if obj.get("age_s", 0.0) < rule["min_age_s"]:
        return "track too young"
    if obj.get("frames_seen", 0) < rule["min_frames"]:
        return "too few frames"
    if obj.get("range_m") is None:
        return "no range"
    sigma = obj.get("range_sigma_m")
    if sigma is None or sigma > rule["max_sigma_m"]:
        return "range too uncertain"
    return None


# ─────────────────────────────── summary ─────────────────────────────────

def _pct(sorted_vals: List[float], q: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))]


def summarise(scenes: List[Dict[str, Any]], ticks: List[Dict[str, Any]],
              rule: Dict[str, float]) -> Dict[str, Any]:
    """Everything the report prints, as data, so it can be tested and logged."""
    out: Dict[str, Any] = {"scenes": len(scenes), "ticks": len(ticks)}
    if len(scenes) >= 2:
        span = scenes[-1]["t"] - scenes[0]["t"]
        frames = scenes[-1]["frame_id"] - scenes[0]["frame_id"]
        out["scene_rate_hz"] = frames / span if span > 0 else None
        inf = sorted(s["inference_ms"] for s in scenes
                     if s.get("inference_ms") is not None)
        lat = sorted(s["latency_s"] for s in scenes)
        if inf:
            out["inference_ms"] = {"p50": _pct(inf, .5), "p95": _pct(inf, .95)}
        out["publish_latency_s"] = {"p50": _pct(lat, .5), "p95": _pct(lat, .95)}
    if ticks:
        ages = sorted(t["age_s"] for t in ticks if t.get("age_s") is not None)
        out["fresh_tick_pct"] = 100.0 * sum(1 for t in ticks if t["fresh"]) \
            / len(ticks)
        if ages:
            out["scene_age_at_tick_s"] = {"p50": _pct(ages, .5),
                                          "p95": _pct(ages, .95),
                                          "max": ages[-1]}

    labels: Dict[str, Dict[str, Any]] = {}
    for s in scenes:
        best: Dict[str, Dict[str, Any]] = {}
        for o in s["objects"]:
            if (o["label"] not in best
                    or o["confidence"] > best[o["label"]]["confidence"]):
                best[o["label"]] = o
        for lab, o in best.items():
            L = labels.setdefault(lab, {"t": [], "bearing": [], "elev": [],
                                        "range": [], "sigma": [], "conf": [],
                                        "ids": {}, "no_id": 0, "sources": {},
                                        "reasons": {}, "found": 0})
            L["t"].append(s["t"])
            L["bearing"].append(o["bearing_deg"])
            L["elev"].append(o["elevation_deg"])
            L["conf"].append(o["confidence"])
            if o.get("range_m") is not None:
                L["range"].append(o["range_m"])
            if o.get("range_sigma_m") is not None:
                L["sigma"].append(o["range_sigma_m"])
            src = o.get("range_source", "none")
            L["sources"][src] = L["sources"].get(src, 0) + 1
            tid = o.get("track_id")
            if tid is None:
                L["no_id"] += 1
            else:
                first, last = L["ids"].get(tid, (s["t"], s["t"]))
                L["ids"][tid] = (first, s["t"])
            why = why_not_found(o, rule)
            if why is None:
                L["found"] += 1
            else:
                L["reasons"][why] = L["reasons"].get(why, 0) + 1

    out["labels"] = {}
    for lab, L in labels.items():
        n = len(L["t"])
        spans = [b - a for a, b in L["ids"].values()]
        out["labels"][lab] = {
            "scenes": n,
            "seen_pct": 100.0 * n / max(1, len(scenes)),
            "track_ids": len(L["ids"]),
            "untracked_pct": 100.0 * L["no_id"] / n,
            "longest_track_s": max(spans) if spans else 0.0,
            "bearing_mean": st.mean(L["bearing"]),
            "bearing_sd": st.pstdev(L["bearing"]),
            "elevation_mean": st.mean(L["elev"]),
            "range_mean": st.mean(L["range"]) if L["range"] else None,
            "range_sd": st.pstdev(L["range"]) if L["range"] else None,
            "sigma_mean": st.mean(L["sigma"]) if L["sigma"] else None,
            "range_sources": L["sources"],
            "confidence_mean": st.mean(L["conf"]),
            "found_pct": 100.0 * L["found"] / n,
            "not_found_reasons": L["reasons"],
        }
    return out


def print_summary(sm: Dict[str, Any], rule: Dict[str, float],
                  max_age_s: float, expect: Dict[str, float],
                  expect_range: Dict[str, float]) -> int:
    """Returns the number of findings that should stop the next step."""
    stop = 0
    print("\nPIPELINE")
    if sm.get("scene_rate_hz") is None:
        print("  too few scenes to measure a rate")
        return 1
    print(f"  scenes          {sm['scenes']} read, "
          f"{sm['scene_rate_hz']:.1f} per second from the detector")
    if "inference_ms" in sm:
        print(f"  inference       {sm['inference_ms']['p50']:.0f} ms median, "
              f"{sm['inference_ms']['p95']:.0f} ms p95")
    print(f"  read to publish {1000 * sm['publish_latency_s']['p50']:.0f} ms "
          f"median, {1000 * sm['publish_latency_s']['p95']:.0f} ms p95")
    if "scene_age_at_tick_s" in sm:
        a = sm["scene_age_at_tick_s"]
        print(f"  age at a 10 Hz tick   median {a['p50']:.2f} s, p95 "
              f"{a['p95']:.2f} s, worst {a['max']:.2f} s")
        print(f"  ticks with a fresh scene   {sm['fresh_tick_pct']:.0f}%   "
              f"(fresh = younger than {max_age_s:g} s)")
        if sm["fresh_tick_pct"] < 90.0:
            stop += 1
            print(f"    TOO STALE. The control loop would be blind on "
                  f"{100 - sm['fresh_tick_pct']:.0f}% of its ticks. A smaller "
                  f"model or --imgsz is the first thing to try.")
    if "main_loop_gap_ms" in sm:
        m = sm["main_loop_gap_ms"]
        print(f"  this tool's own loop   held up {m['max']:.0f} ms at "
              f"worst, {m['p99']:.0f} ms p99 (it sleeps 5 ms)")
        if m["max"] > 150.0:
            stop += 1
            print("    The detector thread is holding the interpreter long "
                  "enough to delay a 10 Hz control loop.")
    print("  These ages start when read() returns. The picture is older "
          "than that by the analog chain;")
    print("  `latency` mode measures how much.")

    if not sm["labels"]:
        print("\nOBJECTS\n  nothing was detected. Check --classes, the "
              "lighting, and the snapshot frames.")
        return stop + 1

    print("\nOBJECTS")
    print(f"  {'label':<12}{'seen':>6}{'ids':>5}{'longest':>9}"
          f"{'bearing':>16}{'range':>18}{'source':>12}{'conf':>6}{'found':>7}")
    for lab, L in sorted(sm["labels"].items(),
                         key=lambda kv: -kv[1]["scenes"]):
        rng = (f"{L['range_mean']:.2f} ±{L['range_sd']:.2f} m"
               if L["range_mean"] is not None else "-")
        src = max(L["range_sources"], key=lambda k: L["range_sources"][k])
        print(f"  {lab:<12}{L['seen_pct']:>5.0f}%{L['track_ids']:>5}"
              f"{L['longest_track_s']:>8.1f}s"
              f"{L['bearing_mean']:>+9.1f} ±{L['bearing_sd']:.1f}°"
              f"{rng:>18}{src:>12}{L['confidence_mean']:>6.2f}"
              f"{L['found_pct']:>6.0f}%")
    print("    seen = share of scenes containing it | ids = distinct track "
          "ids (1 is a stable track)")
    print("    bearing and range are mean ± spread over the run; on a still "
          "object the spread is jitter")
    print(f"    found = share of sightings a search would stop on (track id, "
          f"age >= {rule['min_age_s']:g} s,")
    print(f"            >= {rule['min_frames']:g} frames, range uncertainty "
          f"<= {rule['max_sigma_m']:g} m)")

    for lab, L in sm["labels"].items():
        if L["found_pct"] >= 50.0 or not L["not_found_reasons"]:
            continue
        why = max(L["not_found_reasons"],
                  key=lambda k: L["not_found_reasons"][k])
        print(f"\n  {lab}: a search for it would NOT stop. Most common "
              f"reason: {why} "
              f"({L['not_found_reasons'][why]} of {L['scenes']} sightings).")
        if why == "no range":
            print(f"    There is no size on file for '{lab}', and without a "
                  f"forward rangefinder size is the only range source.")
            print(f"    Either use a label that has one, or add its height "
                  f"to SIZE_PRIORS in perception/geometry.py.")
        elif why == "range too uncertain":
            print(f"    The range is a guess from the object's assumed "
                  f"height, and at this distance the guess is wider than "
                  f"the skill accepts")
            print(f"    (stated ±{L['sigma_mean']:.2f} m against "
                  f"{rule['max_sigma_m']:g} m). Closer objects pass; so "
                  f"would a measured range.")
        elif why == "no track id":
            print(f"    The tracker is not assigning it an id, so every "
                  f"frame looks like a new object.")
        if L["track_ids"] > 3 and L["longest_track_s"] < 2.0:
            print(f"    Its track also keeps changing id "
                  f"({L['track_ids']} ids, longest {L['longest_track_s']:.1f} "
                  f"s).")

    for lab, deg in expect.items():
        L = sm["labels"].get(lab)
        if L is None:
            print(f"\n  --expect {lab}: never detected")
            stop += 1
            continue
        err = L["bearing_mean"] - deg
        verdict = ("good" if abs(err) <= 3.0 else
                   "usable" if abs(err) <= 6.0 else "WRONG")
        print(f"\n  BEARING CHECK  {lab}: measured {L['bearing_mean']:+.1f}° "
              f"against {deg:+.1f}° by tape: off by {err:+.1f}°  ({verdict})")
        if abs(err) > 6.0:
            stop += 1
            print("    More than the calibration should allow. Check that "
                  "the capture size matches camera.json and that")
            print("    the angle was measured from the camera's own axis, "
                  "not from the drone's body.")
    for lab, m in expect_range.items():
        L = sm["labels"].get(lab)
        if L is None or L["range_mean"] is None:
            print(f"\n  --expect-range {lab}: no range was produced")
            continue
        err = L["range_mean"] - m
        print(f"\n  RANGE CHECK  {lab}: estimated {L['range_mean']:.2f} m "
              f"against {m:.2f} m by tape: off by {err:+.2f} m "
              f"({100 * err / m:+.0f}%), stated uncertainty "
              f"±{L['sigma_mean']:.2f} m")
        if L["sigma_mean"] and abs(err) > 2.0 * L["sigma_mean"]:
            print("    The error is more than twice what the estimate "
                  "claims for itself, so the claim is too confident.")
    return stop


# ──────────────────────────────── watch ──────────────────────────────────

def _pairs(spec: Optional[str], what: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in (spec or "").split(","):
        if not item.strip():
            continue
        lab, sep, val = item.partition(":")
        try:
            out[lab.strip()] = float(val)
        except ValueError:
            raise SystemExit(f"{what} must look like chair:30,person:-15")
    return out


def annotate(cv2: Any, frame: Any, objects: List[Dict[str, Any]]) -> Any:
    img = frame.copy()
    h, w = img.shape[:2]
    for o in objects:
        if not o.get("bbox"):
            continue
        x1, y1, x2, y2 = o["bbox"]
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(img, p1, p2, (40, 220, 40), 2)
        tid = f"#{o['track_id']}" if o.get("track_id") is not None else ""
        rng = (f" {o['range_m']:.1f}m" if o.get("range_m") is not None
               else "")
        txt = f"{o['label']}{tid} {o['bearing_deg']:+.0f}deg{rng}"
        cv2.putText(img, txt, (p1[0], max(14, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 40), 1,
                    cv2.LINE_AA)
    return img


def run_watch(args: argparse.Namespace) -> int:
    classes = ([c.strip() for c in args.classes.split(",") if c.strip()]
               if args.classes else None)
    if args.intrinsics:
        intr = CameraIntrinsics.load(args.intrinsics)
        print(f"  intrinsics     {args.intrinsics}: {intr.model}, "
              f"{intr.width_px}x{intr.height_px}, HFOV {intr.hfov_deg:.1f}°")
    else:
        intr = CameraIntrinsics.from_hfov(args.hfov)
        print(f"  intrinsics     NONE: assuming a {args.hfov:g}° field of "
              f"view. Bearings toward the frame edges will be wrong.")

    if args.detector == "synthetic":
        det: Any = SyntheticDetector(size=(intr.width_px, intr.height_px))
    else:
        try:
            from drone_agent.perception.yoloe import YoloeDetector
            src: Any = args.camera
            try:
                src = int(args.camera)
            except ValueError:
                pass
            print(f"  detector       loading {args.model} (the first run "
                  f"downloads the model and a text encoder, and needs the "
                  f"network)")
            det = YoloeDetector(
                model_path=args.model, source=src, classes=classes,
                imgsz=args.imgsz, conf=args.conf, device=args.device,
                width=args.width or intr.width_px,
                height=args.height or intr.height_px)
        except Exception as e:
            print(f"\n  The detector could not start: {e!r}")
            print("  If ultralytics is missing: pip install ultralytics")
            print("  If the camera did not open: `calibrate_camera.py "
                  "devices` shows which index is the capture card.")
            return 2

    known = {}
    if args.object_height and args.target_class:
        known[args.target_class] = float(args.object_height)
    runner = PerceptionRunner(
        det, intrinsics=intr,
        estimator=RangeEstimator(intrinsics=intr, known_heights=known),
        max_age_s=args.max_age, min_confidence=args.min_conf, labels=classes)
    # A bench station: the aircraft on a stand at a measured lens height,
    # the target at a measured distance. No flying, no battery, and no
    # rangefinder either -- the lens height is given rather than sensed,
    # which is the whole point: it removes one unknown from a measurement
    # that has never had a clean one.
    if args.camera_height:
        # Feed the lens height in place of the rangefinder, once: on a
        # stand it does not change.
        runner.set_camera_pose(float(args.camera_height), 0.0)
        print(f"  bench          lens {args.camera_height} m above the "
              f"floor"
              + (f", target taped at {args.truth} m" if args.truth else "")
              + (f", target {args.object_height} m tall" if
                 args.object_height else ""))
    size = getattr(det, "capture_size", None)
    print(f"  capture        {size[0]}x{size[1]}" if size else
          "  capture        size not reported")
    if runner.capture_warning:
        print(f"\n  STOP: {runner.capture_warning}")
        if not args.force:
            det.close()
            return 2
    rule = found_rule()
    expect = _pairs(args.expect, "--expect")
    expect_range = _pairs(args.expect_range, "--expect-range")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = out_dir / f"perception-{stamp}.jsonl"
    snap_dir = out_dir / f"perception-{stamp}-frames"
    cv2 = None
    if args.snapshot_every > 0 and args.detector != "synthetic":
        try:
            import cv2 as _cv2
            cv2 = _cv2
            snap_dir.mkdir(parents=True, exist_ok=True)
        except ImportError:
            pass

    scenes: List[Dict[str, Any]] = []
    ticks: List[Dict[str, Any]] = []
    with open(log_path, "w") as logf:
        logf.write(json.dumps({
            "type": "meta", "drone_agent_version": __version__,
            "argv": sys.argv, "classes": classes, "model": args.model,
            "detector": getattr(det, "name", "?"),
            "intrinsics": intr.to_dict(), "max_age_s": args.max_age,
            "min_confidence": args.min_conf, "found_rule": rule,
            "capture_size": list(size) if size else None}) + "\n")

        print(f"\n  watching for {args.duration:g} s. Nothing is sent to "
              f"any vehicle.\n")
        if args.camera_height and not args.truth:
            print("  --camera-height without --truth: the head range will "
                  "be computed but not scored.\n")
        runner.start()
        t0 = time.monotonic()
        next_tick = t0
        next_print = t0 + 1.0
        next_snap = t0 + 1.0
        seen = 0
        gaps: List[float] = []
        prev = t0
        try:
            while True:
                now = time.monotonic()
                gaps.append(now - prev)
                prev = now
                if now - t0 >= args.duration:
                    break
                scene = runner.latest()
                if scene.frame_id != seen and scene.frame_id > 0:
                    seen = scene.frame_id
                    rec = scene.to_log()
                    rec["t"] = round(now - t0, 3)
                    scenes.append(rec)
                    logf.write(json.dumps({"type": "scene", **rec}) + "\n")
                    frame = getattr(det, "last_frame", None)
                    if (cv2 is not None and frame is not None
                            and now >= next_snap):
                        next_snap = now + args.snapshot_every
                        cv2.imwrite(str(snap_dir / f"t{now - t0:06.2f}.jpg"),
                                    annotate(cv2, frame, rec["objects"]))
                if now >= next_tick:
                    next_tick += 0.1
                    has = scene.t_capture_mono > 0.0
                    ticks.append({
                        "t": round(now - t0, 3),
                        "fresh": runner.fresh_scene(now) is not None,
                        "age_s": scene.age_s(now) if has else None})
                if now >= next_print:
                    next_print += 1.0
                    objs = scenes[-1]["objects"] if scenes else []
                    desc = " | ".join(
                        f"{o['label']}"
                        f"{'#' + str(o['track_id']) if o.get('track_id') is not None else ''}"
                        f" {o['bearing_deg']:+.0f}°"
                        + (f" {o['range_m']:.1f} m" if o.get("range_m")
                           is not None else "")
                        for o in objs[:4]) or "nothing"
                    age = ticks[-1]["age_s"] if ticks else None
                    print(f"  t={now - t0:5.1f}s  {len(scenes):5d} scenes  "
                          f"age {age if age is None else round(age, 2)} s  "
                          f"{desc}")
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\n  stopped early")
        finally:
            runner.stop()
            health = {**runner.health(),
                      **(det.health() if hasattr(det, "health") else {})}
            try:
                det.close()
            except Exception:
                pass

        sm = summarise(scenes, ticks, rule)
        sm["health"] = health
        # This loop sleeps 5 ms and does almost nothing, as a stand-in for
        # the control loop sharing a process with the detector thread. A long
        # gap here is the detector holding the interpreter.
        g = sorted(gaps[1:])
        if g:
            sm["main_loop_gap_ms"] = {"p50": 1000 * _pct(g, .5),
                                      "p99": 1000 * _pct(g, .99),
                                      "max": 1000 * g[-1]}
        logf.write(json.dumps({"type": "summary", **sm}, default=str) + "\n")

    print(f"\n  device         {health.get('device_used') or health.get('device') or '?'}"
          f"   errors {health.get('errors', 0)}"
          + (f"   last error: {health.get('last_error')}"
             if health.get("errors") else ""))
    stop = print_summary(sm, rule, args.max_age, expect, expect_range)
    if args.camera_height:
        stop = bench_summary(scenes, intr, args) or stop
    print(f"\n  log            {log_path}")
    if cv2 is not None:
        print(f"  frames         {snap_dir}/  (annotated, for looking at "
              f"what it saw)")
    return 1 if stop else 0


def bench_summary(scenes: List[Dict[str, Any]], intr: Any,
                  args: argparse.Namespace) -> bool:
    """
    One bench station: the aircraft on a stand, the target taped at a
    measured distance. This is the measurement the head-height range has
    never had -- every earlier figure came from a flight, where the lens
    height was sensed, the aircraft was moving and the target may have
    been too. Here the only unknown is the method.
    """
    import math
    import statistics as st
    got = [o for sc in scenes for o in sc.get("objects") or []
           if o.get("label") == args.target_class]
    heads = [o["range_m"] for o in got
             if o.get("range_m") is not None
             and str(getattr(o.get("range_source"), "value",
                             o.get("range_source"))) == "head_height"]
    others = [str(getattr(o.get("range_source"), "value",
                          o.get("range_source"))) for o in got]
    print(f"\n  bench station")
    print(f"    detections of {args.target_class}   {len(got)}")
    if not got:
        print("    nothing was detected, so there is nothing to score")
        return True
    if not heads:
        print(f"    none got a head-height range; the sources seen were "
              f"{sorted(set(others))}")
        print("    With a tilt in the calibration this means the box could "
              "not be the whole target,")
        print("    or the lens is too close to the target's own height.")
        return True
    med = st.median(heads)
    print(f"    head-height range    median {med:.2f} m, spread "
          f"{st.pstdev(heads) if len(heads) > 1 else 0.0:.2f} m, "
          f"{len(heads)} of {len(got)} frames")
    if not args.truth:
        return False
    print(f"    against the tape     {args.truth} m -> ratio "
          f"{med / args.truth:.2f}  ({100 * (med / args.truth - 1):+.0f}%)")
    h = float(args.object_height or 1.73)
    rise = h - float(args.camera_height)
    if rise > 0.05:
        want = math.degrees(math.atan2(rise, float(args.truth)))
        tops = [o["bbox"][1] for o in got if o.get("bbox")]
        us = [(o["bbox"][0] + o["bbox"][2]) / 2 for o in got if o.get("bbox")]
        if tops:
            raw = intr._camera_angles(st.median(us), st.median(tops))[1]
            print(f"    the tilt this needs  {want - raw:.2f} deg; the "
                  f"calibration says {intr.tilt_up_deg:.2f} deg")
            if abs((want - raw) - intr.tilt_up_deg) > 1.5:
                print("    Those disagree by more than a degree and a half. "
                      "Re-measure the tilt against")
                print("    the wall before trusting any range from this "
                      "camera.")
                return True
    return False


# ─────────────────────────────── latency ─────────────────────────────────

def measure_flash_latency(read_brightness: Callable[[], Optional[float]],
                          show: Callable[[int], None],
                          clock: Callable[[], float],
                          sleep: Callable[[float], None],
                          flips: int = 10, settle_s: float = 1.0,
                          timeout_s: float = 3.0, think_s: float = 0.0,
                          rng: Optional[random.Random] = None
                          ) -> Dict[str, Any]:
    """
    Flash the screen and time how long the change takes to appear in a frame.

    `show(1)` turns the screen white and `show(0)` black; `read_brightness()`
    blocks for the next frame and returns its mean brightness. Each flip is
    timed from the call to show() to the return of the first read() whose
    frame has crossed the midpoint.

    `think_s` sleeps between reads, the way an inference would. It is there
    to expose a capture backend that QUEUES frames: a reader slower than the
    camera then gets older and older pictures, and the measured delay grows
    by whole frame intervals instead of by about half the sleep.
    """
    rng = rng or random.Random(1)

    def level(target: int) -> Optional[float]:
        show(target)
        end = clock() + settle_s
        vals: List[float] = []
        while clock() < end:
            b = read_brightness()
            if b is not None:
                vals.append(b)
        tail = vals[len(vals) // 2:]
        return st.median(tail) if tail else None

    black = level(0)
    white = level(1)
    if black is None or white is None:
        return {"ok": False, "why": "no frames arrived"}
    if white - black < 25.0:
        return {"ok": False, "black": black, "white": white,
                "why": "the camera does not see the screen change"}
    mid = (black + white) / 2.0

    state = 1
    delays: List[float] = []
    missed = 0
    for _ in range(flips):
        sleep(0.25 + 0.35 * rng.random())     # break any lock to frame phase
        target = 0 if state == 1 else 1
        show(target)
        t0 = clock()
        hit = None
        while clock() - t0 < timeout_s:
            b = read_brightness()
            t = clock()
            if b is not None and ((target == 1 and b > mid)
                                  or (target == 0 and b < mid)):
                hit = t - t0
                break
            if think_s > 0:
                sleep(think_s)
        state = target
        if hit is None:
            missed += 1
        else:
            delays.append(hit)
    if not delays:
        return {"ok": False, "black": black, "white": white,
                "why": "no flip was seen within the timeout"}
    delays.sort()
    return {"ok": True, "black": black, "white": white, "n": len(delays),
            "missed": missed, "median_s": st.median(delays),
            "min_s": delays[0], "max_s": delays[-1], "think_s": think_s}


def run_latency(args: argparse.Namespace) -> int:
    from calibrate_camera import (describe_capture, have_gui, open_capture,
                                  require_cv2)
    cv2 = require_cv2()
    if not have_gui(cv2):
        print("  This test has to draw on the screen, and this OpenCV build "
              "has no window support.")
        print("  pip install opencv-python  (not the -headless build)")
        return 2
    import numpy as np
    # Test the capture mode the detector will use. The card's own default is
    # 1920x1080, and a first run of this test measured that mode while the
    # detector was running at the calibrated 720x576.
    w, h = args.width, args.height
    if (w is None or h is None) and args.intrinsics:
        intr = CameraIntrinsics.load(args.intrinsics)
        w, h = intr.width_px, intr.height_px
    cap = open_capture(cv2, args.camera, w, h)
    info = describe_capture(cv2, cap)
    print(f"  capture        {info['width']}x{info['height']} via "
          f"{info['backend']}")
    if w is None:
        print("  note: this is the capture card's default mode. The "
              "detector runs at the calibrated size;")
        print("        add --intrinsics camera.json to time that mode "
              "instead.")
    win = "latency test - point the drone camera at this window"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1200, 800)
    imgs = {0: np.zeros((400, 600, 3), np.uint8),
            1: np.full((400, 600, 3), 255, np.uint8)}

    def show(v: int) -> None:
        cv2.imshow(win, imgs[v])
        cv2.waitKey(1)

    def read_brightness() -> Optional[float]:
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        centre = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
        return float(centre.mean())

    print("\n  Point the drone's camera at the window that just opened so "
          "it fills most of the picture.")
    print("  It will flash. Starting in 3 seconds; keep the camera still.")
    show(0)
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end:
        read_brightness()

    results = []
    for think in (0.0, args.think_ms / 1000.0):
        r = measure_flash_latency(read_brightness, show, time.monotonic,
                                  time.sleep, flips=args.flips,
                                  think_s=think)
        results.append(r)
        if not r["ok"]:
            break
    cap.release()
    cv2.destroyAllWindows()

    fast = results[0]
    if not fast["ok"]:
        print(f"\n  The test could not run: {fast['why']}.")
        if "black" in fast:
            print(f"  Brightness was {fast['black']:.0f} with the window "
                  f"black and {fast['white']:.0f} with it white (of 255).")
            print("  Move the camera closer so the window fills the frame, "
                  "and dim the room if the picture is washed out.")
        return 1
    print(f"\n  SCREEN TO FRAME   median {1000 * fast['median_s']:.0f} ms   "
          f"(min {1000 * fast['min_s']:.0f}, max {1000 * fast['max_s']:.0f}, "
          f"{fast['n']} flips"
          + (f", {fast['missed']} missed" if fast["missed"] else "") + ")")
    print("    This is how old a picture already is when read() hands it "
          "over: this screen's own lag,")
    print("    the camera, the video link, the goggles and the capture "
          "card. The screen's share is")
    print("    typically 10-40 ms, so treat the figure as an upper bound "
          "on the rest.")
    print(f"    The perception gate accepts a scene up to "
          f"{args.max_age:g} s after read(); add this figure to that to "
          f"get the")
    print("    real age of what a skill acts on.")
    if len(results) > 1 and results[1]["ok"]:
        slow = results[1]
        extra = slow["median_s"] - fast["median_s"]
        print(f"\n  WITH A {args.think_ms:g} ms PAUSE BETWEEN READS (as an "
              f"inference would add)   median "
              f"{1000 * slow['median_s']:.0f} ms, "
              f"{1000 * extra:+.0f} ms on the figure above")
        if extra > slow["think_s"] + 0.06:
            print("    That is more than the pause itself. The capture is "
                  "QUEUING frames: a detector slower than")
            print("    the camera is being handed old pictures. Tell me "
                  "this number; the fix is a reader thread")
            print("    that keeps only the newest frame.")
            return 1
        print("    About what the pause alone explains, so read() is "
              "returning the newest frame, not a queued one.")
    return 0


# ───────────────────────────────── main ──────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["watch", "latency"])
    ap.add_argument("--camera", default="0", help="capture index or path")
    ap.add_argument("--width", type=int,
                    help="capture width (default: the calibrated one)")
    ap.add_argument("--height", type=int)
    ap.add_argument("--intrinsics", help="camera.json from calibrate_camera")
    ap.add_argument("--hfov", type=float, default=90.0,
                    help="assumed field of view when there is no "
                         "--intrinsics")
    ap.add_argument("--detector", choices=["yoloe", "synthetic"],
                    default="yoloe")
    ap.add_argument("--model", default="yoloe-11s-seg.pt",
                    help="YOLOE weights; s is the small, fast one")
    ap.add_argument("--classes",
                    help="comma-separated things to look for, in plain "
                         "words: person,chair,bottle")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detector confidence floor")
    ap.add_argument("--min-conf", type=float, default=0.35,
                    help="confidence floor the stack applies afterwards")
    ap.add_argument("--device", help="cpu, mps, ...; default lets the "
                                     "library choose")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--max-age", type=float, default=0.6,
                    help="how old a scene may be and still count as fresh")
    ap.add_argument("--expect",
                    help="bearing check against a tape measure, degrees "
                         "right of the camera axis: chair:30,person:-15")
    ap.add_argument("--camera-height", type=float,
                    help="BENCH: the lens's height above the floor, in "
                         "metres, measured with a tape. Given rather than "
                         "sensed, because on a stand there is no "
                         "rangefinder and the point is to remove one "
                         "unknown from the range measurement")
    ap.add_argument("--truth", type=float,
                    help="BENCH: the tape-measured distance to the target")
    ap.add_argument("--object-height", type=float,
                    help="BENCH: how tall the target really is, in metres")
    ap.add_argument("--target-class", default="person",
                    help="BENCH: which class the measured target is")
    ap.add_argument("--expect-range",
                    help="range check against a tape measure, metres: "
                         "chair:2.0")
    ap.add_argument("--snapshot-every", type=float, default=2.0,
                    help="save an annotated frame this often, s (0 = off)")
    ap.add_argument("--out-dir", default="logs")
    ap.add_argument("--force", action="store_true",
                    help="run even if the capture size does not match the "
                         "calibration")
    ap.add_argument("--flips", type=int, default=10)
    ap.add_argument("--think-ms", type=float, default=120.0,
                    help="latency: pause between reads in the second pass")
    ap.add_argument("--version", action="version",
                    version=f"drone_agent {__version__}")
    args = ap.parse_args()

    print(f"drone_agent {__version__}")
    if args.mode == "latency":
        return run_latency(args)
    return run_watch(args)


if __name__ == "__main__":
    sys.exit(main())
