"""
Running perception alongside the control loop without coupling their rates.

The constraint that shapes this: the tick loop must never block on inference.
It runs at 10 Hz and commands an aircraft; a detector that takes 120 ms on one
frame and 400 ms on the next cannot be allowed to set the control rate. So
perception runs in its own thread and publishes an immutable SceneSnapshot;
the tick loop reads whatever the latest one is and checks its age.

That staleness check is not optional. A detection that is 2 s old describes
where something *was*, and the reference implementations' habit of flying at a
bearing with no age check is how a target that has walked away still attracts
the aircraft. Skills ask for `fresh_scene()` and get None rather than a stale
answer.

Track continuity is maintained here, not in the detector: a detector gives
boxes, a tracker gives ids, and this layer turns a stream of ids into
observations with `first_seen` / `frames_seen` so `is_tracked()` means
something.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .geometry import CameraIntrinsics, RangeEstimator, observation_from_bbox
from .types import EMPTY_SCENE, ObjectObservation, SceneSnapshot


@dataclass
class Detection:
    """What a detector returns for one object in one frame. Deliberately
    minimal: a label, a normalised box, a confidence, and a track id if the
    tracker provided one."""

    label: str
    bbox: tuple                 # normalised (x1, y1, x2, y2)
    confidence: float = 0.0
    track_id: Optional[int] = None


class Detector(Protocol):
    """
    Anything that turns a frame into detections. Implementations live in
    their own modules (yoloe.py) so this package imports no vision
    dependencies unless one is actually used — which keeps offline_check
    runnable on a machine with no camera and no torch.
    """

    name: str

    def detect(self) -> tuple:
        """Returns (t_capture_mono, List[Detection], inference_ms)."""
        ...


class MockDetector:
    """
    Scripted detections, for testing the plumbing without a camera.

    This is what lets the tracking, staleness and range-fusion logic be
    regression-tested at all: the alternative is only ever exercising it in
    flight, which is how the papers end up with perception code that has
    never been tested against a known answer.
    """

    name = "mock"

    def __init__(self, script: List[List[Detection]],
                 period_s: float = 0.1,
                 inference_ms: float = 15.0) -> None:
        self.script = script
        self.period_s = period_s
        self.inference_ms = inference_ms
        self._i = 0
        self._t = 0.0

    def detect(self) -> tuple:
        dets = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        self._t += self.period_s
        return self._t, list(dets), self.inference_ms

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self.script)


class TrackBook:
    """
    Turns per-frame detections into observations with history.

    A detector plus tracker gives a track_id per frame. What a skill needs is
    "has this id been visible long enough to act on", which requires
    remembering when each id first appeared and how many frames it has been
    seen in. That is this class, and it is why `is_tracked()` can mean
    something stricter than "detected once".
    """

    def __init__(self, forget_after_s: float = 2.0) -> None:
        self.forget_after_s = forget_after_s
        self._first: Dict[tuple, float] = {}
        self._count: Dict[tuple, int] = {}
        self._last: Dict[tuple, float] = {}

    def update(self, det: Detection, t_mono: float) -> tuple:
        """Returns (first_seen_mono, frames_seen) for this detection."""
        # Key on (label, track_id). A detector with no tracker gives
        # track_id=None, and then every frame looks like a new object — which
        # is correct: without a tracker there is no track to be continuous.
        key = (det.label, det.track_id)
        if det.track_id is None:
            return t_mono, 1
        last = self._last.get(key)
        if last is None or (t_mono - last) > self.forget_after_s:
            self._first[key] = t_mono
            self._count[key] = 0
        self._count[key] = self._count.get(key, 0) + 1
        self._last[key] = t_mono
        return self._first[key], self._count[key]

    def prune(self, t_mono: float) -> None:
        dead = [k for k, t in self._last.items()
                if (t_mono - t) > self.forget_after_s]
        for k in dead:
            self._first.pop(k, None)
            self._count.pop(k, None)
            self._last.pop(k, None)


class PerceptionRunner:
    """
    Drives a detector in a background thread and publishes snapshots.

    The tick loop only ever calls `latest()` or `fresh_scene()`, both of which
    return immediately.
    """

    def __init__(self, detector: Any,
                 intrinsics: Optional[CameraIntrinsics] = None,
                 estimator: Optional[RangeEstimator] = None,
                 max_age_s: float = 0.6,
                 min_confidence: float = 0.35,
                 labels: Optional[List[str]] = None) -> None:
        self.detector = detector
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.estimator = estimator or RangeEstimator(intrinsics=self.intrinsics)
        self.max_age_s = max_age_s
        self.min_confidence = min_confidence
        self.labels = set(labels) if labels else None

        self.books = TrackBook()
        self._scene: SceneSnapshot = EMPTY_SCENE
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame_id = 0
        # A detector normalises its boxes by the frame IT received. If that
        # frame's aspect differs from the calibrated one the whole geometry
        # is silently wrong, so check once at construction and carry the
        # verdict into status() rather than discovering it in the air.
        self.capture_warning: Optional[str] = None
        size = getattr(detector, "capture_size", None)
        if size:
            self.capture_warning = self.intrinsics.check_capture(
                int(size[0]), int(size[1]))
        self.errors = 0
        self.last_error: Optional[str] = None

        # Boresight range, pushed in from the vehicle each tick. Perception
        # does not read MAVLink; the vehicle owns that.
        self._boresight_m: Optional[float] = None
        self._boresight_trusted = False
        # Where the lens is and which way it points, for the head-height
        # range. Same rule: perception does not read MAVLink.
        self._camera_height_m: Optional[float] = None
        self._pitch_deg = 0.0

    # ---- vehicle -> perception ---------------------------------------

    def set_boresight_range(self, range_m: Optional[float],
                            trusted: bool) -> None:
        self._boresight_m = range_m
        self._boresight_trusted = trusted

    def set_camera_pose(self, height_m: Optional[float],
                        pitch_deg: float = 0.0) -> None:
        """
        The lens's height above the floor, from the down rangefinder, and
        the airframe's pitch. Only the trusted rangefinder: the EKF's own
        height reads about 1/1.2 of it on this airframe, and a height that
        is 20% wrong is a range that is wrong by more than that.
        """
        self._camera_height_m = height_m
        self._pitch_deg = pitch_deg

    # ---- the loop ----------------------------------------------------

    def step(self) -> SceneSnapshot:
        """
        One detect-and-publish cycle. Called by the thread, and directly by
        tests so the whole pipeline is exercisable without threading.
        """
        try:
            t_cap, dets, inf_ms = self.detector.detect()
        except Exception as e:                      # never kill the thread
            self.errors += 1
            self.last_error = repr(e)
            return self._scene

        self._frame_id += 1
        objs: List[ObjectObservation] = []
        for d in dets:
            if d.confidence < self.min_confidence:
                continue
            if self.labels is not None and d.label not in self.labels:
                continue
            obs = observation_from_bbox(
                d.label, d.bbox, self.intrinsics, d.confidence, d.track_id,
                t_cap)
            first, count = self.books.update(d, t_cap)
            obs.first_seen_mono = first
            obs.last_seen_mono = t_cap
            obs.frames_seen = count
            self.estimator.estimate(
                obs, boresight_range_m=self._boresight_m,
                boresight_trusted=self._boresight_trusted,
                camera_height_m=self._camera_height_m,
                pitch_deg=self._pitch_deg)
            objs.append(obs)
        self.books.prune(t_cap)

        scene = SceneSnapshot(
            t_capture_mono=t_cap,
            t_published_mono=time.monotonic(),
            objects=objs, frame_id=self._frame_id,
            detector=getattr(self.detector, "name", "?"),
            inference_ms=inf_ms)
        with self._lock:
            self._scene = scene
        return scene

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="perception")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            before = self.errors
            self.step()
            if getattr(self.detector, "exhausted", False):
                break
            if self.errors != before:
                # A camera that has gone away fails instantly. Without a
                # pause this thread spins on the failure and competes with
                # the control loop for the interpreter.
                self._stop.wait(0.05)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---- perception -> the stack -------------------------------------

    def latest(self) -> SceneSnapshot:
        with self._lock:
            return self._scene

    def fresh_scene(self, now: Optional[float] = None
                    ) -> Optional[SceneSnapshot]:
        """
        The latest snapshot if it is recent enough, else None.

        Returning None rather than a stale scene is the whole point: a skill
        that cannot tell fresh from stale will happily fly at a bearing
        measured two seconds ago.
        """
        now = time.monotonic() if now is None else now
        scene = self.latest()
        if scene.t_capture_mono <= 0.0:
            return None
        return scene if scene.age_s(now) <= self.max_age_s else None

    def objects_for_world(self, now: Optional[float] = None
                          ) -> List[Dict[str, Any]]:
        """Serialised objects for WorldState.objects, or [] when stale."""
        scene = self.fresh_scene(now)
        if scene is None:
            return []
        return [o.to_log() for o in scene.objects]

    def health(self) -> Dict[str, Any]:
        scene = self.latest()
        return {
            "detector": getattr(self.detector, "name", "?"),
            "frames": self._frame_id,
            "errors": self.errors,
            "last_error": self.last_error,
            "latency_s": round(scene.pipeline_latency_s, 4),
            "inference_ms": scene.inference_ms,
            "objects": len(scene.objects),
            "intrinsics_calibrated": self.intrinsics.is_calibrated,
            "capture_warning": self.capture_warning,
            "boresight_calibrated": self.estimator.boresight.calibrated,
        }
