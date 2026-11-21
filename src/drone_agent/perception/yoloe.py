"""
YOLOE adapter — open-vocabulary detection with persistent track ids.

Imported only when used, so the stack stays installable on a machine with no
torch. NOT exercised by offline_check: it needs a camera and a model, so the
plumbing is tested through MockDetector and this file is the thin edge that
has to be verified in flight.

Why open-vocabulary matters here: Baig et al. and the Phadte repo both bound
the LLM to a fixed class list enumerated in the prompt, which is why "find
another drone" is impossible for them — COCO has no drone class. With YOLOE
the class list is set at runtime, so a planner can name a target the system
was never trained on, and that is a real capability difference rather than a
tuning choice.

Tracking: `model.track(persist=True, ...)` keeps ids across frames
(ByteTrack/BoT-SORT). Ids are what make "follow *them*" meaningful rather
than "follow whatever is currently biggest", and PerceptionRunner's TrackBook
turns them into age and frame counts.

Timestamps: the capture time must be when the FRAME was grabbed, not when
inference finished. On an analog path through a capture card the two differ by
tens of milliseconds, and using the wrong one makes a stale detection look
fresh. VideoCapture gives no reliable timestamp, so the grab time is taken
immediately after `read()` returns and the inference duration is reported
separately.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence, Tuple

from .source import Detection


class YoloeDetector:
    """
    Wraps an Ultralytics YOLOE model and a capture device.

    Construct on the thread that will call detect(); some backends dislike
    being moved between threads.
    """

    name = "yoloe"

    def __init__(self, model_path: str = "yoloe-11l-seg.pt",
                 source: Any = 0,
                 classes: Optional[Sequence[str]] = None,
                 imgsz: int = 640,
                 conf: float = 0.25,
                 device: Optional[str] = None,
                 tracker: str = "bytetrack.yaml",
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 warmup: bool = True) -> None:
        try:
            import cv2
            from ultralytics import YOLOE
        except ImportError as e:        # pragma: no cover - needs torch
            raise ImportError(
                "YoloeDetector needs ultralytics and opencv-python. The rest "
                "of the stack does not, which is why this import is here and "
                "not at package level.") from e

        self._cv2 = cv2
        self.model = YOLOE(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.tracker = tracker
        # 'mps' on Apple Silicon; a CoreML export is usually faster still and
        # leaves more unified memory for a local LLM.
        self.device = device
        self.classes: List[str] = list(classes or [])
        if self.classes:
            self.set_classes(self.classes)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open capture source {source!r}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        # Published so PerceptionRunner can check it against the calibration.
        # Boxes leave this class NORMALISED, which hides the frame geometry
        # from everything downstream -- so the one place that still knows it
        # has to say so.
        self.capture_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frames = 0
        self.dropped = 0
        # The most recent frame, kept by reference so a ground tool can save
        # an annotated picture of what the detector was looking at. Nothing
        # in flight reads it.
        self.last_frame: Any = None

        if warmup:
            self._warmup()

    # ---- open vocabulary ---------------------------------------------

    def set_classes(self, classes: Sequence[str]) -> None:
        """
        Change the class list at runtime. This is the capability the fixed-
        prompt implementations lack; a planner asking to find a "drone" sets
        it here rather than being limited to what the model was trained on.
        """
        self.classes = list(classes)
        # The call changed between Ultralytics releases. Current
        # documentation shows set_classes(names) alone; older releases
        # required the text embeddings as a second argument. Try the current
        # form and fall back, so the adapter does not pin a version.
        try:
            self.model.set_classes(self.classes)
        except TypeError:
            self.model.set_classes(
                self.classes, self.model.get_text_pe(self.classes))

    def _warmup(self) -> None:
        ok, frame = self.cap.read()
        if ok:
            try:
                self.model.predict(frame, imgsz=self.imgsz, verbose=False,
                                   device=self.device)
            except Exception:
                pass

    # ---- the Detector protocol ---------------------------------------

    def detect(self) -> Tuple[float, List[Detection], Optional[float]]:
        ok, frame = self.cap.read()
        t_cap = time.monotonic()        # as close to the grab as we can get
        if not ok or frame is None:
            self.dropped += 1
            return t_cap, [], None

        self.frames += 1
        self.last_frame = frame
        h, w = frame.shape[:2]
        t0 = time.monotonic()
        try:
            results = self.model.track(
                frame, imgsz=self.imgsz, conf=self.conf, persist=True,
                tracker=self.tracker, verbose=False, device=self.device)
        except Exception:
            # A single bad inference must not take the thread down; the
            # runner counts errors and the tick loop sees a stale scene,
            # which is the honest outcome.
            raise
        inference_ms = (time.monotonic() - t0) * 1000.0

        dets: List[Detection] = []
        if not results:
            return t_cap, dets, inference_ms
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return t_cap, dets, inference_ms

        names = getattr(results[0], "names", None) or {}
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            cls_i = int(boxes.cls[i].item()) if boxes.cls is not None else -1
            label = names.get(cls_i, str(cls_i))
            confidence = (float(boxes.conf[i].item())
                          if boxes.conf is not None else 0.0)
            tid = None
            if getattr(boxes, "id", None) is not None:
                tid = int(boxes.id[i].item())
            # Normalise to 0..1 so a change of capture resolution does not
            # silently invalidate intrinsics or stored observations.
            dets.append(Detection(
                label=label,
                bbox=(xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h),
                confidence=confidence, track_id=tid))
        return t_cap, dets, inference_ms

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass

    def health(self) -> dict:
        # The device actually in use, which is not the one asked for when
        # the request was "choose for me".
        used = None
        try:
            used = str(getattr(self.model, "device", None)
                       or next(self.model.model.parameters()).device)
        except Exception:
            pass
        return {"frames": self.frames, "dropped": self.dropped,
                "classes": self.classes, "device": self.device,
                "device_used": used, "imgsz": self.imgsz,
                "capture_size": list(self.capture_size)}
