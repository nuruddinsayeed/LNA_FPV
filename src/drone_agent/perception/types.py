"""
What perception hands the rest of the stack.

The design rule, and the thing that separates this from the published work:
**every range carries its source and its uncertainty.** Baig et al.'s visual
encoding annotates each detection with a distance taken from a simulated
depth sensor — exact, free, and with no error model. On FPV hardware a range
is an estimate from one of several methods, each wrong in a different way, and
a skill that brakes at 0.8 m needs to know whether that number came from a
laser or from guessing a person's height.

So an ObjectObservation says where its range came from (`range_source`) and
how much to trust it (`range_sigma_m`). A skill or the arbiter can then refuse
to act on a range it does not trust, which is impossible if the number arrives
bare.

Frames
------
bearing_deg  signed, body frame, +right / -left of the camera boresight.
             Not image pixels: the conversion needs real intrinsics, and the
             reference implementations' hardcoded 90 deg HFOV is exactly the
             error this avoids.
elevation_deg same convention, +up.
range_m      straight-line distance from the camera, metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RangeSource(str, Enum):
    """How a range was obtained, worst to best."""

    NONE = "none"                # bearing only, no range at all
    SIZE_PRIOR = "size_prior"    # assumed real-world size + intrinsics
    MONO_DEPTH = "mono_depth"    # monocular estimator, scale-anchored
    HEAD_HEIGHT = "head_height"  # top of the box + rangefinder + camera tilt
    BORESIGHT = "boresight"      # the forward lidar, on the beam
    DEPTH_SENSOR = "depth"       # a real depth camera

    @property
    def is_metric(self) -> bool:
        """Does this source produce a genuinely metric number, or an
        inference that needs a scale anchor?"""
        return self in (RangeSource.BORESIGHT, RangeSource.DEPTH_SENSOR)


@dataclass
class ObjectObservation:
    """One tracked object, as the planner and skills see it."""

    label: str
    track_id: Optional[int] = None
    confidence: float = 0.0

    bearing_deg: float = 0.0
    elevation_deg: float = 0.0

    range_m: Optional[float] = None
    range_source: RangeSource = RangeSource.NONE
    range_sigma_m: Optional[float] = None

    # Bounding box in NORMALISED image coordinates (0..1), so a change of
    # capture resolution does not silently invalidate stored data.
    bbox: Optional[Tuple[float, float, float, float]] = None

    first_seen_mono: float = 0.0
    last_seen_mono: float = 0.0
    frames_seen: int = 0

    @property
    def age_s(self) -> float:
        return max(0.0, self.last_seen_mono - self.first_seen_mono)

    def staleness_s(self, now: float) -> float:
        return max(0.0, now - self.last_seen_mono)

    def is_tracked(self, min_age_s: float = 0.5,
                   min_frames: int = 3) -> bool:
        """A stable track, not one frame's lucky detection. Skills require
        this before acting, because flying at a bearing derived from a single
        bounding box is how the reference implementations behave and it is not
        defensible."""
        return (self.track_id is not None
                and self.age_s >= min_age_s
                and self.frames_seen >= min_frames)

    def has_usable_range(self, max_sigma_m: float = 0.5,
                         require_metric: bool = False) -> bool:
        if self.range_m is None:
            return False
        if require_metric and not self.range_source.is_metric:
            return False
        if self.range_sigma_m is None:
            return False
        return self.range_sigma_m <= max_sigma_m

    def to_log(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "track_id": self.track_id,
            "confidence": round(self.confidence, 3),
            "bearing_deg": round(self.bearing_deg, 2),
            "elevation_deg": round(self.elevation_deg, 2),
            "range_m": (round(self.range_m, 3)
                        if self.range_m is not None else None),
            "range_source": self.range_source.value,
            "range_sigma_m": (round(self.range_sigma_m, 3)
                              if self.range_sigma_m is not None else None),
            "bbox": ([round(v, 4) for v in self.bbox]
                     if self.bbox else None),
            "age_s": round(self.age_s, 2),
            "frames_seen": self.frames_seen,
        }

    def to_planner_view(self) -> Dict[str, Any]:
        """
        What the LLM sees. Deliberately includes range_source: a planner that
        cannot tell a laser measurement from a height guess will treat them
        alike, and the whole point of the interface is that it should not.
        """
        out: Dict[str, Any] = {
            "label": self.label,
            "track_id": self.track_id,
            "bearing_deg": round(self.bearing_deg, 1),
            "age_s": round(self.age_s, 1),
        }
        if self.range_m is not None:
            out["range_m"] = round(self.range_m, 2)
            out["range_source"] = self.range_source.value
            if self.range_sigma_m is not None:
                out["range_uncertainty_m"] = round(self.range_sigma_m, 2)
        else:
            out["range_m"] = None
            out["range_note"] = "bearing only, no range estimate available"
        return out


@dataclass
class SceneSnapshot:
    """
    One perception result. Immutable once published, so the tick loop always
    reads a self-consistent set rather than a half-updated list.

    `t_capture_mono` is when the FRAME was captured, not when inference
    finished. The difference is the pipeline latency, and reporting the wrong
    one makes a stale detection look fresh.
    """

    t_capture_mono: float
    t_published_mono: float
    objects: List[ObjectObservation] = field(default_factory=list)
    scene_text: Optional[str] = None        # the slow VLM tier, if any
    frame_id: int = 0
    detector: str = ""
    inference_ms: Optional[float] = None
    dropped_frames: int = 0

    @property
    def pipeline_latency_s(self) -> float:
        return max(0.0, self.t_published_mono - self.t_capture_mono)

    def age_s(self, now: float) -> float:
        return max(0.0, now - self.t_capture_mono)

    def find(self, label: str,
             track_id: Optional[int] = None) -> Optional[ObjectObservation]:
        best: Optional[ObjectObservation] = None
        for o in self.objects:
            if o.label != label:
                continue
            if track_id is not None and o.track_id != track_id:
                continue
            if best is None or o.confidence > best.confidence:
                best = o
        return best

    def to_log(self) -> Dict[str, Any]:
        return {
            "t_capture_mono": round(self.t_capture_mono, 4),
            "latency_s": round(self.pipeline_latency_s, 4),
            "frame_id": self.frame_id,
            "detector": self.detector,
            "inference_ms": (round(self.inference_ms, 1)
                             if self.inference_ms is not None else None),
            "dropped_frames": self.dropped_frames,
            "objects": [o.to_log() for o in self.objects],
            "scene": self.scene_text,
        }


EMPTY_SCENE = SceneSnapshot(t_capture_mono=0.0, t_published_mono=0.0)


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def bearing_to_body_xy(bearing_deg: float, range_m: float,
                       yaw_rad: float) -> Tuple[float, float]:
    """
    Camera-relative bearing plus range to a world-frame NED offset.

    Kept here rather than in a skill so there is exactly one place where this
    trigonometry lives; getting the sign wrong points the aircraft at a
    mirror image of the target, which is a hard bug to see in flight.
    """
    world = yaw_rad + math.radians(bearing_deg)
    return range_m * math.cos(world), range_m * math.sin(world)
