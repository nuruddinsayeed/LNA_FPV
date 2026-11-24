"""Perception: frames in, metric tracked observations out.

Nothing here imports a vision library at package level, so the stack stays
importable (and offline_check stays runnable) on a machine with no camera and
no torch. The YOLOE adapter is imported only when used.
"""

from .geometry import (Boresight, CameraIntrinsics, RangeEstimator,  # noqa
                       SIZE_PRIORS, observation_from_bbox)
from .source import (Detection, Detector, MockDetector,  # noqa
                     PerceptionRunner, TrackBook)
from .types import (EMPTY_SCENE, ObjectObservation, RangeSource,  # noqa
                    SceneSnapshot, bearing_to_body_xy, wrap180)

__all__ = ["CameraIntrinsics", "Boresight", "RangeEstimator", "SIZE_PRIORS",
           "observation_from_bbox", "Detection", "Detector", "MockDetector",
           "PerceptionRunner", "TrackBook", "ObjectObservation",
           "RangeSource", "SceneSnapshot", "EMPTY_SCENE",
           "bearing_to_body_xy", "wrap180"]
