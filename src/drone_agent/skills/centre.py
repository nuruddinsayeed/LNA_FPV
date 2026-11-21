"""
Centring -- keep a tracked object on the nose with small heading corrections
that the camera measures.

Why it exists (2026-11-19). This aircraft turns without its EKF knowing.
Over the four flights of 17:59-18:16, with the EKF heading held steady, the
person's bearing walked by 2-4 deg/s more than the EKF's own motion
explains: -3.6, -3.8 and -4.2 deg/s over the three approach steps at 18:14,
-2.2, -3.4 and -4.6 deg/s at 18:16, and both approaches ended with the
person 30-41 deg off the nose and leaving the edge of the picture. At 18:12
face_target ended "still -7 deg off after 4 turns": each turn took out most
of the error, and each 2 s settle after it let 6-9 deg back in.

Not commanding yaw does not keep the heading still; it only leaves the
change uncorrected. The one measurement of that change this stack has is
where the camera sees the target, so the correction is closed on that and
never on the EKF heading.

How. A correction sets the commanded heading to the heading the aircraft
had when the picture was taken, plus a fraction of the bearing seen in it.
The picture is about a quarter of a second old, so the heading is looked up
at that time, not now: otherwise a correction still turning would be
counted twice. Corrections are at least PERIOD_S apart, never larger than
MAX_STEP_DEG, and not made inside DEADBAND_DEG, so the box's own jitter does
not turn into hunting, and a target that walks away across the picture is
not chased. They ride on the position target the skill already
sends, so there is no CONDITION_YAW. Two position targets a second is about
130 B/s, under a fifth of the 735 B/s uplink.

With a fraction GAIN of the bearing removed every PERIOD_S, a steady unseen
turn of r deg/s leaves about r * PERIOD_S / GAIN behind: 2.5 deg at the
3 deg/s of 2026-11-19.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from ..world.state import WorldState, wrap_pi


class Centring:
    """Heading corrections from where the camera sees one tracked object."""

    PERIOD_S = 0.5            # at most two corrections a second
    GAIN = 0.6                # of the bearing, per correction
    DEADBAND_DEG = 2.0        # inside this, leave it: box jitter, not a turn
    # One correction is never a big turn, and together with PERIOD_S this
    # caps what it will follow at 8 deg/s. The aircraft's own unseen turn
    # was 2-4.6 deg/s; a person walking across the nose at 2.5 m is 14 deg/s
    # at 0.6 m/s. Following a moving target is not what centring is for:
    # one that outruns it leaves the cone, and approach stops on that.
    MAX_STEP_DEG = 4.0
    PICTURE_AGE_S = 0.25      # how old the picture is when a skill sees it
    HISTORY_S = 1.5

    def __init__(self, yaw_rad: float) -> None:
        self.yaw_cmd = float(yaw_rad)      # the heading to command, rad
        self._hist: Deque[Tuple[float, float]] = deque()
        self._t_last = -1e9
        self._last_frame: Optional[Tuple[Any, Any, Any]] = None
        self.corrections = 0
        self.turned_deg = 0.0
        self.largest_deg = 0.0
        self.worst_bearing_deg = 0.0

    def note(self, world: WorldState) -> None:
        """Every tick: remember the heading, to look it up by picture time."""
        if world.pose is None:
            return
        self._hist.append((world.t_mono, world.pose.yaw_rad))
        while self._hist and world.t_mono - self._hist[0][0] > self.HISTORY_S:
            self._hist.popleft()

    def heading_at(self, t: float) -> Optional[float]:
        """The EKF heading at time t (the last sample at or before it)."""
        best = None
        for ts, yaw in self._hist:
            if ts <= t:
                best = yaw
            else:
                break
        if best is None and self._hist:
            best = self._hist[0][1]
        return best

    def watch(self, obj: Optional[Dict[str, Any]]) -> None:
        """Keep the worst bearing seen, for the log."""
        if obj is not None and obj.get("bearing_deg") is not None:
            b = float(obj["bearing_deg"])
            if abs(b) > abs(self.worst_bearing_deg):
                self.worst_bearing_deg = b

    def update(self, world: WorldState,
               obj: Optional[Dict[str, Any]]) -> Optional[float]:
        """Make a correction if one is due. Returns it in degrees, or None
        when nothing changed."""
        self.watch(obj)
        if obj is None or world.pose is None:
            return None
        if obj.get("bearing_deg") is None:
            return None
        if world.t_mono - self._t_last < self.PERIOD_S:
            return None
        # The same picture twice is not a second measurement.
        frame = (obj.get("track_id"), obj.get("age_s"), obj.get("bearing_deg"))
        if frame == self._last_frame:
            return None
        b = float(obj["bearing_deg"])
        if abs(b) <= self.DEADBAND_DEG:
            return None
        then = self.heading_at(world.t_mono - self.PICTURE_AGE_S)
        if then is None:
            then = world.pose.yaw_rad
        corr = max(-self.MAX_STEP_DEG, min(self.MAX_STEP_DEG, self.GAIN * b))
        self.yaw_cmd = wrap_pi(then + math.radians(corr))
        self._t_last = world.t_mono
        self._last_frame = frame
        self.corrections += 1
        self.turned_deg += corr
        self.largest_deg = max(self.largest_deg, abs(corr))
        return corr

    def to_log(self) -> Dict[str, Any]:
        return {"corrections": self.corrections,
                "turned_deg": round(self.turned_deg, 1),
                "largest_deg": round(self.largest_deg, 1),
                "worst_bearing_deg": round(self.worst_bearing_deg, 1)}
