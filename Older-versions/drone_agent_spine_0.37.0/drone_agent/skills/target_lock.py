"""
TargetLock — which of the things the detector reports is "the target".

Two flights on 2026-11-16 turned the aircraft the wrong way and lost the
person. In both, the detector was reporting two "person" boxes at the same
time, one near +45 deg and one near -45 deg, and the search and the skill
after it disagreed about which one was meant:

    search:  stopped on track 16 at +48 deg  (range uncertainty 0.32 m)
    face:    turned toward track 4 at -39 deg (range uncertainty 2.0 m)

The search had applied its acceptance rule and rejected the left-hand box as
too uncertain to be a real, rangeable object. The skill after it had no such
rule: it took whichever box was nearest the nose at that instant, which is
not a property of the world at all. Worse, when the two boxes were nearly
symmetric the nearest-the-nose choice flipped between them from frame to
frame, so a median over several frames mixed bearings to two different
objects and meant nothing.

A lock fixes both. The search hands over the track id it accepted; the
skills that follow track that id and nothing else. When the detector's
tracker drops an id and issues a new one, which it does across almost every
turn, the lock re-acquires by direction: a box that appears where the locked
target should now be, allowing for how far the aircraft has turned since,
is the same object; a box 87 deg away is not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..world.state import WorldState


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class TargetLock:
    """Follows one object of one class across frames and across turns."""

    RE_ACQUIRE_DEG = 15.0      # a new id this close to where it should be
    RE_ACQUIRE_AFTER_S = 0.8   # ... counts as the same object, after a gap
    MIN_FRAMES = 3

    def __init__(self, label: str, track_id: Optional[int] = None,
                 bearing_deg: Optional[float] = None,
                 heading_deg: Optional[float] = None,
                 max_range_sigma_m: float = 0.6) -> None:
        self.label = label
        self.track_id = track_id
        self.max_range_sigma_m = max_range_sigma_m
        # Where the handed-over target was, and which way the aircraft was
        # pointing when it was there. Enough to re-acquire it even if the
        # tracker has already dropped the id by the time we first look.
        self.last_bearing = bearing_deg
        self.last_heading = heading_deg
        self.t_last_seen: Optional[float] = None
        self.switches = 0
        self.from_handover = track_id is not None

    # ------------------------------------------------------------------

    def candidates(self, world: WorldState) -> List[Dict[str, Any]]:
        out = []
        for obj in world.objects or []:
            if (obj.get("label") != self.label
                    or obj.get("track_id") is None
                    or obj.get("bearing_deg") is None
                    or obj.get("frames_seen", 0) < self.MIN_FRAMES):
                continue
            out.append(obj)
        return out

    def _rangeable(self, obj: Dict[str, Any]) -> bool:
        """The search's own test for a box worth acting on."""
        sigma = obj.get("range_sigma_m")
        return (obj.get("range_m") is not None and sigma is not None
                and sigma <= self.max_range_sigma_m)

    def _expected_bearing(self, world: WorldState) -> Optional[float]:
        if self.last_bearing is None or self.last_heading is None \
                or world.pose is None:
            return None
        turned = _wrap(world.pose.yaw_deg - self.last_heading)
        return _wrap(self.last_bearing - turned)

    # ------------------------------------------------------------------

    def pick(self, world: WorldState) -> Optional[Dict[str, Any]]:
        """
        This tick's sighting of the locked target, or None. Call it every
        tick: it is what keeps the lock's idea of where the target is
        up to date.
        """
        if world.pose is None:
            return None
        cands = self.candidates(world)
        chosen: Optional[Dict[str, Any]] = None

        if self.track_id is None:
            # Nothing handed over. Choose as the search would: a box with a
            # range it is sure enough of, nearest the nose; only if there is
            # no such box fall back to the nearest of the rest.
            pool = [o for o in cands if self._rangeable(o)] or cands
            if pool:
                chosen = min(pool, key=lambda o: abs(o["bearing_deg"]))
                self.track_id = chosen["track_id"]
        else:
            chosen = next((o for o in cands
                           if o["track_id"] == self.track_id), None)
            if chosen is None:
                # The id is gone. After a moment, take a box that is where
                # this one should be by now.
                stale = (self.t_last_seen is None
                         or world.t_mono - self.t_last_seen
                         >= self.RE_ACQUIRE_AFTER_S)
                want = self._expected_bearing(world)
                if stale and want is not None and cands:
                    near = min(cands,
                               key=lambda o: abs(_wrap(o["bearing_deg"] - want)))
                    if abs(_wrap(near["bearing_deg"] - want)) \
                            <= self.RE_ACQUIRE_DEG:
                        chosen = near
                        self.track_id = near["track_id"]
                        self.switches += 1

        if chosen is not None:
            self.last_bearing = float(chosen["bearing_deg"])
            self.last_heading = world.pose.yaw_deg
            self.t_last_seen = world.t_mono
        return chosen

    def lost_for(self, world: WorldState) -> float:
        """Seconds since the target was last seen. inf if never seen."""
        if self.t_last_seen is None:
            return float("inf")
        return world.t_mono - self.t_last_seen

    def to_log(self) -> Dict[str, Any]:
        return {"label": self.label, "track_id": self.track_id,
                "from_handover": self.from_handover,
                "id_changes": self.switches}
