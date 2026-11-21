"""
WorldState — the single immutable snapshot every layer above the vehicle
reads from.

This is deliberately the only thing skills, the arbiter and (later) the LLM
planner are allowed to look at. Nothing above the vehicle interface touches
pymavlink directly, which is what makes the replay harness possible: feed a
recorded sequence of WorldState snapshots back through the same skill and
arbiter code and you get the same decisions, with no vehicle attached.

When perception arrives it extends this object (tracked objects, obstacle
sectors, scene text). The planner's serialised view is built from here, so
the interface contract stays in one place.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Pose:
    x: float                 # NED north, m
    y: float                 # NED east, m
    z: float                 # NED down, m (negative = above origin)
    yaw_rad: float
    vz: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw_rad)

    @property
    def alt_m(self) -> float:
        """Altitude above the EKF origin."""
        return -self.z

    def horizontal_distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass
class WorldState:
    """One tick's view of the world. Built fresh by the vehicle each tick."""

    t_mono: float                               # monotonic, the one clock
    t_wall: float                               # unix, for human correlation

    pose: Optional[Pose] = None
    pose_age_s: float = math.inf

    armed: bool = False
    mode: Optional[str] = None
    mode_num: Optional[int] = None
    landed_state: Optional[int] = None
    landed_state_age_s: float = math.inf

    # Rangefinders — value plus whether it can be believed.
    rngfnd_down_m: Optional[float] = None
    rngfnd_down_trusted: bool = False
    rngfnd_down_raw_m: Optional[float] = None   # even when distrusted
    rngfnd_down_offset_m: Optional[float] = None    # learned RF-EKF offset
    rngfnd_down_residual_m: Optional[float] = None   # this sample's residual
    rngfnd_down_allowed_m: Optional[float] = None    # threshold used, vz-scaled
    rngfnd_fwd_m: Optional[float] = None
    rngfnd_fwd_trusted: bool = False

    flow_quality: Optional[int] = None

    link_up: bool = False
    link_age_s: float = math.inf
    rssi: Optional[int] = None
    lq: Optional[int] = None
    uplink_bytes: int = 0

    battery_pct: Optional[float] = None
    battery_v: Optional[float] = None

    # Origin captured at handover; everything geofenced is relative to it.
    origin: Optional[Tuple[float, float, float]] = None

    # Filled in by perception later. Present now so the schema is stable and
    # the planner's serialiser does not change shape when vision lands.
    objects: List[Dict[str, Any]] = field(default_factory=list)
    scene: Optional[str] = None

    # ------------------------------------------------------------------

    @property
    def altitude_m(self) -> Optional[float]:
        """
        Best available altitude. The rangefinder is preferred only while
        trusted; once it has misbehaved we fall back to the EKF. Returns None
        when nothing is usable — callers must treat None as unknown, never as
        zero. That distinction is what stopped a bad reading becoming a disarm
        decision.
        """
        if self.rngfnd_down_trusted and self.rngfnd_down_m is not None:
            return self.rngfnd_down_m
        return self.pose.alt_m if self.pose is not None else None

    @property
    def altitude_source(self) -> str:
        if self.rngfnd_down_trusted and self.rngfnd_down_m is not None:
            return "rangefinder"
        return "ekf" if self.pose is not None else "none"

    @property
    def pose_fresh(self) -> bool:
        return self.pose is not None and self.pose_age_s < 1.0

    def distance_from_origin(self) -> Optional[float]:
        if self.pose is None or self.origin is None:
            return None
        return self.pose.horizontal_distance_to(self.origin[0], self.origin[1])

    # ------------------------------------------------------------------

    def ground_evidence(self, ground_alt_m: float,
                        ground_vz_max: float) -> List[str]:
        """
        Independent signals that we are on the ground.

        landed_state is the strong one: Copter derives it from
        ap.land_complete, the autopilot's own land detector (throttle, climb
        rate, attitude). It does not consult the rangefinder, which is exactly
        why it is the signal to lean on after a rangefinder has misbehaved.
        """
        ev: List[str] = []
        if self.landed_state == 1 and self.landed_state_age_s < 5.0:
            ev.append("FC land_complete")        # MAV_LANDED_STATE_ON_GROUND
        if (self.rngfnd_down_trusted and self.rngfnd_down_m is not None
                and self.rngfnd_down_m < ground_alt_m):
            ev.append(f"rangefinder {self.rngfnd_down_m:.2f}m")
        if self.pose is not None and self.pose.vz is not None:
            if (self.pose.alt_m < ground_alt_m
                    and abs(self.pose.vz) < ground_vz_max):
                ev.append(f"EKF {self.pose.alt_m:.2f}m, vz {self.pose.vz:.2f}")
        return ev

    def on_ground(self, ground_alt_m: float,
                  ground_vz_max: float) -> Tuple[bool, List[str]]:
        ev = self.ground_evidence(ground_alt_m, ground_vz_max)
        # The FC's own detector alone is enough; otherwise require two weak
        # signals to agree, so one bad sensor cannot fake a touchdown.
        if "FC land_complete" in ev:
            return True, ev
        return len(ev) >= 2, ev

    # ------------------------------------------------------------------

    def to_log(self) -> Dict[str, Any]:
        """Flat-ish dict for the JSONL recorder."""
        # Deliberately NOT "t_mono": that key belongs to the recorder and
        # means "when this record was written". These two mean "when this
        # state was sampled", and the difference between them is logging
        # latency — which you want to be able to measure, not lose to a dict
        # key collision.
        d: Dict[str, Any] = {
            "t_sample_mono": round(self.t_mono, 4),
            "t_sample_wall": round(self.t_wall, 4),
            "armed": self.armed,
            "mode": self.mode,
            "mode_num": self.mode_num,
            "landed_state": self.landed_state,
            "landed_state_age_s": (round(self.landed_state_age_s, 3)
                                   if math.isfinite(self.landed_state_age_s)
                                   else None),
            "pose_age_s": (round(self.pose_age_s, 3)
                           if math.isfinite(self.pose_age_s) else None),
            "rngfnd_down": self.rngfnd_down_m,
            "rngfnd_down_trusted": self.rngfnd_down_trusted,
            "rngfnd_down_raw": self.rngfnd_down_raw_m,
            "rngfnd_down_offset": self.rngfnd_down_offset_m,
            "rngfnd_down_residual": self.rngfnd_down_residual_m,
            "rngfnd_down_allowed": self.rngfnd_down_allowed_m,
            "rngfnd_fwd": self.rngfnd_fwd_m,
            "rngfnd_fwd_trusted": self.rngfnd_fwd_trusted,
            "flow_quality": self.flow_quality,
            "alt_m": (round(self.altitude_m, 3)
                      if self.altitude_m is not None else None),
            "alt_src": self.altitude_source,
            "link": {
                "up": self.link_up,
                "age_s": (round(self.link_age_s, 3)
                          if math.isfinite(self.link_age_s) else None),
                "rssi": self.rssi,
                "lq": self.lq,
                "uplink_bytes": self.uplink_bytes,
            },
            "batt": {"pct": self.battery_pct, "v": self.battery_v},
            "origin": (list(self.origin) if self.origin is not None else None),
        }
        if self.pose is not None:
            d["pose"] = {
                "x": round(self.pose.x, 3),
                "y": round(self.pose.y, 3),
                "z": round(self.pose.z, 3),
                "yaw_deg": round(self.pose.yaw_deg, 2),
                "vz": self.pose.vz,
                "pitch_deg": self.pose.pitch_deg,
                "roll_deg": self.pose.roll_deg,
            }
        else:
            d["pose"] = None
        if self.objects:
            d["objects"] = self.objects
        if self.scene:
            d["scene"] = self.scene
        return d

    def to_planner_view(self) -> Dict[str, Any]:
        """
        The serialisation the LLM will eventually consume. Metric, compact,
        and carrying only what a mission-level decision needs — no raw
        detections, no pixel coordinates, no trust internals.

        Nothing calls this yet; it is here so the interface contract is fixed
        before the planner is written against it.
        """
        view: Dict[str, Any] = {
            "armed": self.armed,
            "mode": self.mode,
            "altitude_m": (round(self.altitude_m, 2)
                           if self.altitude_m is not None else None),
            "position_known": self.pose_fresh,
            "objects": self.objects,
        }
        if self.pose is not None:
            view["pose"] = {
                "x": round(self.pose.x, 2),
                "y": round(self.pose.y, 2),
                "yaw_deg": round(self.pose.yaw_deg, 1),
            }
        if self.rngfnd_fwd_m is not None and self.rngfnd_fwd_trusted:
            # One 1-2 deg beam. Reported as what it is: a boresight range, not
            # an obstacle map.
            view["boresight_range_m"] = round(self.rngfnd_fwd_m, 2)
        if self.scene:
            view["scene"] = self.scene
        if self.battery_pct is not None:
            view["battery_pct"] = self.battery_pct
        return view


def from_log(rec: Dict[str, Any]) -> WorldState:
    """
    Rebuild a WorldState from a recorded tick. The inverse of
    WorldState.to_log(), and the reason replay is possible at all: the log is
    a complete serialisation of everything the skills and the arbiter were
    allowed to see, so replaying it reproduces their decisions exactly.
    """
    p = rec.get("pose")
    pose = None
    if p:
        pose = Pose(x=p["x"], y=p["y"], z=p["z"],
                    yaw_rad=math.radians(p["yaw_deg"]),
                    vz=p.get("vz"), pitch_deg=p.get("pitch_deg"),
                    roll_deg=p.get("roll_deg"))
    link = rec.get("link") or {}
    batt = rec.get("batt") or {}
    return WorldState(
        t_mono=rec.get("t_sample_mono", rec.get("t_mono", 0.0)),
        t_wall=rec.get("t_sample_wall", rec.get("t_wall", 0.0)),
        pose=pose,
        pose_age_s=_or_inf(rec.get("pose_age_s")),
        armed=bool(rec.get("armed", False)),
        mode=rec.get("mode"),
        mode_num=rec.get("mode_num"),
        landed_state=rec.get("landed_state"),
        landed_state_age_s=_or_inf(rec.get("landed_state_age_s")),
        rngfnd_down_m=rec.get("rngfnd_down"),
        rngfnd_down_trusted=bool(rec.get("rngfnd_down_trusted", False)),
        rngfnd_down_raw_m=rec.get("rngfnd_down_raw"),
        rngfnd_down_offset_m=rec.get("rngfnd_down_offset"),
        rngfnd_down_residual_m=rec.get("rngfnd_down_residual"),
        rngfnd_down_allowed_m=rec.get("rngfnd_down_allowed"),
        rngfnd_fwd_m=rec.get("rngfnd_fwd"),
        rngfnd_fwd_trusted=bool(rec.get("rngfnd_fwd_trusted", False)),
        flow_quality=rec.get("flow_quality"),
        link_up=bool(link.get("up", True)),
        link_age_s=_or_inf(link.get("age_s")),
        rssi=link.get("rssi"), lq=link.get("lq"),
        uplink_bytes=link.get("uplink_bytes", 0),
        battery_pct=batt.get("pct"), battery_v=batt.get("v"),
        objects=rec.get("objects") or [],
        scene=rec.get("scene"),
        origin=(tuple(rec["origin"]) if rec.get("origin") else None),
    )


def _or_inf(v: Any) -> float:
    return math.inf if v is None else float(v)


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
