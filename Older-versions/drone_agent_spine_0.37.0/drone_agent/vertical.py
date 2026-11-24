"""
The vertical story of a flight, from the autopilot's own log.

The ground-station log has told us for two weeks that the EKF reports a
vertical velocity disagreeing with its own altitude: it believes it is
sinking while the rangefinder rises, the controller adds thrust to arrest a
descent that is not happening, and the real height walks up until the
arbiter aborts. Twelve flights have ended that way. What the ground-station
log cannot say is WHY, because it carries no IMU, no vibration and no
estimator-health data.

This module is the arithmetic for answering that from a `.bin`. It is kept
separate from the tool that reads the file so that the arithmetic can be
checked against cases whose answer is known, rather than only against the
one flight it was written for.

The measurement that settles it
-------------------------------
An accelerometer measures specific force, not acceleration. A vehicle
sitting still on the floor, at any attitude, must read exactly -g on the
down axis once its own attitude is used to rotate the reading into the
world frame. So:

  * rotate the body-frame reading into the world's down axis using the
    attitude the autopilot itself logged, which removes tilt -- a hovering
    aircraft leaning 14 deg reads 0.30 m/s2 light on its body Z axis for
    purely geometric reasons, and that is not an error;
  * take a window where the aircraft is demonstrably motionless (on the
    floor, rangefinder at its minimum) but the motors are running;
  * anything other than -9.81 there is the sensor, because the true
    acceleration is zero by construction.

That last window is the control, and it is what turns "the estimator is
biased" into "the accelerometer reads 0.09 m/s2 light at idle and 0.44 m/s2
light at hover, while bolted to a floor it cannot be accelerating through".
"""

from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

G = 9.80665


def down_specific_force(acc: Tuple[float, float, float],
                        roll_deg: float, pitch_deg: float) -> float:
    """
    The body-frame accelerometer reading projected onto the world's down
    axis. For a motionless vehicle this is -g whatever its attitude.

    Only roll and pitch are needed: yaw turns the horizontal axes among
    themselves and leaves the down axis alone.
    """
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    ax, ay, az = acc
    return (az * math.cos(r) * math.cos(p)
            - ax * math.sin(p)
            + ay * math.sin(r) * math.cos(p))


@dataclass
class Window:
    """One stretch of flight, and what the vertical axis did in it."""

    name: str
    t0: float
    t1: float
    n: int = 0
    down_g: float = float("nan")
    vibe_z: float = float("nan")
    rpm: float = float("nan")
    throttle: float = float("nan")
    rangefinder: float = float("nan")
    still: bool = False      # demonstrably not moving: the control

    @property
    def error(self) -> float:
        """How far the down axis is from -g. Positive = reads light, which
        the estimator integrates as a descent."""
        return self.down_g + G

    def to_log(self) -> Dict[str, Any]:
        return {"name": self.name, "t0": round(self.t0, 2),
                "t1": round(self.t1, 2), "n": self.n,
                "down_g": round(self.down_g, 3),
                "error_ms2": round(self.error, 3),
                "vibe_z": round(self.vibe_z, 2), "rpm": round(self.rpm),
                "throttle": round(self.throttle, 3),
                "rangefinder_m": round(self.rangefinder, 2),
                "still": self.still}


def nearest(rows: Sequence[Dict[str, Any]], t: float
            ) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    return min(rows, key=lambda d: abs(d["TimeUS"] - t))


def summarise(imu: Sequence[Dict[str, Any]], att: Sequence[Dict[str, Any]],
              vibe: Sequence[Dict[str, Any]], esc: Sequence[Dict[str, Any]],
              ctun: Sequence[Dict[str, Any]], rfnd: Sequence[Dict[str, Any]],
              name: str, t0_us: float, t1_us: float,
              still: bool = False) -> Window:
    """One window's averages. Times are absolute TimeUS, as logged."""
    def inside(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [d for d in rows if t0_us <= d["TimeUS"] <= t1_us]

    got = inside(imu)
    w = Window(name=name, t0=t0_us, t1=t1_us, n=len(got), still=still)
    if not got:
        return w
    downs = []
    for d in got:
        a = nearest(att, d["TimeUS"])
        downs.append(down_specific_force(
            (d["AccX"], d["AccY"], d["AccZ"]),
            a["Roll"] if a else 0.0, a["Pitch"] if a else 0.0))
    w.down_g = st.mean(downs)
    for field_name, rows, key in (("vibe_z", vibe, "VibeZ"),
                                  ("rpm", esc, "RPM"),
                                  ("throttle", ctun, "ThO"),
                                  ("rangefinder", rfnd, "Dist")):
        vals = [d[key] for d in inside(rows)
                if d.get(key) is not None]
        if vals:
            setattr(w, field_name, st.mean(vals))
    return w


def runs(rows: Sequence[Dict[str, Any]], holds: Any,
         min_s: float = 1.0) -> List[Tuple[float, float]]:
    """
    CONTIGUOUS stretches where `holds(row)` is true, each at least min_s
    long. Contiguous matters: a log can hold two flights with the aircraft
    sitting on the floor between them, and taking the first and last moment
    the aircraft was airborne would average the two flights and the pause
    into one meaningless window. That was the first version of this
    function and it reported a 3.38 m climb for a flight that reached
    1.83 m.
    """
    out: List[Tuple[float, float]] = []
    start = prev = None
    for d in sorted(rows, key=lambda x: x["TimeUS"]):
        if holds(d):
            if start is None:
                start = d["TimeUS"]
            prev = d["TimeUS"]
        elif start is not None:
            if prev - start >= min_s * 1e6:
                out.append((start, prev))
            start = prev = None
    if start is not None and prev - start >= min_s * 1e6:
        out.append((start, prev))
    return out


def longest(spans: Sequence[Tuple[float, float]]
            ) -> Optional[Tuple[float, float]]:
    return max(spans, key=lambda s: s[1] - s[0]) if spans else None


def airborne_spans(rfnd: Sequence[Dict[str, Any]], floor_m: float = 0.12,
                   min_s: float = 2.0) -> List[Tuple[float, float]]:
    """Each separate time the aircraft left the floor."""
    return runs(rfnd, lambda d: (d.get("Dist") or 0.0) > floor_m, min_s)


def airborne_span(rfnd: Sequence[Dict[str, Any]], floor_m: float = 0.12
                  ) -> Optional[Tuple[float, float]]:
    """The longest single flight in the log."""
    return longest(airborne_spans(rfnd, floor_m))


def grounded_spans(rfnd: Sequence[Dict[str, Any]], floor_m: float = 0.12,
                   min_s: float = 2.0) -> List[Tuple[float, float]]:
    """Stretches where the rangefinder sat at the floor for long enough to
    average over. These are the windows where the truth is known."""
    return runs(rfnd, lambda d: (d.get("Dist") or 0.0) <= floor_m, min_s)


def flow_residual(of: Dict[str, Any]) -> Tuple[float, float]:
    """
    What the flow sensor says the aircraft is doing that its gyro does not.

    The sensor measures image motion; the autopilot subtracts the body
    rotation it measured itself. What is left is the aircraft translating.
    On the floor it is translating at exactly zero, so anything left there
    is the sensor, and the estimator believes it: a residual of r rad/s at
    a height of h metres is a velocity of r*h that the aircraft does not
    have, and the position controller flies off sideways to cancel it.
    """
    return (of["flowX"] - of["bodyX"], of["flowY"] - of["bodyY"])


def mean_residual(rows: Sequence[Dict[str, Any]], t0_us: float,
                  t1_us: float) -> Optional[Tuple[float, float, int]]:
    """Mean flow residual over a window, and how many samples it had."""
    sel = [d for d in rows if t0_us <= d["TimeUS"] <= t1_us
           and d.get("flowX") is not None]
    if len(sel) < 10:
        return None
    xs, ys = zip(*(flow_residual(d) for d in sel))
    return st.mean(xs), st.mean(ys), len(sel)


@dataclass
class Verdict:
    """What the windows add up to."""

    windows: List[Window] = field(default_factory=list)
    bias_ms2: float = float("nan")      # the airborne error
    control_ms2: float = float("nan")   # the motionless, motors-on error
    ekf_bias_est: float = float("nan")  # what the EKF worked out for itself
    innovation_m: float = float("nan")  # worst vertical position innovation
    test_ratio: float = float("nan")    # worst height test ratio
    climb_m: float = float("nan")       # how far the rangefinder really rose
    climb_from_to: Optional[Tuple[float, float]] = None

    @property
    def implied_velocity_per_s(self) -> float:
        """A bias integrates. This is what it becomes in one second."""
        return 0.0 if math.isnan(self.bias_ms2) else self.bias_ms2

    def lines(self) -> List[str]:
        out: List[str] = []
        if math.isnan(self.bias_ms2):
            return ["no airborne window was found in this log"]
        out.append(
            f"The down axis reads {self.bias_ms2:+.2f} m/s2 away from -g "
            f"while hovering.")
        if not math.isnan(self.control_ms2):
            out.append(
                f"On the floor with the motors running it already reads "
                f"{self.control_ms2:+.2f} m/s2 away, and there its true "
                f"vertical acceleration is zero, so that part is the "
                f"sensor and not the flying.")
        out.append(
            f"The estimator integrates that as a real descent: "
            f"{self.implied_velocity_per_s:+.2f} m/s of false sink rate per "
            f"second until something corrects it.")
        if not math.isnan(self.ekf_bias_est):
            out.append(
                f"The EKF's own accelerometer-bias state reached "
                f"{self.ekf_bias_est:+.2f} m/s2, which is "
                f"{100.0 * abs(self.ekf_bias_est) / max(1e-9, abs(self.bias_ms2)):.0f}% "
                f"of it: it is learning the bias far slower than the bias "
                f"appears.")
        if not math.isnan(self.innovation_m):
            out.append(
                f"So the height state sits {self.innovation_m:.2f} m away "
                f"from the rangefinder and stays there"
                + (f" (test ratio {self.test_ratio:.2f}, well inside the "
                   f"gate, so the measurement is being accepted and simply "
                   f"outweighed)" if not math.isnan(self.test_ratio) else "")
                + ".")
        if self.climb_from_to is not None:
            out.append(
                f"The rangefinder meanwhile went from "
                f"{self.climb_from_to[0]:.2f} m to "
                f"{self.climb_from_to[1]:.2f} m: the aircraft really climbed "
                f"while the controller believed it was sinking.")
        elif not math.isnan(self.climb_m):
            out.append(
                f"The aircraft really climbed {self.climb_m:+.2f} m while "
                f"the controller believed it was sinking.")
        return out
