"""
Rangefinder trust tracking — one tracker per physical sensor.

The original single-variable version is what produced the interleaving bug:
two DISTANCE_SENSOR streams (downward MTF-01P, forward MT-15) fed one slot, so
consecutive "samples" alternated between sensors pointing in different
directions. Symptoms were the back-to-back 0.79 -> 1.04 m "glitch in 0.00s",
wall distances of 3-4 m appearing as altitude during a yaw, and a corrupted
learned offset.

Fix: route by (orientation, id) and keep independent state per sensor.

Two independent trust tests, because either alone is insufficient:

  1. Rate test. Did it move faster than the aircraft physically could? Caught
     exactly one step (1.22 -> 0.47) in the incident log and nothing at 2 Hz.

  2. Residual test against a LEARNED offset. The sensor sat a steady +0.25 m
     above EKF-implied altitude for a whole stable hold, so a raw-difference
     test would have to tolerate that bias and would have no margin left.
     Against the learned offset:
         stable hold   residuals  -0.05 .. +0.04  (8 samples)
         the failure   residuals  -0.46 then -0.82
     0.35 m flags the bad readings on the first sample and never fires on the
     good ones.

Flying over an obstacle (a table) makes these legitimately disagree and will
distrust the sensor too. That is the safe direction: it only ever makes us
less willing to believe we have landed.

What test 2 is NOT
------------------
It is not an independent cross-check, and the thesis must not call it one.
On this airframe EK3_SRC1_POSZ=2: the EKF's height source IS the rangefinder,
so a rangefinder that lies persistently will drag the EKF with it and the
residual will stay small. What the test actually detects is a rangefinder
deviating FASTER than the EKF's fusion follows — a rate-of-divergence
detector that happens to be expressed as a position residual. That is exactly
why it caught the original incident (a bump producing fast bad data) and
exactly why it false-positived on a commanded climb (sustained vertical
motion produces sustained lag).

A genuinely independent check would need the other height source. The vehicle
has one configured but inactive (EK3_SRC2_POSZ=1, baro), and the GCS only
ever sees the active estimate, so obtaining it would mean streaming
VFR_HUD/baro altitude separately and comparing that. Worth doing if the
cross-check needs to be defensible as independent; the present test earns its
place on the incident it caught, not on an independence it does not have.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config import RangefinderConfig, ForwardRangefinderConfig


@dataclass
class StreamStats:
    """Every stream seen, used or not, so preflight can show what is on the
    link rather than silently discarding it."""

    n: int = 0
    min_m: float = math.inf
    max_m: float = -math.inf
    last_m: float = 0.0
    orientation: Optional[int] = None
    sensor_id: int = 0

    def update(self, d: float) -> None:
        self.n += 1
        self.min_m = min(self.min_m, d)
        self.max_m = max(self.max_m, d)
        self.last_m = d


class RangefinderTracker:
    """Trust state for ONE sensor. Downward instances cross-check against the
    EKF; forward instances skip that test (there is nothing to compare to) and
    rely on the rate test plus pitch gating."""

    def __init__(self, cfg: RangefinderConfig, name: str = "down",
                 cross_check_ekf: bool = True) -> None:
        self.cfg = cfg
        self.name = name
        self.cross_check_ekf = cross_check_ekf

        self.value_m: Optional[float] = None
        self.t_value: float = 0.0

        self.glitches = 0
        self.good_run = 0
        self.low_run = 0
        self.residual_run = 0
        self.residual_widened = 0   # samples where |vz| loosened the test
        self.last_residual: Optional[float] = None
        self.last_allowed: Optional[float] = None
        self._vz_peak = 0.0         # decaying peak of |vz|
        self.offset: Optional[float] = None     # learned (sensor - EKF alt)
        self.offset_n = 0
        self.trusted = True

        self.events: List[str] = []             # drained by the recorder

    # ---------------------------------------------------------------

    def ingest(self, value_m: float, ekf_alt_m: Optional[float],
               vz: Optional[float], now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        prev, prev_t = self.value_m, self.t_value
        self.value_m, self.t_value = value_m, now

        # Decaying peak of |vz|: rises instantly, falls with a time constant.
        if vz is not None:
            tau = max(1e-3, self.cfg.residual_vz_tau_s)
            dt = (now - prev_t) if prev is not None else 0.0
            dt = min(max(dt, 0.0), 5.0)
            self._vz_peak = max(abs(vz), self._vz_peak * math.exp(-dt / tau))

        glitch = False

        # --- 1. rate test -------------------------------------------------
        if prev is not None and self.cfg.min_dt_s <= now - prev_t <= 1.5:
            allowed = self.cfg.max_rate_mps * (now - prev_t) + self.cfg.noise_m
            if abs(value_m - prev) > allowed:
                glitch = True
                self.glitches += 1
                if self.trusted and self.glitches >= self.cfg.unhealthy_glitches:
                    self._distrust(
                        f"{prev:.2f} -> {value_m:.2f} m in {now - prev_t:.2f}s "
                        f"(max plausible {allowed:.2f} m)")

        # --- 2. residual test against the learned offset ------------------
        if self.cross_check_ekf and ekf_alt_m is not None:
            diff = value_m - ekf_alt_m
            if self.offset is None:
                self.offset = diff
                self.offset_n = 1
            else:
                residual = diff - self.offset
                # Allow more disagreement the faster we are moving
                # vertically. Rangefinder height and baro-derived EKF height
                # are independent measurements, and baro lags a climb; a
                # fixed threshold calibrated at hover false-positives on a
                # healthy sensor as soon as the aircraft moves.
                allowed = self.cfg.residual_m
                if vz is not None:
                    allowed += self.cfg.residual_vz_coeff * self._vz_peak
                    if allowed > self.cfg.residual_m:
                        self.residual_widened += 1
                self.last_residual = residual
                self.last_allowed = allowed
                if abs(residual) > allowed:
                    self.residual_run += 1
                    if (self.trusted
                            and self.residual_run >= self.cfg.residual_samples):
                        self._distrust(
                            f"reads {value_m:.2f} m, EKF {ekf_alt_m:.2f} m, "
                            f"residual {residual:+.2f} m vs learned offset "
                            f"{self.offset:+.2f} m (allowed "
                            f"{allowed:.2f} m at vz "
                            f"{vz if vz is not None else 0:+.2f})")
                else:
                    self.residual_run = 0
                    # Learn only from calm, trusted, level flight.
                    if (self.trusted and not glitch
                            and vz is not None and abs(vz) < 0.3):
                        self.offset += self.cfg.offset_alpha * residual
                        self.offset_n += 1

        if glitch or self.residual_run > 0:
            self.good_run = 0
            return

        self.good_run += 1
        if not self.trusted and self.good_run >= self.cfg.recover_samples:
            self.trusted = True
            self.glitches = 0
            self.events.append(f"[RF:{self.name}] consistent and agreeing "
                               f"again - trusted")

    # ---------------------------------------------------------------

    def _distrust(self, why: str) -> None:
        self.trusted = False
        self.events.append(f"[RF:{self.name}] distrusted: {why}")

    def fresh(self, max_age_s: float = 2.0,
              now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        return self.value_m is not None and (now - self.t_value) < max_age_s

    def usable(self, now: Optional[float] = None) -> Optional[float]:
        """The reading if it can be believed, else None. Callers must treat
        None as 'unknown', never as zero."""
        return self.value_m if (self.trusted and self.fresh(now=now)) else None

    def check_low(self) -> Optional[str]:
        """Debounced low-altitude abort. A bump that throws one wild reading
        no longer triggers anything."""
        c = self.cfg
        if not c.abort_enable or self.value_m is None:
            self.low_run = 0
            return None
        if c.abort_require_trust and not self.trusted:
            self.low_run = 0
            return None
        if self.value_m < c.abort_m:
            self.low_run += 1
            if self.low_run >= c.abort_consec:
                return (f"rangefinder {self.value_m:.2f} m below {c.abort_m} m "
                        f"for {self.low_run} consecutive samples")
        else:
            self.low_run = 0
        return None

    def drain_events(self) -> List[str]:
        out, self.events = self.events, []
        return out

    def snapshot(self) -> dict:
        return {
            "value_m": self.value_m,
            "trusted": self.trusted,
            "age_s": (round(time.monotonic() - self.t_value, 3)
                      if self.value_m is not None else None),
            "offset_m": (round(self.offset, 3)
                         if self.offset is not None else None),
            "glitches": self.glitches,
            "residual_widened": self.residual_widened,
        }


class RangefinderRouter:
    """
    Routes DISTANCE_SENSOR messages to the right tracker by orientation, and
    records every stream seen so preflight can report what is actually on the
    link.
    """

    def __init__(self, down_cfg: RangefinderConfig,
                 fwd_cfg: ForwardRangefinderConfig) -> None:
        self.down_cfg = down_cfg
        self.fwd_cfg = fwd_cfg

        self.down = RangefinderTracker(down_cfg, "down", cross_check_ekf=True)
        # The forward sensor reuses the downward thresholds for the rate test
        # but never cross-checks the EKF: there is nothing to compare against.
        self.fwd = RangefinderTracker(down_cfg, "fwd", cross_check_ekf=False)

        self.streams: Dict[Tuple[int, Optional[int]], StreamStats] = {}
        self._locked_down_id: Optional[int] = None
        self._locked_fwd_id: Optional[int] = None
        self.events: List[str] = []
        self.fwd_gated = 0                  # readings rejected by pitch gate

    # ---------------------------------------------------------------

    def ingest(self, sensor_id: int, orientation: Optional[int],
               distance_m: float, ekf_alt_m: Optional[float],
               vz: Optional[float], pitch_deg: Optional[float]) -> None:
        key = (sensor_id, orientation)
        st = self.streams.get(key)
        if st is None:
            st = self.streams[key] = StreamStats(
                orientation=orientation, sensor_id=sensor_id)
        st.update(distance_m)

        if orientation == self.down_cfg.orientation:
            if not self._accept(sensor_id, "down"):
                return
            self.down.ingest(distance_m, ekf_alt_m, vz)
            return

        if self.fwd_cfg.enabled and orientation == self.fwd_cfg.orientation:
            if not self._accept(sensor_id, "fwd"):
                return
            if self._floor_return(distance_m, ekf_alt_m, pitch_deg):
                self.fwd_gated += 1
                return
            if not (self.fwd_cfg.min_m <= distance_m <= self.fwd_cfg.max_m):
                return
            self.fwd.ingest(distance_m, None, vz)

    # ---------------------------------------------------------------

    def _accept(self, sensor_id: int, which: str) -> bool:
        cfg = self.down_cfg if which == "down" else self.fwd_cfg
        if cfg.sensor_id is not None:
            return sensor_id == cfg.sensor_id
        attr = f"_locked_{which}_id"
        locked = getattr(self, attr)
        if locked is None:
            setattr(self, attr, sensor_id)
            self.events.append(
                f"[RF] using DISTANCE_SENSOR id={sensor_id} as the {which} "
                f"sensor; other ids at this orientation ignored")
            return True
        return sensor_id == locked

    def _floor_return(self, distance_m: float, ekf_alt_m: Optional[float],
                      pitch_deg: Optional[float]) -> bool:
        """
        A forward beam on a nose-down airframe eventually hits the floor. With
        height h and effective depression angle a, that happens at h/sin(a).
        Reject readings near that value, otherwise every translation looks
        like an obstacle.
        """
        c = self.fwd_cfg
        if not c.floor_gate_enable or pitch_deg is None or ekf_alt_m is None:
            return False
        # ArduPilot ATTITUDE pitch is positive nose-up; depression is the
        # nose-down component minus whatever the sensor is physically tilted up.
        depression = -pitch_deg - c.mount_pitch_up_deg
        if depression < c.floor_gate_min_pitch_deg:
            return False
        expected = ekf_alt_m / math.sin(math.radians(depression))
        return abs(distance_m - expected) <= c.floor_gate_tol_m

    # ---------------------------------------------------------------

    def drain_events(self) -> List[str]:
        out = self.events + self.down.drain_events() + self.fwd.drain_events()
        self.events = []
        return out

    def describe(self) -> List[str]:
        if not self.streams:
            return ["  no DISTANCE_SENSOR messages seen at all"]
        lines = [f"  DISTANCE_SENSOR streams seen ({len(self.streams)}):"]
        for (sid, orient), s in sorted(self.streams.items(),
                                       key=lambda kv: (kv[0][1] or 0, kv[0][0])):
            if orient == self.down_cfg.orientation:
                role = "USED (down)"
            elif self.fwd_cfg.enabled and orient == self.fwd_cfg.orientation:
                role = "USED (fwd)"
            else:
                role = "ignored"
            lines.append(
                f"    id={sid} orientation={orient}: {s.n} msgs, "
                f"{s.min_m:.2f}-{s.max_m:.2f} m, last {s.last_m:.2f} m "
                f"-> {role}")
        if len(self.streams) > 1:
            lines.append("  (multiple rangefinders - only the downward one "
                         "feeds altitude and the landing logic)")
        if self.fwd_gated:
            lines.append(f"  forward readings rejected by the floor gate: "
                         f"{self.fwd_gated}")
        return lines
