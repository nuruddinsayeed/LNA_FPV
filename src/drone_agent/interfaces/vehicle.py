"""
Vehicle interface — the only module in the package that imports pymavlink.

Two things live here:

  VehicleInterface   a Protocol. Skills and the arbiter are written against
                     this, never against MavlinkVehicle. A ReplayVehicle that
                     feeds recorded WorldStates satisfies the same Protocol,
                     which is what makes offline replay possible.

  MavlinkVehicle     the ArduPilot-over-ELRS implementation.

Setpoint gating
---------------
The actuation methods (``send_position_target``, ``send_condition_yaw``)
require an opaque token. Exactly one token is minted, by the Arbiter, in its
constructor. A skill that tries to command the vehicle directly gets a
BypassError with a stack trace pointing at the offending line.

That is deliberately annoying. "Every setpoint goes through the arbiter" is
only true if violating it fails loudly rather than quietly working.

Mode changes and arming are NOT token-gated: the landing state machine has to
command LAND while the arbiter is already latched into an abort, and gating
that would mean the safety layer could lock itself out of landing.
"""

from __future__ import annotations

import os

# MUST be set before pymavlink is imported. Without it pymavlink builds and
# sends MAVLink1 frames (magic 0xFE) until it happens to parse an inbound v2
# frame, then silently upgrades. The FC is on SERIALx_PROTOCOL=2. It also
# selects the full ardupilotmega v2 dialect; the v1 dialect module is missing
# message IDs this code mutes (e.g. ESC_TELEMETRY_1_TO_4).
os.environ.setdefault("MAVLINK20", "1")

import math
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pymavlink import mavutil

from ..config import (COPTER_MODES, MODE_NAMES, MASK_POSITION_ONLY,
                      MASK_POSITION_YAW, WIRE_SIZE, Config)
from ..world.rangefinder import RangefinderRouter
from ..world.state import Pose, WorldState

if mavutil.mavlink.WIRE_PROTOCOL_VERSION != "2.0":   # pragma: no cover
    print("WARNING: pymavlink loaded in MAVLink1 mode. Set MAVLINK20=1 in the "
          "environment before running.")


# ───────────────────────────── errors ────────────────────────────

class FlightError(Exception):
    """Anything that should end the skill and land the aircraft."""


class LinkLost(FlightError):
    pass


class PilotTakeover(FlightError):
    """The pilot moved the mode switch. Not an error — a handover."""


class BypassError(RuntimeError):
    """Something tried to actuate the vehicle without going through the
    arbiter."""


COPTER_TYPES = {
    mavutil.mavlink.MAV_TYPE_QUADROTOR, mavutil.mavlink.MAV_TYPE_HEXAROTOR,
    mavutil.mavlink.MAV_TYPE_OCTOROTOR, mavutil.mavlink.MAV_TYPE_TRICOPTER,
    mavutil.mavlink.MAV_TYPE_HELICOPTER, mavutil.mavlink.MAV_TYPE_COAXIAL,
    mavutil.mavlink.MAV_TYPE_DODECAROTOR,
}


# ──────────────────────────── protocol ───────────────────────────

@runtime_checkable
class VehicleInterface(Protocol):
    """What everything above this layer is allowed to assume exists."""

    def poll(self) -> WorldState: ...
    def get_state(self) -> WorldState: ...
    def set_mode(self, mode: str, required: bool = True) -> bool: ...
    def arm(self) -> bool: ...
    def disarm(self) -> bool: ...
    def send_position_target(self, token: Any, x: float, y: float, z: float,
                             yaw_rad: Optional[float] = None) -> None: ...
    def send_condition_yaw(self, token: Any, heading_deg: float,
                           rate_dps: float, direction: int) -> None: ...
    def mint_actuation_token(self) -> Any: ...


# ───────────────────────── implementation ────────────────────────

class MavlinkVehicle:
    """ArduPilot over the ELRS MAVLink link."""

    def __init__(self, cfg: Config, verbose: bool = True) -> None:
        self.cfg = cfg
        self.verbose = verbose
        self.master: Optional[Any] = None
        self.mode_map: Dict[str, int] = dict(COPTER_MODES)

        self.rf = RangefinderRouter(cfg.rangefinder, cfg.forward_rf)

        # raw telemetry slots
        self._pos: Optional[tuple] = None
        self._t_pos: float = 0.0
        self._vz: Optional[float] = None
        self._yaw: Optional[float] = None
        self._pitch_deg: Optional[float] = None
        self._roll_deg: Optional[float] = None
        self._hb: Optional[Any] = None
        self._t_hb: float = 0.0
        self._landed_state: Optional[int] = None
        self._t_landed_state: float = 0.0
        self._flow_quality: Optional[int] = None
        self._flow_quality_min: Optional[int] = None
        self._rssi: Optional[int] = None
        self._lq: Optional[int] = None
        self._batt_pct: Optional[float] = None
        self._batt_v: Optional[float] = None

        self._acks: Dict[int, tuple] = {}
        self.msg_counts: Dict[str, int] = {}
        self.uplink_bytes = 0
        self.statustexts: List[str] = []
        # Called with each autopilot STATUSTEXT as it arrives, so the flight
        # log can hold it against its time. The autopilot explains itself in
        # these (EKF resets, failsafes, fence breaches), and a heading
        # sample that jumps 18 deg for one message is unreadable without
        # knowing whether the EKF said anything at that moment.
        self.on_statustext: Optional[Any] = None

        self.expected_mode: Optional[int] = None
        self.origin: Optional[tuple] = None

        self._t_boot = time.monotonic()
        self._t_last_gcs_hb = 0.0
        self._token: Optional[object] = None
        self._token_minted = False
        self._last_state: Optional[WorldState] = None

        # Perception, if attached. The vehicle owns MAVLink and pushes the
        # boresight range INTO perception; perception never reads the link.
        self.perception: Optional[Any] = None

    # ───────────────────────── token ──────────────────────────

    def mint_actuation_token(self) -> object:
        """Called exactly once, by the Arbiter. A second call is a bug: it
        means two things believe they own actuation."""
        if self._token_minted:
            raise BypassError(
                "an actuation token has already been minted — exactly one "
                "Arbiter may own this vehicle")
        self._token_minted = True
        self._token = object()
        return self._token

    def _check_token(self, token: Any, what: str) -> None:
        if self._token is None or token is not self._token:
            raise BypassError(
                f"{what} was called without the arbiter's actuation token. "
                f"Every setpoint must go through Arbiter.submit(); if you are "
                f"writing a skill, return a SetpointRequest from tick() "
                f"instead of commanding the vehicle directly.")

    # ─────────────────────── connection ───────────────────────

    def connect(self) -> None:
        c = self.cfg.link
        self._say(f"Opening {c.connection} as sysid {c.source_system}/"
                  f"{c.source_component}")
        m = mavutil.mavlink_connection(
            c.connection, source_system=c.source_system,
            source_component=c.source_component, autoreconnect=True)

        self._say("Waiting for a valid autopilot heartbeat through the RC "
                  "link...")
        deadline = time.monotonic() + c.heartbeat_wait_s
        hb = None
        while time.monotonic() < deadline:
            msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if msg is None:
                continue
            sysid, compid = msg.get_srcSystem(), msg.get_srcComponent()
            if sysid in (0, c.source_system):
                continue
            if msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                self._say(f"  ignoring non-autopilot heartbeat from "
                          f"{sysid}/{compid} - probably the ELRS radio itself")
                continue
            if compid not in (mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1, 0, 1):
                continue
            m.target_system, m.target_component = sysid, compid
            hb = msg
            break

        if hb is None:
            raise FlightError(
                "No autopilot heartbeat. Check: backpack WiFi joined, UDP "
                "14550 not already bound by another GCS, FC "
                "SERIALx_PROTOCOL=2 and SERIALx_BAUD=460, ELRS TX/RX >=3.5.0 "
                "in MAVLink mode.")

        self.master = m
        self._say(f"Connected. Vehicle SYS {m.target_system} "
                  f"COMP {m.target_component}")
        if hb.type not in COPTER_TYPES:
            raise FlightError(f"Vehicle type {hb.type} is not a multicopter.")
        self.mode_map = self._resolve_mode_map()
        self.send_gcs_heartbeat(force=True)

    def _resolve_mode_map(self) -> Dict[str, int]:
        try:
            dyn = self.master.mode_mapping()
        except Exception:
            dyn = None
        if not dyn:
            self._say("mode_mapping() unavailable - using built-in copter "
                      "table.")
            return dict(COPTER_MODES)
        dyn = {k.upper(): v for k, v in dyn.items()}
        for key in ("GUIDED", "LAND", "LOITER", "ALT_HOLD", "RTL"):
            if dyn.get(key) != COPTER_MODES[key]:
                self._say(f"mode_mapping() disagrees on {key} - using "
                          f"built-in table.")
                return dict(COPTER_MODES)
        merged = dict(COPTER_MODES)
        merged.update(dyn)
        return merged

    # ───────────────────────── streams ────────────────────────

    def configure_streams(self) -> None:
        c = self.cfg.link
        self._say("Configuring telemetry rates for the RC link...")

        if c.mute_enable:
            muted = 0
            for name in c.mute_messages:
                msg_id = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_" + name,
                                 None)
                if msg_id is None:
                    continue
                self._set_message_interval(msg_id, 0)
                muted += 1
                time.sleep(0.12)
                self.poll()
            self._say(f"  muted {muted} default streams (runtime only, resets "
                      f"on reboot)")

        for msg_id, hz, label in (
            (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, c.attitude_hz,
             "ATTITUDE"),
            (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
             c.local_position_hz, "LOCAL_POSITION_NED"),
            (mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR,
             c.distance_sensor_hz, "DISTANCE_SENSOR"),
            (mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE,
             c.extended_sys_state_hz, "EXTENDED_SYS_STATE"),
            (mavutil.mavlink.MAVLINK_MSG_ID_OPTICAL_FLOW, c.optical_flow_hz,
             "OPTICAL_FLOW"),
        ):
            self._set_message_interval(msg_id, hz)
            self._say(f"  {label} -> {hz:.1f} Hz")
            time.sleep(0.15)
            self.poll()

    def _set_message_interval(self, msg_id: int, hz: float) -> None:
        interval = -1 if hz <= 0 else int(1e6 / hz)
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0, msg_id, interval, 0, 0, 0, 0, 0)
        self._count_uplink("COMMAND_LONG")

    # ──────────────────────── telemetry ───────────────────────

    def attach_perception(self, runner: Any) -> None:
        """Attach a PerceptionRunner. world.objects then carries its latest
        fresh snapshot, and [] whenever that snapshot is stale — a skill must
        not be able to act on a two-second-old bearing."""
        self.perception = runner

    def poll(self) -> WorldState:
        """Drain the socket and build this tick's WorldState.

        Non-blocking and never raises — the landing path depends on that.
        """
        self.send_gcs_heartbeat()

        while True:
            try:
                msg = self.master.recv_match(blocking=False)
            except Exception:
                break
            if msg is None:
                break
            try:
                if msg.get_srcSystem() != self.master.target_system:
                    continue
            except Exception:
                continue

            t = msg.get_type()
            self.msg_counts[t] = self.msg_counts.get(t, 0) + 1
            now = time.monotonic()

            if t == "LOCAL_POSITION_NED":
                self._pos = (msg.x, msg.y, msg.z)
                self._t_pos = now
                self._vz = msg.vz
            elif t == "ATTITUDE":
                self._yaw = msg.yaw
                self._pitch_deg = math.degrees(msg.pitch)
                self._roll_deg = math.degrees(msg.roll)
            elif t == "DISTANCE_SENSOR":
                self.rf.ingest(
                    sensor_id=getattr(msg, "id", 0),
                    orientation=getattr(msg, "orientation", None),
                    distance_m=msg.current_distance / 100.0,
                    ekf_alt_m=(-self._pos[2] if self._pos else None),
                    vz=self._vz,
                    pitch_deg=self._pitch_deg)
            elif t == "OPTICAL_FLOW":
                self._flow_quality = msg.quality
                self._flow_quality_min = (
                    msg.quality if self._flow_quality_min is None
                    else min(self._flow_quality_min, msg.quality))
            elif t == "HEARTBEAT":
                self._hb = msg
                self._t_hb = now
            elif t == "EXTENDED_SYS_STATE":
                self._landed_state = msg.landed_state
                self._t_landed_state = now
            elif t == "RADIO_STATUS":
                self._rssi = getattr(msg, "rssi", None)
                self._lq = getattr(msg, "remrssi", None)
            elif t == "RC_CHANNELS":
                self._rssi = getattr(msg, "rssi", None)
            elif t == "BATTERY_STATUS":
                rem = getattr(msg, "battery_remaining", -1)
                self._batt_pct = float(rem) if rem >= 0 else None
                volts = getattr(msg, "voltages", [65535])
                if volts and volts[0] not in (0, 65535):
                    self._batt_v = volts[0] / 1000.0
            elif t == "COMMAND_ACK":
                self._acks[msg.command] = (msg.result, now)
            elif t == "STATUSTEXT":
                txt = (msg.text.strip() if isinstance(msg.text, str)
                       else str(msg.text))
                if txt:
                    self.statustexts.append(txt)
                    self._say(f"  [FC] {txt}")
                    if self.on_statustext is not None:
                        try:
                            self.on_statustext(txt)
                        except Exception:
                            pass    # logging must never break the receive loop

        for ev in self.rf.drain_events():
            self._say(f"  {ev}")

        state = self._build_state()
        if self.perception is not None:
            # Hand the forward lidar to the range estimator, then read back
            # whatever perception has published.
            self.perception.set_boresight_range(
                state.rngfnd_fwd_m, state.rngfnd_fwd_trusted)
            self.perception.set_camera_pose(
                state.rngfnd_down_m if state.rngfnd_down_trusted else None,
                (state.pose.pitch_deg or 0.0)
                if state.pose is not None else 0.0)
            state.objects = self.perception.objects_for_world(state.t_mono)
            scene = self.perception.latest()
            if scene.scene_text:
                state.scene = scene.scene_text
        self._last_state = state
        return self._last_state

    def get_state(self) -> WorldState:
        return self._last_state if self._last_state else self.poll()

    def _build_state(self) -> WorldState:
        now = time.monotonic()
        pose = None
        pose_age = math.inf
        if self._pos is not None and self._yaw is not None:
            pose = Pose(x=self._pos[0], y=self._pos[1], z=self._pos[2],
                        yaw_rad=self._yaw, vz=self._vz,
                        pitch_deg=self._pitch_deg, roll_deg=self._roll_deg)
            pose_age = now - self._t_pos

        link_age = (now - self._t_hb) if self._t_hb else math.inf
        down = self.rf.down
        fwd = self.rf.fwd

        return WorldState(
            t_mono=now,
            t_wall=time.time(),
            pose=pose,
            pose_age_s=pose_age,
            armed=self.is_armed(),
            mode=MODE_NAMES.get(self.current_mode_num() or -1),
            mode_num=self.current_mode_num(),
            landed_state=self._landed_state,
            landed_state_age_s=((now - self._t_landed_state)
                                if self._t_landed_state else math.inf),
            rngfnd_down_m=down.usable(now),
            rngfnd_down_trusted=down.trusted and down.fresh(now=now),
            rngfnd_down_raw_m=down.value_m,
            rngfnd_down_offset_m=(round(down.offset, 4)
                                  if down.offset is not None else None),
            rngfnd_down_residual_m=(round(down.last_residual, 4)
                                    if down.last_residual is not None else None),
            rngfnd_down_allowed_m=(round(down.last_allowed, 4)
                                   if down.last_allowed is not None else None),
            rngfnd_fwd_m=fwd.usable(now),
            rngfnd_fwd_trusted=fwd.trusted and fwd.fresh(now=now),
            flow_quality=self._flow_quality,
            link_up=link_age <= self.cfg.link.link_timeout_s,
            link_age_s=link_age,
            rssi=self._rssi,
            lq=self._lq,
            uplink_bytes=self.uplink_bytes,
            battery_pct=self._batt_pct,
            battery_v=self._batt_v,
            origin=self.origin,
        )

    # ─────────────────────── link helpers ─────────────────────

    def send_gcs_heartbeat(self, force: bool = False) -> None:
        now = time.monotonic()
        if (not force
                and now - self._t_last_gcs_hb < 1.0 / self.cfg.link.gcs_heartbeat_hz):
            return
        self._t_last_gcs_hb = now
        try:
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self._count_uplink("HEARTBEAT")
        except Exception:
            pass

    def check_link(self) -> None:
        if self._t_hb and time.monotonic() - self._t_hb > self.cfg.link.link_timeout_s:
            raise LinkLost(
                f"No heartbeat for {time.monotonic() - self._t_hb:.1f}s.")

    def check_takeover(self) -> None:
        if (not self.cfg.pilot_takeover_abort or self.expected_mode is None
                or self._hb is None):
            return
        if self._hb.custom_mode != self.expected_mode:
            name = MODE_NAMES.get(self._hb.custom_mode,
                                  str(self._hb.custom_mode))
            raise PilotTakeover(f"Mode changed to {name} - pilot has control.")

    def current_mode_num(self) -> Optional[int]:
        return self._hb.custom_mode if self._hb is not None else None

    def current_mode(self) -> Optional[str]:
        return MODE_NAMES.get(self.current_mode_num() or -1)

    def is_armed(self) -> bool:
        if self._hb is None:
            return False
        return bool(self._hb.base_mode
                    & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    # ──────────────────────── commands ────────────────────────

    def send_command(self, command: int, params: List[float],
                     retries: Optional[int] = None,
                     ack_timeout: Optional[float] = None,
                     label: Optional[str] = None,
                     quiet: bool = False) -> Optional[int]:
        retries = retries if retries is not None else self.cfg.link.command_retries
        ack_timeout = (ack_timeout if ack_timeout is not None
                       else self.cfg.link.ack_timeout_s)
        label = label or str(command)
        p = list(params) + [0.0] * (7 - len(params))

        for attempt in range(1, retries + 1):
            self._acks.pop(command, None)
            try:
                self.master.mav.command_long_send(
                    self.master.target_system, self.master.target_component,
                    command, 0, *p[:7])
            except Exception as e:
                self._say(f"  {label}: send failed ({e})")
                return None
            self._count_uplink("COMMAND_LONG")
            t_sent = time.monotonic()

            while time.monotonic() < t_sent + ack_timeout:
                self.poll()
                if command in self._acks:
                    result, t_ack = self._acks.pop(command)
                    if not quiet:
                        tag = ("accepted" if result == 0
                               else f"result {result}")
                        self._say(f"  {label} {tag} "
                                  f"(rtt {(t_ack - t_sent) * 1000:.0f} ms)")
                    return result
                time.sleep(0.02)

            if attempt < retries and not quiet:
                self._say(f"  {label}: no ACK, retry {attempt + 1}/{retries}")

        if not quiet:
            self._say(f"  {label}: no ACK after {retries} attempts")
        return None

    def command_mode_oneshot(self, mode_num: int) -> None:
        """Fire-and-forget mode request. Does not block, does not raise.
        Used by the landing state machine, which must keep trying through a
        dead link."""
        try:
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_num, 0, 0, 0, 0, 0)
            self._count_uplink("COMMAND_LONG")
        except Exception:
            pass

    def set_mode(self, mode: str, required: bool = True,
                 timeout: Optional[float] = None) -> bool:
        timeout = timeout if timeout is not None else self.cfg.preflight.mode_timeout_s
        if mode not in self.mode_map:
            raise FlightError("Mode not available: " + mode)
        target = self.mode_map[mode]
        self._say(f"Changing mode: {mode} (custom_mode={target})")
        self.expected_mode = None

        result = self.send_command(
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            [mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, target],
            label=f"SET_MODE {mode}")

        if result is not None and result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            msg = f"Mode change to {mode} REJECTED (result={result})."
            if required:
                raise FlightError(msg)
            self._say("WARNING: " + msg)
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.poll()
            self.check_link()
            if self.current_mode_num() == target:
                self._say(f"Mode confirmed: {mode}")
                self.expected_mode = target
                return True
            time.sleep(0.05)

        got = self.current_mode() or "unknown"
        msg = f"Mode change to {mode} NOT confirmed (heartbeat reports {got})."
        if required:
            raise FlightError(msg)
        self._say("WARNING: " + msg)
        return False

    def arm(self) -> bool:
        self._say("Arming")
        result = self.send_command(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, [1], label="ARM")
        if result is not None and result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise FlightError(f"Arm REJECTED (result={result}).")
        deadline = time.monotonic() + self.cfg.preflight.arm_timeout_s
        while time.monotonic() < deadline:
            self.poll()
            self.check_link()
            if self.is_armed():
                self._say("Armed (confirmed via heartbeat)")
                return True
            time.sleep(0.05)
        raise FlightError("Arming NOT confirmed within timeout.")

    def disarm(self) -> bool:
        """
        The only disarm path. Gated on confirmed ground contact by its single
        caller in the landing state machine.

        There is deliberately no force-disarm (the 21196 magic value): if the
        autopilot refuses, the autopilot knows something we do not.
        """
        for attempt in range(1, self.cfg.landing.disarm_retries + 1):
            result = self.send_command(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, [0],
                retries=1, label=f"DISARM (attempt {attempt})")
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                self.poll()
                if not self.is_armed():
                    self._say("Disarmed.")
                    return True
                time.sleep(0.05)
            if (result is not None
                    and result != mavutil.mavlink.MAV_RESULT_ACCEPTED):
                self._say("  autopilot refused the disarm - it does not agree "
                          "we are down. Not forcing it.")
        return False

    def takeoff(self, height_m: float) -> None:
        self._say(f"Takeoff to {height_m} m")
        result = self.send_command(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            [0, 0, 0, 0, 0, 0, height_m], label="TAKEOFF")
        if result is None:
            self._say("WARNING: no ACK for takeoff - will verify by altitude.")
        elif result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise FlightError(f"Takeoff REJECTED (result={result}).")

    def command_nav_land(self) -> None:
        try:
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
            self._count_uplink("COMMAND_LONG")
        except Exception:
            pass

    # ────────────────── gated actuation ───────────────────────

    def send_position_target(self, token: Any, x: float, y: float, z: float,
                             yaw_rad: Optional[float] = None) -> None:
        self._check_token(token, "send_position_target")
        mask = MASK_POSITION_ONLY if yaw_rad is None else MASK_POSITION_YAW
        self.master.mav.set_position_target_local_ned_send(
            self._ms_since_boot(),
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            mask, x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            yaw_rad if yaw_rad is not None else 0.0, 0.0)
        self._count_uplink("SET_POSITION_TARGET_LOCAL_NED")

    def send_condition_yaw(self, token: Any, heading_deg: float,
                           rate_dps: float, direction: int) -> Optional[int]:
        self._check_token(token, "send_condition_yaw")
        return self.send_command(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            [heading_deg, abs(rate_dps), direction, 0],
            label=f"CONDITION_YAW {heading_deg:.0f}deg")

    # ──────────────────────── utilities ───────────────────────

    def _ms_since_boot(self) -> int:
        return int((time.monotonic() - self._t_boot) * 1000) & 0xFFFFFFFF

    def _count_uplink(self, name: str, n: int = 1) -> None:
        self.uplink_bytes += WIRE_SIZE.get(name, 40) * n

    def _say(self, line: str) -> None:
        if self.verbose:
            print(line)

    def wait_for(self, predicate, name: str, timeout: float = 8.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.poll()
            if predicate():
                return True
            time.sleep(0.02)
        self._say(f"WARNING: no {name} received.")
        return False

    def report_budget(self) -> List[str]:
        c, s = self.cfg.link, self.cfg.skills
        down = (c.attitude_hz * WIRE_SIZE["ATTITUDE"]
                + c.local_position_hz * WIRE_SIZE["LOCAL_POSITION_NED"]
                + c.distance_sensor_hz * WIRE_SIZE["DISTANCE_SENSOR"]
                * c.rf_streams_expected
                + c.extended_sys_state_hz * WIRE_SIZE["EXTENDED_SYS_STATE"]
                + c.optical_flow_hz * WIRE_SIZE["OPTICAL_FLOW"]
                + 1.0 * WIRE_SIZE["HEARTBEAT"])
        legs = max(1, int(round(s.yaw_total_deg / s.yaw_leg_deg)))
        rot_s = s.yaw_total_deg / s.yaw_rate_dps
        up = (WIRE_SIZE["SET_POSITION_TARGET_LOCAL_NED"] / s.setpoint_period_s
              + c.gcs_heartbeat_hz * WIRE_SIZE["HEARTBEAT"])
        up_rot = (legs * (WIRE_SIZE["SET_POSITION_TARGET_LOCAL_NED"]
                          + WIRE_SIZE["COMMAND_LONG"]) / rot_s
                  + c.gcs_heartbeat_hz * WIRE_SIZE["HEARTBEAT"])

        lines = [
            "Link budget:",
            f"  downlink        {down:6.0f} B/s of {c.downlink_bps} "
            f"({100 * down / c.downlink_bps:.0f}%)",
            f"  uplink (hold)   {up:6.0f} B/s of {c.uplink_bps} "
            f"({100 * up / c.uplink_bps:.0f}%)",
            f"  uplink (rotate) {up_rot:6.0f} B/s of {c.uplink_bps} "
            f"({100 * up_rot / c.uplink_bps:.0f}%)  "
            f"[{legs} legs of {s.yaw_leg_deg:.0f} deg]",
        ]
        if down > c.downlink_bps * 0.6:
            lines.append("  WARNING: downlink over 60% of capacity.")
        if max(up, up_rot) > c.uplink_bps * 0.6:
            lines.append("  WARNING: uplink over 60% of capacity. Raise "
                         "yaw_leg_deg or setpoint_period_s.")
        return lines
