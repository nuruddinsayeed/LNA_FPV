"""
Single source of truth for every tunable in the stack.

Everything here came out of the original guided-hold + rotate script. The
values and the reasoning behind them are preserved; what changed is that they
now live in one place instead of being scattered through a 1500-line module,
so a flight configuration is a thing you can diff, log and cite.

Convention: every duration is seconds, every distance metres, every angle
degrees unless the name ends in _RAD.
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List


# ───────────────────────────── link ──────────────────────────────

@dataclass
class LinkConfig:
    """ELRS 2.4 GHz, 333 Hz Full, MAVLink mode."""

    connection: str = "udpin:0.0.0.0:14550"
    source_system: int = 254            # not 255: the ELRS RX uses 255 as GCS
    # The system id the ELRS receiver sends the RC from when it "acts as a
    # GCS" (its Source SysID setting, default 255). The autopilot accepts RC
    # input only from MAV_GCS_SYSID, so that parameter has to stay equal to
    # THIS, not to source_system above. Setting it to 254 on 15 Nov cost the
    # RC link: "RC not found".
    rc_source_system: int = 255
    source_component: int = 190         # MAV_COMP_ID_MISSIONPLANNER
    gcs_heartbeat_hz: float = 1.0

    downlink_bps: int = 1470
    uplink_bps: int = 735

    heartbeat_wait_s: float = 30.0
    link_timeout_s: float = 4.0
    ack_timeout_s: float = 4.0
    command_retries: int = 3

    # Downlink stream rates. DISTANCE_SENSOR is PER SENSOR: ArduPilot emits one
    # per configured rangefinder backend and SET_MESSAGE_INTERVAL has no
    # per-instance granularity, so N backends cost N times the bandwidth. With
    # the MTF-01P down and the MT-15 forward that is 2 streams.
    attitude_hz: float = 4.0
    local_position_hz: float = 4.0
    distance_sensor_hz: float = 3.0
    extended_sys_state_hz: float = 1.0
    optical_flow_hz: float = 2.0
    # One DISTANCE_SENSOR stream per configured rangefinder backend. The
    # airframe has RNGFND1 only (RNGFND2_TYPE=0), so 1; raise to 2 when the
    # forward MT-15 is fitted. Overstating it only inflates the budget
    # report, but tools/check_params.py reads this against the vehicle.
    rf_streams_expected: int = 1

    mute_enable: bool = True
    mute_messages: List[str] = field(default_factory=lambda: [
        "RAW_IMU", "SCALED_IMU2", "SCALED_IMU3", "SCALED_PRESSURE",
        "SCALED_PRESSURE2", "SCALED_PRESSURE3", "SERVO_OUTPUT_RAW",
        "RC_CHANNELS", "ATTITUDE_QUATERNION", "AHRS", "AHRS2", "SIMSTATE",
        "VIBRATION", "ESC_TELEMETRY_1_TO_4", "TERRAIN_REPORT", "MEMINFO",
        "POWER_STATUS", "MISSION_CURRENT", "NAV_CONTROLLER_OUTPUT",
        "GPS_RAW_INT", "GPS2_RAW", "GLOBAL_POSITION_INT", "SYS_STATUS",
        "BATTERY_STATUS",
    ])


# ────────────────────────── rangefinders ─────────────────────────

@dataclass
class RangefinderConfig:
    """
    Per-sensor tracking. The vehicle publishes one DISTANCE_SENSOR stream per
    backend; feeding them all into one variable produced the interleaving bug
    (a downward sensor 0.7 m off the floor "reading" 4.14 m was the forward
    sensor seeing a wall mid-yaw). Each orientation now gets its own tracker.
    """

    orientation: int = 25               # MAV_SENSOR_ROTATION_PITCH_270 = down
    sensor_id: Any = None               # None -> lock onto first id seen

    # Glitch detection. A fixed jump threshold is useless when the whole
    # envelope is ~0.8 m, so scale the allowed change by elapsed time.
    max_rate_mps: float = 1.5
    noise_m: float = 0.10
    min_dt_s: float = 0.05
    unhealthy_glitches: int = 2

    # EKF cross-check against a LEARNED offset, not a raw difference. The
    # sensor sat a steady +0.25 m above EKF-implied altitude all through a
    # stable hold; against the learned offset the separation is clean
    # (good: -0.05..+0.04, failure: -0.46 then -0.82).
    residual_m: float = 0.35
    residual_samples: int = 2

    # The 0.35 m threshold above was calibrated from a STABLE HOLD, where
    # residuals ran -0.05..+0.04. Under vertical motion EKF height and the
    # rangefinder reading legitimately diverge much further.
    #
    # Why, on the actual airframe (verified against its .param, 2026-11-13):
    # EK3_SRC1_POSZ=2, so the EKF height source IS the rangefinder — but
    # EK3_RNG_M_NSE=0.5 means the filter assumes half a metre of measurement
    # noise and smooths hard, while RNGFND_FILT=0.5 Hz smooths the raw
    # reading less. The estimate therefore LAGS the reading during a climb,
    # by roughly 0.3 m at 0.3 m/s. The clamp_alt flight peaked at a residual
    # of +0.362 on a perfectly healthy sensor, tripping the test for 25 ticks
    # and flipping the altitude source mid-manoeuvre.
    #
    # The lag is proportional to vertical rate, so the allowance is too. At
    # vz=0 the threshold is unchanged, which is where the test matters: the
    # landing logic consults it in quasi-static conditions.
    residual_vz_coeff: float = 1.0      # seconds; allowance per m/s of |vz|

    # The allowance tracks a DECAYING PEAK of |vz|, not the instantaneous
    # value. The EKF's lag is accumulated, so it is still large when the
    # aircraft has momentarily stopped moving: on the 2026-11-13 flight the
    # residual was 0.333 m at vz=-0.04, where an instantaneous-vz allowance
    # collapses to 0.39 m and leaves 0.06 m of margin. The peak decays with
    # this time constant, matched to how long the estimate takes to catch up.
    residual_vz_tau_s: float = 1.0
    offset_alpha: float = 0.05
    recover_samples: int = 6

    # Low-altitude abort
    abort_m: float = 0.15
    abort_enable: bool = True
    abort_consec: int = 3
    abort_require_trust: bool = True


@dataclass
class ForwardRangefinderConfig:
    """
    MT-15, forward. 1-2 deg beam: a tape measure, not an obstacle sensor.

    Pitch gating matters. A quad pitches nose-down to translate; at 1.5 m and
    10 deg the beam hits the floor at ~8.6 m, at 20 deg at ~4.4 m. Without
    gating those floor returns read as obstacles.
    """

    enabled: bool = False               # off until the MT-15 is fitted
    orientation: int = 0                # MAV_SENSOR_ROTATION_NONE = forward
    sensor_id: Any = None
    mount_pitch_up_deg: float = 0.0     # physical uptilt of the sensor body
    max_m: float = 12.0                 # derated from 15 for indoor reflectance
    min_m: float = 0.05

    # Reject a reading consistent with the expected floor intersection.
    floor_gate_enable: bool = True
    floor_gate_tol_m: float = 0.75
    floor_gate_min_pitch_deg: float = 4.0


# ───────────────────────── flight envelope ───────────────────────

@dataclass
class EnvelopeConfig:
    """
    The hard limits the arbiter enforces. Nothing above this layer can widen
    them; a skill that asks for more gets clamped and the clamp is logged.
    """

    alt_min_m: float = 0.30
    # Sized to the room, not to the aircraft. The flight room has a 2.50 m
    # ceiling and the 2.2" airframe overshot a clamped setpoint by up to
    # 0.85 m, so the old default of 2.00 put the worst-case peak at 2.85 m
    # -- 0.35 m INTO the ceiling -- on any flight where --alt-max was
    # forgotten. At 1.20 the peak is 2.05 m. Raise it only after the
    # overshoot has been re-measured smaller, or in a taller room.
    #
    # FRAMES. This clamp acts on setpoints, which are in the EKF's frame.
    # The breach abort (alt_max + breach_margin) is judged on the best
    # estimate, which is the rangefinder: real height. They are not the
    # same number -- see ekf_under_read_climb_m.
    alt_max_m: float = 1.20

    # The room. Nothing in the arbiter uses this; check_params does, to
    # decide whether the autopilot's altitude fence can actually stop the
    # aircraft before the ceiling.
    ceiling_m: float = 2.50
    # How far BELOW the real height the EKF altitude has read while the
    # aircraft climbed fast. Measured on the 2.2" airframe in three
    # clamp_alt flights: at real heights of 1.83, 1.70 and 2.05 m the EKF
    # read 1.47, 1.38 and 1.53 m (0.36, 0.32, 0.51 m low). In a steady hover
    # the gap is 0.06 m plus about 2.1 s x the EKF's vertical-velocity
    # error, up to 0.34 m across three baseline sweeps. The autopilot's
    # fence is judged on the EKF's number, so this much real height has to
    # be added to FENCE_ALT_MAX to know where the aircraft will be when it
    # trips.
    ekf_under_read_climb_m: float = 0.50
    ekf_under_read_static_m: float = 0.06

    fence_radius_m: float = 4.0         # from the captured origin
    max_step_m: float = 2.0             # single setpoint jump from current pos

    max_speed_mps: float = 0.5          # advisory; logged, used by velocity skills
    max_yaw_rate_dps: float = 45.0

    # Staleness: a position estimate older than this is not a position estimate.
    pose_max_age_s: float = 0.80
    pose_stale_hold_s: float = 2.0      # hold this long, then land

    # Watchdog on SKILL TICKS, not on setpoints. Rotate legs deliberately go
    # quiet on the uplink for ~2.25 s (a position target mid-leg cancels the
    # CONDITION_YAW slew), so a setpoint watchdog would fire every leg.
    tick_watchdog_s: float = 1.0

    drift_abort_m: float = 1.5
    drift_abort_enable: bool = True

    # Past this multiple of any envelope limit, a request is treated as
    # nonsense and REJECTED rather than clamped. Clamping is right for a
    # small overshoot; for "fly to 10 m" it silently converts a bad command
    # into a legal one (fly to the ceiling) that nobody asked for. Set 0 to
    # clamp everything, as before.
    gross_violation_factor: float = 2.0

    # A single gross violation is blocked and the mission continues — it is
    # probably a transient. This many CONSECUTIVE ones means the caller is
    # broken and will keep producing nonsense, so the arbiter aborts and the
    # aircraft lands. Set 0 to never escalate.
    gross_violation_abort_after: int = 3

    # --- state-based envelope, as opposed to setpoint-based ---------------
    # Clamping a setpoint bounds the COMMAND, not the aircraft. On the
    # 2026-11-13 flight a request for 1.68 m was clamped to 1.20 m and
    # transmitted, and the vehicle's own altitude controller then overshot to
    # 1.48 m — 0.28 m past the limit the arbiter had just enforced. An
    # envelope enforced on requests is therefore NOT an envelope on the
    # vehicle's state, and the thesis should not claim otherwise.
    #
    # So measure the real thing too: how far outside the envelope the
    # aircraft actually got, and for how long. Transient overshoot from the
    # autopilot's own controller is expected and only recorded; a sustained
    # excursion means the aircraft is not where the arbiter believes and
    # aborts.
    breach_margin_m: float = 0.40       # tolerates normal controller overshoot
    breach_abort_after: int = 10        # consecutive breaching ticks -> abort


# ──────────────────────────── landing ────────────────────────────

@dataclass
class LandingConfig:
    total_timeout_s: float = 180.0
    retry_period_s: float = 2.5
    nav_escalate_every: int = 3
    ground_alt_m: float = 0.25
    ground_vz_max: float = 0.20
    ground_confirm_s: float = 2.0
    disarm_after_ground_s: float = 3.0
    disarm_retries: int = 4


# ─────────────────────────── preflight ───────────────────────────

@dataclass
class PreflightConfig:
    enabled: bool = True
    window_s: float = 6.0
    max_command_rtt_s: float = 1.5
    min_attitude_rate_hz: float = 2.5
    min_position_rate_hz: float = 2.5
    min_heartbeat_hz: float = 0.5

    mode_timeout_s: float = 10.0
    arm_timeout_s: float = 15.0
    climb_timeout_s: float = 30.0
    min_climb_m: float = 0.15

    # The ALT_HOLD -> GUIDED handover exists for the case where a flow-only
    # EKF cannot produce a horizontal position estimate on the ground (no
    # motion, below the sensor's minimum working distance) and GUIDED pre-arm
    # fails with "Need Position Estimate".
    #
    # OFF by default: it requires a human pilot to climb on the sticks, and
    # if nobody does, ArduPilot auto-disarms after DISARM_DELAY (10 s by
    # default) with the aircraft still on the ground. When GUIDED arms fine,
    # auto-takeoff is the simpler and more repeatable path. Turn this on with
    # --handover only when GUIDED refuses to pre-arm.
    handover_enabled: bool = False
    handover_confirm: bool = True       # wait for Enter before arming
    handover_arm_mode: str = "ALT_HOLD"
    handover_verify_mode: str = "LOITER"
    handover_min_alt_m: float = 0.50
    handover_timeout_s: float = 120.0
    handover_pose_samples: int = 8
    handover_pose_max_spread_m: float = 0.35


# ────────────────────────── skill defaults ───────────────────────

@dataclass
class SkillDefaults:
    takeoff_height_m: float = 0.80
    settle_s: float = 6.0
    hover_s: float = 8.0

    setpoint_period_s: float = 2.0      # GUIDED holds the last target
    tick_hz: float = 10.0

    yaw_total_deg: float = 360.0
    yaw_rate_dps: float = 20.0
    yaw_dir: int = 1
    yaw_leg_deg: float = 45.0           # 8 XY corrections per revolution
    leg_settle_s: float = 2.0           # quiet window for the EKF to reconverge
    yaw_tolerance_deg: float = 8.0
    # How many SUCCESSIVE heading samples must sit inside the tolerance for
    # a leg to count as complete. 1 is the rule every flight up to 0.14
    # used, and it is the default again.
    #
    # 0.15.0 and 0.15.1 shipped with 2, on a wrong diagnosis. On the last
    # leg of every 360 deg sweep flown so far (6 of 6) the heading steps
    # 16-27 deg in one sample and the old rule ends the leg there. That
    # looked like a bad sample. It is the EKF re-aligning its yaw to the
    # compass -- the autopilot says so: "EKF3 IMU0 MAG0 ground mag anomaly,
    # yaw re-aligned" -- so the heading frame itself moved, and waiting for
    # a second sample does not undo that. What 2 did do was end every leg
    # one attitude period later, about 1.6 s per sweep, with no measured
    # benefit. It stays available for a stream that really does produce a
    # stray sample.
    yaw_confirm_samples: int = 1
    # If the heading stops changing while inside the tolerance (a stalled
    # attitude stream repeats one value) accept it after this long instead
    # of waiting for the leg to time out. Longer than the 0.42 s the bad
    # sample above stayed current.
    yaw_confirm_fallback_s: float = 1.0

    # move: command real metres instead of EKF metres. Flow velocity is
    # scaled by the height the EKF believes, which reads below the
    # rangefinder, so an uncorrected move goes further than it is told
    # (measured over six flights: 24% further, tape-measured). The
    # correction divides the commanded distance by the measured
    # rangefinder-to-EKF height ratio.
    move_scale_by_height: bool = True
    move_scale_min: float = 0.9
    move_scale_max: float = 1.6
    yaw_settle_s: float = 3.0
    rotation_timeout_pad_s: float = 25.0
    reassert_position_before_yaw: bool = True

    flow_quality_warn: int = 50         # advisory only, never aborts


# ──────────────────────────── top level ──────────────────────────

@dataclass
class Config:
    link: LinkConfig = field(default_factory=LinkConfig)
    rangefinder: RangefinderConfig = field(default_factory=RangefinderConfig)
    forward_rf: ForwardRangefinderConfig = field(
        default_factory=ForwardRangefinderConfig)
    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    landing: LandingConfig = field(default_factory=LandingConfig)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    skills: SkillDefaults = field(default_factory=SkillDefaults)

    # WHICH AIRCRAFT. There is more than one, they behave differently, and
    # a log that does not say which one flew is not evidence of anything.
    # Two airframes share this stack: a 2.2" cinewhoop (downward rangefinder
    # + optical flow) and a 3.5" cinewhoop (the same, plus a forward MT-15).
    # Their hover throttle, climb authority and sensor count all differ, so
    # a figure measured on one does not transfer to the other. Set it with
    # --airframe; it lands in the log's meta record.
    airframe: str = "unknown"
    pilot_takeover_abort: bool = True
    print_period_s: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialised into the first record of every log, so a flight is
        reproducible from its own log file."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """
        Rebuild the configuration a flight was flown with.

        Replay MUST use this rather than a fresh default. A skill's timing
        comes from the config (leg dwells, settle times, setpoint period), so
        replaying a flight under today's defaults runs a different skill than
        the one that was recorded — which shows up as a confusing
        stream_exhausted rather than an obvious error.

        Unknown keys are ignored, so a log from an older version still loads.
        """
        sections = {
            "link": LinkConfig, "rangefinder": RangefinderConfig,
            "forward_rf": ForwardRangefinderConfig,
            "envelope": EnvelopeConfig, "landing": LandingConfig,
            "preflight": PreflightConfig, "skills": SkillDefaults,
        }
        kwargs: Dict[str, Any] = {}
        for name, klass in sections.items():
            sub = d.get(name) or {}
            valid = {f.name for f in fields(klass)}
            vals = {k: v for k, v in sub.items() if k in valid}
            # A key missing from a log means the log PREDATES it, and the
            # right value is how the stack behaved then -- not today's
            # default. Without this, replaying an old flight silently runs
            # a different skill from the one that flew.
            for k, v in LEGACY_DEFAULTS.get(name, {}).items():
                if k not in sub:
                    vals[k] = v
            kwargs[name] = klass(**vals)
        # Derived, not listed. A hardcoded tuple here means every new
        # top-level field is silently dropped on replay until someone
        # remembers to add it, and the symptom is a replay that differs
        # from the flight for no visible reason. `airframe` was lost
        # exactly that way.
        for f in fields(cls):
            if f.name in sections or f.name in kwargs:
                continue
            if f.name in d:
                kwargs[f.name] = d[f.name]
        return cls(**kwargs)


# What the stack did before a setting existed, keyed like the config. Used
# only when rebuilding the configuration of a log that lacks the key.
LEGACY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # Before 0.15.0 a leg completed on the first heading sample inside the
    # tolerance.
    # Before 0.20.0 `move` commanded EKF metres, not real metres.
    "skills": {"yaw_confirm_samples": 1, "move_scale_by_height": False},
}

DEFAULT = Config()


# ───────────────────────── MAVLink constants ─────────────────────

COPTER_MODES = {
    "STABILIZE": 0, "ACRO": 1, "ALT_HOLD": 2, "AUTO": 3, "GUIDED": 4,
    "LOITER": 5, "RTL": 6, "CIRCLE": 7, "LAND": 9, "DRIFT": 11,
    "SPORT": 13, "FLIP": 14, "AUTOTUNE": 15, "POSHOLD": 16, "BRAKE": 17,
    "THROW": 18, "AVOID_ADSB": 19, "GUIDED_NOGPS": 20, "SMART_RTL": 21,
    "FLOWHOLD": 22, "FOLLOW": 23, "ZIGZAG": 24, "SYSTEMID": 25,
    "AUTOROTATE": 26, "AUTO_RTL": 27,
}
MODE_NAMES = {v: k for k, v in COPTER_MODES.items()}

MASK_POSITION_ONLY = 0b110111111000
MASK_POSITION_YAW = 0b100111111000

# Approximate MAVLink2 wire sizes, for the link budget report.
WIRE_SIZE = {
    "SET_POSITION_TARGET_LOCAL_NED": 65,
    "COMMAND_LONG": 45,
    "HEARTBEAT": 21,
    "ATTITUDE": 40,
    "LOCAL_POSITION_NED": 40,
    "DISTANCE_SENSOR": 50,
    "EXTENDED_SYS_STATE": 14,
    "OPTICAL_FLOW": 38,
}
