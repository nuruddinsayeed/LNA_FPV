"""
The executor: preflight, the ALT_HOLD -> GUIDED handover, and the tick loop.

Three responsibilities, kept together because they share the same failure
handling: anything that goes wrong here lands the aircraft.

The handover
------------
On a flow-only airframe the EKF cannot produce a horizontal position estimate
on the ground — there is no motion for the flow sensor to observe and the
aircraft is below the sensor's minimum working distance — so GUIDED's pre-arm
check fails with "Need Position Estimate". The sequence is therefore:

    arm in ALT_HOLD -> pilot climbs -> verify the estimate is actually
    healthy -> hand over to GUIDED

and that handover is a first-class, logged step rather than a mode flip on the
sticks that the script hopes it noticed. "Verify" means a real criterion:
LOCAL_POSITION_NED arriving fresh, altitude above a floor, and N consecutive
samples whose spread is small enough that the estimate is not wandering.

Everything from here raises up into a landing. There is no code path from an
abort to a disarm.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from .config import Config
from .interfaces.vehicle import (FlightError, LinkLost, MavlinkVehicle,
                                 PilotTakeover)
from .recording.recorder import FlightRecorder
from .safety.arbiter import Arbiter, AbortRequested, SetpointRequest
from .safety.landing import LandingController
from .skills.base import Skill, SkillResult, SkillStatus
from .world.state import WorldState


class Executor:

    def __init__(self, vehicle: MavlinkVehicle, cfg: Config,
                 recorder: FlightRecorder, verbose: bool = True) -> None:
        self.vehicle = vehicle
        self.cfg = cfg
        self.recorder = recorder
        self.verbose = verbose

        # Constructing the Arbiter mints the vehicle's one actuation token.
        self.arbiter = Arbiter(vehicle, cfg)
        self.landing = LandingController(
            vehicle, cfg, on_event=recorder.event, verbose=verbose,
            on_tick=self._record_landing_tick)

        self._t_last_print = 0.0

        # Receive-only: every autopilot STATUSTEXT becomes a timed event in
        # the flight log.
        vehicle.on_statustext = (
            lambda txt: recorder.event("statustext", {"text": txt}))

    # ═══════════════════════════ preflight ═══════════════════════

    def preflight(self) -> None:
        for line in self.vehicle.report_budget():
            self._say(line)
        self.recorder.note("link budget reported")

        self.vehicle.connect()
        self.vehicle.configure_streams()

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            self.vehicle.poll()
            time.sleep(0.02)

        if self.vehicle.is_armed():
            raise FlightError("Vehicle is already armed - refusing to start.")

        if not self.vehicle.wait_for(lambda: self.vehicle._yaw is not None,
                                     "ATTITUDE"):
            raise FlightError("No ATTITUDE stream - yaw control would be "
                              "blind.")
        if not self.vehicle.wait_for(lambda: self.vehicle._pos is not None,
                                     "LOCAL_POSITION_NED"):
            raise FlightError("No LOCAL_POSITION_NED - no position estimate.")

        if self.cfg.preflight.enabled:
            self._link_check()

    def _link_check(self) -> None:
        from pymavlink import mavutil

        c = self.cfg.preflight
        self._say(f"\nMeasuring link quality over {c.window_s:.0f}s...")
        for k in list(self.vehicle.msg_counts):
            self.vehicle.msg_counts[k] = 0

        t0 = time.monotonic()
        while time.monotonic() - t0 < c.window_s:
            self.vehicle.poll()
            time.sleep(0.02)
        window = time.monotonic() - t0

        counts = self.vehicle.msg_counts
        att_hz = counts.get("ATTITUDE", 0) / window
        pos_hz = counts.get("LOCAL_POSITION_NED", 0) / window
        hb_hz = counts.get("HEARTBEAT", 0) / window
        ess = counts.get("EXTENDED_SYS_STATE", 0)

        self._say(f"  ATTITUDE           {att_hz:.1f} Hz")
        self._say(f"  LOCAL_POSITION_NED {pos_hz:.1f} Hz")
        self._say(f"  HEARTBEAT          {hb_hz:.1f} Hz")
        self._say(f"  EXTENDED_SYS_STATE {ess} msgs")
        self._say(f"  OPTICAL_FLOW       {counts.get('OPTICAL_FLOW', 0)} msgs "
                  f"(quality={self.vehicle._flow_quality})")
        for line in self.vehicle.rf.describe():
            self._say(line)

        t_sent = time.monotonic()
        self.vehicle.send_command(
            mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
            [mavutil.mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION],
            retries=2, label="RTT probe")
        rtt = time.monotonic() - t_sent
        self._say(f"  command round trip {rtt * 1000:.0f} ms")

        report = {"attitude_hz": round(att_hz, 2),
                  "position_hz": round(pos_hz, 2),
                  "heartbeat_hz": round(hb_hz, 2),
                  "extended_sys_state_msgs": ess,
                  "command_rtt_s": round(rtt, 3),
                  "rf_streams": len(self.vehicle.rf.streams)}
        self.recorder.event("preflight", report)

        problems: List[str] = []
        if att_hz < c.min_attitude_rate_hz:
            problems.append(f"ATTITUDE only {att_hz:.1f} Hz")
        if pos_hz < c.min_position_rate_hz:
            problems.append(f"LOCAL_POSITION_NED only {pos_hz:.1f} Hz")
        if rtt > c.max_command_rtt_s:
            problems.append(f"command round trip {rtt:.2f}s exceeds "
                            f"{c.max_command_rtt_s}s")
        if hb_hz < c.min_heartbeat_hz:
            problems.append(f"heartbeat only {hb_hz:.1f} Hz")

        if ess == 0:
            # Not fatal, but say so: it is the best touchdown signal we have.
            self._say("  NOTE: no EXTENDED_SYS_STATE. Touchdown detection "
                      "will fall back to rangefinder + EKF agreement, which "
                      "is weaker.")

        if problems:
            for p in problems:
                self._say("  FAIL: " + p)
            raise FlightError("Link quality insufficient for guided flight.")
        self._say("  Link OK.")

    # ═══════════════════════════ handover ════════════════════════

    def launch(self, handover: Optional[bool] = None,
               takeoff_height_m: Optional[float] = None) -> None:
        """
        Get the aircraft airborne and into GUIDED with a trusted estimate.

        Default is auto-takeoff: arm in GUIDED and use MAV_CMD_NAV_TAKEOFF.
        Pass handover=True only when GUIDED refuses to pre-arm on the ground
        — that path needs a pilot on the throttle and will auto-disarm if it
        does not get one.
        """
        use_handover = (self.cfg.preflight.handover_enabled
                        if handover is None else handover)
        if use_handover:
            self._launch_handover()
        else:
            self._launch_guided(takeoff_height_m)

        world = self.vehicle.poll()
        if world.pose is not None:
            self.arbiter.set_origin(world.pose.x, world.pose.y, world.pose.z)
            self.recorder.event("origin_captured", {
                "x": round(world.pose.x, 3), "y": round(world.pose.y, 3),
                "z": round(world.pose.z, 3),
                "yaw_deg": round(world.pose.yaw_deg, 1)})
        self._report_rf_vs_ekf(world)

    def _launch_guided(self, takeoff_height_m: Optional[float]) -> None:
        height = (takeoff_height_m if takeoff_height_m is not None
                  else self.cfg.skills.takeoff_height_m)
        self._say("\n>>> Pilot: stay on the sticks. Flipping out of GUIDED "
                  "aborts the script and returns control immediately. <<<\n")
        self.recorder.event("launch_begin", {"mode": "guided_auto_takeoff",
                                             "height_m": height})
        self.vehicle.set_mode("GUIDED")
        self.vehicle.arm()
        self.vehicle.takeoff(height)
        climb = self._wait_for_climb(height)
        self.landing.mark_airborne()
        self.recorder.event("launch_end", {"result": "airborne", **climb})

    def _launch_handover(self) -> None:
        c = self.cfg.preflight
        self._say("\n" + "=" * 70)
        self._say("HANDOVER SEQUENCE (flow-only: GUIDED cannot pre-arm on the "
                  "ground)")
        self._say("=" * 70)
        self._say(f"  1. arming in {c.handover_arm_mode}")
        self._say(f"  2. PILOT: climb above {c.handover_min_alt_m:.1f} m and "
                  f"hold")
        self._say(f"  3. estimate verified -> {c.handover_verify_mode} -> "
                  f"GUIDED")
        self._say("")
        self._say(f"  This path needs YOU on the throttle. ArduPilot "
                  f"auto-disarms after")
        self._say(f"  DISARM_DELAY (10 s by default) if the throttle stays at "
                  f"minimum,")
        self._say(f"  so raise it as soon as the aircraft arms.")
        self._say(">>> Keep your hand on the mode switch throughout. <<<\n")
        self.recorder.event("handover_begin", {
            "arm_mode": c.handover_arm_mode,
            "verify_mode": c.handover_verify_mode,
            "min_alt_m": c.handover_min_alt_m})

        if c.handover_confirm:
            try:
                input("  Ready on the sticks? Press Enter to arm "
                      "(Ctrl-C to abort): ")
            except EOFError:
                pass        # non-interactive: carry on

        self.vehicle.set_mode(c.handover_arm_mode)
        self.vehicle.arm()

        self._say(f"Armed in {c.handover_arm_mode}. CLIMB NOW; waiting for a "
                  f"healthy position estimate...")
        ok, detail = self._wait_for_healthy_estimate()
        if not ok:
            raise FlightError(f"handover aborted: {detail.get('reason')}")

        self.landing.mark_airborne()
        self.recorder.event("estimate_healthy", detail)
        self._say(f"Estimate healthy: {detail}")

        # LOITER first: if the EKF is lying, the aircraft will show it here,
        # under a mode the pilot can take back instantly.
        if c.handover_verify_mode:
            self.vehicle.set_mode(c.handover_verify_mode, required=False)
            t_end = time.monotonic() + 3.0
            while time.monotonic() < t_end:
                self.vehicle.poll()
                time.sleep(0.05)

        self.vehicle.set_mode("GUIDED")
        self.recorder.event("handover_end", {"result": "guided"})
        self._say("GUIDED accepted. The script has control.")

    def _wait_for_healthy_estimate(self) -> tuple:
        """
        Real criterion, not a sleep: fresh LOCAL_POSITION_NED, altitude above
        the floor, and N consecutive samples whose spread stays small.
        """
        c = self.cfg.preflight
        deadline = time.monotonic() + c.handover_timeout_s
        samples: List[tuple] = []
        t_armed = time.monotonic()
        best_alt = 0.0

        while time.monotonic() < deadline:
            world = self.vehicle.poll()
            if not world.armed:
                held = time.monotonic() - t_armed
                if best_alt < c.handover_min_alt_m:
                    # The aircraft never left the ground. This is almost
                    # always ArduPilot's own auto-disarm: throttle at minimum
                    # for DISARM_DELAY seconds after arming.
                    return False, {
                        "reason": (
                            f"auto-disarmed on the ground after {held:.0f}s "
                            f"without a climb (max altitude {best_alt:.2f} m "
                            f"of the {c.handover_min_alt_m:.1f} m needed)"),
                        "cause": "likely DISARM_DELAY - throttle stayed at "
                                 "minimum",
                        "fix": "raise the throttle as soon as it arms, or "
                               "drop --handover and use auto-takeoff",
                        "armed_for_s": round(held, 1),
                        "max_alt_m": round(best_alt, 3)}
                return False, {"reason": f"disarmed during handover after "
                                         f"{held:.0f}s",
                               "max_alt_m": round(best_alt, 3)}
            try:
                self.vehicle.check_link()
            except LinkLost as e:
                return False, {"reason": str(e)}

            alt = world.altitude_m
            if alt is not None:
                best_alt = max(best_alt, alt)
            if (world.pose is None or not world.pose_fresh
                    or alt is None or alt < c.handover_min_alt_m):
                samples.clear()
                self._progress(world, "WAIT-HANDOVER")
                time.sleep(0.05)
                continue

            samples.append((world.pose.x, world.pose.y))
            if len(samples) > c.handover_pose_samples:
                samples.pop(0)

            if len(samples) == c.handover_pose_samples:
                xs = [s[0] for s in samples]
                ys = [s[1] for s in samples]
                spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                if spread <= c.handover_pose_max_spread_m:
                    return True, {"alt_m": round(alt, 2),
                                  "spread_m": round(spread, 3),
                                  "samples": len(samples),
                                  "flow_quality": world.flow_quality}
            self._progress(world, "WAIT-HANDOVER")
            time.sleep(0.05)

        return False, {"reason": "timed out waiting for a healthy position "
                                 "estimate"}

    def _wait_for_climb(self, height_m: float) -> Dict[str, Any]:
        """
        Wait until the best height estimate (the rangefinder) reaches 85% of
        the takeoff height. This is the 0.14 decision, restored exactly.

        0.15.0 and 0.15.1 judged the climb on the EKF's altitude instead, on
        the theory that the hold point is captured from the EKF so the test
        should be too. The theory was wrong about the autopilot: with
        WP_RFND_USE=1 a GUIDED takeoff climbs to the requested height ABOVE
        THE GROUND AS THE RANGEFINDER SEES IT, so the EKF finishes at
        (height - rangefinder/EKF gap), not at height. Flown on the 2.2in
        airframe on 2026-11-15:

          the EKF never climbed 0.68 m in one flight (ground reading 0.12 m,
          gap 0.16 m, climb 0.625 m), so the wait ran its full 30 s with the
          aircraft hovering before the sweep began;

          the other two captured their hold later in the climb, flew at 1.05
          and 0.98 m instead of 0.78-0.88 m, and one reached 1.30 m against
          a 1.2 m ceiling.

        It also did not buy what it was for. The commanded height became
        repeatable (0.72-0.75 m, from 0.53-0.70) but the REAL height got
        less repeatable (0.88-1.05 m, from 0.78-0.88): the real height is
        the commanded one plus an offset that follows the EKF's vertical-
        velocity error, and crossing a rangefinder threshold cancels part
        of that offset where crossing an EKF threshold does not.

        The returned dict is a record only. It decides nothing.
        """
        c = self.cfg.preflight
        world = self.vehicle.poll()
        start_alt = world.altitude_m or 0.0
        ekf0 = world.pose.alt_m if world.pose is not None else None
        goal = height_m * 0.85
        best = start_alt
        deadline = time.monotonic() + c.climb_timeout_s

        def report(w: WorldState, timed_out: bool) -> Dict[str, Any]:
            out: Dict[str, Any] = {"climb_frame": "best_estimate",
                                   "climb_timed_out": timed_out}
            if w.pose is not None and ekf0 is not None:
                out["ekf_climb_m"] = round(w.pose.alt_m - ekf0, 3)
                out["ekf_ground_m"] = round(ekf0, 3)
            if w.altitude_m is not None:
                out["alt_best_m"] = round(w.altitude_m, 3)
            return out

        while time.monotonic() < deadline:
            world = self.vehicle.poll()
            self.vehicle.check_link()
            self.vehicle.check_takeover()
            if not world.armed:
                raise FlightError("Vehicle disarmed during takeoff.")

            # Sticky: once the FC says we are off the ground we stay airborne
            # until landing is confirmed, whatever any sensor says later.
            if world.landed_state in (2, 3):     # TAKEOFF, IN_AIR
                self.landing.mark_airborne()

            alt = world.altitude_m
            if alt is not None:
                best = max(best, alt)
                if alt - start_alt > c.min_climb_m:
                    self.landing.mark_airborne()
                if alt >= goal:
                    self.landing.mark_airborne()
                    self._say(f"Reached {alt:.2f} m")
                    return report(world, False)
            time.sleep(0.05)

        if best - start_alt < c.min_climb_m and not self.landing.airborne:
            raise FlightError(
                f"Vehicle never left the ground (max {best:.2f} m from "
                f"{start_alt:.2f} m). Aborting.")
        self.landing.mark_airborne()
        self._say(f"Takeoff height not fully reached (max {best:.2f} m) but "
                  f"airborne.")
        return report(world, True)

    def _report_rf_vs_ekf(self, world: WorldState) -> None:
        rf = world.rngfnd_down_raw_m
        if rf is None or world.pose is None:
            return
        ekf = world.pose.alt_m
        diff = rf - ekf
        learned = self.vehicle.rf.down.offset
        self._say(f"  rangefinder {rf:.2f} m vs EKF-implied {ekf:.2f} m "
                  f"(offset {diff:+.2f} m, learned "
                  f"{learned if learned is None else f'{learned:+.2f}'} m)")
        if abs(diff) > 0.15:
            # If the EKF were fusing the rangefinder for height these would
            # track each other closely. A persistent bias this size suggests
            # EK3_SRC1_POSZ is on baro, which indoors drifts and cannot see
            # the floor - meaning EKF altitude and rangefinder altitude are
            # two genuinely different numbers all flight.
            self._say(f"  NOTE: {abs(diff):.2f} m standing disagreement "
                      f"between rangefinder and EKF height. Check "
                      f"EK3_SRC1_POSZ (should be RangeFinder for this "
                      f"airframe) and RNGFND1_GNDCLEAR.")
        self.recorder.event("rf_vs_ekf", {"rf_m": round(rf, 3),
                                          "ekf_m": round(ekf, 3),
                                          "offset_m": round(diff, 3)})

    # ═══════════════════════════ tick loop ═══════════════════════

    def run_skill(self, skill: Skill) -> SkillResult:
        period = 1.0 / self.cfg.skills.tick_hz
        world = self.vehicle.poll()

        pre = skill.check_preconditions(world)
        self.recorder.skill_start(skill.name, skill.params,
                                  {"ok": pre.ok, "reason": pre.reason,
                                   "detail": pre.detail})
        if not pre.ok:
            skill.status = SkillStatus.FAILED
            skill._reason = f"precondition: {pre.reason}"
            skill._detail = pre.detail
            result = skill.result(world)
            self.recorder.skill_result(result)
            self._say(f"[{skill.name}] precondition FAILED: {pre.reason}")
            return result

        self._say(f"\n--- {skill.name} ---")
        skill.start(world)
        t_prev = world.t_mono

        try:
            while True:
                world = self.vehicle.poll()
                dt = world.t_mono - t_prev
                t_prev = world.t_mono

                self.vehicle.check_link()
                self.vehicle.check_takeover()

                # Rangefinder low-altitude abort, debounced and trust-gated.
                low = self.vehicle.rf.down.check_low()
                if low:
                    raise AbortRequested(low)

                status = skill.tick(world, dt)
                requested = skill.request

                decision = self.arbiter.submit(requested, world)

                self.recorder.tick(
                    world=world, skill_name=skill.name,
                    skill_status=status.value,
                    requested=requested.to_log(),
                    applied=decision.applied.to_log(),
                    arbiter_actions=decision.actions,
                    blocked=decision.blocked,
                    arbiter_counters=dict(self.arbiter.counters))

                if decision.intervened:
                    self._say(f"  [ARB] {', '.join(decision.actions) or 'blocked'}")

                self._progress(world, skill.name)

                if status in (SkillStatus.SUCCEEDED, SkillStatus.FAILED,
                              SkillStatus.ABORTED):
                    break
                time.sleep(max(0.0, period - (time.monotonic() - world.t_mono)))

        except AbortRequested as e:
            skill.on_abort(world, e.reason)
            skill.abort(e.reason)
        except PilotTakeover:
            skill.on_abort(world, "pilot takeover")
            skill.abort("pilot takeover")
            result = skill.result(world)
            self.recorder.skill_result(result)
            raise
        except (LinkLost, FlightError) as e:
            skill.on_abort(world, str(e))
            skill.abort(str(e))

        result = skill.result(world)
        self.recorder.skill_result(result)
        self._say(f"[{skill.name}] {result.status.value}"
                  f"{': ' + result.reason if result.reason else ''}")
        if result.metrics:
            self._say(f"  metrics: {result.metrics}")
        return result

    # ═══════════════════════════ helpers ═════════════════════════

    def land(self, reason: str) -> bool:
        return self.landing.force_land(reason)

    def _record_landing_tick(self, world: WorldState) -> None:
        # Rate-limited to the skill tick rate: the landing loop polls faster.
        now = time.monotonic()
        if now - getattr(self, "_t_last_land_tick", 0.0) < (
                1.0 / max(1.0, self.cfg.skills.tick_hz)):
            return
        self._t_last_land_tick = now
        self.recorder.tick(world, "land", "running", None, None, [], False)

    def _progress(self, world: WorldState, tag: str) -> None:
        now = time.monotonic()
        if now - self._t_last_print < self.cfg.print_period_s:
            return
        self._t_last_print = now
        p = world.pose
        pos = (f"{p.x:6.2f} {p.y:6.2f} {p.z:6.2f}" if p else "   n/a")
        yaw = f"  yaw {p.yaw_deg:6.1f}" if p else ""
        rf = ""
        if world.rngfnd_down_raw_m is not None:
            rf = (f"  RF {world.rngfnd_down_raw_m:.2f}m"
                  f"{'' if world.rngfnd_down_trusted else '!'}")
        fq = ""
        if world.flow_quality is not None:
            warn = ("!" if world.flow_quality
                    < self.cfg.skills.flow_quality_warn else "")
            fq = f"  flow {world.flow_quality:3d}{warn}"
        self._say(f"[{tag}] {pos}{yaw}{rf}{fq}")

    def _say(self, line: str) -> None:
        if self.verbose:
            print(line)

    def summary(self) -> Dict[str, Any]:
        elapsed = max(1e-6, time.monotonic() - self.recorder._t0)
        return {
            "uplink_bps_avg": round(self.vehicle.uplink_bytes / elapsed, 1),
            "elapsed_s": round(elapsed, 2),
            "messages": dict(sorted(self.vehicle.msg_counts.items())),
            "arbiter": self.arbiter.snapshot(),
            "rangefinder_down": self.vehicle.rf.down.snapshot(),
            "rangefinder_fwd": self.vehicle.rf.fwd.snapshot(),
            "fwd_floor_gated": self.vehicle.rf.fwd_gated,
            "airborne_at_exit": self.landing.airborne,
        }
