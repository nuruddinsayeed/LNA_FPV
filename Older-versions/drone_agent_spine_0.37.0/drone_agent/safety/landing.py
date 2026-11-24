"""
The landing state machine.

Land at any cost. Command LAND, verify it engaged, keep re-commanding until it
does, watch the descent, confirm touchdown from independent signals, then
disarm. Survives link dropouts. Never disarms in the air.

This is the direct descendant of the incident fix, and the two rules it
encodes are the ones that matter:

  1. An abort is a LANDING, never a disarm. The original branched to disarm()
     on a single altitude reading; during a rotation the aircraft bumped
     something, the rangefinder lied, and the script tried to disarm a flying
     aircraft. The autopilot refused, the script exited, and the drone was
     left flying with nobody commanding it. That branch does not exist here.

  2. There is no force-disarm (the 21196 magic value). A refused disarm is the
     autopilot being right.

Airborne is sticky: set when takeoff is confirmed, cleared only by confirmed
touchdown. No instantaneous sensor reading can flip it.
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Optional

from ..config import Config, MODE_NAMES
from ..world.state import WorldState


class LandingController:

    def __init__(self, vehicle: Any, cfg: Config,
                 on_event: Optional[Callable[[str, dict], None]] = None,
                 verbose: bool = True) -> None:
        self.vehicle = vehicle
        self.cfg = cfg
        self.on_event = on_event or (lambda kind, payload: None)
        self.verbose = verbose

        self.airborne = False
        self.active = False

    # ------------------------------------------------------------------

    def mark_airborne(self) -> None:
        self.airborne = True

    def force_land(self, reason: str, allow_disarm: bool = True) -> bool:
        """Returns True if the aircraft ended up disarmed on the ground."""
        if self.active:
            return False
        self.active = True
        c = self.cfg.landing
        self.vehicle.expected_mode = None       # stop takeover detection

        land_mode = self.vehicle.mode_map.get("LAND", 9)
        guided_mode = self.vehicle.mode_map.get("GUIDED", 4)

        self._say("\n" + "=" * 70)
        self._say(f"FORCE LAND: {reason}")
        self._say("=" * 70)
        self.on_event("land_begin", {"reason": reason})

        deadline = time.monotonic() + c.total_timeout_s
        t_next_cmd = 0.0
        attempts = 0
        confirmed = False
        t_ground_since: Optional[float] = None
        t_ground_confirmed: Optional[float] = None
        warned_link = False

        try:
            while time.monotonic() < deadline:
                world = self.vehicle.poll()
                mode_num = world.mode_num
                link_up = world.link_up

                # ---- already down? --------------------------------------
                if not world.armed:
                    if self.airborne and not confirmed:
                        self._say("Vehicle is disarmed but we never confirmed "
                                  "a landing. Check the aircraft.")
                        self.on_event("land_end",
                                      {"result": "disarmed_unconfirmed"})
                    else:
                        self._say("Vehicle disarmed - on the ground.")
                        self.on_event("land_end", {"result": "disarmed"})
                    self.airborne = False
                    return True

                # ---- pilot took a mode of their own? --------------------
                if mode_num is not None and mode_num not in (land_mode,
                                                             guided_mode):
                    name = MODE_NAMES.get(mode_num, str(mode_num))
                    self._say(f"Mode is {name} - pilot or autopilot failsafe "
                              f"has control. Standing down, not fighting it.")
                    self.on_event("land_end", {"result": "pilot_has_control",
                                               "mode": name})
                    return False

                # ---- link state -----------------------------------------
                if not link_up:
                    if not warned_link:
                        warned_link = True
                        self._say("Link is down. Still transmitting LAND in "
                                  "case it comes back; ArduPilot RC failsafe "
                                  "should be acting meanwhile.")
                elif warned_link:
                    warned_link = False
                    self._say("Link recovered.")

                # ---- get into LAND and keep it there --------------------
                if mode_num == land_mode:
                    if not confirmed:
                        confirmed = True
                        self._say("LAND mode CONFIRMED from heartbeat. "
                                  "Descending.")
                else:
                    if confirmed:
                        self._say("Dropped out of LAND - re-commanding.")
                        confirmed = False
                    if time.monotonic() >= t_next_cmd:
                        attempts += 1
                        t_next_cmd = time.monotonic() + c.retry_period_s
                        self._say(f"Commanding LAND (attempt {attempts})...")
                        self.vehicle.command_mode_oneshot(land_mode)
                        if attempts % c.nav_escalate_every == 0:
                            # Different code path in the autopilot; worth
                            # trying when DO_SET_MODE is not taking.
                            self._say("  escalating: MAV_CMD_NAV_LAND")
                            self.vehicle.command_nav_land()

                # ---- touchdown detection --------------------------------
                down, ev = world.on_ground(c.ground_alt_m, c.ground_vz_max)
                if down:
                    if t_ground_since is None:
                        t_ground_since = time.monotonic()
                        self._say(f"Ground evidence: {', '.join(ev)}")
                    elif (t_ground_confirmed is None
                          and time.monotonic() - t_ground_since
                          >= c.ground_confirm_s):
                        t_ground_confirmed = time.monotonic()
                        self.airborne = False
                        self._say(f"TOUCHDOWN confirmed ({', '.join(ev)}). "
                                  f"Waiting {c.disarm_after_ground_s:.0f}s for "
                                  f"the FC's own auto-disarm before "
                                  f"commanding one.")
                        self.on_event("touchdown", {"evidence": ev})
                else:
                    if t_ground_since is not None:
                        self._say("Ground evidence lost - still airborne.")
                    t_ground_since = None
                    t_ground_confirmed = None

                # ---- disarm, only once genuinely down -------------------
                if (allow_disarm and t_ground_confirmed is not None
                        and time.monotonic() - t_ground_confirmed
                        >= c.disarm_after_ground_s):
                    self._say("Commanding disarm.")
                    if self.vehicle.disarm():
                        self.airborne = False
                        self.on_event("land_end", {"result": "landed"})
                        return True
                    t_ground_confirmed = time.monotonic()   # back off, retry

                self._progress(world)
                time.sleep(0.1)

            self._say("\n" + "!" * 70)
            self._say("FORCE LAND TIMED OUT and the vehicle is still armed.")
            self._say("TAKE MANUAL CONTROL NOW - flip your mode switch.")
            self._say("!" * 70)
            self.on_event("land_end", {"result": "timeout"})
            return False
        finally:
            self.active = False

    # ------------------------------------------------------------------

    _t_last_progress = 0.0

    def _progress(self, world: WorldState) -> None:
        now = time.monotonic()
        if now - self._t_last_progress < self.cfg.print_period_s:
            return
        self._t_last_progress = now
        alt = world.altitude_m
        rf = world.rngfnd_down_raw_m
        self._say(
            f"[LAND] mode={world.mode} "
            f"alt={f'{alt:.2f}' if alt is not None else ' n/a'} "
            f"vz={f'{world.pose.vz:.2f}' if world.pose and world.pose.vz is not None else ' n/a'} "
            f"rf={f'{rf:.2f}' if rf is not None else ' n/a'}"
            f"{'' if world.rngfnd_down_trusted else '(distrusted)'} "
            f"landed_state={world.landed_state}")

    def _say(self, line: str) -> None:
        if self.verbose:
            print(line)
