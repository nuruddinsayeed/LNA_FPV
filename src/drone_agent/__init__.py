"""
Ground-station control stack for an indoor GPS-denied ArduPilot quad.

The version below is bumped whenever the tool surface changes. It is printed
by every tool and written into each flight log's meta record, so a log always
says which build produced it and a stale extracted copy announces itself
instead of failing with "unrecognized arguments".
"""

__version__ = "0.40.0"

# What changed, newest first — the tool surface, not the internals.
#
# 0.24.1 The head range now includes the aircraft's own pitch, which
#        carries the camera with it: 0.7 to 1.9 deg nose-down while
#        holding station, worth about 0.2 m at 2.5 m. On the three
#        tape-measured flights with a 1.80 m target it brings the six
#        looks to a median of 2.40 m against 2.57 m, and the tilt those
#        looks imply to 15.0 deg.
#        `range_check.py --truth` now solves for the tilt the looks would
#        need, so it can be compared against the one measured at a wall:
#        tilt and target height are two unknowns and the tape pins the
#        answer, so agreement between the two independent measurements is
#        the test of the method rather than an assumption inside it. It
#        also refuses to invent a target height.
#        `calibrate_camera.py tilt` refuses the three numbers printed in
#        the documentation, because they were run verbatim once and the
#        resulting "measurement" was 16.17 deg of nothing.
# 0.24.0 The tape measurement settled the range question, and the answer
#        was no. Three flights at 20:25-20:27 with the person standing
#        2.54-2.60 m away and the sideways step tape-measured at 0.53-
#        0.55 m: parallax said 1.49, 2.41 and 2.16 m, -42%, -6% and -16%,
#        and its stated band missed the truth in one of the three.
#        WHY. The person's bearing, measured again once the aircraft had
#        returned to where look A was taken, had moved by 10 to 16 deg --
#        with the person standing still. The aircraft's heading changed
#        during the manoeuvre by far more than the EKF reported (it
#        reported -4.3, +0.3 and +1.3 deg). The parallax signal itself is
#        only about 12 deg, so the heading error swamps it. Parallax
#        ranging needs the heading to about a degree between the two
#        looks, and this aircraft's heading is not that good.
#        WHAT ELSE THE SAME DATA SHOWS. The camera points UP, by 10-18 deg
#        on two independent estimates, and nothing in a checkerboard
#        calibration can see that. It is why the person's feet are below
#        the frame in every single flight, and it inflates every bearing
#        of an object below the axis.
#        NEW, and it needs no flying: `calibrate_camera.py tilt` measures
#        that angle against a wall. CameraIntrinsics then reports angles in
#        the AIRFRAME's frame, and offers height_range(): the range from
#        the elevation of the TOP of a box, the camera's height above the
#        floor and how tall the object is. One look, no movement, no
#        heading, and it does not care that the bottom of the object is
#        out of frame. tools/range_check.py re-measures flights already
#        flown with it. On the three tape-measured flights, with the tilt
#        at 13 deg, the six looks give 2.38-2.72 m against 2.57 m: median
#        error 0.12 m, worst 0.19 m, against parallax's 0.42 m. The tilt
#        is the whole accuracy: 1 deg of it is 0.25 m at this range, so it
#        has to be measured, not guessed.
#        Nothing in the flight path changes until a tilt is in the
#        calibration file. NOT flown.
# 0.23.0 REMOVED the altitude trim of 0.22.0, at the pilot's instruction
#        and rightly. It did stop the climb, and it did it by flying the
#        aircraft at the floor. Three flights at 17:33-17:38: the trim
#        reached its authority against the envelope floor in two of them,
#        the commanded altitude sat at the 0.30 m minimum, the EKF
#        altitude averaged 0.19 m in one flight, and a measure_range was
#        aborted with the rangefinder reading 0.13 m. Position went with
#        it: optical flow quality fell from 56 to 35 as the aircraft came
#        down, and the worst horizontal excursion of any flight so far,
#        1.15 m, happened in the same run.
#        The loop was wrong in a way the simulation could not show. It is
#        an integrator with no damping against a plant that lags by more
#        than the second I modelled, so it wound up: it kept integrating
#        while the aircraft had not yet responded, and by the time it had,
#        the correction was already at its limit. A second fault: the
#        surface-step rebase drove the target to -0.14 m, which is not a
#        height at all. Holding a constant commanded altitude is wrong,
#        but this was worse, and worse in the dangerous direction.
#        Flight behaviour is now exactly 0.21.0's. The altitude finding
#        itself stands and is recorded in the project notes; a correct fix
#        needs damping, a rate limit tied to the measured response, and
#        most likely the autopilot's own onboard log to explain the
#        estimator fault instead of papering over it.
# 0.22.0 THE ALTITUDE TRIM, and it is the fix for the thing that has been
#        ending flights all day. Every holding skill captured one altitude
#        and commanded that number unchanged for as long as it ran. The
#        EKF's vertical velocity disagrees with its own altitude on this
#        aircraft, the autopilot adds thrust to cancel a descent that is
#        not happening, and the real height walks upward while the
#        commanded number sits still. Four flights at 14:42-15:48 ended
#        with the safety layer landing the aircraft, one from 2.33 m under
#        a 2.50 m ceiling.
#        Now a slow integrator from the rangefinder biases the commanded
#        altitude until the REAL height is the one that was captured:
#        0.4 per second, at most 0.25 m/s of command movement and 0.60 m
#        of authority, with an 8 cm deadband. It is one-sided on purpose.
#        A rangefinder reading long can only mean the aircraft is high;
#        reading short can mean that or a table underneath, so the loop
#        descends freely and climbs at most 10 cm past where it started.
#        A reading that steps faster than the aircraft can move is the
#        surface changing, not the aircraft: the target moves with it
#        (bounded, and only downward from where it started) and climbing
#        is blocked while it is measuring anything but the floor. In the
#        14:42 flight the rangefinder fell 1.30 -> 0.47 m in half a second
#        and rose to 2.33 m; run through the loop, that produces no climb
#        command at all. On the three flights whose drift was smooth the
#        loop reaches -0.25 m of correction within 3-5 s, open loop.
#        skills.altitude_trim turns it off; logs from before 0.22.0 replay
#        under the old rule. analyze_flight prints an ALTITUDE TRIM
#        section. NOT flown.
# 0.21.0 FIX, mine, and the reason two flights of 2026-11-16 turned the
#        wrong way and lost the person. The detector was reporting TWO
#        "person" boxes, and the search and the skill after it disagreed
#        about which was meant. The search applied its acceptance rule and
#        stopped on the box at +48 deg whose range it was sure of (sigma
#        0.32-0.45 m); face_target had no such rule and took whichever box
#        was nearest the nose at that instant, which was the other one, at
#        -39 deg, with a range sigma of 0.66-2.0 m. With two near-symmetric
#        boxes that choice also flipped frame to frame, so a median over
#        several frames mixed bearings to two different objects.
#        Now: rotate_search reports the track id it accepted; the executor
#        hands each step's result to the next; face_target and
#        measure_range follow that track and nothing else (target_lock.py),
#        re-acquiring by direction when the tracker issues a new id across
#        a turn, and refusing a box 87 deg from where the target should be.
#        Run standalone, with nothing handed over, they now choose as the
#        search would. Replayed on the two failed flights, both now command
#        the first turn toward the person (+48 and +45 deg) instead of away
#        from her.
#        Also: face_target no longer throws away sightings taken while it
#        settles after a turn, and a target that is already centred when
#        the detector loses it is reported as faced instead of failed --
#        at 15:02 the person was within a degree of the nose on four
#        frames and the flight failed anyway. measure_range steps with a
#        tighter arrival tolerance and now uses a short or timed-out step
#        rather than abandoning the measurement, since the distance
#        covered is measured, not assumed. NOT flown.
# 0.20.0 NEW SKILL measure_range: how far away a target is, by parallax.
#        Look while still, step sideways, look again, step back; the
#        bearing change and the real baseline give the range, with no prior
#        on the object's size and no need for the whole object to be in
#        frame. The detector's own range is unusable for this: four
#        flights facing the SAME standing person read 3.2, 4.0, 4.1 and
#        7.1 m, because her feet were below the bottom of the picture
#        every time. Every answer carries a bound on its error, computed
#        from the spread of its own bearing samples; a target too far for
#        the step is refused with the limit named rather than guessed at,
#        and the aircraft steps back to where it started in every ending,
#        including the failures. run_mission: measure_range
#        --target-class X [--baseline M], or --then-measure after
#        --then-face. In simulation at 3 m with a 0.5 m step: median error
#        0.18 m, worst 0.41 m over twelve runs, band contained the truth
#        12 times out of 12.
#        FIX, mine: `move` commanded EKF metres, not real ones. Six
#        tape-measured flights (2026-11-16) went 24% further than asked:
#        0.5 m -> 0.51-0.71 m, 1.0 m -> 1.25-1.31 m, while the rangefinder
#        read 1.20x the EKF height in the same flights. Flow velocity is
#        scaled by the height the EKF believes, so the EKF-frame goal is
#        now the asked-for distance divided by that measured ratio. On
#        those six flights the error would have fallen to 8% +- 10%.
#        skills.move_scale_by_height turns it off; logs from before 0.20.0
#        replay under the old rule. NOT flown.
# 0.19.0 NEW SKILL move: a straight displacement of at most 1 m relative
#        to the nose (forward_m, right_m), holding heading and height. The
#        first skill that translates the aircraft. The target is fed in
#        sub-steps of 0.25 m, one every 1.2 s, because the autopilot flies
#        to a position target at whatever speed its own parameters allow
#        and a target 0.25 m away cannot be taken fast. Refused before it
#        starts if it would end outside the fence radius. One plan may
#        move 2 m in total. The planner's prompt now says what moving can
#        and cannot do (a measured distance, not a named place or object).
#        run_mission: move --forward/--right. Distances are in the EKF
#        frame; the real distance can be longer where the EKF height reads
#        low. Tested on a simulated aircraft and link. NOT flown.
#        From the first face_target flights (2026-11-16 02:22-02:30, five
#        flights, two of them from an instruction with two steps): 4 of 4
#        ended within 3.5 deg of a standing person, in 1 to 3 turns; the
#        fifth gave up after four turns on a person who walked away, and
#        landed. After a ~45 deg turn the target was up to 10 deg from
#        where the first bearing put it, in both directions; the second
#        look absorbed it. The person's box was cut off by the frame in
#        every final view, and the size-prior range read 3.2 to 7.1 m.
# 0.18.0 NEW SKILL face_target: turn in place until an object that is
#        already in view is straight ahead. Every search so far (8 of 8
#        flights that found a person) ended with the person 38-44 deg off
#        the nose and cut off by the frame, where its range is a guess.
#        The skill looks while still, turns by the median bearing, and
#        looks again, up to four times; it never uses a bearing taken
#        while turning (the picture is ~0.25 s old), so an error in the
#        camera model or in the EKF heading costs another round, not the
#        result. It reports whether the box is still cut off. The yaw
#        mechanics are rotate_search's. run_mission: rotate_search
#        --target-class X --then-face flies search then face without the
#        language model; the planner can plan it (the skill is in the
#        catalogue and the prompt). FIX, mine: a plan stopped after ANY
#        step with a target_class whose result was not worded "target
#        acquired"; that rule is for a search only. Tested on a simulated
#        aircraft (old pictures, 17.5 deg/s, a camera 15% wrong, hand-over
#        mid-turn, target lost). NOT flown.
#        From the 0.17.1 benches (qwen3:8b, 33 runs each): with reasoning
#        33/33 valid on the first attempt, median 7.9 s; without
#        (--no-think) 33/33 valid, median 1.6 s, but 6 runs first returned
#        a PARTIAL plan for a request they should refuse and were turned
#        back by the validator ("go up to three metres" used all three
#        attempts).
# 0.17.1 FIX, mine, from the first bench against a real model (qwen3:8b
#        on Ollama, 33 runs). (1) The reply limit was 800 tokens. A
#        reasoning model thinks first, and on the longer instructions it
#        used all 800 thinking and returned nothing: 6 of 33 runs had no
#        plan, and "look around for a person" needed a second attempt
#        every time, 32 s instead of about 8. The limit is now 4096
#        (--max-tokens), a reply cut off at the limit is reported as that
#        and is not asked for again, and --no-think asks the server to
#        turn thinking off (reasoning_effort "none", Ollama). (2) The
#        duration estimate ignored rate_dps, so "turn slowly" at 5 deg/s
#        passed as 40 s when it would take about 94 s; the estimate is now
#        turning time plus 2.75 s per leg, and the prompt says what slow
#        costs. Flight code is as 0.17.0.
# 0.17.0 THE PLANNER. One instruction in plain language becomes a short
#        list of skill calls, made by a language model ON THE GROUND,
#        before the camera is opened and before the vehicle is connected.
#        The model's reply is untrusted text: parsed, checked against
#        every skill's schema, the room's altitude limit, a step count and
#        a one-battery duration, and sent back with the reasons when it
#        fails (twice at most). A request no skill can do must come back
#        as a refusal, and a refusal does not fly. The operator is shown
#        the plan and types y. In the air nothing asks the model again:
#        the steps run in order through the same executor, arbiter and
#        landing as before, the plan stops at a step that fails, and a
#        search that ends without its target skips what was planned after
#        it. drone_agent/planner.py (standard library only, any
#        OpenAI-compatible server); run_mission.py --instruction;
#        tools/plan_mission.py: `models`, `plan`, and `bench` (a file of
#        instructions, each N times: valid, refused-what-it-must,
#        repeatable, seconds). Naming a skill on the command line flies
#        exactly as 0.16.2. NOT run against the real model, NOT flown.
#        From the first three perception flights (2026-11-15 20:08-20:10,
#        all three stopped on a person 0.5-0.6 s after the first
#        sighting): the control loop is not slowed by the detector.
# 0.16.2 FIX, mine: calibrate_camera verify, live. It kept any six frames
#        that had a board in them, 0.4 s apart, without waiting for the
#        board to stop. `intrinsics` has always waited; `verify` did not.
#        An interlaced analog feed combs a moving board, so six pictures
#        taken in motion read as a bad calibration: the 2.2in camera's
#        file read POOR (5.20 px median) from six views grabbed in about
#        ten seconds, and the result cannot say whether the file or the
#        pictures are at fault. For scale, a PERFECT calibration of that
#        lens with the corner noise its own fit reports (0.78 px) shows
#        1.2-1.5 px on still views. verify now takes a view only when the
#        board has been still for --stable-s and is somewhere the earlier
#        views are not, says what it is waiting for, prints every view's
#        figure with where the board was, gives the result as an angle as
#        well as in pixels, and gives no verdict on fewer than three
#        views. Flight code is as 0.16.1.
# 0.16.1 FIX, mine, two of them in run_mission's perception path, found by
#        reading it against the ground tool before any perception flight.
#        (1) The detector was loaded AFTER takeoff: seconds of model
#        loading (a download, the first time) with the aircraft hovering.
#        (2) The camera was opened at the capture card's default 1920x1080
#        instead of the calibrated 720x576, which the calibration does not
#        fit; the runner noticed and only printed a warning. Perception now
#        starts before the MAVLink connection is opened, at the calibrated
#        size, and must be publishing scenes before anything is armed. If
#        it is not, the flight does not go (exit code 3, "NOT FLYING").
#        Default model yoloe-11s-seg.pt, the one the ground test measured;
#        --device and --imgsz. A failing camera no longer spins the
#        perception thread. watch_perception: `latency` times the
#        calibrated capture mode when given --intrinsics, and `watch`
#        reports how long its own loop was held up by the detector thread.
#        analyze_flight: PERCEPTION section (what was seen, when, in which
#        direction, and the picture's age recovered from the shift between
#        turning and holding). No change to flight behaviour without
#        --perception. NOT flown.
# 0.16.0 tools/watch_perception.py: perception on the ground. Flight code is
#        as 0.15.2. `watch` runs the detector, calibrated geometry and
#        staleness gate exactly as a flight would, with no vehicle and no
#        MAVLink, and reports rate, scene age at a 10 Hz tick, track
#        stability, bearing against a tape measure, and the share of
#        sightings rotate_search would stop on (its thresholds are read
#        from the skill, and offline_check pins the two to agree).
#        `latency` flashes the screen and times the change through the
#        analog chain; a second pass with a paced reader exposes a capture
#        backend that queues frames. The YOLOE adapter follows the current
#        documented set_classes() form with a fallback to the older one,
#        and keeps its last frame for annotated snapshots. The adapter is
#        still UNRUN against a real model: that is what this tool is for.
# 0.15.2 Flies as 0.14 did. The user flew 0.15.1 and reported that 0.14 was
#        better; three logs agree and both causes were mine.
#        (1) TAKEOFF, reverted to the 0.14 decision line for line. 0.15.0
#        judged the climb on the EKF's altitude; but with WP_RFND_USE=1 the
#        autopilot's own takeoff goes to the height the RANGEFINDER reads,
#        so the EKF finishes short by the gap between them. One flight
#        waited out the full 30 s timeout before starting (EKF climbed
#        0.625 m of the 0.68 m it was waiting for), and the others were
#        handed over later in the climb and flew at 1.05 and 0.98 m instead
#        of 0.78-0.88 m, one reaching 1.30 m. It made the commanded height
#        repeatable and the real height less so.
#        (2) LEG RULE, default back to 1 sample. The "bad heading sample"
#        that 0.15.0's two-sample rule was built for is the EKF re-aligning
#        its yaw to the compass; with STATUSTEXT logged (the one 0.15 change
#        that earned its place) all six sweeps show "EKF3 IMU0 MAG0 ground
#        mag anomaly, yaw re-aligned" at that instant. The rule added
#        1.6 s per sweep for nothing measured. yaw_confirm_samples=2
#        remains available.
#        Replay confirms the claim: the three 0.14 flights replayed under
#        0.15.2's default configuration make identical decisions.
#        analyze_flight attributes the heading step to the EKF, keeps that
#        leg out of the pooled yaw figures, reports takeoff time, flags a
#        flight the autopilot called low or critical on battery, and
#        compares REAL height between flights instead of the commanded one.
# 0.15.1 CORRECTION, mine, twice, both found by the user on the aircraft.
#        (1) MAV_GCS_SYSID must stay 255. With ELRS in MAVLink mode the
#        sticks arrive as MAVLink from the receiver, system 255, and the
#        autopilot takes RC input only from its GCS system id: setting 254
#        as check_params demanded gave "RC not found". It follows that the
#        GCS failsafe watches the receiver and cannot detect a dead script,
#        so FS_GCS_ENABLE=0 is no longer a BLOCK and the README no longer
#        claims that tier. (2) An altitude fence in the default above-home
#        frame tripped ~10 cm off the ground at a 1.4 m limit: with no
#        global position there is no home to measure from. check_params now
#        detects RC-over-MAVLink from the serial setup, blocks a GCS sysid
#        that would drop the RC, and blocks a fence frame that needs a
#        position the vehicle does not have. FENCE_ALT_MAX_TP=2 (above
#        origin) is the untested candidate.
# 0.15.0 From three baseline sweeps on the 2.2in airframe (2026-11-15).
#        CORRECTION, mine: the altitude fence is judged on the EKF's
#        altitude, which reads below the real height on this airframe (in
#        three clamp_alt flights: 1.47 / 1.38 / 1.53 m at real heights of
#        1.83 / 1.70 / 2.05 m). The FENCE_ALT_MAX=1.8 recommended in 0.14.1
#        would not have tripped in any of them and leaves 0.2 m under a
#        2.50 m ceiling. check_params now reasons about the fence in REAL
#        height against envelope.ceiling_m and blocks one that cannot stop
#        the aircraft; 1.4 is the value that would have tripped in all three.
#        A yaw leg now needs TWO successive heading samples inside the
#        tolerance: on the last leg of 3 sweeps out of 3 the autopilot
#        reported one sample ~18 deg ahead of both neighbours, which ended
#        the leg with the aircraft 10-13 deg short. Takeoff is judged
#        complete in the EKF's frame, the frame the hold is captured in:
#        the same 0.80 m request had produced hold altitudes of 0.53, 0.70
#        and 0.69 m. Autopilot STATUSTEXT is recorded as timed events.
#        analyze_flight finds heading samples that step out and straight
#        back and keeps them out of the yaw figures (it had reported the one
#        above as a systematic leg-8 overshoot), and explains the height
#        error by the EKF's vertical-velocity error instead of reading the
#        2:1 rangefinder-to-EKF slope as an EKF that sees half of every
#        height change. A log that predates a setting replays under the
#        rule it flew (Config.LEGACY_DEFAULTS); replay_flight --set runs
#        the recorded sensors through a changed rule.
# 0.14.1 The room has a 2.50 m ceiling, which the stack did not know.
#        Default envelope.alt_max_m 2.00 -> 1.20: with the 0.85 m overshoot
#        measured on the 2.2" airframe the old default put the worst-case
#        peak at 2.85 m, inside the ceiling, on any flight that forgot
#        --alt-max. check_params now reads FENCE_TYPE as the bitmask it is
#        (no more FENCE_RADIUS warning when the circle is off), BLOCKS a
#        FENCE_MARGIN that is not smaller than FENCE_ALT_MAX and any fence
#        or failsafe action that can enter RTL indoors, warns when the
#        autopilot fence sits below the arbiter's breach abort, and notes a
#        FENCE_ALT_MAX below its documented 10 m minimum. offline_check's
#        ceiling test requested a hardcoded 3.0 m; it now scales with the
#        ceiling, since 3.0 m is past the gross-violation factor at 1.2.
# 0.14.0 TWO AIRFRAMES, which the logs did not record. A 2.2" cinewhoop and
#        a 3.5" with a forward MT-15 share this stack; their hover
#        throttle, climb authority and sensor count differ, so a figure
#        from one does not describe the other -- and every flight log so
#        far is silent about which flew. Config.airframe + --airframe,
#        carried into the log meta; analyze_flight shows it and REFUSES to
#        read its pooled spread as repeatability across a mixed set (on
#        seeded logs, mixing inflates the spread 0.130 -> 0.450 m).
#        Config.from_dict now derives its scalar fields from the dataclass
#        instead of a hardcoded list, which is how `airframe` was silently
#        dropped on replay the first time.
#        check_params reports EVERY rangefinder backend with its
#        orientation, and BLOCKS when two are oriented down: ArduPilot
#        picks the height source by orientation, so a forward MT-15
#        mislabelled as down feeds the altitude controller its distance to
#        the nearest wall, and the aircraft will not climb because it
#        believes it is already there.
# 0.13.1 report the view count the fit ACTUALLY used, not the number
#        captured, when outliers were dropped; readable error for a missing
#        --from-dir (each release extracts to its own folder, so saved
#        frames are usually one directory behind).
# 0.13.0 FIX, mine, two of them in --model auto.
#        (1) It ranked models by REPROJECTION ERROR, which is the quantity
#        the fit minimises, so it rewards whichever model had more freedom
#        and is worthless when the two land within noise: a real run chose
#        pinhole over fisheye on 1.047 vs 1.066 px and gave away 6 deg of
#        field. Ranking is now by COLLINEARITY -- a straight line must stay
#        straight after undistortion, which the fit never optimises for. On
#        a synthetic 95.4 deg lens the two models tie at 0.327 vs 0.328 px
#        reprojection while bow separates them 0.65 vs 0.56 (worst 2.67 vs
#        0.94), and the bow-ranked choice recovers 95.4 deg where the other
#        reports 90.6.
#        (2) No outlier rejection. PAL is interlaced and its fields are
#        20 ms apart, so a hand-held board combs every horizontal edge and
#        one such view corrupts a 20-view fit -- good median bow, terrible
#        worst case. Views beyond median + 4*MAD are now dropped and the
#        fit repeated (--keep-outliers to disable); on seeded data it found
#        exactly the two poisoned views and halved reprojection error.
#        verify now names the worst frames and judges on the median, since
#        one bad frame is a bad frame and not a bad calibration.
# 0.12.1 FIX, mine: verify captured at whatever the card offered instead of
#        the resolution the calibration was fitted at. A card that pads 4:3
#        analog into a 16:9 frame displaces every normalised coordinate, so
#        a geometrically PERFECT calibration measured ~10 px of board bow
#        and was reported POOR. verify now requests the calibrated size and
#        refuses to judge on an aspect mismatch. The same hazard was live in
#        flight: detector boxes are normalised by the frame the detector
#        got, so CameraIntrinsics.check_capture() now gates it and
#        PerceptionRunner carries the verdict in health(). A stretch or a
#        uniform rescale is harmless; only padding breaks it.
# 0.12.0 calibrate_camera gains `verify`, because a calibration that wrote a
#        file is not necessarily a calibration you can fly on. A pinhole fit
#        to a 150 deg lens reaches 0.126 px reprojection error while being
#        35 deg wrong about the field of view, so the fit's own residual
#        cannot be the gate. Two tests that do catch it: EDGE LINEARITY
#        (deg/px at the edge over deg/px at the centre -- 0.97 for the true
#        model, 0.18 for the wrong one) and COLLINEARITY (undistort the
#        board's corners; a straight line must stay straight -- 0.000 px vs
#        8.5 px bow). Both pinned in offline_check. --expect-hfov compares
#        against the lens's spec, which also surfaces a capture-card crop.
# 0.11.3 probe separates the CAPTURE rate from the loop rate (board search
#        costs more than the capture, so the old single figure understated
#        the device) and names the pixel format, flagging a resolution that
#        is not a native analog mode. New `modes` scan: requests each
#        candidate resolution and reports what the device clamps to, since
#        that is the only way to learn a UVC device's real mode list.
# 0.11.2 tools/make_board.py generates the checkerboard target as a
#        print-ready PDF, sized per paper so a 150 deg lens still resolves
#        ~30 px per square at 0.5 m, with the --board / --square-mm
#        arguments printed on the page itself.
# 0.11.1 calibrate_camera gains a `devices` mode: scans every capture index,
#        saves a frame from each, and classifies what it holds, so "wrong
#        index", "right device but no video" and "no camera permission" stop
#        looking identical. The content test uses neighbour decorrelation,
#        not variance, because a checkerboard filling the frame has the same
#        mean and spread as analog static.
# 0.11.0 CORRECTION: the pinhole+radial model is invalid past ~100 deg and
#        the undistortion used a single Newton step (5.2 deg out vs cv2).
#        Fisheye (Kannala-Brandt) model added and auto-selected, solvers
#        iterate and are cross-validated against cv2, non-monotonic fits are
#        rejected. calibrate_camera gains probe / --from-dir / auto-capture /
#        --model. The "31 deg" figure was wrong: it compared two pinholes.
# 0.10.0 perception: ObjectObservation carries its range's SOURCE and
#        uncertainty, not a bare number; calibrated intrinsics (an assumed
#        90 deg FOV misplaces an edge target on a 150 deg lens by 31 deg);
#        boresight matching so the forward lidar is only used for a target
#        actually on the beam; threaded runner with a staleness gate;
#        TrackBook so is_tracked() means a track and not a flicker; YOLOE
#        adapter; tools/calibrate_camera.py
# 0.9.1  analyze_flight --compare carries the numbers a repeat test produces
#        (breach in both frames, RF trust margin, duration) and pools the
#        spread across flights, which is what a single flight cannot give
# 0.9.0  residual allowance tracks a DECAYING PEAK of |vz| rather than the
#        instantaneous value (EKF lag is accumulated, so it is largest when
#        the aircraft has momentarily stopped); analyze_flight reports the
#        trust margin and distinguishes a MODIFIED setpoint from an
#        observation, which the rate-limiter note wrongly applied to both
# 0.8.0  tools/check_params.py: verifies the vehicle's .param against what
#        the stack assumes, after a flight finding was misdiagnosed from a
#        comment that had EK3_SRC1_POSZ wrong. Corrected: the residual test
#        is a rate-of-divergence detector, not an independent cross-check,
#        because the EKF height source IS the rangefinder on this airframe.
#        rf_streams_expected now 1 to match the vehicle.
# 0.7.0  state-based envelope: tracks where the AIRCRAFT actually went, not
#        just what was commanded, and aborts on a sustained breach;
#        rangefinder residual threshold widens with |vz| (it false-positived
#        on a healthy sensor during a commanded climb); rangefinder offset,
#        residual and threshold now in the log so trust decisions are
#        auditable
# 0.6.0  arbiter escalates after N consecutive gross violations (lands);
#        --inject-repeat to exercise it; --version on every tool and the
#        build recorded in each log
# 0.5.0  replay fault injection (--inject, --inject-sweep, --list-faults,
#        --inject-not-immediate), --max-step, origin carried in the log,
#        fence fault fixed to measure from the origin
# 0.4.0  gross-violation rejection, injected requests marked immediate,
#        arbiter timers moved to the world clock
# 0.3.0  replay harness, analyze_flight, degradation sweeps
# 0.2.0  auto-takeoff default, handover opt-in via --handover
# 0.1.0  the spine: skills, arbiter, landing, recorder
