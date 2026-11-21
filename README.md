# drone_agent

Ground-station control stack for an indoor GPS-denied ArduPilot quad over an
ELRS MAVLink link. This is the spine: deterministic skills, a safety arbiter,
and synchronised logging. No LLM, no vision — those attach to it later, and
the shape is already cut for them.

## Layout

```
drone_agent/
├── config.py                  every tunable, serialised into each log
├── interfaces/vehicle.py      VehicleInterface protocol + MavlinkVehicle
├── world/
│   ├── state.py               WorldState — the one snapshot everything reads
│   └── rangefinder.py         per-sensor trust tracking and routing
├── safety/
│   ├── arbiter.py             envelope clamp, fence, staleness, watchdog, e-stop
│   └── landing.py             force-land state machine
├── skills/
│   ├── base.py                Skill ABC, SkillResult, registry
│   ├── hover.py               hold station (+ HoldingSkill base)
│   ├── rotate_search.py       leg-wise yaw sweep
│   └── climb.py               altitude change
├── faults.py                  setpoint fault injection (shared by CLI+replay)
├── perception/
│   ├── types.py               ObjectObservation, SceneSnapshot, RangeSource
│   ├── geometry.py            intrinsics, boresight, range fusion
│   ├── source.py              threaded runner, TrackBook, MockDetector
│   └── yoloe.py               open-vocabulary adapter (needs torch)
├── interfaces/replay.py       ReplayVehicle + degradations + replay runner
├── recording/recorder.py      one JSONL stream, one monotonic clock
├── executor.py                preflight, ALT_HOLD→GUIDED handover, tick loop
run_mission.py                 CLI
tools/offline_check.py         full-stack test with no vehicle
tools/analyze_flight.py        read a log: interventions, skills, link, timing
tools/replay_flight.py         re-fly a log offline; sweep degradations
tools/check_params.py          vehicle .param vs the stack's assumptions
tools/calibrate_camera.py      intrinsics + lidar boresight, via the analog path
```

## Install and verify

```bash
pip install pymavlink
python tools/offline_check.py      # must be all-green before any flight
python run_mission.py --version    # which build this copy is
```

Every tool takes `--version` and prints its build on startup, and the version
is written into each flight log's meta record. `analyze_flight.py` flags a log
written by a different build than the one reading it. If a flag is
"unrecognized", check the version before anything else — it is almost always
a stale extracted copy rather than a bug.

`offline_check.py` runs the real arbiter, skills and recorder against a
simulated vehicle. Run it after every change to the stack.

## Running

```bash
python run_mission.py --list                        # skills + JSON schemas
python run_mission.py hover --duration 10
python run_mission.py rotate_search --sweep 360     # the smoke test
```

SITL (no ELRS, no handover):

```bash
python run_mission.py hover --duration 10 \
    --connection udpin:0.0.0.0:14551 --auto-takeoff
```

Bench tests for the safety layer, **props off**:

**Props off does not work for this, and never did.** An injection only fires
once a skill is ticking, which means airborne. With no thrust the takeoff
check times out after 30 s, the skill never starts, and the fault never
fires. A props-off run tests connection, streams, mode changes, arming and
rangefinder routing — nothing in flight.

Inject in **replay** instead. It fires the fault against a recorded real
flight: real pose, real noise, real timing, the real arbiter, no aircraft.

```bash
python tools/replay_flight.py logs/<flight>.jsonl --list-faults
python tools/replay_flight.py logs/<flight>.jsonl --inject altitude
python tools/replay_flight.py logs/<flight>.jsonl --inject clamp_alt \
    --inject-sweep 15:35
```

The sweep is the useful one: it moves the injection tick across a range and
reports, per tick, whether the command was transmitted. That is how you tell
an envelope that held from a rate limiter that happened to fire first.

Then SITL, which runs the real autopilot with simulated physics so you also
see how the vehicle responds to a clamped command.

**What is worth flying, and what is not.** The three reject faults
(`altitude`, `step`, `nan`) send nothing to the aircraft — a flight test of
those is an aircraft hovering normally, which replay already proves. Only the
clamp faults command real movement, so only they teach you anything in the
air: does ArduPilot accept the clamped target, and does the vehicle stop
where the arbiter said. One flight of `clamp_alt` with the ceiling narrowed
to the room covers that:

```bash
python run_mission.py hover --duration 12 --alt-max 1.2 --inject clamp_alt
```

`clamp_step` and `fence` exercise the same clamp code path horizontally;
narrow both limits before trying them indoors:

```bash
python run_mission.py hover --duration 12 --max-step 0.6 --fence 0.5 \
    --inject clamp_step
```

Isolating the fence clamp needs `fence_radius < max_step` — near the origin
the step limit is otherwise the tighter constraint and both clamps fire.

Each injection produces a visible divergence between `setpoint_requested` and
`setpoint_applied` in the log. That divergence is the safety-intervention
metric.

**Reject vs clamp.** A clamp is right for a small overshoot — a controller
overrunning, a goal just past the fence. It is wrong for nonsense: clamping
"fly to 10 m" produces "fly to the ceiling", a legal command nobody asked for.
Past `envelope.gross_violation_factor` (default 2×) of any limit, the request
is rejected and nothing is sent. A hallucinated planner parameter looks
exactly like a gross violation, which is why this distinction exists.

**Escalation.** One gross violation is treated as a transient: the tick is
blocked and the mission continues. `gross_violation_abort_after` consecutive
ones (default 3) means the caller is not going to recover, so the arbiter
aborts and the aircraft lands. Any clampable request resets the run, since a
caller asking for something merely out of bounds is still in touch with
reality. Exercise it with `--inject-repeat N`:

```bash
python tools/replay_flight.py logs/<flight>.jsonl --inject altitude --inject-repeat 1  # succeeds
python tools/replay_flight.py logs/<flight>.jsonl --inject altitude --inject-repeat 3  # lands
```

This is the backstop beneath the planner's own retry budget, for when that is
exhausted or absent.

**A clamp bounds the command, not the aircraft.** This is the finding from the
2026-11-13 `clamp_alt` flight and it belongs in the thesis as a limitation of
setpoint-level enforcement. A request for 1.68 m was clamped to 1.20 m and
transmitted; the vehicle's own altitude controller then overshot to 1.79 m by
the rangefinder, 1.48 m in the commanded frame. So the arbiter also measures
where the aircraft actually is:

- `peak_alt_breach_m` — excursion past `alt_max` on the system's best
  altitude estimate (rangefinder while trusted, EKF otherwise). This is the
  physical number and the one that can abort.
- `peak_alt_breach_ekf_m` — the same in the commanded LOCAL_NED frame, which
  measures how well the vehicle tracked the clamped setpoint.

They differ because the EKF smooths the rangefinder heavily
(`EK3_RNG_M_NSE=0.5` on this airframe) so the estimate lags the reading during
vertical motion — about 0.4 m at the rates flown. `breach_margin_m` (0.40)
tolerates ordinary controller overshoot; `breach_abort_after` consecutive
breaching ticks past it (10, ~1 s) aborts.

Contributing factors on this airframe, from `tools/check_params.py`:
`WP_SPD_UP=2.5` m/s and `WP_ACC_Z=1` m/s² are outdoor-scale limits on GUIDED
climbs, and `MOT_THST_HOVER=0.169` is very low, so small throttle changes
produce large vertical accelerations. Lowering `WP_SPD_UP` is the first thing
to try if the overshoot needs reducing.

**The rangefinder residual test widens with vertical speed.** The 0.35 m
threshold was calibrated from a stable hold, where residuals ran
-0.05..+0.04. During the same commanded climb a healthy sensor reached +0.362
and was distrusted for 25 ticks, flipping the altitude source to EKF
mid-manoeuvre — a 0.3 m step change in reported altitude. The allowance is
now `residual_m + residual_vz_coeff * |vz|`, unchanged at hover, which is
where the landing logic actually consults it. The learned offset, the
residual and the threshold used are written to every tick so the decision can
be audited instead of inferred.

On the 2026-11-13 clamp_alt flight the residual reached 0.371 m against a
base threshold of 0.35 m — within 0.060 m of distrusting a healthy sensor.
The allowance therefore tracks a **decaying peak** of |vz| rather than the
instantaneous value, because the EKF's lag is accumulated and is largest just
after the aircraft stops moving: at vz=-0.04 the residual was still 0.333 m.
That roughly doubles the worst margin (+0.060 → +0.117 m on that flight's
data), which is an improvement and not a fix. Residuals on this airframe reach
0.37 m in ordinary flight, against a threshold calibrated from a stable hold
where they ran -0.05..+0.04, so the base value needs recalibrating from a
dedicated vertical profile rather than inferring it from one flight.
`analyze_flight.py` prints the worst margin every flight and warns below
0.10 m.

**That test is not an independent cross-check**, and the thesis must not call
it one. With `EK3_SRC1_POSZ=2` the EKF's height source *is* the rangefinder,
so a sensor that lies persistently drags the EKF with it and the residual
stays small. What the test detects is a rangefinder deviating faster than the
EKF's fusion follows — which is why it caught the original incident (a bump,
fast bad data) and why it false-positived on a sustained climb. A genuinely
independent check would mean streaming baro altitude separately
(`EK3_SRC2_POSZ=1` exists on the vehicle but is inactive, and the GCS sees
only the active estimate).

**Injected requests are marked `immediate`.** Without that the uplink rate
limiter can swallow a clamped command before it reaches the aircraft, and the
log shows a clamp that proves nothing. `analyze_flight.py` reports
`[TRANSMITTED]` or `[suppressed by the rate limiter]` for every intervention,
so a hollow test is visible rather than reassuring.
`replay_flight.py --inject-not-immediate` reproduces the hollow case on
purpose, which is worth a line in the methodology: on the 2026-11-11 flights
only 1 tick in 6 would have transmitted.

**The arbiter's timers run on the world clock**, not the wall clock — the
rate limiter, the staleness grace window and the tick watchdog all read
`world.t_mono`. In flight that *is* `time.monotonic()`; in replay it is the
recorded sample time, so replay reproduces the rate limiter instead of
collapsing it into the replay's own runtime. All three 2026-11-11 flights
replay with an exact match on transmitted setpoint count.

## Analysis and replay

Read a flight:

```bash
python tools/analyze_flight.py logs/<flight>.jsonl
python tools/analyze_flight.py logs/*.jsonl --compare   # repeats + spread
python tools/analyze_flight.py logs/<flight>.jsonl --csv ticks.csv
```

Re-fly it offline through the real skills and the real arbiter:

```bash
python tools/replay_flight.py logs/<flight>.jsonl           # faithful replay
python tools/replay_flight.py logs/<flight>.jsonl \
    --sweep range --values -0.6,-0.3,0,0.3,0.6 --repeats 5
```

Sweeps: `range` (rangefinder bias, m), `pose` (x/y gaussian sigma, m),
`latency` (extra pose age, s), `flow` (quality scale), `drift` (m/s).

`--compare` is for repeats of an identical command: one row per flight with
the breach in both frames, the rangefinder trust margin and the duration, then
the mean and spread pooled across them. Three `clamp_alt` flights on
2026-11-13 overshot the ceiling by 0.50, 0.63 and 0.85 m from the same
command — a 0.35 m spread on a 0.66 m mean, which no single flight reveals and
which belongs in any claim about the envelope.

Replay rebuilds the `Config` from the log's `meta` record, because skill
timing lives in the config — replaying under today's defaults runs a
*different* skill than the one recorded. `--default-config` overrides that if
you specifically want to test a config change against old data.

Open loop (the default) replays pose straight from the log and is
reproducible to the bit: it answers "what would the stack have decided, given
exactly what it saw?". `--closed-loop` advances pose with an approximate
first-order model instead — useful for asking whether a different controller
would have flown better, as long as any result says the dynamics are
approximate.

Why this matters for the evaluation: ten flights per condition is achievable,
eight hundred planner evaluations is not. Record once, replay many. It also
separates perception variance (between conditions, costs flight time) from
decision variance (repeat runs on identical input, free).

## Perception

```bash
python tools/calibrate_camera.py intrinsics --camera 0 --out camera.json
python tools/calibrate_camera.py boresight  --camera 0 --out camera.json \
    --connection udpin:0.0.0.0:14550
python run_mission.py rotate_search --sweep 360 \
    --perception yoloe --classes person,drone --intrinsics camera.json \
    --target-class person
```

Run `probe` first, always — it reports which device you actually opened, what
it delivers, and whether a board is detectable in it. Index 1 on a Mac is as
likely to be a Continuity Camera as a capture card.

**Check a calibration before trusting it**, and keep one file per camera
(the 3.5" airframe's lens measured 88.8°, the 2.2" one 110.3°):

```bash
python tools/calibrate_camera.py verify --camera 1 --out camera_2.2in.json
```

Hold the board still. `verify` takes a view by itself once the board has
stopped, six of them in six different parts of the picture, and says what it
is waiting for in between. It prints each view's figure and the result as an
angle. Before 0.16.2 it grabbed any six frames without waiting, so a board in
motion could read as a bad calibration.

**Test perception on the ground before it flies.** `run_mission.py
--perception` is the only other way to run this code, and it flies.
`tools/watch_perception.py` runs the same detector, geometry and staleness
gate with no vehicle and never opens a MAVLink connection:

```bash
python tools/watch_perception.py watch --camera 1 --intrinsics camera.json \
    --classes person,chair --duration 30 --expect chair:30
python tools/watch_perception.py latency --camera 1 --intrinsics camera.json
```

`watch` reports the detector's rate, how old a scene is when a 10 Hz loop
reads it, whether each object keeps one track id, the bearing against a
tape-measured angle, and what share of sightings `RotateSearch` would
actually stop on, with the reason for the rest. `latency` flashes the screen
with the drone's camera pointed at it and times how long the change takes to
appear in a frame: the age a picture already has when `read()` returns, which
no timestamp in the stack can see. Give it `--intrinsics` so it times the
capture mode the detector uses, not the card's default.

First run on the 2.2" airframe's video chain (2026-11-15, yoloe-11s on the
Mac's CPU): 20 scenes/s, 43 ms inference, a scene 0.07 s old at a tick and
never older than 0.15 s, a person held on one track for 29.8 s and accepted
by the search rule on 98% of sightings. The picture is at most ~180 ms old
when `read()` returns (measured at the card's default mode; that figure
includes this Mac's display delay, so it is an upper bound), and a slow
reader adds only its own pause: the capture hands out the newest frame.

**In flight, perception starts before the vehicle is connected.** With
`--perception yoloe`, `run_mission.py` loads the detector, opens the camera
at the calibrated size and waits for scenes to flow before it opens the
MAVLink link. If any of that fails (no camera, a capture size the
calibration does not fit, a `--target-class` missing from `--classes`) it
prints `NOT FLYING` and exits with code 3 without arming. Before 0.16.1 the
detector was loaded after takeoff and the camera opened at the card's
default size; no flight was made that way. `analyze_flight.py` prints a
PERCEPTION section for such a flight: what was seen, when, in which
direction, and how old the pictures were by the time they were used.

**Calibrate through the full analog path** — camera, VTX, VRX, capture card.
The card may crop, scale or deinterlace, and each changes the effective focal
length. Against assuming a 90° pinhole, a calibrated 150° fisheye places a
target differently by **8° at 0.7 of frame width, 13° at 0.8, 23° at 0.95** —
a bearing that wrong points the aircraft at something that is not there.
Without `--intrinsics` the stack assumes an FOV, says so on startup, and
records `intrinsics_calibrated: false` in the log.

**The lens model matters more than the FOV number.** The standard
pinhole+radial model is valid to roughly 90–100° of field. Past that the
polynomial stops being invertible: fitted to a 150° lens it yields a
pixel-to-angle mapping that rises to 71° and then *falls*, which no real lens
does. An FPV lens is an equidistant fisheye (Kannala–Brandt), so `--model`
defaults to `auto`: both are fitted, the better reprojection error wins, and
**any fit whose mapping is non-monotonic is rejected rather than saved** — a
calibration that looks authoritative and produces meaningless bearings is
worse than an honestly assumed FOV.

The undistortion solvers are pure Python (the module has no cv2 dependency),
and `offline_check` cross-validates both against `cv2.undistortPoints` and
`cv2.fisheye.undistortPoints` whenever cv2 is present: 0.026° and 0.000°
worst disagreement. An earlier single-Newton-step version was 5.2° out.

**Practical note for a wide lens:** it shrinks everything, so a small board is
undetectable. A 0.25 m board at 0.5 m spans about 50 px — 5 px per square.
Print A2/A1 with 50–80 mm squares and fill a third of the frame. `intrinsics`
auto-captures once the board is still and has moved far enough from every
previous view, so no keypresses are needed, and `--from-dir` calibrates from
saved images if the preview window is awkward.

**Every range carries its source and uncertainty.** This is the deliberate
difference from the published encodings, where each detection arrives with an
exact distance from a simulated depth sensor and no error model. Here:

| source | metric? | from |
|---|---|---|
| `boresight` | yes | the forward lidar, only for a target *on the beam* |
| `depth` | yes | a real depth camera |
| `mono_depth` | no | monocular estimator, needs a scale anchor |
| `size_prior` | no | assumed real-world height + intrinsics |
| `none` | — | bearing only |

The boresight check matters: the MT-15's beam is 1–2°, about a 5–10 cm spot at
3 m, so it measures the target only when the target is centred. An off-beam
detection silently taking the lidar's reading would be a confidently wrong
number, so `Boresight.contains()` decides and the fallback is a size prior
with a wider sigma. `RangeEstimator.disagreement()` compares the two when both
exist — a free runtime confidence signal.

**Perception never sets the control rate.** It runs in its own thread and
publishes an immutable `SceneSnapshot`; the tick loop reads the latest and
checks its age. `fresh_scene()` returns `None` rather than stale data, and
`world.objects` is `[]` when the snapshot is older than `max_age_s` — a skill
must not be able to act on a two-second-old bearing.

**A flicker is not a track.** `TrackBook` accumulates `first_seen` and
`frames_seen` per `(label, track_id)`, so `is_tracked()` can require a real
track. `RotateSearch` needs a persistent id, ≥0.5 s of age, ≥3 frames, *and* a
range whose stated sigma is small enough to act on. A sighting it cannot range
is recorded in `metrics["sightings"]` and does **not** stop the sweep — "saw
it but could not range it" and "never saw it" are different failures with
different fixes, and the failure taxonomy needs to tell them apart.

The vehicle owns MAVLink and pushes the boresight range *into* perception;
perception never reads the link. `MockDetector` drives the whole pipeline from
a script, which is how tracking, staleness and range fusion get
regression-tested without a camera.

## The three invariants

**1. Every setpoint goes through the arbiter.**
`MavlinkVehicle.send_position_target` and `send_condition_yaw` require an
opaque token. Exactly one is minted, by `Arbiter.__init__`. Anything else
raises `BypassError` with a stack trace pointing at the offending line.
Skills return a `SetpointRequest` from `tick()`; they never touch the vehicle.

Mode changes and arming are deliberately *not* token-gated — the landing state
machine must be able to command LAND while the arbiter is already latched into
an abort.

**2. Every skill returns a structured result.**
`SkillResult` carries status, reason, detail, final pose, elapsed time, tick
count and skill-specific metrics. `to_planner_view()` is what the LLM will get
back from a tool call. This is the feedback channel the reference
implementations lack, and it is what makes replanning possible.

**3. One clock, one log.**
`time.monotonic()` throughout. Each record carries `t_mono` (write time) and
each tick additionally `t_sample_mono` (when the state was sampled); their
difference is logging latency. Loading needs no parsing work:

```python
import pandas as pd, json
rows  = [json.loads(l) for l in open("logs/<flight>.jsonl")]
ticks = pd.json_normalize([r for r in rows if r["type"] == "tick"])
ticks[ticks["arb.intervened"]].groupby(ticks["arb.actions"].astype(str)).size()
```

## The camera points up

An FPV camera is mounted nose-up and a checkerboard calibration cannot see
it: the board was wherever it was held, so every angle the calibration gives
is in the camera's own frame. On this airframe the tilt is somewhere between
10 and 18 deg, which is why a person's feet are below the bottom of the
picture in every flight, and why a bearing measured to something below the
axis reads larger than it is.

```bash
python tools/calibrate_camera.py tilt --camera 1 --out camera.json
# stand the aircraft level facing a wall. The saved frame has a red line
# where the camera's horizon falls and a yellow v-scale down the side:
# measure the wall at the red line (H), the lens off the floor (C), and
# read off the scale where the wall meets the floor (V).
python tools/calibrate_camera.py tilt --camera 1 --out camera.json \
    --mark-height H --camera-height C --junction-v V
```

The floor line is at height zero, so two features at two known heights in
one picture give the distance and the tilt together and nobody measures the
distance to the wall — which was got wrong twice here, both times by
running the numbers out of this README instead of measuring. The tool now
refuses those numbers, reports what one notch of the scale does to the
answer (+-0.01 of v is about +-5 cm at 2.4 m), and refuses a floor line
read so high it would put the wall out of the room. `--distance D` in place
of `--junction-v` still takes a measured distance.

With the tilt in the file, bearings are in the airframe's frame and the
range can come from the top of the box instead of its size:

    range = (how tall the object is - how high the camera is) / tan(elevation of its top)

The aircraft's own pitch moves the camera with it and is included per tick.
Below 0.35 m of rise (the target's height minus the camera's) there is no
answer at all and the estimator says so rather than returning a number:
measured on 2026-11-17, the error per degree of tilt goes from 0.07 m with
the camera at 0.70 m to 0.26 m at 1.38 m, and above the target's own height
the geometry inverts. The aircraft reached a rangefinder reading of 1.83 m
in those flights, which is over a person's head.

One look, no movement, no heading, and it does not care that the bottom of
the object is out of frame. `tools/range_check.py` applies it to flights
already flown. Measure the tilt to about a degree: at 2.5 m, one degree of
tilt error is a quarter of a metre of range.

**Measured, 2026-11-16.** The floor line in this airframe's own frame sits
at v=0.981, which with the lens 0.66 m up and the mark at 1.24 m puts the
wall at 2.40 m and the tilt at **+13.56 deg**. Independently, three flights
with the person tape-measured at 2.57 m need a tilt of **13.5 deg** for the
head range to come out right. Those two numbers are measured separately and
agree to 0.06 deg. Over the six looks in those flights the head range gives
a median of 2.57 m against 2.57 m of tape, median error 0.13 m, worst
0.36 m — against +17% to +44% for the size prior and -42%, -16%, -6% for
parallax on the same looks.

## How far away is it

```bash
python run_mission.py rotate_search --sweep 360 --alt-max 1.2 \
    --airframe 2.2in-cinewhoop --perception yoloe --camera 1 \
    --intrinsics camera.json --classes person --target-class person \
    --then-face --then-measure
```

`measure_range` looks, steps 0.5 m sideways (`--baseline`, 0.3 to 1.0 m),
looks again and steps back. The bearing change over a known baseline gives
the range without any prior on the object's size and without needing the
whole object in frame, which is what makes the detector's own range useless
for a person this camera sees from below. The result carries a bound on its
error; a longer step measures further, more tightly. A target too far for
the step is refused with the limit named. The aircraft returns to where it
started in every ending, including the failures. Flown on 2026-11-16 against
a tape, and it does not work: all three flights under-read, by 6% to 42%,
because the heading moved 10-16 deg during the manoeuvre while the EKF
reported 0-4 deg. Kept, documented, not relied on. The head-height range
above replaced it.

## Getting closer to something

```bash
python run_mission.py rotate_search --sweep 360 --alt-max 1.2 \
    --airframe 2.2in-cinewhoop --perception yoloe --camera 1 \
    --intrinsics camera.json --classes person --target-class person \
    --target-height 1.73 --then-face --then-approach --standoff 1.5
```

`approach` closes on the locked target until it is `--standoff` metres away,
measuring the range again between every step. It is the first skill that
acts on a range, so what may drive it is restricted on purpose: a
`head_height` range or a lidar on the beam, and nothing else. A size-prior
range is **refused** — it read 25% to 31% long against a tape, and a stack
that thinks it has 2.5 m when it has 1.9 m is pointed at a person. Without
a camera tilt in the calibration there is no head range, so the skill
refuses to move at all rather than falling back.


## Planner

```bash
export DRONE_LLM_URL=http://HOST:PORT/v1     # everything before /chat/completions
export DRONE_LLM_MODEL=NAME
python tools/plan_mission.py models                       # reachable? which names?
python tools/plan_mission.py plan "look around for a person"
python tools/plan_mission.py bench tools/instructions_example.txt --repeat 5
python tools/plan_mission.py grade tools/eval_set.jsonl --repeat 5   # graded
python run_mission.py --instruction "look around for a person" \
    --alt-max 1.2 --airframe 2.2in-cinewhoop \
    --perception yoloe --camera 1 --intrinsics camera.json
```

The model is asked once, on the ground. Its reply is checked against each
skill's schema and the envelope; a reply that fails is returned to it with
the reasons, twice at most. The plan is printed and flies only after `y`.
Takeoff and landing are not skills and are not the model's to plan. A
request the skills cannot do (anything that needs horizontal movement, for
now) must come back as a refusal, and partial plans are not accepted.

In the air the plan is fixed. Steps run in order; the first step that does
not succeed ends the plan, and a search that ends without its target skips
the steps planned after it. Every case ends in a landing. The instruction,
every reply from the model, the validation errors and the operator's answer
are in the flight log (`plan`, `plan_confirmation` events).

A model that reasons before it answers needs room for it: the reply limit is
4096 tokens (`--max-tokens`), and `--no-think` asks the server to turn the
reasoning off, which is several times faster. Run `bench` both ways and keep
the one that still plans correctly.

`bench` is the quick evaluation: each instruction N times, reporting how
often the reply was valid, how often a doable request was planned and an
impossible one (a line starting with `!`) refused, how often the same
instruction gave the same plan, and the time per plan.

### The graded evaluation

`bench` cannot say whether a valid plan is the *right* plan, and a number
in a thesis cannot rest on reading the log by hand. `grade` can:

```bash
python tools/plan_mission.py grade tools/eval_set.jsonl --repeat 5 \
    --classes person,chair --alt-max 1.2
python tools/plan_mission.py grade tools/eval_set.jsonl --repeat 5 \
    --classes person,chair --alt-max 1.2 --no-think     # and again, faster
```

Each instruction carries a rubric that is checked in code — 45 of them over
ten categories, from one skill with one number to three-step compositions,
German, deliberately vague requests, and 17 that must be refused:

```json
{"id": "search-face", "category": "sequence",
 "text": "find a person and turn to face them",
 "must": [{"skill": "rotate_search", "params": {"target_class": "person"}},
          {"skill": "face_target",   "params": {"target_class": "person"}}],
 "forbid": ["move"], "max_steps": 3}
```

`must` is an ordered **subsequence**, so an extra hover at the end is fine
but face-then-search is not: several plans are right, and marking against
one reference plan would measure conformity to its author's taste rather
than correctness. Numbers are matched with a tolerance (`{"approx": 180,
"tol": 20}`, `{"min": 5}`), because "half way around" does not specify
180.0000, and a parameter left at its default does not count as one the
model set. `refuse: true` cases accept nothing but a refusal with a stated
reason — a plan that does *part* of an impossible instruction is wrong,
which is the rule the system prompt states and the one worth measuring.

Two fields exist because the first run of this set found four rubrics
wrong rather than four model failures. `must_any` holds alternative
readings, any one of which passes: "turn around" is a full circle to one
reader and an about-face to another, and marking one reading wrong measures
the rubric's author. `or_refuse` says a refusal is acceptable *and so is* a
plan meeting `must`: asked to "fly right up to my face as close as you can
get", the model planned `approach` with `standoff_m: 1.2` — the schema's
minimum and the skill's hard floor — and asked to "spin as fast as the
motors will go" it planned `rate_dps: 45`, the schema's maximum. The limits
live in the schema precisely so the planner cannot breach them, so a
planner that clamps to them has understood the request and declined the
unsafe part. Calling that a failure measured an opinion.

The report is a pass rate overall and per category, with the commonest
reasons printed under any instruction that failed. The offline suite checks
that every rubric — every *reading* of every rubric — is satisfiable by a
plan the validator accepts, so a case nobody can pass cannot quietly
subtract from the score.

### Re-scoring without re-running the model

```bash
python tools/plan_mission.py regrade tools/eval_set.jsonl logs/graded-*.jsonl
```

Correcting a rubric must not cost a fresh 225-plan run, and a score that
moved because the rubric moved has to be reproducible from the same
replies. `regrade` re-marks a saved `grade` log against the rubrics as they
now stand, reports the old and new scores side by side, and names every
mark that moved and which way.

## Safety

RC and MAVLink share one radio. If the link dies the script dies **and**
ArduPilot's RC failsafe fires — that is the real safety net, not this code.
Keep a pilot on the sticks with a mode switch ready, every flight.

Everything the arbiter does is ground-side and dies with the script. Check
what the autopilot will do on its own:

```bash
python tools/check_params.py vehicle.param --alt-max 1.2
```

It reads an exported `.param` file and flags where the vehicle contradicts
what this stack assumes, at three severities: BLOCK (the stack will misbehave
or a claim here is false of this vehicle), WARN, INFO. This exists because a
2026-11-13 flight finding was misdiagnosed from a comment that assumed
`EK3_SRC1_POSZ=1` while the airframe had `2` — assumptions about the vehicle
belong in a checker that reads the vehicle.

On the airframes as flown (ELRS in MAVLink mode, optical flow, no GPS) the
autopilot-level tier is thinner than an outdoor vehicle's, and three things
that look like fixes are not. All three were learned on 2026-11-15 by
getting them wrong:

- **`MAV_GCS_SYSID` must stay 255.** The sticks arrive as MAVLink from the
  ELRS receiver, which acts as a GCS with system id 255, and the autopilot
  takes RC input only from its GCS system id. Setting it to this stack's 254
  produced "RC not found".
- **The GCS failsafe cannot detect a dead script.** It watches that same
  system id, which is the receiver, and the receiver keeps talking while the
  script is dead. If the script or the Mac dies with the radio up, the
  aircraft holds its last GUIDED target. The cover for that is the pilot.
- **An altitude fence in the default frame does not work.**
  `FENCE_ALT_MAX_TP=1` (above home) needs a home, and this vehicle has no
  position source to set one from; with it the fence tripped about 10 cm off
  the ground at a 1.4 m limit. `FENCE_ALT_MAX_TP=2` (above origin) reads the EKF's own altitude
  and is the only candidate; it has not been flown yet. Either way
  `FENCE_ALT_MAX` is in the EKF's frame, which reads below the real height
  (see `envelope.ekf_under_read_climb_m`).

What does work on its own is the RC failsafe: `FS_THR_ENABLE=3` lands the
aircraft when the radio drops.
