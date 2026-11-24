"""
Pixels to metres. The part that must be right before any of it is useful.

Three jobs:

  bearing   from a bounding box centre, using REAL intrinsics. The reference
            implementations assume a 90 deg HFOV and compute bearing from the
            fraction of the frame; on an FPV lens of 150-170 deg with heavy
            barrel distortion that is wrong everywhere and badly wrong at the
            edges. A wrong bearing points the aircraft at something that is
            not there, so this is calibrated, not assumed.

  range     from whatever source is available, each with an uncertainty:
              boresight lidar  metric, tight, but only ON the beam
              size prior       needs an assumed real-world size
              mono depth       relative, needs a scale anchor
            The fusion picks the best available and says which it used.

  boresight the pixel the forward lidar actually points at. It is not the
            geometric image centre, and finding it is a bench measurement
            (move a target through the frame, note where the lidar reading
            jumps from background to target).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .types import ObjectObservation, RangeSource


@dataclass
class CameraIntrinsics:
    """
    Pixel -> ray angles, in NORMALISED units so a change of capture
    resolution does not invalidate a calibration.

    Two models, and picking the wrong one is not a detail:

      pinhole   the standard Brown-Conrady radial model. Valid to perhaps
                90-100 deg of field. Past that the polynomial stops being
                invertible: with k1=-0.21, k2=0.04 at 150 deg the pixel-to-
                angle mapping is non-monotonic, peaking at 71 deg and then
                FALLING, which no real lens does. A calibration that reports
                a wide FOV under this model is fitting nonsense.

      fisheye   Kannala-Brandt equidistant, which is what an FPV lens
                actually is. r_d = theta + k1*theta^3 + k2*theta^5 +
                k3*theta^7 + k4*theta^9. Use this for anything beyond ~100
                deg.

    Calibrate through the FULL analog path -- camera, VTX, VRX, capture card.
    The card may crop, scale or deinterlace, and each changes the effective
    focal length.
    """

    fx: float = 0.5
    fy: float = 0.5
    cx: float = 0.5
    cy: float = 0.5
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    model: str = "pinhole"              # "pinhole" | "fisheye"
    width_px: int = 720
    height_px: int = 576
    reproj_err_px: Optional[float] = None
    # How far the camera points above the airframe's own forward axis, in
    # degrees. An FPV camera is mounted nose-up, and nothing in a
    # checkerboard calibration can see that: the board is wherever it was
    # held. Until it is measured this is 0.0 and every angle below is in the
    # camera's frame, which is what every flight up to 2026-11-16 used.
    #
    # The aircraft's own pitch is NOT included here: it moves the camera
    # too, but it is a per-tick quantity, so height_range() takes it
    # separately.
    #
    # It is not a cosmetic offset. A tilted camera makes an object BELOW the
    # optical axis read at a larger azimuth than it really has, because the
    # azimuth is measured in a plane that is itself tilted. At the 15 deg
    # depression a standing person is seen at, a 15 deg tilt inflates the
    # azimuth by about a fifth, and bearing differences -- which is what a
    # parallax range is -- by the same.
    tilt_up_deg: float = 0.0
    note: str = ""

    @classmethod
    def from_hfov(cls, hfov_deg: float, aspect: float = 4 / 3,
                  model: str = "fisheye", **kw: Any) -> "CameraIntrinsics":
        """
        Fallback when no calibration exists. Defaults to the fisheye model
        with zero higher-order terms, i.e. a pure equidistant lens: for a
        wide FPV lens that is a far better approximation than a distortion-
        free pinhole, which cannot represent a 150 deg field at all.

        Flagged in `note`, because an assumed FOV is exactly the shortcut
        this module exists to avoid.
        """
        if model == "fisheye":
            # equidistant: r_d = f * theta, so half-width maps to half-FOV
            f = 0.5 / math.radians(hfov_deg / 2.0)
        else:
            f = 0.5 / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx=f, fy=f * aspect, cx=0.5, cy=0.5, model=model,
                   note=f"UNCALIBRATED: assumed {hfov_deg:g} deg HFOV "
                        f"({model})", **kw)

    @property
    def hfov_deg(self) -> float:
        """Full horizontal field, from the actual pixel-to-angle mapping
        rather than from fx alone, so it is right for either model."""
        left = self.angles(0.0, self.cy)[0]
        right = self.angles(1.0, self.cy)[0]
        return abs(right - left)

    @property
    def vfov_deg(self) -> float:
        """Full vertical field, the same way. One normalised row is this
        over 1, which is what a loose box edge costs in degrees."""
        top = self.angles(self.cx, 0.0)[1]
        bottom = self.angles(self.cx, 1.0)[1]
        return abs(top - bottom)

    @property
    def is_calibrated(self) -> bool:
        return not self.note.startswith("UNCALIBRATED")

    @property
    def is_fisheye(self) -> bool:
        return self.model == "fisheye"

    # ---- pixel -> ray ------------------------------------------------

    def _undistort_pinhole(self, x: float, y: float,
                           iters: int = 8) -> Tuple[float, float]:
        """
        Invert the radial polynomial by fixed-point iteration -- the same
        thing cv2.undistortPoints does. A single step (an earlier mistake
        here) is off by up to 5 deg near the frame edge.
        """
        if self.k1 == 0.0 and self.k2 == 0.0:
            return x, y
        xu, yu = x, y
        for _ in range(iters):
            r2 = xu * xu + yu * yu
            scale = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            if abs(scale) < 1e-9:
                return x, y
            xu, yu = x / scale, y / scale
        return xu, yu

    def _theta_from_rd(self, r_d: float, iters: int = 12) -> float:
        """Solve r_d = theta + k1 t^3 + k2 t^5 + k3 t^7 + k4 t^9 for theta
        (Kannala-Brandt), by Newton from theta = r_d."""
        if r_d <= 1e-12:
            return 0.0
        t = r_d
        for _ in range(iters):
            t2 = t * t
            f = (t * (1.0 + t2 * (self.k1 + t2 * (self.k2 + t2 *
                 (self.k3 + t2 * self.k4)))) - r_d)
            df = (1.0 + t2 * (3.0 * self.k1 + t2 * (5.0 * self.k2 + t2 *
                  (7.0 * self.k3 + t2 * 9.0 * self.k4))))
            if abs(df) < 1e-12:
                break
            step = f / df
            t -= step
            if abs(step) < 1e-12:
                break
        return max(0.0, t)

    def angles(self, u: float, v: float) -> Tuple[float, float]:
        """
        Normalised pixel -> (bearing_deg, elevation_deg), + right / + up,
        in the AIRFRAME's frame when the camera's tilt has been measured and
        in the camera's own frame when it has not.
        """
        return self._rotate(*self._camera_angles(u, v))

    def _rotate(self, bearing: float, elevation: float) -> Tuple[float, float]:
        """Camera frame -> airframe, about the camera's right axis."""
        if abs(self.tilt_up_deg) < 1e-9:
            return bearing, elevation
        a, e = math.radians(bearing), math.radians(elevation)
        th = math.radians(self.tilt_up_deg)
        x = math.cos(e) * math.sin(a)
        y = -math.sin(e)                       # y down
        z = math.cos(e) * math.cos(a)
        yb = y * math.cos(th) - z * math.sin(th)
        zb = y * math.sin(th) + z * math.cos(th)
        return (math.degrees(math.atan2(x, zb)),
                math.degrees(math.atan2(-yb, math.hypot(x, zb))))

    def _camera_angles(self, u: float, v: float) -> Tuple[float, float]:
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy

        if self.is_fisheye:
            r_d = math.hypot(x, y)
            theta = self._theta_from_rd(r_d)
            if r_d <= 1e-12:
                return 0.0, 0.0
            # Ray direction from the incidence angle and the image azimuth.
            s = math.sin(theta)
            dx, dy, dz = s * x / r_d, s * y / r_d, math.cos(theta)
            if dz <= 1e-9:          # at or beyond 90 deg off-axis
                dz = 1e-9
            return (math.degrees(math.atan2(dx, dz)),
                    -math.degrees(math.atan2(dy, dz)))

        xu, yu = self._undistort_pinhole(x, y)
        return math.degrees(math.atan(xu)), -math.degrees(math.atan(yu))

    def height_range(self, v_top: float, object_height_m: float,
                     camera_height_m: float, u: float = 0.5,
                     pitch_deg: float = 0.0) -> Optional[float]:
        """
        Range to an upright object from the elevation of its TOP, the height
        of the camera above the floor, and how tall the object is.

            range = (object height - camera height) / tan(elevation of top)

        Unlike a range from the object's apparent SIZE, this does not care
        that the bottom of the object is outside the frame, which for a
        person seen by a nose-up camera it always is. Unlike a range from
        parallax, it uses no baseline and no heading, so the aircraft's yaw
        estimate cannot spoil it. It does need the camera's tilt: without
        it, None.
        """
        if abs(self.tilt_up_deg) < 1e-9:
            return None
        # The aircraft's own pitch carries the camera with it, and unlike
        # the mount it changes tick by tick: 0.7 to 1.9 deg nose-down on
        # this airframe holding station, which at 2.5 m is about 0.2 m.
        saved, self.tilt_up_deg = self.tilt_up_deg, \
            self.tilt_up_deg + pitch_deg
        try:
            bearing, elev = self.angles(u, v_top)
        finally:
            self.tilt_up_deg = saved
        rise = object_height_m - camera_height_m
        t = math.tan(math.radians(elev))
        if abs(t) < 1e-6 or rise / t <= 0.0:
            return None
        # That is the distance along the line of sight's ground projection
        # at this bearing; the range to the object is the same thing, since
        # both are measured from the camera.
        return (rise / t) / max(1e-6, math.cos(math.radians(bearing))) \
            * math.cos(math.radians(bearing))

    def check_capture(self, width: int, height: int) -> Optional[str]:
        """
        Is this calibration valid for a capture of this size? Returns None if
        it is, or a description of the problem.

        These intrinsics are normalised, so a pure rescale is harmless: fx/w
        and cx/w are unchanged whether the card delivers 720x576 or twice
        that. A change of ASPECT is not harmless, and the distinction is the
        whole point of this method.

        A capture card asked for 16:9 while fed 4:3 analog does one of two
        things. If it STRETCHES, each axis scales uniformly and normalised
        intrinsics still hold. If it PILLARBOXES -- pads with black bars --
        then the picture no longer starts at u=0, every normalised coordinate
        is displaced, and a geometrically perfect calibration produces about
        10 px of residual curvature on a checkerboard. Nothing downstream can
        detect that on its own; it just quietly misplaces targets.

        Since the two are indistinguishable from the frame size alone, an
        aspect change has to be reported rather than guessed at.
        """
        if width <= 0 or height <= 0:
            return f"capture reports a {width}x{height} frame"
        if (width, height) == (self.width_px, self.height_px):
            return None
        cal_ar = self.width_px / self.height_px
        cap_ar = width / height
        if abs(cap_ar - cal_ar) / cal_ar < 0.01:
            return None                 # uniform rescale, intrinsics hold
        return (f"calibrated at {self.width_px}x{self.height_px} "
                f"(aspect {cal_ar:.3f}) but capturing {width}x{height} "
                f"(aspect {cap_ar:.3f}). If the card pillarboxes, every "
                f"angle is wrong. Capture at the calibrated size, or "
                f"recalibrate at this one.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str) -> "CameraIntrinsics":
        d = json.loads(Path(path).read_text())
        d.pop("boresight", None)        # lives beside us, not in us
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @staticmethod
    def load_boresight(path: str) -> "Boresight":
        """The boresight stored alongside a calibration, or an uncalibrated
        default pointing at the image centre."""
        try:
            d = json.loads(Path(path).read_text()).get("boresight")
        except (OSError, json.JSONDecodeError):
            d = None
        if not d:
            return Boresight()
        return Boresight(u=d.get("u", 0.5), v=d.get("v", 0.5),
                         beam_deg=d.get("beam_deg", 2.0),
                         calibrated=bool(d.get("calibrated", False)))

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


@dataclass
class Boresight:
    """
    Where the forward lidar points, in normalised image coordinates, and how
    wide its beam is.

    The MT-15 is a 1-2 deg beam: at 3 m that is a 5-10 cm spot. Centring a
    target in the image puts it on the beam ONLY if the sensor is boresighted
    with the camera and the camera is near level. Both are mount decisions,
    and both are measured here rather than assumed.
    """

    u: float = 0.5
    v: float = 0.5
    beam_deg: float = 2.0
    calibrated: bool = False

    def contains(self, obs: ObjectObservation,
                 intr: CameraIntrinsics) -> bool:
        """Is this object's centre within the beam?"""
        if obs.bbox is None:
            return False
        bu = (obs.bbox[0] + obs.bbox[2]) / 2.0
        bv = (obs.bbox[1] + obs.bbox[3]) / 2.0
        b1, e1 = intr.angles(bu, bv)
        b2, e2 = intr.angles(self.u, self.v)
        return math.hypot(b1 - b2, e1 - e2) <= self.beam_deg / 2.0


# Assumed real-world heights, metres, with a spread. The spread IS the
# uncertainty: people are 1.5-1.95 m, so a range from a person's bbox height
# is good to roughly +/-12%, and saying so is the point.
SIZE_PRIORS: Dict[str, Tuple[float, float]] = {
    "person": (1.72, 0.11),
    "chair": (0.90, 0.15),
    "door": (2.03, 0.06),
    "doorway": (2.03, 0.06),
    "table": (0.74, 0.08),
    "laptop": (0.25, 0.05),
    "bottle": (0.25, 0.06),
    "tv": (0.60, 0.15),
    # A quad's "height" is its visual extent, which varies with attitude far
    # more than a standing person's does; the spread reflects that.
    "drone": (0.28, 0.10),
    "quadcopter": (0.28, 0.10),
}


@dataclass
class RangeEstimator:
    """
    Fuses whatever range sources exist into one number per object, tagged with
    its provenance.

    Priority is by trustworthiness, not availability: a laser on the beam
    beats a height guess every time, and the estimator says which it used so
    nothing downstream has to assume.
    """

    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    boresight: Boresight = field(default_factory=Boresight)
    size_priors: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(SIZE_PRIORS))

    # Fractional uncertainty on a metric lidar reading, plus a floor.
    lidar_frac: float = 0.02
    lidar_floor_m: float = 0.03
    # Monocular depth, once it exists: generous until measured.
    mono_frac: float = 0.25

    # Head-height ranging. Only for things that stand on the floor: the
    # method measures the height of the box's top above the lens, and that
    # is a distance only if the object's base is at zero. A laptop on a
    # desk is 0.25 m tall and 0.9 m up, and would read as very far away.
    floor_standing: Tuple[str, ...] = ("person", "chair", "door", "doorway",
                                       "table")
    # How well the camera's tilt is known, in degrees, and how loosely the
    # detector puts the top edge of the box, in normalised rows. The second
    # is measured: over the six looks of 2026-11-16 the within-look spread
    # of the top edge was 0.0103 of a frame height.
    tilt_sigma_deg: float = 0.3
    box_top_sigma_v: float = 0.0103
    # Below this much rise (object height minus camera height) the
    # geometry is too flat to mean anything: the range divides by the
    # elevation of the box top, so as the aircraft climbs toward the
    # target's own height the answer runs away. Measured 2026-11-17 on the
    # first approach flights: 0.07 m of range error per degree with the
    # camera at 0.70 m, 0.26 m/deg at 1.38 m, and above the target's own
    # height there is no answer at all. The aircraft reached a rangefinder
    # reading of 1.83 m in those flights, which is over a person's head.
    min_rise_m: float = 0.35
    # The method's whole premise is that the TOP of the box is the top of
    # the object. Measured 2026-11-17, that premise fails silently: at
    # 1.3-1.5 m the detector returned a box 0.38 of the frame tall whose
    # bottom was nowhere near the floor, when a 1.73 m person at that
    # distance should fill more than the whole frame -- a torso, not a
    # person, and its top is not a head. So the box's angular height is
    # checked against what the derived range implies, and a box that
    # cannot be the whole object is not ranged this way. A box clipped at
    # the bottom of the frame is exempt: the feet being out of shot is
    # normal here and says nothing about the top.
    min_box_fraction: float = 0.55
    frame_edge: float = 0.985
    # Everything the three terms below do not model: posture, where the
    # detector decides hair ends, the pitch not being read at the instant
    # the frame was captured. Measured, same six looks, person tape-measured
    # at 2.57 m: the looks scattered over 0.18 m, which is 7% and is larger
    # than the within-look jitter. Without this the stated sigma is the
    # precision of one look and gets mistaken for the accuracy.
    residual_frac: float = 0.07
    # A measured height for one label, when there is one. The head range is
    # proportional to (object height - camera height), so with the camera at
    # 1.1 m a 0.11 m spread on a person's height is 19% of the answer, while
    # a tape-measured height is a few centimetres. Worth asking for.
    known_heights: Dict[str, float] = field(default_factory=dict)

    def estimate(self, obs: ObjectObservation,
                 boresight_range_m: Optional[float] = None,
                 boresight_trusted: bool = False,
                 mono_depth_m: Optional[float] = None,
                 camera_height_m: Optional[float] = None,
                 pitch_deg: float = 0.0) -> ObjectObservation:
        """Fill range_m / range_source / range_sigma_m on `obs`."""

        # 1. The forward lidar, but only if the object is actually on the
        #    beam. A 2 deg beam measures whatever is in front of it, which is
        #    the target only when the target is centred.
        if (boresight_range_m is not None and boresight_trusted
                and self.boresight.contains(obs, self.intrinsics)):
            obs.range_m = boresight_range_m
            obs.range_source = RangeSource.BORESIGHT
            obs.range_sigma_m = max(self.lidar_floor_m,
                                    boresight_range_m * self.lidar_frac)
            return obs

        # 2. The top of the box against the floor the rangefinder is
        #    measuring. Measured on 2026-11-16 against a tape at 2.57 m:
        #    median error 0.13 m over six looks, where the size prior was
        #    0.65 m out and parallax 0.41 m. It needs the camera's tilt, so
        #    on an uncalibrated mount it simply does not fire.
        r = self.from_head_height(obs, camera_height_m, pitch_deg)
        if r is not None:
            obs.range_m, obs.range_sigma_m = r
            obs.range_source = RangeSource.HEAD_HEIGHT
            return obs

        # 3. Monocular depth, if a scale-anchored estimator is running.
        if mono_depth_m is not None:
            obs.range_m = mono_depth_m
            obs.range_source = RangeSource.MONO_DEPTH
            obs.range_sigma_m = max(0.10, mono_depth_m * self.mono_frac)
            return obs

        # 4. Assumed real-world size. Needs calibrated intrinsics to mean
        #    anything: range scales directly with focal length, so an assumed
        #    FOV produces a confidently wrong number.
        r = self.from_size_prior(obs)
        if r is not None:
            obs.range_m, obs.range_sigma_m = r
            obs.range_source = RangeSource.SIZE_PRIOR
            return obs

        obs.range_m = None
        obs.range_source = RangeSource.NONE
        obs.range_sigma_m = None
        return obs

    def object_height(self, label: str) -> Optional[Tuple[float, float]]:
        """(height, spread) for a label: measured if given, else the prior."""
        if label in self.known_heights:
            return self.known_heights[label], 0.03
        return self.size_priors.get(label)

    def from_head_height(self, obs: ObjectObservation,
                         camera_height_m: Optional[float],
                         pitch_deg: float = 0.0
                         ) -> Optional[Tuple[float, float]]:
        """
        Range from the elevation of the top of the box.

            range = (object height - camera height) / tan(elevation of top)

        No baseline, no movement, and no dependence on heading, which is
        what defeated parallax on this airframe. The bottom of the object
        being out of frame -- always true of a person under a nose-up
        camera -- does not matter, which is what defeats the size prior.
        """
        if (camera_height_m is None or obs.bbox is None
                or obs.label not in self.floor_standing
                or abs(self.intrinsics.tilt_up_deg) < 1e-9):
            return None
        h = self.object_height(obs.label)
        if h is None:
            return None
        real_h, spread = h
        rise = real_h - camera_height_m
        if rise < self.min_rise_m:
            return None
        u = (obs.bbox[0] + obs.bbox[2]) / 2.0
        rng = self.intrinsics.height_range(obs.bbox[1], real_h,
                                           camera_height_m, u=u,
                                           pitch_deg=pitch_deg)
        if rng is None or rng <= 0.0 or abs(rise) < 1e-6:
            return None
        # Could this box be the whole object at that range? The object
        # spans from the floor to real_h, so its top and bottom subtend
        # known elevations; the box has to cover most of that span unless
        # it runs off the bottom of the frame.
        if obs.bbox[3] < self.frame_edge:
            top_e = math.degrees(math.atan2(rise, rng))
            bot_e = math.degrees(math.atan2(-camera_height_m, rng))
            want = abs(top_e - bot_e)
            got = abs(self.intrinsics.angles(u, obs.bbox[1])[1]
                      - self.intrinsics.angles(u, obs.bbox[3])[1])
            if want > 1e-6 and got / want < self.min_box_fraction:
                return None
        # Three independent errors, added in quadrature.
        #  - the object's height, which enters through the RISE, not the
        #    height: at a 1.1 m camera height that is three times as
        #    sensitive as the size prior. Flying lower shrinks it.
        #  - the row the detector puts the top edge on.
        #  - the tilt.
        from_height = rng * abs(spread / rise)
        e = math.radians(max(1.0, math.degrees(math.atan2(rise, rng))))
        per_deg = abs(rise) * math.radians(1.0) / max(1e-6, math.sin(e) ** 2)
        rows_deg = self.box_top_sigma_v * self.intrinsics.vfov_deg
        from_rows = per_deg * rows_deg
        from_tilt = per_deg * self.tilt_sigma_deg
        sigma = math.sqrt(from_height ** 2 + from_rows ** 2
                          + from_tilt ** 2
                          + (rng * self.residual_frac) ** 2)
        return rng, max(0.05, sigma)

    def from_size_prior(self, obs: ObjectObservation
                        ) -> Optional[Tuple[float, float]]:
        prior = self.size_priors.get(obs.label)
        if prior is None or obs.bbox is None:
            return None
        real_h, spread = prior
        box_h_norm = abs(obs.bbox[3] - obs.bbox[1])
        if box_h_norm <= 1e-4:
            return None
        # Pinhole: range = real_height * fy / normalised_box_height
        rng = real_h * self.intrinsics.fy / box_h_norm
        # Uncertainty from the size spread, plus a term for the box itself
        # being a few pixels loose, plus a penalty if intrinsics are assumed.
        frac = spread / real_h + 0.05
        if not self.intrinsics.is_calibrated:
            frac += 0.20
        return rng, max(0.10, rng * frac)

    def disagreement(self, obs: ObjectObservation,
                     boresight_range_m: Optional[float]) -> Optional[float]:
        """
        Difference between the size-prior range and the lidar range for the
        same object, when both exist.

        This is a free runtime confidence signal: when two independent methods
        disagree by more than their combined uncertainty, something is wrong
        with one of them and nothing downstream should be translating.
        """
        if boresight_range_m is None:
            return None
        sp = self.from_size_prior(obs)
        if sp is None:
            return None
        return abs(sp[0] - boresight_range_m)


def observation_from_bbox(label: str, bbox: Tuple[float, float, float, float],
                          intr: CameraIntrinsics, confidence: float = 0.0,
                          track_id: Optional[int] = None,
                          t_mono: float = 0.0) -> ObjectObservation:
    """Build an observation from a normalised bbox, with real intrinsics."""
    u = (bbox[0] + bbox[2]) / 2.0
    v = (bbox[1] + bbox[3]) / 2.0
    bearing, elevation = intr.angles(u, v)
    return ObjectObservation(
        label=label, track_id=track_id, confidence=confidence,
        bearing_deg=bearing, elevation_deg=elevation, bbox=tuple(bbox),
        first_seen_mono=t_mono, last_seen_mono=t_mono, frames_seen=1)
