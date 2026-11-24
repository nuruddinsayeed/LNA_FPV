#!/usr/bin/env python3
"""
Calibrate the camera through the FULL analog path, and find the lidar
boresight.

Start here, always:

    # which index is the capture card? scans them all, saves a frame from each
    python tools/calibrate_camera.py devices

    # then confirm that one device: is a board detectable in what it delivers?
    python tools/calibrate_camera.py probe --camera 1

Index 1 on a Mac is as likely to be a Continuity Camera as a capture card,
and a calibration run that finds no board tells you nothing about which of
those went wrong. `devices` separates "wrong index" from "right device, no
video" from "no camera permission", which otherwise look identical.

Then either:

    # auto-capture: hold the board still, it grabs views by itself
    python tools/calibrate_camera.py intrinsics --camera 1 --out camera.json

    # no GUI, or capture frames however is convenient and calibrate after
    python tools/calibrate_camera.py intrinsics --from-dir frames/ --out camera.json

    python tools/calibrate_camera.py boresight --camera 1 --out camera.json \\
        --connection udpin:0.0.0.0:14550

Why through the full path: the capture card may crop, scale or deinterlace,
and each of those changes the effective focal length. Calibrating the camera
alone gives a number that does not describe what the detector sees.

Why at all: with a 150 deg FPV lens, assuming a 90 deg HFOV misplaces a
target at 0.95 of frame width by 31 degrees.

Board geometry: --board is INNER CORNERS, not squares. A printed board of
10x7 squares has 9x6 inner corners. Getting this wrong means nothing is ever
detected, with no other symptom.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_agent import __version__
from drone_agent.perception.geometry import Boresight, CameraIntrinsics


def require_cv2() -> Any:
    try:
        import cv2
    except ImportError:
        print("this tool needs opencv-python:  pip install opencv-python")
        raise SystemExit(2)
    return cv2


def have_gui(cv2: Any) -> bool:
    """A headless opencv build has no imshow. Detect it rather than crashing
    halfway through a calibration run."""
    try:
        cv2.namedWindow("__probe__")
        cv2.destroyWindow("__probe__")
        return True
    except Exception:
        return False


def open_capture(cv2: Any, source: str, width: Optional[int] = None,
                 height: Optional[int] = None) -> Any:
    src: Any = source
    try:
        src = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"could not open capture source {source!r}.")
        print("  On macOS, index 0 is usually the built-in camera and 1+ are")
        print("  USB devices or a Continuity Camera. Try `probe` on each "
              "index.")
        raise SystemExit(2)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def describe_capture(cv2: Any, cap: Any) -> dict:
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "backend": cap.getBackendName(),
    }


# ───────────────────────── board detection ───────────────────────

def find_board(cv2: Any, gray: Any, pattern: Tuple[int, int],
               exhaustive: bool = True) -> Tuple[bool, Any]:
    """
    Robust checkerboard detection for a noisy analog feed.

    findChessboardCornersSB handles blur, noise and uneven lighting far
    better than the classic detector and needs no subpixel refinement pass.
    The classic one is kept as a fallback because SB occasionally refuses a
    view the old one accepts.
    """
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = 0
        if exhaustive and hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
        if found:
            return True, corners

    found, corners = cv2.findChessboardCorners(
        gray, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    return found, corners


def prepare(cv2: Any, frame: Any, equalize: bool = True) -> Any:
    """Grayscale, and optionally equalise. An analog feed is noisy and often
    poorly exposed; CLAHE recovers a board that is otherwise undetectable in
    shadow."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if equalize:
        gray = cv2.createCLAHE(clipLimit=2.0,
                               tileGridSize=(8, 8)).apply(gray)
    return gray


def board_objp(pattern: Tuple[int, int], square_m: float) -> Any:
    import numpy as np
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * square_m


def corner_centroid(corners: Any) -> Tuple[float, float]:
    pts = corners.reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


class ViewGate:
    """
    Decides when a detected board is worth keeping as a view: it has stopped
    moving, and it is somewhere the views already kept are not.

    `intrinsics` has always captured this way. `verify` did not: it kept any
    six frames with a board in them, 0.4 s apart, so a board in motion (which
    an interlaced analog feed combs) or six pictures of one pose could decide
    the verdict. A first verify of a real calibration read POOR that way and
    the result could not say whether the file or the six pictures were at
    fault.
    """

    def __init__(self, stable_s: float = 0.6, min_move_px: float = 60.0,
                 jitter_px: float = 6.0) -> None:
        self.stable_s, self.min_move_px = stable_s, min_move_px
        self.jitter_px = jitter_px
        self.kept: List[Tuple[float, float]] = []
        self._last: Optional[Tuple[float, float]] = None
        self._since: Optional[float] = None
        self.why = "no board in view"

    def offer(self, centroid: Optional[Tuple[float, float]],
              now: float) -> bool:
        """True when this frame should be kept. `why` says what it is
        waiting for otherwise."""
        if centroid is None:
            self._last, self._since = None, None
            self.why = "no board in view"
            return False
        moved = (self._last is None
                 or math.hypot(centroid[0] - self._last[0],
                               centroid[1] - self._last[1]) > self.jitter_px)
        self._since = now if moved else (self._since
                                         if self._since is not None else now)
        self._last = centroid
        if any(math.hypot(centroid[0] - k[0], centroid[1] - k[1])
               <= self.min_move_px for k in self.kept):
            self.why = ("this position is already covered: move the board "
                        "to another part of the picture")
            return False
        if moved or now - self._since < self.stable_s:
            self.why = "the board is moving: hold it still"
            return False
        self.kept.append(centroid)
        self._since = None
        self.why = "view taken: move the board to another part of the picture"
        return True


# ───────────────────────── device scan ───────────────────────────

@contextlib.contextmanager
def quiet_stderr():
    """
    OpenCV's capture backend prints a C++ error for every index that does
    not exist. Scanning six indices buries the scan's own output in it, so
    silence stderr at the file-descriptor level — redirecting sys.stderr in
    Python does not catch output from C++.
    """
    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def mac_camera_names() -> List[str]:
    """
    OpenCV exposes no device names, only indices. macOS does, via
    system_profiler. The order is usually the order AVFoundation enumerates
    in, but that is not guaranteed, so this is printed as a hint and the
    sample frames are what actually identify the device.
    """
    if sys.platform != "darwin":
        return []
    try:
        out = subprocess.run(["system_profiler", "-json", "SPCameraDataType"],
                             capture_output=True, text=True,
                             timeout=15).stdout
        data = json.loads(out)
        return [str(item.get("_name", "?"))
                for item in data.get("SPCameraDataType", [])]
    except Exception:
        return []


ANALOG_MODES = {
    (720, 576): "PAL 720x576, native analog (4:3 on display, non-square px)",
    (704, 576): "PAL 704x576, native analog",
    (768, 576): "PAL 768x576, square-pixel analog",
    (720, 480): "NTSC 720x480, native analog (4:3 on display)",
    (704, 480): "NTSC 704x480, native analog",
    (640, 480): "640x480, 4:3 — either a rescale or a webcam mode",
}


def describe_analog(w: int, h: int) -> str:
    """
    Say whether this resolution is something an analog chain natively
    produces. A capture card that upscales to 1080p costs inference time for
    no extra information, and the same resolution is what a webcam or a
    Continuity Camera reports, so the number is worth naming either way.
    """
    known = ANALOG_MODES.get((w, h))
    if known:
        return known
    if h == 0:
        return f"{w}x{h}"
    ar = w / h
    name = {1.78: "16:9", 1.33: "4:3", 1.6: "16:10", 1.5: "3:2"}.get(
        round(ar, 2), f"{ar:.2f}:1")
    if abs(ar - 16 / 9) < 0.02:
        return (f"{w}x{h} ({name}) — NOT a native analog mode. Either the "
                f"card is upscaling 4:3 analog, or this is a webcam")
    return f"{w}x{h} ({name}) — not a native analog mode; probably a rescale"


def run_modes(args: argparse.Namespace) -> int:
    """
    Ask the device for each candidate resolution and report what it actually
    returns. A UVC device silently clamps to its nearest supported mode, so
    the only way to learn the real mode list is to request and read back.
    """
    cv2 = require_cv2()
    candidates = [(720, 576), (720, 480), (704, 576), (768, 576),
                  (640, 480), (800, 600), (1024, 768), (1280, 720),
                  (1280, 960), (1920, 1080)]
    print(f"\n  requesting each mode from index {args.camera!r}; a UVC device"
          f" clamps to\n  its nearest supported one, so what comes back is "
          f"what it really has\n")
    seen = {}
    for w, h in candidates:
        with quiet_stderr():
            cap = open_capture_quiet(cv2, args.camera, w, h)
            if cap is None:
                print("  could not open the device")
                return 2
            gw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            gh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok = False
            t0 = time.monotonic()
            n = 0
            while n < 10 and time.monotonic() - t0 < 2.0:
                r, f = cap.read()
                if r and f is not None:
                    ok = True
                    gh, gw = f.shape[0], f.shape[1]
                    n += 1
                else:
                    time.sleep(0.03)
            rate = n / max(time.monotonic() - t0, 1e-3)
            cap.release()
        mark = "" if (gw, gh) == (w, h) else f"  -> clamped to {gw}x{gh}"
        status = f"{rate:4.1f} fps" if ok else "no frames"
        print(f"    asked {w}x{h:<5} got {gw}x{gh:<10} {status}{mark}")
        if ok:
            seen[(gw, gh)] = max(seen.get((gw, gh), 0.0), rate)

    print()
    if not seen:
        print("  nothing delivered frames at any mode.")
        return 1
    print("  modes this device actually delivers:")
    for (w, h), rate in sorted(seen.items()):
        print(f"    {w}x{h:<10} {rate:4.1f} fps   {describe_analog(w, h)}")
    nat = [m for m in seen if m in ANALOG_MODES]
    if nat:
        w, h = sorted(nat)[0]
        print(f"\n  Prefer the native mode. Calibrate AND run inference at "
              f"the same one:")
        print(f"    python tools/calibrate_camera.py intrinsics --camera "
              f"{args.camera} --width {w} --height {h} \\")
        print(f"        --board {args.board} --out camera.json")
    return 0


def open_capture_quiet(cv2: Any, source: str, width: Optional[int] = None,
                       height: Optional[int] = None) -> Optional[Any]:
    """open_capture but returns None instead of exiting, for scanning."""
    src: Any = source
    try:
        src = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def classify_frame(frame: Any) -> str:
    """
    What a capture card delivers says whether the analog chain is up:

      black           the card is fine, nothing is reaching it — VTX off,
                      wrong channel, or the goggles are not driving AV out
      static          the receiver is on but has no lock on a VTX
      live picture    there is a signal

    This distinguishes "wrong device index" from "right device, no video",
    which otherwise look identical: an empty calibration run either way.
    """
    import numpy as np
    arr = np.asarray(frame)
    mean = float(arr.mean())
    # Per-channel, then the largest: a solid colour has zero variance WITHIN
    # each channel but plenty across them (blue 200/0/0 gives std 94 if the
    # array is flattened), so a flattened std reads a no-sync blue screen as
    # a picture.
    if arr.ndim == 3:
        std = max(float(arr[:, :, c].std()) for c in range(arr.shape[2]))
    else:
        std = float(arr.std())
    if std < 6.0 and mean < 12.0:
        return f"BLACK (mean {mean:.0f}, std {std:.0f}) — no video reaching it"
    if std < 6.0:
        return f"FLAT (mean {mean:.0f}, std {std:.0f}) — a solid colour, no picture"

    # Variance alone cannot tell static from a high-contrast picture: a
    # checkerboard filling the frame has the same mid mean and large spread
    # as noise, and the board is exactly what will be held up here.
    # Spatial correlation can. For uncorrelated noise the spread of
    # horizontally adjacent differences is sqrt(2) x the pixel spread; in any
    # real image neighbours resemble each other, so the ratio is far lower.
    g = arr[:, :, 0].astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)
    d = float(np.diff(g, axis=1).std())
    ratio = d / (np.sqrt(2.0) * max(float(g.std()), 1e-6))
    if ratio > 0.80 and 60.0 < mean < 195.0:
        return (f"STATIC (mean {mean:.0f}, std {std:.0f}, neighbour "
                f"decorrelation {ratio:.2f}) — receiving, no VTX lock")
    return f"picture (mean {mean:.0f}, std {std:.0f}, structure {1-ratio:.2f})"


def run_devices(args: argparse.Namespace) -> int:
    cv2 = require_cv2()
    names = mac_camera_names()
    if names:
        print("\n  macOS lists these video devices:")
        for i, n in enumerate(names):
            print(f"    {i}  {n}")
        print("  (order usually matches the indices below — confirm with the "
              "sample frames)")

    print(f"\n  scanning indices 0..{args.list_max}\n")
    live: List[int] = []
    for idx in range(args.list_max + 1):
        with quiet_stderr():
            cap = cv2.VideoCapture(idx)
            opened = bool(cap.isOpened())
            info = describe_capture(cv2, cap) if opened else None
            frame = None
            if opened:
                # The first read after opening is often empty while the
                # device starts streaming; give it a few tries.
                for _ in range(8):
                    ok, f = cap.read()
                    if ok and f is not None:
                        frame = f
                        break
                    time.sleep(0.05)
            cap.release()

        if not opened:
            print(f"    {idx}  —  did not open")
            continue
        assert info is not None
        head = (f"    {idx}  {info['width']}x{info['height']} "
                f"@{info['fps']:g}fps  {info['backend']}")
        if frame is None:
            print(f"{head}  opened but delivered NO frames")
            continue
        shot = str(Path(args.out).with_suffix(f".dev{idx}.png"))
        cv2.imwrite(shot, frame)
        print(f"{head}")
        print(f"        {classify_frame(frame)}")
        print(f"        sample  {shot}")
        live.append(idx)

    print()
    if not live:
        print("  No device delivered a frame.")
        if sys.platform == "darwin":
            print("  On macOS a terminal with no camera permission opens "
                  "devices and reads")
            print("  nothing, exactly like this. Check System Settings > "
                  "Privacy & Security >")
            print("  Camera and enable your terminal app, then run this "
                  "again.")
        return 1
    print(f"  Open the sample frames and pick the index showing the FPV "
          f"feed, then:")
    print(f"    python tools/calibrate_camera.py probe --camera "
          f"{live[-1]} --board {args.board}")
    return 0


# ──────────────────────────── probe ──────────────────────────────

def run_probe(args: argparse.Namespace) -> int:
    cv2 = require_cv2()
    cap = open_capture(cv2, args.camera, args.width, args.height)
    info = describe_capture(cv2, cap)
    print(f"  device         index/path {args.camera!r}")
    print(f"  backend        {info['backend']}")
    print(f"  resolution     {info['width']}x{info['height']}")
    print(f"  reported fps   {info['fps']}")

    pattern = tuple(int(v) for v in args.board.split("x"))
    print(f"  looking for    {pattern[0]}x{pattern[1]} INNER corners "
          f"(a {pattern[0]+1}x{pattern[1]+1} square board)")

    native = describe_analog(info["width"], info["height"])
    print(f"  format         {native}")

    frames = 0
    found_n = 0
    read_s = 0.0
    t0 = time.monotonic()
    saved: Optional[str] = None
    last_frame = None
    while frames < args.probe_frames:
        tr = time.monotonic()
        ok, frame = cap.read()
        read_s += time.monotonic() - tr
        if not ok or frame is None:
            break
        frames += 1
        last_frame = frame
        gray = prepare(cv2, frame, args.equalize)
        found, _ = find_board(cv2, gray, pattern, exhaustive=False)
        if found:
            found_n += 1
        if saved is None:
            saved = str(Path(args.out).with_suffix(".probe.png"))
            cv2.imwrite(saved, frame)
    dt = time.monotonic() - t0
    cap.release()

    # Two different rates, and conflating them is misleading: board search
    # costs more than the capture does, so the loop rate understates what the
    # device can actually deliver.
    print(f"  frames read    {frames} in {dt:.1f}s")
    print(f"  capture rate   {frames/max(read_s,1e-3):.1f} fps "
          f"(read() alone; the loop ran at "
          f"{frames/max(dt,1e-3):.1f} fps including board search)")
    if frames == 0:
        print("\n  NOTHING WAS READ from this device. It opened but delivers "
              "no frames.")
        print("  Try another index, check the capture card is selected as a "
              "video device,")
        print("  and on macOS confirm camera permission for your terminal "
              "in System Settings.")
        print("  `devices` mode scans every index and says which deliver "
              "frames.")
        return 1

    print(f"  board detected {found_n}/{frames} frames")
    if saved:
        print(f"  sample frame   {saved}  (open it: is this the FPV feed?)")
    if last_frame is not None:
        state = classify_frame(last_frame)
        print(f"  content        {state}")
        # A detected board is proof there is a picture, whatever the
        # statistics say, so this advice is only for the case where nothing
        # was found.
        if found_n == 0 and any(k in state for k in ("BLACK", "STATIC", "FLAT")):
            print("\n  This device is working but there is no picture on it. "
                  "The analog chain,")
            print("  in the order it fails: drone battery in (camera and VTX "
                  "are powered by it),")
            print("  goggles on the VTX's band and channel, goggles' AV-OUT "
                  "into the capture")
            print("  card, card into the Mac. A black frame means the card "
                  "is fine and nothing")
            print("  is reaching it; static means the goggles are receiving "
                  "but have no lock.")
            return 1

    if found_n == 0:
        print("\n  No board found in any frame. In order of likelihood:")
        print(f"    0. The board is too SMALL in frame. A 150 deg lens "
              f"shrinks everything:")
        print(f"       a 0.25 m board at 0.5 m spans about 50 px, which is "
              f"~5 px per square")
        print(f"       and undetectable. Print it large (A2/A1, 50-80 mm "
              f"squares) and hold")
        print(f"       it close enough to fill a third of the frame.")
        print(f"    1. --board is wrong. It is INNER CORNERS: a board of "
              f"{pattern[0]+1}x{pattern[1]+1} squares")
        print(f"       has {pattern[0]}x{pattern[1]} inner corners. Count "
              f"them on yours.")
        print("    2. The whole board is not in frame, or it is too small / "
              "too far.")
        print("    3. This is the wrong device — open the sample frame above.")
        print("    4. The feed is too dark or too noisy; try --equalize "
              "(default on) and more light.")
        return 1
    print("\n  Ready. Run the intrinsics mode.")
    return 0


# ─────────────────────────── intrinsics ──────────────────────────

def _fit_pinhole(cv2: Any, obj_points: List[Any], img_points: List[Any],
                 size: Tuple[int, int]) -> Tuple[CameraIntrinsics, float]:
    err, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, size, None, None)
    w, h = size
    return CameraIntrinsics(
        fx=float(K[0, 0]) / w, fy=float(K[1, 1]) / h,
        cx=float(K[0, 2]) / w, cy=float(K[1, 2]) / h,
        k1=float(dist[0][0]), k2=float(dist[0][1]), model="pinhole",
        width_px=w, height_px=h, reproj_err_px=round(float(err), 4)), float(err)


def _fit_fisheye(cv2: Any, obj_points: List[Any], img_points: List[Any],
                 size: Tuple[int, int]) -> Tuple[CameraIntrinsics, float]:
    """
    Kannala-Brandt. This is the right model for an FPV lens: the standard
    radial polynomial stops being invertible past roughly 100 deg of field,
    where it produces a pixel-to-angle mapping that rises and then FALLS.
    """
    import numpy as np
    w, h = size
    objp = [o.reshape(1, -1, 3).astype(np.float64) for o in obj_points]
    imgp = [i.reshape(1, -1, 2).astype(np.float64) for i in img_points]
    K = np.zeros((3, 3)); D = np.zeros((4, 1))
    # These constants moved to the cv2 top level in OpenCV 5; older builds
    # have them under cv2.fisheye. Look in both rather than assuming.
    def flag(name: str, default: int) -> int:
        return getattr(cv2.fisheye, name, getattr(cv2, name, default))
    flags = (flag("CALIB_RECOMPUTE_EXTRINSIC", 1)
             | flag("CALIB_FIX_SKEW", 8))
    err, K, D, _, _ = cv2.fisheye.calibrate(
        objp, imgp, size, K, D, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50,
                  1e-6))
    return CameraIntrinsics(
        fx=float(K[0, 0]) / w, fy=float(K[1, 1]) / h,
        cx=float(K[0, 2]) / w, cy=float(K[1, 2]) / h,
        k1=float(D[0][0]), k2=float(D[1][0]), k3=float(D[2][0]),
        k4=float(D[3][0]), model="fisheye",
        width_px=w, height_px=h, reproj_err_px=round(float(err), 4)), float(err)


def _per_view_bow(intr: CameraIntrinsics, img_points: List[Any],
                  pattern: Tuple[int, int],
                  size: Tuple[int, int]) -> List[Optional[float]]:
    """Collinearity residual per view, under a given fitted model."""
    w, h = size
    return [_collinearity(intr, ip, pattern, w, h) for ip in img_points]


def calibrate_from_points(cv2: Any, obj_points: List[Any],
                          img_points: List[Any],
                          size: Tuple[int, int], note: str,
                          model: str = "auto",
                          pattern: Optional[Tuple[int, int]] = None,
                          drop_outliers: bool = True) -> Tuple[
                              CameraIntrinsics, float]:
    """
    Fit the requested model, or with "auto" fit both and keep the better one.

    Selection is on COLLINEARITY, not reprojection error. Reprojection error
    is the quantity the fit minimises, so comparing models by it rewards
    whichever had more freedom and is near-useless when the two land within
    noise of each other -- a 1.047 vs 1.066 px difference is not evidence,
    yet it is enough to flip the choice to the model that is physically
    wrong for a wide lens. Collinearity is different in kind: a straight
    line in the world must stay straight after undistortion, and the fit
    never optimises for that, so it discriminates.

    Outlier views are dropped before the final fit. On an interlaced analog
    feed a hand-held board moving between the two fields combs every
    horizontal edge, displacing corners by a field offset. One such view
    corrupts a 20-view fit, and the symptom is exactly what was seen here:
    a good median bow with a terrible worst case.
    """
    def fit_all(objp: List[Any], imgp: List[Any]) -> List[tuple]:
        out = []
        for nm, fn in (("pinhole", _fit_pinhole), ("fisheye", _fit_fisheye)):
            if model not in ("auto", nm):
                continue
            try:
                intr, err = fn(cv2, objp, imgp, size)
            except Exception as e:
                print(f"    {nm} fit failed: {e}")
                continue
            mono = _monotonic(intr)
            bows = ([b for b in _per_view_bow(intr, imgp, pattern, size)
                     if b is not None] if pattern else [])
            med = st.median(bows) if bows else None
            worst = max(bows) if bows else None
            msg = (f"    {nm:<8} reprojection {err:.3f} px, "
                   f"hfov {intr.hfov_deg:.1f} deg, "
                   f"mapping {'monotonic' if mono else 'NON-MONOTONIC'}")
            if med is not None:
                msg += f", bow {med:.2f} px median / {worst:.2f} worst"
            print(msg)
            out.append({"name": nm, "intr": intr, "err": err, "mono": mono,
                        "med": med, "worst": worst})
        return out

    results = fit_all(obj_points, img_points)
    if not results:
        raise RuntimeError("no model could be fitted")

    def score(r: dict) -> tuple:
        # monotonic first (a non-monotonic mapping is not a calibration),
        # then median bow, then reprojection error as the last resort.
        return (0 if r["mono"] else 1,
                r["med"] if r["med"] is not None else 1e9,
                r["err"])

    results.sort(key=score)
    best = results[0]

    # Drop outlier views under the chosen model and refit. Threshold is
    # median + 4*MAD, which is robust to the outliers it is looking for in a
    # way that a mean and standard deviation are not.
    if drop_outliers and pattern and len(img_points) >= 8:
        bows = _per_view_bow(best["intr"], img_points, pattern, size)
        good = [b for b in bows if b is not None]
        if len(good) >= 8:
            med = st.median(good)
            mad = st.median([abs(b - med) for b in good]) or 1e-6
            limit = max(med + 4.0 * mad, med * 2.0, 0.5)
            keep = [i for i, b in enumerate(bows)
                    if b is not None and b <= limit]
            dropped = len(img_points) - len(keep)
            if dropped and len(keep) >= 8:
                worst_list = sorted(
                    ((b, i) for i, b in enumerate(bows) if b is not None),
                    reverse=True)[:dropped]
                print(f"\n    dropping {dropped} outlier view(s) over "
                      f"{limit:.2f} px bow "
                      f"(median {med:.2f}): "
                      + ", ".join(f"#{i+1} at {b:.1f} px"
                                  for b, i in worst_list))
                print(f"    refitting on {len(keep)} views")
                refit = fit_all([obj_points[i] for i in keep],
                                [img_points[i] for i in keep])
                if refit:
                    refit.sort(key=score)
                    if score(refit[0]) < score(best):
                        best = refit[0]
                        note = f"{note}, {dropped} outlier view(s) dropped"
                    else:
                        print("    the refit was not better; keeping all "
                              "views")

    name, intr, err = best["name"], best["intr"], best["err"]
    if len(results) > 1:
        why = ("lower board bow" if best["med"] is not None
               else "lower reprojection error")
        print(f"    -> keeping {name} ({why})")
    rank = 1e6 if not best["mono"] else 0.0
    if rank >= 1e6:
        # Every candidate fit is invalid over the frame. Saving one would
        # hand the stack bearings that rise and then fall, which is worse
        # than an honestly assumed FOV because it looks calibrated.
        raise RuntimeError(
            f"the best fit ({name}) has a NON-MONOTONIC pixel-to-angle "
            f"mapping, so its bearings are meaningless. For a lens this "
            f"wide use --model fisheye; if that also fails, the views do "
            f"not constrain the fit (capture more, covering the frame "
            f"corners).")
    intr.note = f"{note} [{name}]"
    return intr, err


def _monotonic(intr: CameraIntrinsics) -> bool:
    """Does pixel position map to a monotonically increasing angle? A model
    fitted outside its validity range does not, and a bearing taken from it
    is meaningless."""
    prev = None
    for i in range(51):
        u = intr.cx + (0.5 - 0.0) * i / 50.0
        if u > 1.0:
            break
        a = intr.angles(min(u, 1.0), intr.cy)[0]
        if prev is not None and a < prev - 1e-6:
            return False
        prev = a
    return True


def report_intrinsics(intr: CameraIntrinsics, err: float, n: int) -> None:
    # n is how many views were CAPTURED. If outliers were dropped the fit
    # used fewer, and the note records how many -- so say so here rather
    # than printing a count the fit did not actually use.
    dropped = 0
    mark = "outlier view(s) dropped"
    if mark in intr.note:
        try:
            dropped = int(intr.note.split(mark)[0].strip().split()[-1])
        except (ValueError, IndexError):
            dropped = 0
    if dropped:
        print(f"\n  views used     {n - dropped} of {n} "
              f"({dropped} dropped as outliers)")
    else:
        print(f"\n  views used     {n}")
    print(f"  model          {intr.model}")
    print(f"  resolution     {intr.width_px}x{intr.height_px}")
    print(f"  HFOV           {intr.hfov_deg:.1f} deg")
    print(f"  reprojection   {err:.3f} px"
          f"{'   (>1 px: recapture, more varied angles)' if err > 1.0 else ''}")
    print(f"  distortion     k1 {intr.k1:+.4f}  k2 {intr.k2:+.4f}"
          + (f"  k3 {intr.k3:+.4f}  k4 {intr.k4:+.4f}"
             if intr.is_fisheye else ""))
    naive = CameraIntrinsics.from_hfov(90.0, model="pinhole")
    print("  bearing error avoided, vs assuming a 90 deg pinhole:")
    for u in (0.7, 0.8, 0.9, 0.95):
        print(f"    u={u:.2f}  calibrated {intr.angles(u, intr.cy)[0]:+6.1f} "
              f"deg   assumed {naive.angles(u, 0.5)[0]:+6.1f} deg   "
              f"error {abs(intr.angles(u, intr.cy)[0] - naive.angles(u, 0.5)[0]):5.1f} deg")


# ──────────────────────────── verify ─────────────────────────────

def _collinearity(intr: CameraIntrinsics, corners: Any,
                  pattern: Tuple[int, int], w: int, h: int) -> Optional[float]:
    """
    The acid test for a calibration, and the only one that uses ground truth
    you actually have.

    A checkerboard row is a straight line of points on a plane. Under ANY
    correct pinhole projection a straight line in the world stays straight
    in the image. So: take the detected corners, map each through the fitted
    model to a ray, re-project those rays onto an ideal distortion-free
    plane, and the rows and columns must come out straight. Whatever
    curvature remains is calibration error, measured against geometry rather
    than against the fit's own residual.

    This catches what reprojection error cannot. Reprojection error is the
    fit grading its own homework: a wrong lens model can still fit the
    observed corners closely while extrapolating nonsense elsewhere.

    Returns the worst perpendicular deviation in pixels of an equivalent
    rectified image, or None if the view reaches angles where the rectified
    plane is not usable.
    """
    import numpy as np
    cols, rows = pattern
    pts = corners.reshape(-1, 2)
    if len(pts) != cols * rows:
        return None

    f_virt = intr.fx * w          # keep the result in comparable pixels
    rect = np.zeros((len(pts), 2), float)
    for i, (u_px, v_px) in enumerate(pts):
        az, el = intr.angles(float(u_px) / w, float(v_px) / h)
        if abs(az) > 80.0 or abs(el) > 80.0:
            return None           # tan() explodes; this view proves nothing
        rect[i, 0] = f_virt * math.tan(math.radians(az))
        rect[i, 1] = -f_virt * math.tan(math.radians(el))

    grid = rect.reshape(rows, cols, 2)
    worst = 0.0
    lines = [grid[r, :, :] for r in range(rows)] + \
            [grid[:, c, :] for c in range(cols)]
    for line in lines:
        c0 = line.mean(axis=0)
        d = line - c0
        # principal direction by SVD, then perpendicular distances to it
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        n = np.array([-vt[0, 1], vt[0, 0]])
        worst = max(worst, float(np.abs(d @ n).max()))
    return worst


def list_images(spec: str) -> List[Path]:
    """
    Images from a directory, with a readable error instead of a traceback.
    Each release extracts to its own folder, so a --from-dir that worked
    last version often points at nothing this version -- that is a path
    problem and should say so plainly.
    """
    d = Path(spec).expanduser()
    if not d.exists():
        raise SystemExit(
            f"no such directory: {d}\n"
            f"  Saved frames and camera.json live wherever you ran the "
            f"PREVIOUS version,\n"
            f"  not in this release's folder. Find them with:\n"
            f"    find ~ -maxdepth 6 -type d -name frames 2>/dev/null\n"
            f"  then pass the full path to --from-dir.")
    if not d.is_dir():
        raise SystemExit(f"--from-dir needs a directory, but {d} is a file")
    files = sorted(q for q in d.iterdir()
                   if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if not files:
        raise SystemExit(f"{d} holds no .png/.jpg/.bmp images")
    return files


VERIFY_CAPTURE_S = 90.0


def run_verify(args: argparse.Namespace) -> int:
    cv2 = require_cv2()
    path = Path(args.out)
    if not path.exists():
        print(f"no calibration at {path}. Run intrinsics first.")
        return 2
    intr = CameraIntrinsics.load(str(path))
    raw = json.loads(path.read_text())

    print(f"\n  file           {path}")
    print(f"  model          {intr.model}"
          + ("" if intr.is_fisheye else
             "   <-- only valid to ~100 deg of field"))
    print(f"  calibrated at  {intr.width_px}x{intr.height_px}")
    print(f"  fx, fy         {intr.fx:.4f}, {intr.fy:.4f}  (normalised)")
    print(f"  cx, cy         {intr.cx:.4f}, {intr.cy:.4f}")
    print(f"  k1..k4         {intr.k1:+.4f} {intr.k2:+.4f} "
          f"{intr.k3:+.4f} {intr.k4:+.4f}")
    if intr.reproj_err_px is not None:
        verdict = ("good" if intr.reproj_err_px < 0.5 else
                   "acceptable for analog" if intr.reproj_err_px < 1.0 else
                   "HIGH — suspect blurred or mis-detected views")
        print(f"  reprojection   {intr.reproj_err_px:.3f} px   ({verdict})")
    if raw.get("note"):
        print(f"  note           {raw['note']}")

    # --- 1. principal point sanity -----------------------------------
    off_x = abs(intr.cx - 0.5) * intr.width_px
    off_y = abs(intr.cy - 0.5) * intr.height_px
    print(f"\n  principal point {off_x:.0f} px, {off_y:.0f} px from centre")
    if off_x > 0.12 * intr.width_px or off_y > 0.12 * intr.height_px:
        print("    SUSPECT: a lens is rarely mounted that far off centre.")
        print("    Usually means the views did not cover the frame evenly.")

    # --- 2. monotonicity and the angle table -------------------------
    print(f"\n  derived HFOV   {intr.hfov_deg:.1f} deg "
          f"(compare with your lens's spec)")
    print("  pixel -> angle across the half-width:")
    prev = -1e9
    mono = True
    for frac in (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        u = intr.cx + frac * (1.0 - intr.cx)
        az = intr.angles(u, intr.cy)[0]
        flag = ""
        if az < prev - 1e-6:
            mono = False
            flag = "   <-- FALLS. no lens does this; the fit is invalid"
        prev = az
        print(f"    u={frac:4.2f} of half-width   {az:6.2f} deg{flag}")
    if not mono:
        print("\n  REJECT this calibration. Re-run with --model fisheye.")
        return 1

    # Edge compression. A pinhole fit to a wide lens saturates: the mapping
    # stays technically monotonic but the degrees-per-pixel collapses toward
    # the frame edge, so the model under-reports every peripheral angle. This
    # is the failure that reprojection error cannot see -- such a fit can sit
    # at 0.13 px and still be 35 deg wrong about the field of view.
    def _slope(a: float, b: float) -> float:
        ua = intr.cx + a * (1.0 - intr.cx)
        ub = intr.cx + b * (1.0 - intr.cx)
        return abs(intr.angles(ub, intr.cy)[0] - intr.angles(ua, intr.cy)[0])

    centre_slope = _slope(0.0, 0.1)
    edge_slope = _slope(0.9, 1.0)
    comp = edge_slope / max(centre_slope, 1e-9)
    print(f"\n  edge linearity {comp:.2f} "
          f"(deg/px at the edge, over deg/px at the centre)")
    if comp < 0.45:
        print("    SATURATED. The model flattens out at the frame edge, "
              "which is what a")
        print("    pinhole fit does to a wide lens. Peripheral angles will "
              "read too small —")
        print("    exactly the error this module exists to prevent.")
        if not intr.is_fisheye:
            print("    Re-run intrinsics with --model fisheye.")
        else:
            print("    Re-run with more views at the frame EDGES.")
        return 1
    if comp < 0.70:
        print("    marginal — the edges are under-constrained. More views "
              "in the frame corners would help.")

    if args.expect_hfov:
        err_deg = abs(intr.hfov_deg - args.expect_hfov)
        print(f"  vs spec        lens is {args.expect_hfov:g} deg, "
              f"calibration says {intr.hfov_deg:.1f} deg "
              f"(off by {err_deg:.1f})")
        if err_deg > 20.0:
            print("    That gap is too large to be lens tolerance. Either "
                  "the model is wrong,")
            print("    or the capture card is cropping — which would be "
                  "real and worth knowing.")
            return 1
        if err_deg > 10.0:
            print("    Larger than I would expect. A capture card crop "
                  "would do this.")

    # --- 3. cross-check my solver against cv2 ------------------------
    try:
        import numpy as np
        w, h = intr.width_px, intr.height_px
        K = np.array([[intr.fx * w, 0, intr.cx * w],
                      [0, intr.fy * h, intr.cy * h], [0, 0, 1]], float)
        test = np.array([[[0.15 * w, 0.2 * h]], [[0.5 * w, 0.5 * h]],
                         [[0.85 * w, 0.8 * h]], [[0.95 * w, 0.5 * h]]], float)
        if intr.is_fisheye:
            D = np.array([intr.k1, intr.k2, intr.k3, intr.k4], float)
            und = cv2.fisheye.undistortPoints(test, K, D)
        else:
            D = np.array([intr.k1, intr.k2, 0.0, 0.0], float)
            und = cv2.undistortPoints(test, K, D)
        worst = 0.0
        for i in range(len(test)):
            u_px, v_px = test[i, 0]
            az, el = intr.angles(u_px / w, v_px / h)
            xu, yu = und[i, 0]
            az_cv = math.degrees(math.atan(xu))
            el_cv = -math.degrees(math.atan(yu))
            worst = max(worst, abs(az - az_cv), abs(el - el_cv))
        ok = worst < 0.05
        print(f"\n  vs OpenCV      worst disagreement {worst:.3f} deg"
              f"   ({'ok' if ok else 'PROBLEM'})")
        if not ok:
            print("    The stack's own solver disagrees with cv2 on this "
                  "model. Do not fly on it.")
            return 1
    except Exception as exc:
        print(f"\n  vs OpenCV      skipped ({exc})")

    # --- 4. collinearity against the board ---------------------------
    pattern = tuple(int(v) for v in args.board.split("x"))
    frames: List[Any] = []
    names: List[str] = []
    if args.from_dir:
        files = list_images(args.from_dir)
        for q in files[:args.max_views]:
            img = cv2.imread(str(q))
            if img is not None:
                frames.append(img)
                names.append(q.name)
        src = f"{len(frames)} images from {args.from_dir}"
    elif args.camera is not None and not args.no_capture:
        # Default to the resolution this calibration was fitted at. Asking
        # for nothing lets the card pick, and a card that upscales 4:3
        # analog into a 16:9 frame invalidates the normalised intrinsics --
        # which showed up as a POOR straightness verdict on a calibration
        # that was in fact fine.
        want_w = args.width or intr.width_px
        want_h = args.height or intr.height_px
        cap = open_capture(cv2, args.camera, want_w, want_h)
        got = describe_capture(cv2, cap)
        bad = intr.check_capture(got["width"], got["height"])
        if bad:
            cap.release()
            print(f"\n  straightness   NOT RUN")
            print(f"    {bad}")
            print(f"    asked for {want_w}x{want_h}, got "
                  f"{got['width']}x{got['height']}.")
            print(f"    `modes --camera {args.camera}` lists what this "
                  f"device really offers.")
            return 1
        if (got["width"], got["height"]) != (intr.width_px, intr.height_px):
            print(f"\n  note: capturing {got['width']}x{got['height']}, "
                  f"calibrated at {intr.width_px}x{intr.height_px} — same "
                  f"aspect, so the normalised intrinsics still hold")
        print(f"\n  capturing 6 board views for the straightness test.")
        print(f"    Hold the board STILL until a view is taken, then move "
              f"it to a different part of")
        print(f"    the picture: the middle, each side, the corners. Up to "
              f"{VERIFY_CAPTURE_S:.0f} s.")
        gate = ViewGate(args.stable_s, args.min_move_px)
        t0 = time.monotonic()
        t_said = t0
        while (len(frames) < 6
               and time.monotonic() - t0 < min(args.timeout_s,
                                               VERIFY_CAPTURE_S)):
            ok, f = cap.read()
            if not ok or f is None:
                break
            g = prepare(cv2, f, args.equalize)
            found, corners = find_board(cv2, g, pattern, exhaustive=False)
            now = time.monotonic()
            c = corner_centroid(corners) if found else None
            if gate.offer(c, now):
                frames.append(f)
                names.append(f"view {len(frames)} at ({c[0]:.0f}, "
                             f"{c[1]:.0f})")
                print(f"    {names[-1]}")
                t_said = now
            elif now - t_said > 4.0:
                print(f"      ... {gate.why}")
                t_said = now
        cap.release()
        src = f"{len(frames)} live views, each held still"
    else:
        src = "skipped"

    if not frames:
        print(f"\n  straightness   {src} — no views, test not run")
        print("    This is the strongest check there is. Run it:")
        print(f"      python tools/calibrate_camera.py verify --camera "
              f"{args.camera} --board {args.board}")
        return 0

    devs = []
    per_view: List[Tuple[float, str]] = []
    for i, f in enumerate(frames):
        g = prepare(cv2, f, args.equalize)
        found, corners = find_board(cv2, g, pattern)
        if not found:
            continue
        d = _collinearity(intr, corners, pattern, f.shape[1], f.shape[0])
        if d is not None:
            devs.append(d)
            per_view.append((d, names[i] if i < len(names)
                             else f"view {i+1}"))

    print(f"\n  straightness   {src}, {len(devs)} usable")
    if not devs:
        print("    no view produced a usable measurement")
        return 0
    worst = max(devs)
    med = st.median(devs)
    for d, nm in per_view:
        print(f"      {d:6.2f} px   {nm}")
    print(f"    board lines bow by {med:.2f} px median, {worst:.2f} px worst")
    # The same figure as an angle, because a bearing is what this file is
    # used for and pixels do not say how much it matters.
    f_virt = intr.fx * frames[0].shape[1]
    print(f"    as an angle near the middle of the picture: "
          f"{math.degrees(math.atan(med / f_virt)):.1f} deg median, "
          f"{math.degrees(math.atan(worst / f_virt)):.1f} deg worst")
    if len(devs) < 3:
        print(f"    Only {len(devs)} view(s): not enough to judge a "
              f"calibration on. Run it again and")
        print(f"    give it the board in at least three different places.")
        return 0
    print(f"    (after undistortion a straight line must stay straight; "
          f"what is left is error)")
    # The median is what the model is worth; a far worse single view is a bad
    # FRAME, not a bad model, and naming it is what lets it be removed.
    per_view.sort(reverse=True)
    if per_view and per_view[0][0] > max(2.0 * med, med + 1.0):
        print(f"    worst views, far above the {med:.2f} px median — these "
              f"are bad FRAMES, not a bad model:")
        for d, nm in per_view[:4]:
            if d <= max(2.0 * med, med + 1.0):
                break
            print(f"      {d:6.2f} px   {nm}")
        print(f"    An interlaced analog feed combs a moving board, which "
              f"displaces corners by a")
        print(f"    field offset. Delete those frames and re-run "
              f"intrinsics --from-dir, or just")
        print(f"    re-run it: outlier views are now dropped "
              f"automatically.")
    if med < 1.0 and worst < 4.0:
        print("    VERDICT: good. The model explains the lens.")
    elif med < 2.0:
        print("    VERDICT: usable. Acceptable for an analog feed.")
    else:
        print("    VERDICT: POOR. The residual curvature is large.")
        print("    Most likely: too few views, all near the centre, or all")
        print("    fronto-parallel. Re-run intrinsics pushing the board into")
        print("    the frame CORNERS and tilting it 30-45 deg.")
        return 1
    return 0


def run_intrinsics_from_dir(args: argparse.Namespace) -> int:
    cv2 = require_cv2()
    pattern = tuple(int(v) for v in args.board.split("x"))
    objp = board_objp(pattern, args.square_mm / 1000.0)

    files = list_images(args.from_dir)
    print(f"  {len(files)} images in {args.from_dir}")

    obj_points: List[Any] = []
    img_points: List[Any] = []
    size: Optional[Tuple[int, int]] = None
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"    {f.name}: unreadable")
            continue
        gray = prepare(cv2, img, args.equalize)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            print(f"    {f.name}: {gray.shape[::-1]} differs from {size} — "
                  f"skipped (mixing resolutions invalidates a calibration)")
            continue
        found, corners = find_board(cv2, gray, pattern)
        print(f"    {f.name}: {'board found' if found else 'no board'}")
        if found:
            obj_points.append(objp.copy())
            img_points.append(corners)

    if len(obj_points) < 6 or size is None:
        print(f"\n  only {len(obj_points)} usable views — need 6 minimum, "
              f"15-20 for a good fit")
        return 1
    try:
        intr, err = calibrate_from_points(
        cv2, obj_points, img_points, size,
        f"checkerboard {args.board} @ {args.square_mm:g}mm, "
        f"{len(obj_points)} views from {args.from_dir}", args.model,
        pattern=pattern, drop_outliers=not args.keep_outliers)
    except RuntimeError as e:
        print(f"\n  CALIBRATION REJECTED: {e}")
        return 1
    report_intrinsics(intr, err, len(obj_points))
    _merge_save(args.out, intrinsics=intr)
    print(f"\n-> {args.out}")
    return 0


def run_intrinsics(args: argparse.Namespace) -> int:
    if args.from_dir:
        return run_intrinsics_from_dir(args)

    cv2 = require_cv2()
    pattern = tuple(int(v) for v in args.board.split("x"))
    objp = board_objp(pattern, args.square_mm / 1000.0)
    gui = have_gui(cv2) and not args.no_gui
    if not gui:
        print("  no GUI (headless opencv or --no-gui): auto-capture only, "
              "no preview")

    cap = open_capture(cv2, args.camera, args.width, args.height)
    print(f"  capture: {describe_capture(cv2, cap)}")
    if gui:
        print("  SPACE capture   c calibrate   q quit   "
              "(auto-capture also running)")
    print(f"  hold the board still; a view is taken automatically once it has "
          f"moved >{args.min_move_px:.0f} px from every previous one")

    obj_points: List[Any] = []
    img_points: List[Any] = []
    centroids: List[Tuple[float, float]] = []
    size: Optional[Tuple[int, int]] = None
    frames = 0
    found_frames = 0
    stable_since: Optional[float] = None
    last_centroid: Optional[Tuple[float, float]] = None
    frame_dir = Path(args.save_frames) if args.save_frames else None
    if frame_dir:
        frame_dir.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + args.timeout_s
    while time.monotonic() < deadline and len(obj_points) < args.max_views:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames += 1
        gray = prepare(cv2, frame, args.equalize)
        size = gray.shape[::-1]
        # Exhaustive search is slow; run it only every Nth frame so the
        # preview stays responsive.
        heavy = (frames % max(1, args.search_every) == 0)
        found, corners = find_board(cv2, gray, pattern, exhaustive=heavy)
        if found:
            found_frames += 1

        take = False
        if found:
            c = corner_centroid(corners)
            moved = (last_centroid is None
                     or ((c[0] - last_centroid[0]) ** 2
                         + (c[1] - last_centroid[1]) ** 2) ** 0.5 > 6.0)
            # Stable = the board has stopped moving. Capturing mid-motion on
            # an interlaced analog feed gives smeared corners and a bad fit.
            stable_since = time.monotonic() if moved else (stable_since
                                                           or time.monotonic())
            last_centroid = c
            held = time.monotonic() - (stable_since or time.monotonic())
            far_enough = all(
                ((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2) ** 0.5
                > args.min_move_px for p in centroids)
            take = (not moved and held >= args.stable_s and far_enough)
        else:
            stable_since = None
            last_centroid = None

        key = 255
        if gui:
            view = frame.copy()
            if found:
                cv2.drawChessboardCorners(view, pattern, corners, True)
            for p in centroids:
                cv2.circle(view, (int(p[0]), int(p[1])), 5, (0, 200, 255), 2)
            cv2.putText(view, f"views {len(obj_points)}/{args.max_views}   "
                              f"{'BOARD' if found else 'no board'}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow("calibrate", view)
            key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap.release()
            if gui:
                cv2.destroyAllWindows()
            return 1
        if key == ord("c"):
            break
        if found and (take or key == ord(" ")):
            obj_points.append(objp.copy())
            img_points.append(corners)
            centroids.append(corner_centroid(corners))
            if frame_dir:
                cv2.imwrite(str(frame_dir / f"view_{len(obj_points):03d}.png"),
                            frame)
            print(f"    view {len(obj_points)} at "
                  f"({centroids[-1][0]:.0f}, {centroids[-1][1]:.0f})"
                  f"{' [auto]' if take else ''}")
            stable_since = None

    cap.release()
    if gui:
        cv2.destroyAllWindows()

    print(f"\n  frames read {frames}, board seen in {found_frames}")
    if not obj_points:
        print("  No views captured. Run `probe` first — it reports whether "
              "this device")
        print("  delivers frames at all and whether your --board geometry "
              "matches.")
        return 1
    if len(obj_points) < 6 or size is None:
        print(f"  only {len(obj_points)} views — need 6 minimum, 15-20 for a "
              f"good fit")
        if frame_dir:
            print(f"  frames kept in {frame_dir}; add more and re-run with "
                  f"--from-dir {frame_dir}")
        return 1

    try:
        intr, err = calibrate_from_points(
        cv2, obj_points, img_points, size,
        f"checkerboard {args.board} @ {args.square_mm:g}mm, "
        f"{len(obj_points)} views, full analog path", args.model,
        pattern=pattern, drop_outliers=not args.keep_outliers)
    except RuntimeError as e:
        print(f"\n  CALIBRATION REJECTED: {e}")
        return 1
    report_intrinsics(intr, err, len(obj_points))
    _merge_save(args.out, intrinsics=intr)
    print(f"\n-> {args.out}")
    return 0


# ───────────────────────────── tilt ──────────────────────────────

MAX_WALL_M = 8.0        # past this, one row of reading error is metres


def solve_from_floor(intr: CameraIntrinsics, junction_v: float,
                     mark_height: float, camera_height: float
                     ) -> Tuple[float, bool]:
    """
    The lens-to-wall distance that puts BOTH the marked line and the
    wall-floor junction at the rows they occupy in the picture.

    Two features at two known heights, one picture, two unknowns: the
    distance and the tilt. The mark's depression below the camera's axis
    grows as the aircraft gets closer; the floor's, read off the lens
    distortion, shrinks. One distance makes them consistent.

    Returns (distance_m, pinned). `pinned` means the search ran into a
    bound: no distance fits, so the heights or the row are wrong.
    """
    rise = mark_height - camera_height
    lo, hi = 0.1, 60.0
    e_j = intr._camera_angles(intr.cx, junction_v)[1]

    def diff(d: float) -> float:
        from_mark = math.degrees(math.atan2(rise, d))
        from_floor = -e_j + math.degrees(math.atan2(-camera_height, d))
        return from_mark - from_floor

    if diff(lo) <= 0.0 or diff(hi) >= 0.0:
        return 0.5 * (lo + hi), True
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if diff(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


def run_tilt(args: argparse.Namespace) -> int:
    """
    Measure how far above the airframe's forward axis the camera points.

    A checkerboard calibration cannot see this: the board was wherever it
    was held, so every angle it gives is in the camera's own frame. On an
    FPV airframe the camera is mounted nose-up, and that makes an object
    below the optical axis read at a larger azimuth than it has -- which
    is why a person's feet are never in frame, and part of why a bearing
    difference between two positions does not give the range it should.

    The measurement needs no flying and no board. Stand the aircraft level,
    facing a wall a measured distance away. This prints a frame with a line
    drawn across it where the camera's own horizon lies. Read off the height
    on the wall that the line crosses, and give it back.
    """
    cv2 = require_cv2()
    try:
        intr = CameraIntrinsics.load(args.out)
    except (OSError, json.JSONDecodeError):
        print(f"  no calibration at {args.out}. Run intrinsics first: the "
              f"tilt is measured against the lens model, not instead of it.")
        return 2

    # The image row whose elevation is zero in the camera's own frame.
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if intr._camera_angles(intr.cx, mid)[1] > 0.0:
            lo = mid
        else:
            hi = mid
    v0 = 0.5 * (lo + hi)

    if args.mark_height is None or (args.distance is None
                                    and args.junction_v is None):
        cap = open_capture(cv2, args.camera, intr.width_px, intr.height_px)
        frame = None
        for _ in range(15):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        cap.release()
        if frame is None:
            print("  no frame from the camera.")
            return 2
        h, w = frame.shape[:2]
        y = int(round(v0 * h))
        cv2.line(frame, (0, y), (w, y), (0, 0, 255), 2)
        cv2.line(frame, (int(intr.cx * w), y - 25),
                 (int(intr.cx * w), y + 25), (0, 0, 255), 2)
        cv2.putText(frame, "camera horizon", (10, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        out = str(Path(args.out).with_name("tilt.png"))
        cv2.imwrite(out, frame)
        print(f"  frame saved to {out}, with the camera's horizon drawn "
              f"across it at v={v0:.3f}")
        print("")
        print("  Now, without moving anything:")
        print("    1. mark the wall where the RED line falls, and measure "
              "that height")
        print("    2. measure the height of the lens above the same floor")
        print("    3. read off the YELLOW scale where the wall meets the "
              "floor")
        print("")
        print("  then run, with your own numbers in metres:")
        print(f"    python tools/calibrate_camera.py tilt --camera "
              f"{args.camera} --out {args.out} \\")
        print("        --mark-height H --camera-height C --junction-v V")
        print("")
        print("  The floor line replaces measuring the distance to the "
              "wall: two features at")
        print("  two known heights in one picture give the distance and the "
              "tilt together. To")
        print("  measure the distance instead, pass --distance D and leave "
              "--junction-v out.")
        return 0

    if args.distance is None and args.junction_v is None:
        print("  give either --junction-v (read off the yellow scale where "
              "the wall meets the")
        print("  floor) or --distance (lens to wall, measured).")
        return 2

    if args.junction_v is not None:
        # Two points on the wall at known heights -- the marked line, and
        # the floor, which is at zero -- seen in one picture. The distance
        # and the tilt are whatever puts both of them where they are.
        e_j = intr._camera_angles(intr.cx, args.junction_v)[1]
        dist, pinned = solve_from_floor(
            intr, args.junction_v, args.mark_height, args.camera_height)
        args.distance = dist
        print(f"  the floor line at v={args.junction_v:.3f} is "
              f"{e_j:+.2f} deg below the camera's axis")
        print(f"  with the lens {args.camera_height:.2f} m up and the mark "
              f"at {args.mark_height:.2f} m, the wall is "
              f"{args.distance:.2f} m away")
        # One row of reading error, in metres of wall. Close in this is
        # centimetres; far out one row is the whole answer, which is the
        # real reason the far cases below get refused.
        near = solve_from_floor(intr, min(1.0, args.junction_v + 0.01),
                                args.mark_height, args.camera_height)[0]
        far = solve_from_floor(intr, max(0.0, args.junction_v - 0.01),
                               args.mark_height, args.camera_height)[0]
        print(f"  read one notch of the scale out (+-0.01) and that becomes "
              f"{min(near, far):.2f}-{max(near, far):.2f} m")
        if pinned:
            print("  no distance puts the floor line and the mark where they "
                  "are: the two heights")
            print("  cannot both be right, or the row read is not the "
                  "wall-floor junction.")
            return 2
        if not (0.5 < args.distance < MAX_WALL_M):
            print("  which is not a believable distance for a room: check "
                  "the row you read, and")
            print("  the heights. The floor line sits LOW in the frame, "
                  "usually past v=0.9; a row")
            print("  read too high puts the wall metres away and the tilt "
                  "degrees out.")
            return 2

    if (abs(args.distance - 2.0) < 1e-9
            and abs(args.mark_height - 0.70) < 1e-9
            and abs(args.camera_height - 0.12) < 1e-9):
        print("  those are the three numbers out of the documentation, not "
              "measurements of anything.")
        print("  Measure your own. The tilt is only as good as they are, "
              "and at 2.5 m one")
        print("  degree of it is a quarter of a metre of range.")
        return 2
    rise = args.mark_height - args.camera_height
    tilt = math.degrees(math.atan2(rise, args.distance))
    print(f"  wall at {args.distance:.3f} m, line at "
          f"{args.mark_height:.3f} m, lens at {args.camera_height:.3f} m")
    print(f"  the camera's horizon falls {rise:+.3f} m above the lens at "
          f"{args.distance:.3f} m")
    print(f"  camera tilt     {tilt:+.2f} deg above the airframe's forward "
          f"axis")
    if tilt < -2.0:
        print("  that is nose-DOWN, which is unusual for an FPV mount; "
              "check which height is which.")
    if abs(tilt) > 45.0:
        print("  over 45 deg: check the distance and the heights.")
        return 2
    # What it changes, in the terms it will be used in.
    for depression in (10.0, 15.0, 20.0):
        a_cam = -20.0
        intr.tilt_up_deg = 0.0
        intr.tilt_up_deg = tilt
        b, _ = intr._rotate(a_cam, -depression)
        print(f"    an object {depression:.0f} deg below the axis at "
              f"{a_cam:+.0f} deg now reads {b:+.1f} deg")
    intr.tilt_up_deg = round(tilt, 2)
    _merge_save(args.out, intrinsics=intr)
    print("")
    print(f"  written to {args.out}. Bearings from now on are in the "
          f"airframe's frame.")
    print("  Re-run `verify` if you want the field of view printed again; "
          "the lens model itself is unchanged.")
    return 0


# ─────────────────────────── boresight ───────────────────────────

def run_boresight(args: argparse.Namespace) -> int:
    cv2 = require_cv2()
    if not have_gui(cv2) or args.no_gui:
        print("  boresight needs the preview window to mark the target, and "
              "no GUI is available.")
        print("  Install the non-headless build:  pip install "
              "opencv-python")
        return 2

    master = None
    if args.connection:
        import os
        os.environ.setdefault("MAVLINK20", "1")
        from pymavlink import mavutil
        print(f"  opening {args.connection} for the forward rangefinder...")
        master = mavutil.mavlink_connection(args.connection,
                                            source_system=254)
        if master.wait_heartbeat(timeout=30) is None:
            print("  no heartbeat; continuing without live range")
            master = None
        else:
            print("  heartbeat received")
    else:
        print("  no --connection: you can still mark the pixel by hand, but "
              "without the")
        print("  range readings there is nothing to tell you WHERE the beam "
              "lands.")

    cap = open_capture(cv2, args.camera, args.width, args.height)
    hits: List[Tuple[float, float]] = []
    click: List[Tuple[int, int]] = []
    last_range: Optional[float] = None
    background: Optional[float] = None

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            click.append((x, y))

    cv2.namedWindow("boresight")
    cv2.setMouseCallback("boresight", on_mouse)

    print("  Move a small target slowly across the frame.")
    print("    b      record the current reading as BACKGROUND (empty frame)")
    print("    click  mark the target when the reading JUMPS to its range")
    print("    u      undo the last mark")
    print("    s      save    q  quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("  capture stopped")
            break
        h, w = frame.shape[:2]

        if master is not None:
            while True:
                m = master.recv_match(type="DISTANCE_SENSOR", blocking=False)
                if m is None:
                    break
                if getattr(m, "orientation", None) == args.orientation:
                    last_range = m.current_distance / 100.0

        while click:
            x, y = click.pop()
            hits.append((x / w, y / h))
            print(f"    marked ({x/w:.3f}, {y/h:.3f})"
                  f"{f' at {last_range:.2f} m' if last_range else ''}"
                  f"  [{len(hits)}]")

        view = frame.copy()
        txt = (f"range {last_range:.2f} m" if last_range is not None
               else "range n/a")
        if background is not None:
            txt += f"   bg {background:.2f} m"
        cv2.putText(view, f"{txt}   marks {len(hits)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        for u, v in hits:
            cv2.circle(view, (int(u * w), int(v * h)), 6, (0, 255, 0), 2)
        if hits:
            mu = st.median(u for u, _ in hits)
            mv = st.median(v for _, v in hits)
            cv2.drawMarker(view, (int(mu * w), int(mv * h)), (0, 0, 255),
                           cv2.MARKER_CROSS, 28, 2)
        cv2.imshow("boresight", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("b"):
            background = last_range
            print(f"    background = {background}")
        elif key == ord("u") and hits:
            hits.pop()
            print(f"    undone [{len(hits)}]")
        elif key == ord("s"):
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return 1

    cap.release()
    cv2.destroyAllWindows()

    if len(hits) < 3:
        print(f"  only {len(hits)} marks — need at least 3 for a median")
        return 1
    u = st.median(x for x, _ in hits)
    v = st.median(y for _, y in hits)
    spread = max(((x - u) ** 2 + (y - v) ** 2) ** 0.5 for x, y in hits)
    bs = Boresight(u=round(u, 4), v=round(v, 4), beam_deg=args.beam_deg,
                   calibrated=True)
    print(f"\n  boresight at ({u:.4f}, {v:.4f}) of the frame")
    print(f"  offset from image centre ({u - 0.5:+.4f}, {v - 0.5:+.4f})")
    print(f"  mark spread {spread:.4f} normalised "
          f"{'(tight)' if spread < 0.05 else '(loose — add marks)'}")
    print(f"  beam {args.beam_deg:g} deg")
    _merge_save(args.out, boresight=bs)
    print(f"-> {args.out}")
    return 0


def _merge_save(path: str, intrinsics: Optional[CameraIntrinsics] = None,
                boresight: Optional[Boresight] = None) -> None:
    """Keep whichever half of the file we are not writing."""
    data: dict = {}
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            data = {}
    if intrinsics is not None:
        data.update(intrinsics.to_dict())
    if boresight is not None:
        data["boresight"] = {"u": boresight.u, "v": boresight.v,
                             "beam_deg": boresight.beam_deg,
                             "calibrated": True}
    p.write_text(json.dumps(data, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode",
                    choices=["devices", "modes", "probe", "verify", "tilt",
                             "intrinsics", "boresight"])
    ap.add_argument("--camera", default="0")
    ap.add_argument("--list", action="store_true",
                    help="with probe: scan every index instead of opening one")
    ap.add_argument("--list-max", type=int, default=5,
                    help="highest index the devices scan tries")
    ap.add_argument("--out", default="camera.json")
    ap.add_argument("--distance", type=float,
                    help="tilt: lens to wall, in metres")
    ap.add_argument("--junction-v", type=float,
                    help="tilt: the row on the yellow scale where the wall "
                         "meets the floor. Replaces --distance")
    ap.add_argument("--mark-height", type=float,
                    help="tilt: height on the wall where the drawn line "
                         "falls, in metres")
    ap.add_argument("--camera-height", type=float, default=0.0,
                    help="tilt: height of the lens above the same floor")
    ap.add_argument("--board", default="9x6",
                    help="INNER CORNERS, cols x rows (a 10x7-square board is "
                         "9x6)")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--model", choices=["auto", "pinhole", "fisheye"],
                    default="auto",
                    help="lens model. auto fits both and keeps the better "
                         "reprojection error, rejecting any fit whose "
                         "pixel-to-angle mapping is non-monotonic. An FPV "
                         "lens past ~100 deg needs fisheye.")
    ap.add_argument("--width", type=int, help="request this capture width")
    ap.add_argument("--height", type=int)
    ap.add_argument("--no-equalize", dest="equalize", action="store_false",
                    help="skip CLAHE; it usually helps on a noisy feed")
    ap.add_argument("--no-gui", action="store_true",
                    help="auto-capture with no preview window")

    ap.add_argument("--from-dir", help="calibrate from saved images instead "
                                       "of a live camera")
    ap.add_argument("--save-frames", help="also write each captured view here")
    ap.add_argument("--max-views", type=int, default=20)
    ap.add_argument("--min-move-px", type=float, default=60.0,
                    help="a new view must be this far from every previous one")
    ap.add_argument("--stable-s", type=float, default=0.6,
                    help="board must be still this long before auto-capture")
    ap.add_argument("--search-every", type=int, default=3,
                    help="run the exhaustive search every Nth frame")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--probe-frames", type=int, default=60)
    ap.add_argument("--expect-hfov", type=float,
                    help="verify: your lens's spec HFOV in degrees, "
                         "checked against what the calibration implies")
    ap.add_argument("--keep-outliers", action="store_true",
                    help="intrinsics: do not drop outlier views. An "
                         "interlaced analog feed combs a moving board, and "
                         "one such view corrupts the whole fit, so dropping "
                         "is the default.")
    ap.add_argument("--no-capture", action="store_true",
                    help="verify: skip the live straightness test")

    ap.add_argument("--connection",
                    help="MAVLink connection for the rangefinder")
    ap.add_argument("--orientation", type=int, default=0,
                    help="DISTANCE_SENSOR orientation of the forward sensor")
    ap.add_argument("--beam-deg", type=float, default=2.0)
    ap.add_argument("--version", action="version",
                    version=f"drone_agent {__version__}")
    args = ap.parse_args()

    print(f"drone_agent {__version__}")
    if args.mode == "devices" or (args.mode == "probe" and args.list):
        return run_devices(args)
    if args.mode == "modes":
        return run_modes(args)
    if args.mode == "verify":
        return run_verify(args)
    if args.mode == "probe":
        return run_probe(args)
    if args.mode == "intrinsics":
        return run_intrinsics(args)
    if args.mode == "tilt":
        return run_tilt(args)
    return run_boresight(args)


if __name__ == "__main__":
    sys.exit(main())
