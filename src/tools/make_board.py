#!/usr/bin/env python3
"""
Generate a checkerboard calibration target as a print-ready PDF.

    python tools/make_board.py                      # A2, 10x7 squares @ 55 mm
    python tools/make_board.py --paper a1           # bigger, for more range
    python tools/make_board.py --paper a4 --square-mm 25

Defaults are chosen for a 150 deg FPV lens: at 0.5 m the A2 board spans
about 300 px of a 720-wide capture, which is ~30 px per square and
comfortably detectable. An A4 board has to be held at ~0.25 m to reach the
same, which is close enough that framing the board corners becomes awkward.

The printed square size does NOT affect the calibration. Scaling the object
points scales only the per-view translation; the focal length, principal
point and distortion coefficients come out identical. So a printer that
scales the page by 2% costs nothing here. Measure it anyway and pass
--square-mm to the calibrator, because the number lands in camera.json and
a future metric use of the same board would need it to be right.

What DOES matter is flatness: the solver assumes every corner lies on one
plane. Mount the print on foam board, a clipboard or stiff card. A couple of
millimetres of paper bow at A2 size is a real error term.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAPER_MM = {
    "a0": (841.0, 1189.0),
    "a1": (594.0, 841.0),
    "a2": (420.0, 594.0),
    "a3": (297.0, 420.0),
    "a4": (210.0, 297.0),
    "letter": (215.9, 279.4),
    "tabloid": (279.4, 431.8),
}

# Square size per paper, picked so a 10x7-square board (9x6 inner corners)
# leaves a white margin on all four sides. findChessboardCornersSB wants that
# quiet border; a board bled to the page edge detects far less reliably.
DEFAULT_SQUARE_MM = {
    "a0": 110.0, "a1": 78.0, "a2": 55.0, "a3": 38.0,
    "a4": 27.0, "letter": 27.0, "tabloid": 38.0,
}


def build(path: Path, paper: str, cols: int, rows: int, square_mm: float,
          landscape: bool) -> dict:
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    pw, ph = PAPER_MM[paper]
    if landscape:
        pw, ph = ph, pw

    board_w = cols * square_mm
    board_h = rows * square_mm
    if board_w > pw or board_h > ph:
        raise SystemExit(
            f"a {cols}x{rows} board of {square_mm:g} mm squares is "
            f"{board_w:g}x{board_h:g} mm, which does not fit "
            f"{paper.upper()} {'landscape' if landscape else 'portrait'} "
            f"({pw:g}x{ph:g} mm). Reduce --square-mm or use larger paper.")

    mx = (pw - board_w) / 2.0
    my = (ph - board_h) / 2.0
    if min(mx, my) < 8.0:
        print(f"  warning: margin is only {min(mx, my):.1f} mm. The detector "
              f"wants a white border;")
        print(f"           10 mm or more is safer.")

    c = canvas.Canvas(str(path), pagesize=(pw * mm, ph * mm))
    c.setTitle(f"calibration board {cols}x{rows} squares, {square_mm:g} mm")

    c.setFillColorRGB(0, 0, 0)
    for r in range(rows):
        for col in range(cols):
            # (0,0) black puts a black square in the bottom-left; the parity
            # is irrelevant to the detector, only the inner-corner count is.
            if (r + col) % 2 == 0:
                x = mx + col * square_mm
                y = my + r * square_mm
                c.rect(x * mm, y * mm, square_mm * mm, square_mm * mm,
                       stroke=0, fill=1)

    # Printed on the page so the board can never be separated from the two
    # numbers the calibrator needs. --board is inner corners, and getting it
    # wrong means nothing is ever detected with no other symptom.
    label = (f"{cols}x{rows} squares  |  {square_mm:g} mm  |  "
             f"calibrate with:  --board {cols-1}x{rows-1} "
             f"--square-mm {square_mm:g}")
    c.setFont("Helvetica", 8)
    if my >= 12.0:
        c.drawCentredString(pw * mm / 2.0, 4 * mm, label)
    c.showPage()
    c.save()
    return {"paper": paper, "page_mm": (pw, ph), "board_mm": (board_w, board_h),
            "margin_mm": (mx, my), "inner": (cols - 1, rows - 1)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", default="a2",
                    choices=sorted(PAPER_MM), help="page size")
    ap.add_argument("--squares", default="10x7",
                    help="SQUARES, cols x rows (inner corners are one less "
                         "each way)")
    ap.add_argument("--square-mm", type=float,
                    help="square edge in mm (default: fills the chosen paper)")
    ap.add_argument("--portrait", action="store_true",
                    help="portrait instead of landscape")
    ap.add_argument("--out", default="calibration_board.pdf")
    args = ap.parse_args()

    try:
        cols, rows = (int(v) for v in args.squares.lower().split("x"))
    except ValueError:
        raise SystemExit("--squares must look like 10x7")
    if cols < 3 or rows < 3:
        raise SystemExit("a board needs at least 3x3 squares")
    if cols == rows:
        print("  note: a square board is rotationally ambiguous. An "
              "asymmetric one (10x7)")
        print("        lets the detector fix the orientation.")

    square = args.square_mm or DEFAULT_SQUARE_MM.get(args.paper, 25.0)
    info = build(Path(args.out), args.paper, cols, rows, square,
                 landscape=not args.portrait)

    bw, bh = info["board_mm"]
    mx, my = info["margin_mm"]
    ic, ir = info["inner"]
    print(f"\n  wrote          {args.out}")
    print(f"  paper          {args.paper.upper()} "
          f"{'portrait' if args.portrait else 'landscape'} "
          f"{info['page_mm'][0]:g}x{info['page_mm'][1]:g} mm")
    print(f"  board          {cols}x{rows} squares of {square:g} mm "
          f"= {bw:g}x{bh:g} mm")
    print(f"  margins        {mx:.1f} mm x {my:.1f} mm")
    print(f"\n  Print at 100% / 'actual size' — NOT 'fit to page'.")
    print(f"  Mount it flat and rigid, matte paper, no lamination.")
    print(f"\n  Then calibrate with:")
    print(f"    python tools/calibrate_camera.py probe --camera 1 "
          f"--board {ic}x{ir}")
    print(f"    python tools/calibrate_camera.py intrinsics --camera 1 "
          f"--board {ic}x{ir} --square-mm {square:g} --out camera.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
