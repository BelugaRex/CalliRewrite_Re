"""
KanjiVG → Robot Calligraphy Trajectory
=======================================
Parses kanjivg.xml and converts any CJK character into a list of stroke
waypoints in robot world-frame coordinates, ready for the Viser simulator.

Robot / canvas coordinate convention (matches XuanPaperCanvas in calligraphy_sim.py):
  world_x  →  canvas pixel x  (LEFT ←→ RIGHT)
  world_y  →  canvas pixel y  (UP   ↕  DOWN,  negative = up)

KanjiVG SVG canvas is 109 × 109 units, origin top-left.
  svg_x  0 → 109  maps to  world_x  PAPER_LEFT → PAPER_RIGHT
  svg_y  0 → 109  maps to  world_y  PAPER_TOP  → PAPER_BOTTOM

Usage
-----
  # Python API
  from kanjivg_to_trajectory import KanjiVGParser
  parser = KanjiVGParser("data/kanjivg.xml")
  strokes = parser.get_trajectory("永")   # list of stroke waypoint lists

  # CLI — preview a character and print waypoints
  python kanjivg_to_trajectory.py 永
  python kanjivg_to_trajectory.py 福 --preview   # saves preview PNG
"""

from __future__ import annotations
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Robot workspace constants (keep in sync with calligraphy_sim.py)
# ---------------------------------------------------------------------------
PAPER_X_CENTER = 0.55       # robot x at paper centre
PAPER_Y_CENTER = 0.00       # robot y at paper centre
PAPER_HALF     = 0.075      # half-width of the character bounding box in metres

Z_SAFE  =  0.025            # pen-up clearance height
Z_MED   = -0.003            # normal contact
Z_HEAVY = -0.006            # heavy press (stroke ends, 捺 foot)

# Stroke-type pressure profiles: (start, mid, end) pressure values
# Keyed on the KanjiVG ns0:type attribute (SVG stroke classifier)
_STROKE_PRESSURE: dict[str, tuple[float, float, float]] = {
    "㇐": (0.8, 0.7, 1.1),   # 横 heng  – horizontal
    "㇑": (1.1, 0.8, 1.2),   # 竖 shu   – vertical
    "㇔": (1.0, 1.1, 0.5),   # 点 dian  – dot
    "㇒": (1.0, 0.7, 0.3),   # 撇 pie   – left-falling
    "㇏": (0.7, 1.1, 1.4),   # 捺 na    – right-falling (heavy foot)
    "㇆": (1.1, 0.9, 1.3),   # 横折钩
    "㇚": (1.1, 0.9, 1.3),   # 竖钩
    "㇇": (1.0, 0.8, 0.3),   # 横撇
}
_DEFAULT_PRESSURE = (0.9, 0.8, 0.9)

KVG_NS = "http://kanjivg.tagaini.net"
SVG_SIZE = 109.0            # KanjiVG canvas size in SVG units


# ---------------------------------------------------------------------------
# Minimal SVG path parser (M, L, C, Q, Z commands only)
# ---------------------------------------------------------------------------

def _parse_svg_path(d: str, n_interp: int = 8) -> list[tuple[float, float]]:
    """Convert an SVG path string to a list of (x, y) points in SVG units."""
    # Split into (command-letter, [numbers...]) groups, respecting implicit repetition
    token_re = re.compile(r"([MLCQZmlcqz])|"
                          r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
    raw = [(m.group(1), m.group(2)) for m in token_re.finditer(d)]

    points: list[tuple[float, float]] = []
    cx, cy = 0.0, 0.0
    cmd = "M"
    nums: list[float] = []
    idx = 0

    while idx < len(raw):
        letter, num = raw[idx]
        if letter:
            cmd = letter
            idx += 1
            nums = []
            continue
        # accumulate numbers until we have enough for one implicit repetition
        nums.append(float(num))
        idx += 1

        arg_counts = {"M": 2, "m": 2, "L": 2, "l": 2,
                      "C": 6, "c": 6, "Q": 4, "q": 4, "Z": 0, "z": 0}
        need = arg_counts.get(cmd, 0)
        if need == 0 or len(nums) < need:
            continue

        args, nums = nums[:need], nums[need:]

        if cmd in ("M", "m"):
            x, y = args
            if cmd == "m":
                x += cx; y += cy
            points.append((x, y))
            cx, cy = x, y
            cmd = "L" if cmd == "M" else "l"

        elif cmd in ("L", "l"):
            x, y = args
            if cmd == "l":
                x += cx; y += cy
            points.append((x, y))
            cx, cy = x, y

        elif cmd in ("C", "c"):
            x1, y1, x2, y2, x, y = args
            if cmd == "c":
                x1 += cx; y1 += cy
                x2 += cx; y2 += cy
                x  += cx; y  += cy
            P0 = np.array([cx, cy])
            P1 = np.array([x1, y1])
            P2 = np.array([x2, y2])
            P3 = np.array([x, y])
            for k in range(1, n_interp + 1):
                t = k / (n_interp + 1)
                pt = ((1-t)**3*P0 + 3*(1-t)**2*t*P1
                      + 3*(1-t)*t**2*P2 + t**3*P3)
                points.append((float(pt[0]), float(pt[1])))
            points.append((x, y))
            cx, cy = x, y

        elif cmd in ("Q", "q"):
            x1, y1, x, y = args
            if cmd == "q":
                x1 += cx; y1 += cy
                x  += cx; y  += cy
            P0 = np.array([cx, cy])
            P1 = np.array([x1, y1])
            P2 = np.array([x, y])
            for k in range(1, n_interp + 1):
                t = k / (n_interp + 1)
                pt = (1-t)**2*P0 + 2*(1-t)*t*P1 + t**2*P2
                points.append((float(pt[0]), float(pt[1])))
            points.append((x, y))
            cx, cy = x, y

    return points


# ---------------------------------------------------------------------------
# Coordinate transform: SVG units → robot world frame
# ---------------------------------------------------------------------------

def _svg_to_world(sx: float, sy: float) -> tuple[float, float]:
    """Map KanjiVG SVG (0–109) to robot world (x, y) in metres."""
    # Normalise to [0, 1] then centre on paper
    nx = sx / SVG_SIZE
    ny = sy / SVG_SIZE
    wx = PAPER_X_CENTER + PAPER_HALF * (2.0 * nx - 1.0)   # left→right
    wy = PAPER_Y_CENTER + PAPER_HALF * (2.0 * ny - 1.0)   # top→bottom
    return round(wx, 4), round(wy, 4)


# ---------------------------------------------------------------------------
# Pressure profile along a stroke
# ---------------------------------------------------------------------------

def _pressure_z(idx: int, total: int, profile: tuple[float, float, float]
                ) -> tuple[float, float]:
    """Return (z, pressure) for waypoint idx out of total."""
    p_start, p_mid, p_end = profile
    if total <= 1:
        return Z_MED, p_mid
    t = idx / (total - 1)
    if t < 0.2:
        p = p_start + (p_mid - p_start) * (t / 0.2)
    elif t > 0.8:
        p = p_mid + (p_end - p_mid) * ((t - 0.8) / 0.2)
    else:
        p = p_mid
    # Heavy press → deeper contact
    z = Z_HEAVY if p >= 1.3 else Z_MED
    return z, round(float(p), 2)


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class KanjiVGParser:
    def __init__(self, xml_path: str | Path = "data/kanjivg.xml"):
        self._tree = ET.parse(str(xml_path))
        self._root = self._tree.getroot()

    def _find_kanji(self, char: str) -> ET.Element | None:
        code = f"{ord(char):05x}"
        for kanji in self._root:
            kid = kanji.get("id", "")
            # Match exact id like "kvg:kanji_06c38" (no variant suffix)
            if kid == f"kvg:kanji_{code}":
                return kanji
        # Fallback: accept first variant
        for kanji in self._root:
            if f"kvg:kanji_{code}" in kanji.get("id", ""):
                return kanji
        return None

    def _collect_paths(self, element: ET.Element) -> list[tuple[str, str]]:
        """Return list of (stroke_type, path_d) for all <path> descendants."""
        results = []
        for path in element.iter("path"):
            d = path.get("d", "")
            stype = path.get(f"{{{KVG_NS}}}type", "")
            if d:
                results.append((stype, d))
        return results

    def get_trajectory(self, char: str, n_interp: int = 6,
                       ) -> list[list[tuple[float, float, float, float]]]:
        """
        Return trajectory for *char* as a list of strokes.
        Each stroke = list of (world_x, world_y, z, pressure) waypoints.
        The first and last waypoint of every stroke have z=Z_SAFE, pressure=0
        (pen-up approach / lift-off).
        """
        kanji = self._find_kanji(char)
        if kanji is None:
            raise ValueError(f"Character '{char}' (U+{ord(char):04X}) not found in KanjiVG.")

        stroke_paths = self._collect_paths(kanji)
        trajectory: list[list[tuple]] = []

        for stype, d in stroke_paths:
            svg_pts = _parse_svg_path(d, n_interp=n_interp)
            if len(svg_pts) < 2:
                continue

            # Pick pressure profile from stroke type (use first classifier token)
            type_key = stype.split("/")[0] if stype else ""
            profile = _STROKE_PRESSURE.get(type_key, _DEFAULT_PRESSURE)

            stroke: list[tuple] = []
            # Pen-down approach at first contact point
            wx0, wy0 = _svg_to_world(*svg_pts[0])
            stroke.append((wx0, wy0, Z_SAFE, 0.0))   # approach

            for k, (sx, sy) in enumerate(svg_pts):
                wx, wy = _svg_to_world(sx, sy)
                z, p = _pressure_z(k, len(svg_pts), profile)
                stroke.append((wx, wy, z, p))

            # Pen-up lift-off at last point
            wx_last, wy_last = _svg_to_world(*svg_pts[-1])
            stroke.append((wx_last, wy_last, Z_SAFE, 0.0))

            trajectory.append(stroke)

        return trajectory

    def available(self, char: str) -> bool:
        return self._find_kanji(char) is not None


# ---------------------------------------------------------------------------
# Offline preview: render strokes onto a canvas PNG
# ---------------------------------------------------------------------------

def preview(char: str, xml_path: str, out_png: str | None = None) -> None:
    from PIL import Image, ImageDraw

    parser = KanjiVGParser(xml_path)
    trajectory = parser.get_trajectory(char)

    RES = 512
    # Canvas: world x maps to image x, world y maps to image y
    x_min = PAPER_X_CENTER - PAPER_HALF
    y_min = PAPER_Y_CENTER - PAPER_HALF
    span = PAPER_HALF * 2

    img = Image.new("RGB", (RES, RES), (251, 247, 238))
    draw = ImageDraw.Draw(img)
    # guide grid
    m = int(RES * 0.04)
    draw.rectangle([m, m, RES-m, RES-m], outline=(200, 80, 80, 80), width=2)
    draw.line([(RES//2, m), (RES//2, RES-m)], fill=(200, 80, 80), width=1)
    draw.line([(m, RES//2), (RES-m, RES//2)], fill=(200, 80, 80), width=1)

    colors = [(20, 20, 20), (40, 40, 180), (180, 40, 40), (40, 140, 40),
              (140, 40, 140), (40, 140, 140), (140, 100, 40), (80, 80, 80)]

    for s_idx, stroke in enumerate(trajectory):
        col = colors[s_idx % len(colors)]
        prev_px = None
        for pt in stroke:
            wx, wy, z, p = pt
            if z >= 0:          # pen-up, skip drawing
                prev_px = None
                continue
            px = int((wx - x_min) / span * RES)
            py = int((wy - y_min) / span * RES)
            r = max(2, int(2 + p * 3))
            draw.ellipse([px-r, py-r, px+r, py+r], fill=col)
            if prev_px:
                draw.line([prev_px, (px, py)], fill=col, width=max(1, r-1))
            prev_px = (px, py)

    out = out_png or f"/tmp/preview_{char}.png"
    img.save(out)
    print(f"Saved preview → {out}  ({len(trajectory)} strokes)")
    for i, s in enumerate(trajectory):
        contact = [pt for pt in s if pt[2] < 0]
        print(f"  stroke {i+1}: {len(contact)} contact waypoints")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("char", help="Chinese character to parse, e.g. 永")
    ap.add_argument("--xml", default=str(Path(__file__).parent / "data" / "kanjivg.xml"),
                    help="Path to kanjivg.xml")
    ap.add_argument("--preview", action="store_true", help="Save a preview PNG")
    ap.add_argument("--out", default=None, help="Output PNG path (with --preview)")
    args = ap.parse_args()

    parser = KanjiVGParser(args.xml)
    traj = parser.get_trajectory(args.char)
    print(f"'{args.char}' → {len(traj)} strokes")
    for i, stroke in enumerate(traj):
        contact = [pt for pt in stroke if pt[2] < 0]
        p_vals = [pt[3] for pt in contact]
        print(f"  stroke {i+1}: {len(contact)} pts, "
              f"pressure [{min(p_vals):.2f}, {max(p_vals):.2f}]")

    if args.preview:
        preview(args.char, args.xml, args.out)
