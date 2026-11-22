"""
Stage 3 (100pts): seq_extract .npz -> Viser calligraphy trajectory adapter.

Input:  .npz from CalliRewrite seq_extract/test.py
        strokes_data: (N, 7) float  [flag, x1, y1, x2, y2, r2, s2]
          flag=0: draw stroke, flag=1: pen-up / end
          (x1,y1) cursor start in [0,1], (x2,y2) relative endpoint, r2/s2: width/pressure params
        init_cursors: (K, 2)  initial cursor positions for K sub-sequences

Output: list of strokes compatible with CALLIGRAPHY_TRAJECTORIES dict:
        each stroke = list of (x, y, z, pressure) waypoints in robot world frame

Robot world frame assumptions (matching calligraphy_panda_viser_sim.py):
  x  = 0.42 .. 0.57  (reach, forward)
  y  = -0.09 .. 0.09 (lateral, left-right)
  z  = 0.025          pen-up safe height
  z  = -0.004 .. -0.006  contact (negative = below paper surface)
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Robot workspace mapping constants (calibrated to match viser scene)
# ---------------------------------------------------------------------------
PAPER_X_CENTER = 0.50      # robot x at paper centre
PAPER_Y_CENTER = 0.00      # robot y at paper centre
PAPER_X_HALF   = 0.065     # half-extent along x (reach axis)
PAPER_Y_HALF   = 0.085     # half-extent along y (lateral axis)

Z_SAFE   =  0.025          # pen-up height
Z_LIGHT  = -0.002          # light contact (thin stroke)
Z_HEAVY  = -0.006          # heavy contact (thick stroke)

# Pressure clamping
P_MIN    =  0.3
P_MAX    =  1.4


def _norm_to_robot(nx: float, ny: float) -> tuple[float, float]:
    """Map [0,1] normalised canvas coords to robot (x, y) in metres.

    Canvas convention from seq_extract: origin top-left, y down.
    Robot convention: x increases away from robot base (forward),
                      y increases to the left.
    We map: canvas_x (left->right) -> robot_y (left->right)
            canvas_y (top->bottom) -> robot_x (far->near, i.e. inverted)
    """
    robot_x = PAPER_X_CENTER + PAPER_X_HALF * (1.0 - 2.0 * ny)  # top=far, bottom=near
    robot_y = PAPER_Y_CENTER + PAPER_Y_HALF * (2.0 * nx - 1.0)  # left=-y, right=+y
    return float(robot_x), float(robot_y)


def _pressure_from_width(r2: float, s2: float, idx: int, total: int) -> tuple[float, float]:
    """Derive z-contact depth and pressure from stroke width/pressure params.

    r2: relative width parameter  [0, ~1]
    s2: stroke pressure/speed param [0, ~1]
    idx/total: position within stroke for start/end taper
    """
    # Taper: heavier at start, lighter mid, heavier at finish (calligraphy convention)
    taper = 1.0
    if total > 2:
        rel = idx / (total - 1)
        if rel < 0.15:
            taper = 0.6 + rel / 0.15 * 0.7   # ramp up
        elif rel > 0.85:
            taper = 0.6 + (1.0 - rel) / 0.15 * 0.7  # ramp down

    pressure = float(np.clip(0.5 + r2 * 0.8 + s2 * 0.3, P_MIN, P_MAX)) * taper
    z = float(np.clip(Z_LIGHT + (Z_HEAVY - Z_LIGHT) * r2, Z_HEAVY, Z_LIGHT))
    return z, float(np.clip(pressure, P_MIN, P_MAX))


def load_npz(npz_path: str | Path) -> dict:
    data = np.load(str(npz_path), allow_pickle=True)
    return {
        "strokes_data":  data["strokes_data"],   # (N, 7)
        "init_cursors":  data["init_cursors"],    # (K, 2)
        "image_size":    int(data["image_size"]),
        "round_length":  data["round_length"],
        "init_width":    float(data["init_width"]),
    }


def npz_to_trajectories(npz_path: str | Path, n_interp: int = 6) -> list[list[tuple]]:
    """Convert a seq_extract .npz into a list of stroke waypoint lists.

    Each stroke is a list of (robot_x, robot_y, z, pressure) tuples.
    Pen-up transitions are NOT included as strokes; they are implicit
    between consecutive strokes (the Viser player already handles this
    by interpolating from the last point of stroke N to the first of N+1).

    n_interp: number of interpolated intermediate waypoints per segment.
    """
    npz = load_npz(npz_path)
    strokes_raw = npz["strokes_data"]   # (N, 7): flag x1 y1 x2 y2 r2 s2
    init_cursors = npz["init_cursors"]  # (K, 2)

    trajectories: list[list[tuple]] = []
    current_stroke: list[tuple] = []

    # Reconstruct absolute cursor position from relative deltas
    # init_cursors gives starting position; subsequent points are cumulative deltas
    K = len(init_cursors)
    cursor = np.array(init_cursors[0], dtype=np.float64)  # start with first cursor

    for i, row in enumerate(strokes_raw):
        flag = int(row[0])
        x1, y1 = float(row[1]), float(row[2])
        x2, y2 = float(row[3]), float(row[4])
        r2,  s2 = float(row[5]), float(row[6])

        if flag == 1:
            # Pen-up: save current stroke, start new one
            if len(current_stroke) >= 2:
                trajectories.append(current_stroke)
            current_stroke = []
            # Update cursor to absolute position (x1,y1 is new start after pen-up)
            cursor = np.array([x1, y1])
            continue

        # flag == 0: draw
        start = cursor.copy()
        end   = np.array([x1 + x2, y1 + y2])  # x1,y1=current; x2,y2=delta endpoint
        # Clamp to canvas
        end = np.clip(end, 0.0, 1.0)

        # Add start point (pen-down) if stroke is empty
        if len(current_stroke) == 0:
            rx, ry = _norm_to_robot(start[0], start[1])
            current_stroke.append((rx, ry, Z_SAFE, 0.0))   # approach
            z, p = _pressure_from_width(r2, s2, 0, n_interp + 2)
            current_stroke.append((rx, ry, z, p))           # contact

        # Interpolate intermediate waypoints along the segment
        for k in range(1, n_interp + 1):
            t = k / (n_interp + 1)
            pt = start + t * (end - start)
            rx, ry = _norm_to_robot(float(pt[0]), float(pt[1]))
            z, p = _pressure_from_width(r2, s2, k, n_interp + 2)
            current_stroke.append((rx, ry, z, p))

        cursor = end

    # Flush last stroke
    if len(current_stroke) >= 2:
        # Add pen-up at end
        last = current_stroke[-1]
        current_stroke.append((last[0], last[1], Z_SAFE, 0.0))
        trajectories.append(current_stroke)

    return trajectories


def trajectories_to_dict_entry(char_name: str, npz_path: str | Path) -> str:
    """Generate a Python source snippet to paste into CALLIGRAPHY_TRAJECTORIES."""
    trajs = npz_to_trajectories(npz_path)
    lines = [f'    # {char_name} (auto-generated from seq_extract)\n']
    lines.append(f'    "{char_name}": [\n')
    for stroke in trajs:
        lines.append('        [\n')
        for pt in stroke:
            lines.append(f'            ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}, {pt[3]:.2f}),\n')
        lines.append('        ],\n')
    lines.append('    ],\n')
    return ''.join(lines)


# ---------------------------------------------------------------------------
# CLI: python seq_to_trajectory.py <npz_file> [char_name]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python seq_to_trajectory.py <seq_data.npz> [char_name]")
        print("       Prints trajectory waypoints as JSON or as Python source.")
        sys.exit(1)

    npz_path = sys.argv[1]
    char_name = sys.argv[2] if len(sys.argv) > 2 else Path(npz_path).stem

    trajs = npz_to_trajectories(npz_path)
    print(f"# Loaded {len(trajs)} strokes from {npz_path}")
    for i, s in enumerate(trajs):
        print(f"#   stroke {i+1}: {len(s)} waypoints")

    print()
    print("# Python snippet for CALLIGRAPHY_TRAJECTORIES:")
    print(trajectories_to_dict_entry(char_name, npz_path))
