"""
seq_extract_to_trajectory.py
============================
Bridge: seq_extract .npz output → calligraphy_sim Viser trajectory format.

seq_extract saves strokes_raw_out with shape (N, 7):
  col 0: flag  — 1 = pen-up / end of stroke, 0 = pen-down
  col 1-2: (x1, y1) — stroke start, normalised [0, 1] on image canvas
  col 3-4: (x2, y2) — stroke end, normalised [0, 1]
  col 5:   r2       — end width (normalised)
  col 6:   s2       — stroke intensity / darkness [0, 1]

calligraphy_sim expects a list of strokes; each stroke is a list of
  (world_x, world_y, world_z, pressure) tuples where:
  world_x  — robot forward reach  [0.37, 0.73] m (paper centre 0.55)
  world_y  — robot lateral        [-0.18, 0.18] m
  world_z  — 0.025 = pen-up safe, -0.003 = contact, -0.006 = heavy press
  pressure — [0, 1.4+], scaled from s2
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


# Paper extent in robot world frame (must match calligraphy_sim.py)
PAPER_X_CENTER = 0.55
PAPER_Y_CENTER = 0.0
PAPER_HALF = 0.18   # half-width of the written area (not the full paper)

Z_PEN_UP   =  0.025
Z_CONTACT  = -0.003
Z_HEAVY    = -0.006

PRESSURE_SCALE = 1.4   # max pressure mapped from s2=1.0


def _img_to_world(nx: float, ny: float) -> tuple[float, float]:
    """Map normalised image coords [0,1]² → robot world (x, y)."""
    # image x (col) → robot world_x (forward reach): left=small x, right=large x
    world_x = PAPER_X_CENTER + (nx - 0.5) * 2.0 * PAPER_HALF
    # image y (row) → robot world_y: top=negative y, bottom=positive y
    world_y = PAPER_Y_CENTER + (ny - 0.5) * 2.0 * PAPER_HALF
    return float(np.clip(world_x, 0.37, 0.73)), float(np.clip(world_y, -0.18, 0.18))


def _norm_to_world(nx: float, ny: float) -> tuple[float, float]:
    """Map seq_extract normalised coords [-1,1]² → robot world (x, y).

    seq_extract uses a [-1,1] canvas where (0,0) is centre.
    We map that linearly onto PAPER_HALF around PAPER_{X,Y}_CENTER.
    """
    world_x = PAPER_X_CENTER + nx * PAPER_HALF
    world_y = PAPER_Y_CENTER + ny * PAPER_HALF
    return float(np.clip(world_x, 0.37, 0.73)), float(np.clip(world_y, -0.18, 0.18))


def npz_to_trajectory(npz_path: str | Path, samples: int = 8) -> list[list[tuple]]:
    """
    Load a seq_extract .npz file and return a Viser trajectory.

    Parameters
    ----------
    npz_path : path to the .npz produced by seq_extract save_seq_data()
    samples  : number of intermediate points to interpolate per stroke segment

    Returns
    -------
    List of strokes; each stroke is a list of (world_x, world_y, z, pressure).
    """
    data = np.load(npz_path, allow_pickle=True)
    strokes_raw: np.ndarray = data["strokes_data"]   # (N, 7)
    # col 0: flag (1=pen-up, 0=pen-down)
    # col 1-2: (x1,y1) stroke start in [-1,1]
    # col 3-4: (x2,y2) stroke end   in [-1,1]
    # col 5:   r2  end width
    # col 6:   s2  intensity/pressure

    strokes: list[list[tuple]] = []
    current_stroke: list[tuple] = []

    for row in strokes_raw:
        flag = int(row[0])
        x1, y1 = float(row[1]), float(row[2])
        x2, y2 = float(row[3]), float(row[4])
        intensity = float(np.clip(abs(row[6]), 0.0, 1.5))
        pressure = min(intensity, 1.0) * PRESSURE_SCALE

        if flag == 1:
            # pen-up: finalise current stroke
            if current_stroke:
                last = current_stroke[-1]
                current_stroke.append((last[0], last[1], Z_PEN_UP, 0.0))
                strokes.append(current_stroke)
                current_stroke = []
            continue

        wx1, wy1 = _norm_to_world(x1, y1)
        wx2, wy2 = _norm_to_world(x2, y2)
        z = Z_HEAVY if pressure > 0.9 * PRESSURE_SCALE else Z_CONTACT

        if not current_stroke:
            current_stroke.append((wx1, wy1, Z_PEN_UP, 0.0))

        for i in range(samples + 1):
            t = i / samples
            wx = wx1 + (wx2 - wx1) * t
            wy = wy1 + (wy2 - wy1) * t
            current_stroke.append((wx, wy, z, pressure))

    if current_stroke:
        last = current_stroke[-1]
        current_stroke.append((last[0], last[1], Z_PEN_UP, 0.0))
        strokes.append(current_stroke)

    return strokes


def load_seq_extract_trajectory(npz_path: str | Path) -> list[list[tuple]]:
    """Public entry point used by calligraphy_sim.py."""
    return npz_to_trajectory(npz_path)


# ---------------------------------------------------------------------------
# CLI: preview the trajectory as ASCII art
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(description="Convert seq_extract .npz → Viser trajectory and preview")
    ap.add_argument("npz", help="Path to seq_extract output .npz file")
    ap.add_argument("--samples", type=int, default=8, help="Interpolation samples per segment (default 8)")
    args = ap.parse_args()

    path = Path(args.npz)
    if not path.exists():
        print(f"Error: {path} not found.", file=sys.stderr)
        sys.exit(1)

    traj = npz_to_trajectory(path, samples=args.samples)
    print(f"Loaded: {len(traj)} strokes")
    for i, stroke in enumerate(traj):
        pts_down = [(x, y) for x, y, z, _ in stroke if z < 0]
        if pts_down:
            xs = [p[0] for p in pts_down]
            ys = [p[1] for p in pts_down]
            print(f"  stroke {i+1:2d}: {len(stroke):3d} pts | "
                  f"x [{min(xs):.3f}, {max(xs):.3f}]  y [{min(ys):.3f}, {max(ys):.3f}]")

    # ASCII art preview (40×20 grid)
    W, H = 40, 20
    grid = [["·"] * W for _ in range(H)]
    for stroke in traj:
        for x, y, z, _ in stroke:
            if z < 0:
                col = int((x - 0.37) / (0.73 - 0.37) * (W - 1))
                row = int((y + 0.18) / 0.36 * (H - 1))
                col = max(0, min(W - 1, col))
                row = max(0, min(H - 1, row))
                grid[row][col] = "█"

    print("\nASCII preview (left=small x, top=negative y):")
    print("+" + "-" * W + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * W + "+")
