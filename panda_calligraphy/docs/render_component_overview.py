#!/usr/bin/env python3
"""Rebuild the overview from a live Viser capture and base-frame replay data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "calligraphy_panda_viser_sim.py"
BRUSH_SOURCE = HERE.parent / "deformable_brush_physics.py"


def generate_data():
    import numpy as np

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE.parent))
    from calligraphy_panda_viser_sim import (
        CALLIGRAPHY_TRAJECTORIES, PandaKinematics, XuanPaperCanvas,
    )
    from deformable_brush_physics import RealisticDeformableTuft

    kinematics = PandaKinematics()
    brush = RealisticDeformableTuft()
    paper = XuanPaperCanvas()
    q = kinematics.NOMINAL_REST_Q.copy()
    previous_tip = kinematics.forward_kinematics(q)[2]
    best = None
    samples = 0
    contacts = 0
    for stroke in CALLIGRAPHY_TRAJECTORIES["yi"]:
        for p0, p1 in zip(stroke[:-1], stroke[1:]):
            for alpha in np.linspace(0, 1, 29)[1:]:
                blend = alpha * alpha * (3 - 2 * alpha)
                target = np.asarray(p0[:3]) * (1 - blend) + np.asarray(p1[:3]) * blend
                q = kinematics.solve_ik(target, q)
                hand, rotation, tip = kinematics.forward_kinematics(q)
                tuft = brush.compute_deformation(
                    hand + 0.180 * rotation[:, 2], tip, rotation,
                    (tip - previous_tip) / 0.015,
                )
                previous_tip = tip.copy()
                samples += 1
                if tuft["is_contact"]:
                    contacts += 1
                    paper.deposit_ink(*tuft["actual_tip"][:2],
                                      tuft["footprint_radius"], tuft["normal_force"])
                    if best is None or tuft["normal_force"] > best["normal_force"]:
                        best = tuft
    assert best is not None
    raw = np.asarray(paper.image.convert("RGB"))
    horizontal = np.rot90(raw, 1)
    ink = horizontal.max(axis=2) < 80
    rows, columns = np.nonzero(ink)
    assert columns.max() - columns.min() > 3 * (rows.max() - rows.min()), "yi must be horizontal"
    # Verify the exact display transform: raw (+X right, +Y down) becomes
    # displayed (+Y right, +X up). No points or robot configurations are rotated.
    for row, column in [(0, 0), (123, 456), (1023, 1023)]:
        np.testing.assert_array_equal(horizontal[raw.shape[1] - 1 - column, row],
                                      raw[row, column])
    output = HERE / "overview_components.npz"
    np.savez_compressed(output, canvas=raw, horizontal_canvas=horizontal,
                        tuft_vertices=best["vertices"], tuft_faces=best["faces"],
                        tuft_bristles=best["bristles"], actual_tip=best["actual_tip"])
    metadata = {
        "source": "calligraphy_panda_viser_sim.py",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "brush_source_sha256": hashlib.sha256(BRUSH_SOURCE.read_bytes()).hexdigest(),
        "scene_image": "viser_yi_finished.jpg",
        "scene_method": "Existing actual Viser browser capture, 600x600 viewport crop at x=340,y=70",
        "replay": {"samples": samples, "contact_samples": contacts,
                   "samples_per_segment": 28, "velocity_interval_s": 0.015},
        "canvas_display": {"rotation_degrees_counterclockwise": 90,
                           "horizontal_axis": "+Y_Panda_base",
                           "vertical_axis": "+X_Panda_base",
                           "paper_X_bounds_m": [0.37, 0.73],
                           "paper_Y_bounds_m": [-0.18, 0.18],
                           "ink_horizontal_span_px": int(columns.max() - columns.min()),
                           "ink_vertical_span_px": int(rows.max() - rows.min()),
                           "simulation_coordinates_changed": False},
        "contact_detail": {"compression_mm": float(best["penetration"] * 1000),
                           "normal_force_estimate_N": float(best["normal_force"])},
        "note": "The live scene, complete replay canvas and peak-contact detail are separate views, not synchronized frames.",
    }
    (HERE / "overview_components.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def render(data_path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    data = np.load(data_path, allow_pickle=False)
    metadata = json.loads((HERE / "overview_components.json").read_text())
    background = "#f4f6f8"
    ink = "#213142"
    muted = "#596777"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.edgecolor": "#9ba7b3"})
    fig = plt.figure(figsize=(18, 7.6), facecolor=background)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.0, 0.9),
                           left=0.025, right=0.975, top=0.78, bottom=0.22,
                           wspace=0.17)
    fig.text(0.035, 0.945, "Panda calligraphy | component overview", fontsize=25, weight="bold")
    fig.text(0.035, 0.885,
             "Live Viser workspace  /  horizontal yi in Panda-base coordinates  /  computed brush contact",
             fontsize=12.5, color=muted)

    scene_ax = fig.add_subplot(grid[0, 0])
    with Image.open(HERE / "viser_yi_finished.jpg") as screenshot:
        scene = np.asarray(screenshot.convert("RGB"))[70:670, 340:940]
    assert scene.shape == (600, 600, 3)
    scene_ax.imshow(scene)
    scene_ax.set_axis_off()
    scene_ax.set_title("Panda + drawing table", fontsize=15, weight="bold", pad=16)
    scene_ax.text(0.5, -0.075, "Actual Viser capture after drawing yi",
                  transform=scene_ax.transAxes, ha="center", color=muted, fontsize=11)

    canvas_ax = fig.add_subplot(grid[0, 1], facecolor=background)
    canvas_ax.imshow(data["horizontal_canvas"], origin="upper",
                     extent=(-0.18, 0.18, 0.37, 0.73), interpolation="nearest")
    canvas_ax.set_title("Horizontal yi · Panda base frame", fontsize=15, weight="bold", pad=16)
    canvas_ax.set_xlabel(r"$Y_B$ (m)  /  +Y to the right", labelpad=10)
    canvas_ax.set_ylabel(r"$X_B$ (m)", labelpad=8)
    canvas_ax.set_xticks([-0.18, 0.0, 0.18])
    canvas_ax.set_yticks([0.37, 0.55, 0.73])
    canvas_ax.set_aspect("equal")
    canvas_ax.text(0.5, -0.22, "Completed source replay · canvas rotated 90° CCW",
                   transform=canvas_ax.transAxes, ha="center", color=muted, fontsize=10.5)

    tuft_ax = fig.add_subplot(grid[0, 2], projection="3d", facecolor=background)
    tip = data["actual_tip"]
    vertices = (data["tuft_vertices"] - tip) * 1000
    tuft_ax.add_collection3d(Poly3DCollection(vertices[data["tuft_faces"]],
                                            facecolors="#252525", edgecolor="none", shade=True))
    tuft_ax.add_collection3d(Line3DCollection((data["tuft_bristles"] - tip) * 1000,
                                             colors="#795b44", linewidths=0.65))
    tuft_ax.add_collection3d(Poly3DCollection(
        [np.array([[-18, -18, 0], [18, -18, 0], [18, 18, 0], [-18, 18, 0]])],
        facecolor="#e6ddc9", alpha=0.65))
    tuft_ax.set(xlim=(-18, 18), ylim=(-18, 18), zlim=(0, 50),
                xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    tuft_ax.set_xticks([-15, 0, 15]); tuft_ax.set_yticks([-15, 0, 15]); tuft_ax.set_zticks([0, 20, 40])
    tuft_ax.set_box_aspect((1, 1, 1.4))
    tuft_ax.view_init(elev=17, azim=-65)
    tuft_ax.tick_params(labelsize=8)
    for axis in (tuft_ax.xaxis, tuft_ax.yaxis, tuft_ax.zaxis):
        axis.pane.fill = False
        axis._axinfo["grid"]["color"] = (0.75, 0.78, 0.82, 0.35)
    tuft_ax.set_title("Deformable tuft at contact", fontsize=15, weight="bold", pad=16)
    contact = metadata["contact_detail"]
    tuft_ax.text2D(0.5, 0.94,
                    f"Compression {contact['compression_mm']:.1f} mm\n"
                    f"Force estimate {contact['normal_force_estimate_N']:.2f} N",
                    transform=tuft_ax.transAxes, ha="center", fontsize=10.5, color=muted)
    tuft_ax.text2D(0.5, -0.12, "Computed geometry · axes relative to contact tip",
                    transform=tuft_ax.transAxes, ha="center", fontsize=10.5, color=muted)

    fig.text(0.035, 0.055,
             "World = Panda base frame. The 90° canvas rotation changes the presentation only; the trajectory and robot pose are unchanged.",
             fontsize=11, color=muted)
    fig.text(0.035, 0.023,
             "Scene, completed canvas and contact detail show separate moments. Brush force is a model estimate, not a calibrated measurement.",
             fontsize=10.5, color=muted)
    output = HERE / "expected_simulation.png"
    fig.savefig(output, dpi=150, facecolor=background)
    plt.close(fig)
    print("Saved", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-python", default=sys.executable,
                        help="Python with NumPy, Pillow and Matplotlib")
    parser.add_argument("--plot-data", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.plot_data:
        render(args.plot_data)
        return
    data_path = generate_data()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/callirewrite-overview-matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp/callirewrite-overview-cache")
    subprocess.run([args.plot_python, str(Path(__file__).resolve()),
                    "--plot-data", str(data_path)], check=True, env=env)


if __name__ == "__main__":
    main()
