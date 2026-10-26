#!/usr/bin/env python3
"""Render an audited, source-derived diagram of RealisticDeformableTuft.

Example from the repository root (no packages are installed by this script):
  env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
    panda_calligraphy/venv/bin/python \
    panda_calligraphy/docs/render_brush_explainer.py \
    --plot-python /home/mihawk/anaconda3/envs/calli_rl2/bin/python

The current interpreter computes the actual tuft meshes. A separate interpreter
with Matplotlib may draw them, avoiding changes to the viewer's environment.
All measurements are source-model outputs, not calibrated physical measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "deformable_brush_physics.py"


def generate_data(output_dir: Path) -> Path:
    import numpy as np

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE.parent))
    from deformable_brush_physics import RealisticDeformableTuft

    model = RealisticDeformableTuft()
    # Proper rotation: the local +Z tool axis points down towards the paper.
    rotation = np.diag([1.0, -1.0, -1.0])
    tool_dir = rotation[:, 2]
    scenarios = [
        ("lifted", 0.008, np.zeros(3)),
        ("pressed", -0.006, np.zeros(3)),
        ("moving", -0.006, np.array([0.030, 0.0, 0.0])),
    ]
    arrays = {}
    report = {
        "source": "panda_calligraphy/deformable_brush_physics.py",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "model": "RealisticDeformableTuft (instantaneous parametric geometry)",
        "disclaimer": "Normal force is a model estimate, not calibrated measurement. "
                      "Stiffness changes the force estimate, not the mesh or footprint.",
        "clamp_scope": "Loft-ring vertices and bristle points use max(paper_z, z). "
                       "The apex is P3. Actual-tip contact clamping applies only for "
                       "penetration > 1e-4 m; examples avoid the 0–0.1 mm deadband.",
        "parameters": {"length_m": model.length,
                       "root_radius_m": model.root_radius,
                       "stiffness_parameter": model.stiffness,
                       "splay_factor": model.splay_factor,
                       "paper_z_m": model.paper_z},
        "scenarios": {},
    }
    for key, height, velocity in scenarios:
        nominal = np.array([0.0, 0.0, model.paper_z + height])
        root = nominal - model.length * tool_dir
        result = model.compute_deformation(root, nominal, rotation, velocity)
        speed = float(np.linalg.norm(velocity[:2]))
        motion = np.r_[velocity[:2] / speed, 0.0] if speed > 1e-4 else np.zeros(3)
        # Control points reconstructed from the exact branches in the source;
        # mesh vertices and bristle segments below come directly from the method.
        p0 = root
        p1 = root + model.length / 3.0 * tool_dir
        p2 = (result["actual_tip"] - model.length / 3.0 * motion
              if result["is_contact"]
              else root + 2.0 * model.length / 3.0 * tool_dir)
        p3 = result["actual_tip"]
        controls = np.array([p0, p1, p2, p3])
        s = np.linspace(0.0, 1.0, 151)[:, None]
        spine = ((1-s)**3 * p0 + 3*(1-s)**2*s * p1
                 + 3*(1-s)*s**2 * p2 + s**3 * p3)
        delta = float(result["penetration"])
        expected_force = (650.0 * model.stiffness * delta**1.25
                          if result["is_contact"] else 0.0)
        expected_radius = (0.0018 + 0.048 * model.splay_factor * np.sqrt(delta)
                           if result["is_contact"] else 0.0)
        lag = float(np.linalg.norm(result["actual_tip"][:2] - nominal[:2]))
        assert result["vertices"].shape == (141, 3)
        assert result["faces"].shape == (266, 3)
        assert result["bristles"].shape == (108, 2, 3)
        assert result["vertices"][:, 2].min() >= model.paper_z - 1e-8
        assert result["bristles"][..., 2].min() >= model.paper_z - 1e-8
        assert np.isclose(result["normal_force"], expected_force)
        assert np.isclose(result["footprint_radius"], expected_radius)
        assert np.allclose(root + model.length * tool_dir, nominal)
        for name in ("vertices", "faces", "bristles", "actual_tip"):
            arrays[key + "_" + name] = result[name]
        arrays[key + "_nominal"] = nominal
        arrays[key + "_root"] = root
        arrays[key + "_spine"] = spine
        arrays[key + "_controls"] = controls
        report["scenarios"][key] = {
            "nominal_height_above_paper_mm": float(height * 1000.0),
            "horizontal_speed_mm_s": speed * 1000.0,
            "contact": bool(result["is_contact"]),
            "compression_mm": delta * 1000.0,
            "normal_force_estimate_N": float(result["normal_force"]),
            "footprint_radius_mm": float(result["footprint_radius"] * 1000.0),
            "lag_mm": lag * 1000.0,
            "actual_tip_m": result["actual_tip"].tolist(),
            "minimum_mesh_z_m": float(result["vertices"][:, 2].min()),
            "mesh_vertices": len(result["vertices"]),
            "mesh_triangles": len(result["faces"]),
            "bristle_segments": len(result["bristles"]),
        }
    arrays["paper_z"] = np.array(model.paper_z)
    data_path = output_dir / "brush_deformation_data.npz"
    np.savez_compressed(data_path, **arrays)
    (output_dir / "brush_deformation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return data_path


def render(data_path: Path) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

    output_dir = data_path.parent
    data = np.load(data_path, allow_pickle=False)
    report = json.loads((output_dir / "brush_deformation_metrics.json").read_text())
    paper_z = float(data["paper_z"])
    ink = "#213044"
    muted = "#617185"
    teal = "#007F8B"
    hair = "#7C4D28"
    red = "#BA4749"
    orange = "#CE8B22"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(16.5, 11.5), facecolor="#FFFFFF")
    fig.text(0.5, 0.962, "How the brush deforms", ha="center", va="top",
             fontsize=23, fontweight="bold")
    fig.text(0.5, 0.925,
             "Actual RealisticDeformableTuft meshes  ·  45 mm tuft  ·  vertical tool  ·  paper at z = 2 mm",
             ha="center", va="top", fontsize=13.5, color=muted)
    legend = [Line2D([0], [0], color=hair, lw=5, alpha=.55, label="Core mesh + bristles"),
              Line2D([0], [0], color=teal, lw=2.7, label="Cubic Bézier spine"),
              Line2D([0], [0], color=red, marker="D", mfc="white", lw=1.2, ls="--", label="Rigid / nominal tip"),
              Line2D([0], [0], color=orange, lw=2, label="Model footprint extent")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .903),
               ncol=4, frameon=False, fontsize=11.5, handlelength=1.8,
               columnspacing=1.9)

    def side(points):
        p = np.asarray(points)
        return np.stack([p[..., 0]*1000, (p[..., 2]-paper_z)*1000], axis=-1)

    def callout(ax, text, xy, xytext, color=ink, align="left", size=11.5):
        ax.annotate(text, xy=xy, xytext=xytext, fontsize=size, color=color,
                    ha=align, va="center", linespacing=1.25,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": .95,
                                "shrinkA": 4, "shrinkB": 4})

    for index, (key, title) in enumerate([
            ("lifted", "1  Lifted"), ("pressed", "2  Pressed · stationary"),
            ("moving", "3  Moving on paper")]):
        left = .044 + index * .319
        ax = fig.add_axes([left, .345, .283, .462])
        ax.set_xlim(-29, 29)
        ax.set_ylim(-15, 63)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#FFFFFF")
        for side_name in ("top", "right"):
            ax.spines[side_name].set_visible(False)
        ax.spines["left"].set_color("#D9E0E6")
        ax.spines["bottom"].set_color("#D9E0E6")
        ax.set_xticks([-20, 0, 20])
        ax.set_yticks([0, 15, 30, 45, 60])
        ax.tick_params(axis="both", labelsize=10, length=3)
        ax.grid(axis="y", color="#EEF1F4", lw=.7, zorder=0)
        ax.set_xlabel("Horizontal offset (mm)", fontsize=10.5, labelpad=6)
        if index == 0:
            ax.set_ylabel("Height above paper (mm)", fontsize=10.5, labelpad=8)
        else:
            ax.set_yticklabels([])
        middle = left + .283 / 2
        fig.text(middle, .844, title, fontsize=16, ha="center", fontweight="bold")
        metrics = report["scenarios"][key]
        if key == "lifted":
            details = "8 mm air gap  ·  no contact  ·  F = 0"
        else:
            details = (f"δ = 6 mm  ·  F ≈ {metrics['normal_force_estimate_N']:.2f} N"
                       f"  ·  r ≈ {metrics['footprint_radius_mm']:.2f} mm")
        fig.text(middle, .818, details, fontsize=11.5, ha="center", color=muted)
        ax.axhspan(-15, 0, facecolor="#F8F4EB", edgecolor="none", zorder=0)
        ax.axhline(0, color="#9D8E70", lw=1.6, zorder=3)
        if index == 0:
            ax.text(-27.5, -2.3, "Paper surface  zₚ = 2 mm", fontsize=10, color="#7C705B")
        vertices = side(data[key + "_vertices"])
        triangles = vertices[data[key + "_faces"]]
        ax.add_collection(PolyCollection(triangles, facecolors="#B88C5E",
                                         edgecolors="#815A35", linewidths=.20,
                                         alpha=.27, zorder=4))
        bristles = side(data[key + "_bristles"])
        ax.add_collection(LineCollection(bristles, colors=hair,
                                        linewidths=.63, alpha=.58, zorder=5))
        root = side(data[key + "_root"])
        nominal = side(data[key + "_nominal"])
        actual = side(data[key + "_actual_tip"])
        spine = side(data[key + "_spine"])
        controls = side(data[key + "_controls"])
        # The ferrule is a context marker; tuft geometry above is actual output.
        ax.add_patch(Rectangle((root[0]-7, root[1]), 14, 6.2,
                               facecolor="#384252", edgecolor="#253041", zorder=6))
        ax.plot([root[0], nominal[0]], [root[1], nominal[1]],
                 color=red, lw=1.1, ls=(0, (3, 3)), alpha=.65, zorder=2)
        ax.plot(spine[:, 0], spine[:, 1], color=teal, lw=2.6, zorder=8)
        ax.scatter(*root, s=27, color=teal, zorder=9)
        ax.scatter(*actual, s=40, color=teal, edgecolor="white", linewidth=.8, zorder=12)
        ax.scatter(*nominal, s=42, marker="D", facecolor="white", edgecolor=red,
                    linewidth=1.3, zorder=11)
        if metrics["contact"]:
            radius = metrics["footprint_radius_mm"]
            ax.add_patch(Ellipse(actual, 2*radius, 1.5, facecolor="#E9B858",
                                 edgecolor=orange, alpha=.5, linewidth=1.3, zorder=7))
        if key == "lifted":
            callout(ax, "Root / ferrule rim", root, (9, 54))
            callout(ax, "Straight spine", (0, 29), (9, 30), color=teal)
            callout(ax, "Actual = nominal tip", actual, (7, 11), color=teal)
            ax.annotate("", xy=(-13, 8), xytext=(-13, 0),
                        arrowprops={"arrowstyle": "<->", "color": muted, "lw": 1})
            ax.text(-15, 4, "8 mm", va="center", ha="right", fontsize=11, color=muted)
        elif key == "pressed":
            callout(ax, "Root lowered", root, (9, 43))
            callout(ax, "Compressed spine", (0, 20), (8, 25), color=teal)
            callout(ax, "Lower-tuft\nsplay", (-4.8, .2), (-26, 10), color=hair)
            callout(ax, "Actual tip on paper", actual, (7, 6), color=teal, size=10.5)
            callout(ax, "Nominal tip\nbelow paper", nominal, (7, -10), color=red)
        else:
            ax.plot(controls[:, 0], controls[:, 1], color="#8995A1", lw=.9,
                     ls=(0, (2, 3)), alpha=.85, zorder=1)
            ax.scatter(controls[1:3, 0], controls[1:3, 1], s=25, facecolor="white",
                        edgecolor="#7E8A98", linewidth=1, zorder=10)
            callout(ax, "P₀  root", root, (10, 42), size=11)
            callout(ax, "P₁  tool tangent", controls[1], (8, 24), size=10.5)
            callout(ax, "P₂", controls[2], (-27, 9), size=11.5)
            callout(ax, "P₃  actual tip", actual, (-26, -10), color=teal, size=11)
            callout(ax, "Nominal tip", nominal, (7, -10), color=red, size=11)
            ax.annotate("", xy=(15, 52), xytext=(-5, 52),
                        arrowprops={"arrowstyle": "->", "color": teal, "lw": 2})
            ax.text(5, 55, "Motion +X  ·  30 mm/s", ha="center", fontsize=11.5, color=teal)
            ax.annotate("", xy=(actual[0], -2.7), xytext=(nominal[0], -2.7),
                        arrowprops={"arrowstyle": "->", "color": orange, "lw": 1.5})
            ax.text(-3.75, -5.3, "7.5 mm lag", ha="center", fontsize=10.5, color=orange)

    # Full-width cubic description, separated from the four compact formula cards.
    fig.text(.5, .268,
             r"Cubic spine:  $C(s)=\sum_{i=0}^{3}\binom{3}{i}(1-s)^{3-i}s^iP_i$",
             ha="center", fontsize=13.5, color=teal)
    fig.text(.5, .240,
             "In contact: P₀ = root;  P₁ = root + (L/3) d;  P₂ = actual tip − (L/3) m;  P₃ = actual tip",
             ha="center", fontsize=12)
    fig.text(.5, .216,
             "d = tool axis; m = horizontal motion unit vector (zero when stationary). Dashed gray line shows the control polygon.",
             ha="center", fontsize=10.5, color=muted)
    cards = [
        ("CONTACT / CLAMP", [r"$\delta=\max(0,\ z_p-z_{nom})$",
                              r"Contact only if $\delta>10^{-4}$ m",
                              r"Ring / fiber $z\leftarrow\max(z_p,z)$"]),
        ("FORCE ESTIMATE", [r"$F_n=650\,k\,\delta^{1.25}$",
                              r"$k=1.2$; does not change geometry",
                              "Reported model value; not calibrated"]),
        ("TIP LAG", [r"$\ell=\min(0.018,\ 1.25\delta)$",
                      r"$tip_{xy}=nom_{xy}-\ell\,\hat v_{xy}$",
                      "Applied if planar speed > 0.1 mm/s"]),
        ("SPLAY / FOOTPRINT", [r"$r_{stamp}=0.0018+0.048a\sqrt{\delta}$",
                                r"$\Delta r_{core}(s)=0.042a\sqrt{\delta}\,s^{1.6}$",
                                r"$a=1.4$; fiber coefficient is 0.046"]),
    ]
    for i, (title, lines) in enumerate(cards):
        x = .034 + i * .2425
        fig.add_artist(FancyBboxPatch((x, .073), .231, .119,
                                      boxstyle="round,pad=0.006,rounding_size=0.006",
                                      transform=fig.transFigure, facecolor="#F3F6F8",
                                      edgecolor="none", zorder=-1))
        fig.text(x+.011, .174, title, fontsize=11.7, fontweight="bold", color=ink)
        for j, line in enumerate(lines):
            fig.text(x+.011, .148-j*.026, line, fontsize=11.5, color=ink)
    fig.text(.5, .043,
             "Instantaneous parametric geometry, not measured brush mechanics. Formula lengths / δ are in metres; displayed dimensions are mm.",
             ha="center", fontsize=11, color=muted)
    fig.text(.5, .022,
             "Ring and fiber points clamp to paper; the apex is P₃. Examples avoid the 0–0.1 mm contact deadband. Orange extent is a stamp estimate, not the mesh boundary.",
             ha="center", fontsize=10.2, color=muted)
    fig.savefig(output_dir / "brush_deformation_explained.png", dpi=150,
                facecolor=fig.get_facecolor())
    fig.savefig(output_dir / "brush_deformation_explained.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--plot-python", default=sys.executable,
                        help="Interpreter with NumPy and Matplotlib, if different from viewer Python")
    parser.add_argument("--plot-data", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.plot_data:
        render(args.plot_data.resolve())
        return
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = generate_data(output_dir)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/callirewrite-brush-matplotlib")
    subprocess.run([args.plot_python, str(Path(__file__).resolve()),
                    "--plot-data", str(data_path)], check=True, env=env)
    metrics = json.loads((output_dir / "brush_deformation_metrics.json").read_text())
    print(json.dumps({"status": "pass", "source_sha256": metrics["source_sha256"],
                      "scenarios": metrics["scenarios"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
