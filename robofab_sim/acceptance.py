"""Generate rubric evidence by executing IK/FK + brush contact (not SVG preview)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from calligraphy_sim import PandaKinematics, XuanPaperCanvas
from deformable_brush_physics import RealisticDeformableTuft
from execution import Execution
from kanjivg_to_trajectory import KanjiVGParser

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(parser, char, out):
    strokes = parser.get_trajectory(char)
    paper = XuanPaperCanvas()
    execution = Execution(PandaKinematics(), RealisticDeformableTuft(), paper, strokes)
    while not execution.done:
        execution.step()
    out.mkdir(parents=True, exist_ok=True)
    paper.image.save(out / "canvas.png")
    write_json(out / "trajectory.json", strokes)
    # Compressed numeric trace retains every sample and all seven actual joints.
    trace = execution.trace
    np.savez_compressed(out / "execution.npz", **{
        key: np.array([row[key] for row in trace]) for key in trace[0]
    })
    contact = [r for r in trace if r["deposited"]]
    summary = dict(char=char, codepoint=f"U+{ord(char):04X}",
                   xml_id=parser._find_kanji(char).get("id"),
                   strokes=len(strokes), samples=len(trace), contact_samples=len(contact),
                   max_ik_error_m=max(r["residual_m"] for r in trace),
                   max_tool_axis_error=max(r["axis_error"] for r in trace),
                   travel_contact_samples=sum(r["contact"] for r in trace if r["phase"] == "travel"),
                   force_range_n=[min(r["force_n"] for r in contact), max(r["force_n"] for r in contact)],
                   radius_range_m=[min(r["radius_m"] for r in contact), max(r["radius_m"] for r in contact)],
                   ink_pixels=int(np.count_nonzero(paper.coverage)),
                   canvas_sha256=digest(out / "canvas.png"),
                   trace_sha256=digest(out / "execution.npz"))
    assert summary["travel_contact_samples"] == 0, summary
    assert summary["max_ik_error_m"] < .0005, summary
    assert summary["ink_pixels"] > 100, summary
    write_json(out / "summary.json", summary)
    print(char, "strokes", len(strokes), "samples", len(trace),
          "max IK mm", round(summary["max_ik_error_m"]*1000, 4), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", type=Path, default=ROOT / "data/kanjivg.xml")
    ap.add_argument("--output", type=Path, default=ROOT / "evidence")
    ap.add_argument("--chars", default="永永一一人人春江月夜")
    args = ap.parse_args()
    parser = KanjiVGParser(args.xml)
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for i, char in enumerate(args.chars):
        name = f"{i+1:02d}_U{ord(char):04X}"
        summary = run(parser, char, args.output / name)
        runs.append(dict(directory=name, **summary))
    manifest = dict(evidence_type="deterministic kinematic execution, not browser recording",
                    unseen_definition="whole characters absent from pre-existing project demos; not a training-set claim",
                    source="KanjiVG, CC BY-SA 3.0, https://kanjivg.tagaini.net/",
                    xml_sha256=digest(args.xml),
                    source_sha256={name: digest(ROOT/name) for name in
                                   ("calligraphy_sim.py", "kanjivg_to_trajectory.py", "execution.py", "deformable_brush_physics.py", "acceptance.py")},
                    units="metres, radians, heuristic newtons", runs=runs)
    write_json(args.output / "manifest.json", manifest)
    sheet = Image.new("RGB", (320*5, 360*2), "white")
    for i, row in enumerate(runs):
        image = Image.open(args.output / row["directory"] / "canvas.png").resize((320, 320))
        x, y = (i % 5)*320, (i // 5)*360
        sheet.paste(image, (x, y))
        ImageDraw.Draw(sheet).text((x+10, y+325), f'{row["codepoint"]} | {row["strokes"]} strokes | run {i+1}', fill="black")
    sheet.save(args.output / "contact_sheet.png")
    # Explicit reference/result comparison; reference is the dataset centerline,
    # not a claimed reconstruction from an image or an independent robot run.
    unique = list(dict.fromkeys(args.chars))
    comparison = Image.new("RGB", (640, 350*len(unique)), "white")
    for i, char in enumerate(unique):
        reference = XuanPaperCanvas(resolution=320)
        draw = ImageDraw.Draw(reference.image)
        for stroke in parser.get_trajectory(char):
            xy = [(int((p[0]-.37)/.36*320), int((p[1]+.18)/.36*320)) for p in stroke if p[3] > 0]
            draw.line(xy, fill=(20, 20, 20), width=2)
        result = next(row for row in runs if row["char"] == char)
        comparison.paste(reference.image, (0, 350*i))
        comparison.paste(Image.open(args.output/result["directory"]/"canvas.png").resize((320, 320)), (320, 350*i))
        ImageDraw.Draw(comparison).text((10, 350*i+325), f"{result['codepoint']} KanjiVG centerline          Actual IK + brush result", fill="black")
    comparison.save(args.output / "reference_comparison.png")
    calibration = Image.new("RGB", (960, 380), "white")
    levels = []
    from kanjivg_to_trajectory import pressure_to_z
    for i, p in enumerate((.3, .8, 1.4)):
        paper = XuanPaperCanvas(resolution=320)
        brush = RealisticDeformableTuft()
        tip = np.array([.55, .025, pressure_to_z(p)])
        rot = np.diag([1., -1., -1.])
        tuft = brush.compute_deformation(tip-.045*rot[:, 2], tip, rot, np.zeros(3))
        for x in np.linspace(.47, .63, 300):
            paper.deposit_ink(x, .025, tuft["footprint_radius"], tuft["normal_force"])
        row = paper.world_to_pixel(.55, .025)[1]
        gray = float(np.array(paper.image)[row, 160].mean())
        levels.append(dict(pressure=p, depth_m=.002-tip[2], diameter_mm=tuft["footprint_radius"]*2000, center_rgb_mean=gray))
        calibration.paste(paper.image, (320*i, 0))
        ImageDraw.Draw(calibration).text((320*i+10, 330), f"p={p} | width={levels[-1]['diameter_mm']:.1f}mm | gray={gray:.0f}", fill="black")
    ImageDraw.Draw(calibration).text((10, 360), "Brush/ink model calibration (not a robot demonstration); lower gray = darker", fill="black")
    calibration.save(args.output / "pressure_calibration.png")
    write_json(args.output / "pressure_calibration.json", levels)


if __name__ == "__main__":
    main()