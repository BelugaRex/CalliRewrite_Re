"""Regression suite + independent validation of every saved executed joint pose."""
import hashlib
import json
import subprocess
import unittest
from pathlib import Path
import numpy as np
from calligraphy_sim import PandaKinematics
from execution import plan_samples
from kanjivg_to_trajectory import KanjiVGParser

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    evidence = ROOT / "evidence"
    manifest = json.loads((evidence/"manifest.json").read_text())
    assert manifest["xml_sha256"] == sha(ROOT/"data/kanjivg.xml")
    for name, value in manifest["source_sha256"].items():
        assert sha(ROOT/name) == value, f"Stale evidence source: {name}"
    assert "".join(r["char"] for r in manifest["runs"]) == "永永一一人人春江月夜"
    parser, kin = KanjiVGParser(ROOT/"data/kanjivg.xml"), PandaKinematics()
    initial_tip = kin.forward_kinematics(kin.NOMINAL_REST_Q)[2]
    canvases = {}
    total, max_error = 0, 0.0
    for row in manifest["runs"]:
        folder = evidence/row["directory"]
        assert sha(folder/"canvas.png") == row["canvas_sha256"]
        assert sha(folder/"execution.npz") == row["trace_sha256"]
        expected_strokes = parser.get_trajectory(row["char"])
        np.testing.assert_equal(json.loads((folder/"trajectory.json").read_text()), expected_strokes)
        expected = plan_samples(expected_strokes, initial_tip)
        with np.load(folder/"execution.npz", allow_pickle=False) as t:
            assert len(t["q"]) == row["samples"] == len(expected)
            np.testing.assert_allclose(t["target"], [s.target for s in expected], atol=1e-12)
            assert not np.any(t["contact"][t["phase"] == "travel"])
            assert not np.any(t["deposited"] & ~t["contact"])
            assert set(t["stroke"][t["deposited"]]) == set(range(row["strokes"]))
            limits = np.array(kin.JOINT_LIMITS)
            assert np.all((t["q"] >= limits[:, 0]) & (t["q"] <= limits[:, 1]))
            fk = [kin.forward_kinematics(q) for q in t["q"]]
            tips = np.array([p[2] for p in fk])
            axes = np.array([p[1][:, 2] for p in fk])
            np.testing.assert_allclose(tips, t["tip"], atol=1e-12)
            residuals = np.linalg.norm(tips-t["target"], axis=1)
            np.testing.assert_allclose(residuals, t["residual_m"], atol=1e-12)
            assert residuals.max() < .0005
            assert np.linalg.norm(axes-[0, 0, -1], axis=1).max() < .005
            total += len(tips)
            max_error = max(max_error, residuals.max())
        if row["char"] in canvases:
            assert row["canvas_sha256"] == canvases[row["char"]], "Repeatability failed"
        canvases[row["char"]] = row["canvas_sha256"]
    calibration = json.loads((evidence/"pressure_calibration.json").read_text())
    assert np.all(np.diff([r["diameter_mm"] for r in calibration]) > 0)
    assert np.all(np.diff([r["center_rgb_mean"] for r in calibration]) < -10)
    browser = evidence/"browser"
    capture = json.loads((browser/"capture.json").read_text())
    assert sha(browser/"demo.mp4") == capture["video_sha256"]
    for name, value in capture["source_sha256"].items():
        assert sha(ROOT/name) == value, f"Stale browser capture source: {name}"
    timing = json.loads((browser/"timing.json").read_text())["frames"]
    assert len(timing) == capture["frames"]
    assert np.all(np.diff([r["seconds"] for r in timing]) > 0)
    assert set(capture["runs"]) == {str(i) for i in range(1, 11)}
    for i, char in enumerate("永永一一人人春江月夜", 1):
        label = capture["runs"][str(i)]["label"]
        assert label.startswith(f"U+{ord(char):04X} | run {i}/10 | sample ")
        current, end = map(int, label.rsplit(" ", 1)[1].split("/"))
        assert current == end, f"Incomplete browser run: {i}"
        assert (browser/"snapshots"/f"run_{i:02d}.jpg").exists()
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                               "format=duration:stream=width,height,codec_name", "-of", "json", str(browser/"demo.mp4")]))
    assert float(probe["format"]["duration"]) >= 30
    assert probe["streams"][0]["width"] == 1280
    print(f"EVIDENCE_OK: 10 runs, {total} FK poses, max IK {max_error*1000:.6f} mm, "
          f"10 completed browser runs, video {float(probe['format']['duration']):.2f}s")


if __name__ == "__main__":
    validate()