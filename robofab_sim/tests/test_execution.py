import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calligraphy_sim import PandaKinematics, XuanPaperCanvas
from deformable_brush_physics import RealisticDeformableTuft
from kanjivg_to_trajectory import KanjiVGParser, _parse_svg_path, pressure_to_z
from execution import Execution, plan_samples


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = KanjiVGParser(ROOT / "data/kanjivg.xml")

    def test_new_characters(self):
        for char, count in zip("春江月夜", (9, 6, 4, 8)):
            strokes = self.parser.get_trajectory(char)
            self.assertEqual(len(strokes), count)
            for stroke in strokes:
                a = np.array(stroke)
                self.assertTrue(np.all(np.isfinite(a)))
                self.assertEqual(a[0, 3], 0)
                self.assertEqual(a[-1, 3], 0)
                self.assertGreater(a[0, 2], .002)
            z = [p[2] for s in strokes for p in s if p[3] > 0]
            self.assertGreater(len(set(z)), 3)

    def test_svg_commands_and_rejection(self):
        self.assertEqual(_parse_svg_path("M1,2 l3,4 Z")[-1], (1, 2))
        for d in ("M0 0 H5", "M0 0 C1 2", "L1 2", "M0$0 L1 2", "M0 0 Z1 2"):
            with self.assertRaises(ValueError):
                _parse_svg_path(d)
        for char in ("", "春江", "😀"):
            with self.assertRaises(ValueError):
                self.parser.get_trajectory(char)

    def test_pressure_and_ink(self):
        brush = RealisticDeformableTuft()
        radii, gray = [], []
        for pressure in (.3, .8, 1.4):
            paper = XuanPaperCanvas()
            tip = np.array([.55, .025, pressure_to_z(pressure)])
            rot = np.diag([1., -1., -1.])
            tuft = brush.compute_deformation(tip-.045*rot[:, 2], tip, rot, np.zeros(3))
            paper.deposit_ink(*tip[:2], tuft["footprint_radius"], tuft["normal_force"])
            radii.append(tuft["footprint_radius"])
            row = paper.world_to_pixel(.55, .025)[1]
            pixel = np.array(paper.image)[row, 512].mean()
            gray.append(pixel)
            before = np.array(paper.image)
            paper.deposit_ink(*tip[:2], tuft["footprint_radius"], tuft["normal_force"])
            np.testing.assert_array_equal(np.array(paper.image), before)
            paper.deposit_ink(99, 99, .005, 3)
            np.testing.assert_array_equal(np.array(paper.image), before)
        self.assertTrue(np.all(np.diff(radii) > 0))
        self.assertTrue(np.all(np.diff(gray) < -10), gray)

    def test_world_to_viser_image_rows(self):
        paper = XuanPaperCanvas(resolution=100)
        top = paper.world_to_pixel(.55, -.05)
        bottom = paper.world_to_pixel(.55, .05)
        self.assertGreater(top[1], bottom[1])
        self.assertEqual(paper.world_to_pixel(.55, .18)[1], 0)
        self.assertIsNone(paper.world_to_pixel(.55, -.18))

    def test_safe_transitions(self):
        k = PandaKinematics()
        samples = plan_samples(self.parser.get_trajectory("春"), k.forward_kinematics(k.NOMINAL_REST_Q)[2])
        for sample in samples:
            if sample.phase == "travel":
                self.assertGreaterEqual(sample.target[2], .025-1e-12)
                self.assertEqual(sample.pressure, 0)
        self.assertEqual(samples[-1].phase, "lift")
        self.assertAlmostEqual(samples[-1].target[2], .025)
        with self.assertRaises(ValueError):
            plan_samples([], [0, 0, 0])

    def test_execution_repeat_and_penup(self):
        k = PandaKinematics()
        outputs = []
        for _ in range(2):
            paper = XuanPaperCanvas()
            run = Execution(k, RealisticDeformableTuft(), paper, self.parser.get_trajectory("一"))
            while not run.done:
                tuft = run.step()
                self.assertIs(run.last_tuft, tuft)
            outputs.append(np.array(paper.image))
            self.assertIsNone(run.step())
            self.assertLess(max(row["residual_m"] for row in run.trace), .0005)
            self.assertLess(max(row["axis_error"] for row in run.trace), .005)
            self.assertFalse(any(row["contact"] for row in run.trace if row["phase"] == "travel"))
            self.assertFalse(any(row["deposited"] for row in run.trace if not row["contact"]))
            self.assertEqual({row["stroke"] for row in run.trace if row["deposited"]}, {0})
            q = np.array([row["q"] for row in run.trace])
            limits = np.array(k.JOINT_LIMITS)
            self.assertTrue(np.all(q >= limits[:, 0]))
            self.assertTrue(np.all(q <= limits[:, 1]))
        np.testing.assert_array_equal(*outputs)

    def test_fk_against_urdf(self):
        import yourdfpy
        candidates = list(Path.home().joinpath(".cache/robot_descriptions").glob("**/panda.urdf"))
        if not candidates:
            self.skipTest("Optional official Panda URDF not cached")
        urdf = yourdfpy.URDF.load(str(candidates[0]), load_meshes=False)
        k = PandaKinematics()
        rng = np.random.default_rng(42)
        for _ in range(12):
            q = k.NOMINAL_REST_Q + rng.uniform(-.2, .2, 7)
            urdf.update_cfg({f"panda_joint{i+1}": float(v) for i, v in enumerate(q)})
            T = urdf.get_transform("panda_hand")
            hand, rot, tip = k.forward_kinematics(q)
            np.testing.assert_allclose(hand, T[:3, 3], atol=1e-9)
            np.testing.assert_allclose(rot, T[:3, :3], atol=1e-9)
            np.testing.assert_allclose(tip, T[:3, 3]+.225*T[:3, 2], atol=1e-9)


if __name__ == "__main__":
    unittest.main()