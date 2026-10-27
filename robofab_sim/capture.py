"""Record real Viser browser renders, with a same-tick canvas inset.

Requires a connected WebGL-capable browser. Does not fabricate a 3D render.
"""
import json
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


class BrowserCapture:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.frames = self.directory / "frames"
        self.frames.mkdir(parents=True, exist_ok=True)
        for old_frame in self.frames.glob("*.jpg"):
            old_frame.unlink()
        self.start = time.monotonic()
        self.timings = []
        root = Path(__file__).parent
        self.sources = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                ("calligraphy_sim.py", "execution.py", "kanjivg_to_trajectory.py", "deformable_brush_physics.py", "capture.py")}

    def frame(self, client, canvas, label):
        rgb = client.camera.get_render(height=720, width=960, transport_format="jpeg", timeout=20)
        image = Image.new("RGB", (1280, 760), (251, 247, 238))
        image.paste(Image.fromarray(np.asarray(rgb)[..., :3]), (0, 40))
        image.paste(canvas.resize((320, 320)), (960, 100))
        draw = ImageDraw.Draw(image)
        draw.text((16, 12), "VISER BROWSER RENDER | Panda IK/FK + deformable brush | " + label, fill="black")
        draw.text((970, 72), "SAME-TICK INK CANVAS", fill="black")
        draw.text((970, 440), "KanjiVG automatic strokes", fill="black")
        draw.text((970, 462), "Heuristic pressure / ink", fill="black")
        name = f"{len(self.timings):06d}.jpg"
        image.save(self.frames / name, quality=87)
        self.timings.append(dict(file=name, seconds=time.monotonic()-self.start, label=label))

    def finish(self):
        if not self.timings:
            raise RuntimeError("No real browser frames were captured")
        (self.directory / "timing.json").write_text(json.dumps(dict(
            capture_method="Viser CameraHandle.get_render from connected WebGL browser; same-tick canvas inset",
            frames=self.timings), indent=2) + "\n")
        # Variable durations preserve measured elapsed time, not an invented FPS.
        lines = []
        for i, row in enumerate(self.timings):
            duration = self.timings[i+1]["seconds"]-row["seconds"] if i+1 < len(self.timings) else 2.0
            lines.extend((f"file 'frames/{row['file']}'", f"duration {duration:.6f}"))
        lines.append(f"file 'frames/{self.timings[-1]['file']}'")
        concat = self.directory / "frames.txt"
        concat.write_text("\n".join(lines) + "\n")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(concat), "-vf", "fps=20", "-c:v", "libx264", "-crf", "24",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.directory / "demo.mp4")], check=True)
        snapshots = self.directory / "snapshots"
        snapshots.mkdir(exist_ok=True)
        runs = {}
        for row in self.timings:
            run = row["label"].split(" | ")[1].split()[1].split("/")[0]
            runs[run] = row
        for number, row in runs.items():
            shutil.copyfile(self.frames / row["file"], snapshots / f"run_{int(number):02d}.jpg")
        video = self.directory / "demo.mp4"
        (self.directory / "capture.json").write_text(json.dumps(dict(
            method="real Viser connected-browser renders + same-tick canvas inset",
            source_sha256=self.sources, frames=len(self.timings), runs=runs,
            video_sha256=hashlib.sha256(video.read_bytes()).hexdigest()), indent=2) + "\n")