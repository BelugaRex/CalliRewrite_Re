#!/usr/bin/env python3
"""Build README media from the recorded Viser browser session (requires ffmpeg)."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
STAGES = [("approach", 0.4, "1  Approach above the paper"),
          ("contact", 3.5, "2  Press and make the first mark"),
          ("sweep", 8.5, "3  Sweep across the paper"),
          ("lift", 15.9, "4  Lift to finish yi")]


def run(arguments):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-frames", type=Path,
                        help="Optional folder with CUA JPEGs and timing.json.")
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg must be on PATH")
    video = HERE / "viser_yi_recording.mp4"
    if args.from_frames:
        folder = args.from_frames.resolve()
        frames = json.loads((folder / "timing.json").read_text())
        manifest = ["ffconcat version 1.0"]
        for index, frame in enumerate(frames[:-1]):
            manifest += ["file '" + str(folder / frame["file"]) + "'",
                         "duration " + str((frames[index + 1]["ms"] - frame["ms"]) / 1000)]
        manifest.append("file '" + str(folder / frames[-1]["file"]) + "'")
        concat = folder / "frames.ffconcat"
        concat.write_text("\n".join(manifest) + "\n")
        run(["-safe", "0", "-f", "concat", "-i", concat,
             "-vf", "fps=12", "-c:v", "libx264", "-crf", "22",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", video])
    if not video.is_file():
        parser.error("Missing viser_yi_recording.mp4; supply --from-frames to encode a recording")

    # Keep the actual robot/table canvas and omit the control sidebar in motion media.
    # Full browser-context JPEGs are preserved separately next to this script.
    run(["-i", video, "-vf",
         "fps=7,crop=600:600:340:70,split[a][b];[a]palettegen[p];[b][p]paletteuse",
         "-loop", "0", HERE / "viser_yi_drawing.gif"])

    inputs = []
    filters = []
    for index, (name, seconds, title) in enumerate(STAGES):
        inputs += ["-ss", str(seconds), "-i", video]
        filters.append(
            f"[{index}:v]crop=600:600:340:70,pad=600:660:0:60:color=0xf4f6f8,"
            f"drawtext=font='DejaVu Sans':text='{title}':x=20:y=13:"
            f"fontsize=23:fontcolor=0x1f2937,"
            f"drawtext=font='DejaVu Sans':text='t = {seconds:.1f} s':x=20:y=40:"
            f"fontsize=14:fontcolor=0x64748b[stage{index}]"
        )
    filters.append("[stage0][stage1][stage2][stage3]xstack=inputs=4:layout=0_0|600_0|0_660|600_660[out]")
    run([*inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
         "-frames:v", "1", HERE / "viser_yi_sequence.png"])
    print("Built viser_yi_drawing.gif and viser_yi_sequence.png from", video)


if __name__ == "__main__":
    main()
