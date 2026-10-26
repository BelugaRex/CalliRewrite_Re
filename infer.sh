#!/usr/bin/env bash
# infer.sh — one-shot: render char image → CalliRewrite inference → launch Viser
# Usage: bash infer.sh 福
# Requires: seq_extract/.venv (TF 2.10) and .venv (PyTorch/Viser)

set -e
CHAR="${1:-永}"
REPO="$(cd "$(dirname "$0")" && pwd)"
SE_VENV="$REPO/seq_extract/.venv/bin/python"
SIM_VENV="/home/rex/ISDN5240/.venv/bin/python"
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# 1. Render character to grayscale PNG
NAME="infer_$(echo "$CHAR" | od -A n -t x1 | tr -d ' \n')"
IMGDIR="$REPO/seq_extract/sample_inputs/${NAME}_dir"
mkdir -p "$IMGDIR"
$SE_VENV - <<PYEOF
from PIL import Image, ImageDraw, ImageFont
img = Image.new("L", (256,256), 255)
ImageDraw.Draw(img).text((28,10), "$CHAR", fill=0,
    font=ImageFont.truetype("$FONT", 200))
img.save("$IMGDIR/$NAME.png")
print("Rendered: $IMGDIR/$NAME.png")
PYEOF

# 2. CalliRewrite inference
echo "Running seq_extract inference..."
cd "$REPO/seq_extract"
CUDA_VISIBLE_DEVICES=4 "$SE_VENV" test.py \
    --input "sample_inputs/${NAME}_dir" \
    --model new_train_phase_2 2>&1 | grep -E "strokes_raw_out|average_stroke|round_lengths"

NPZ="$REPO/seq_extract/outputs/sampling/__new_train_phase_2/seq_data/${NAME}.npz"

# 3. Launch Viser simulator with CalliRewrite trajectory
echo ""
echo "Starting Viser simulator → http://localhost:8080"
echo "Press Ctrl+C to stop."
cd "$REPO/robofab_sim"
exec "$SIM_VENV" calligraphy_sim.py --npz "$NPZ"
