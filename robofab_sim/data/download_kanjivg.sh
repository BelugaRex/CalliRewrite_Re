#!/bin/bash
# Download KanjiVG stroke-order data (required for Stage 3 automatic trajectory generation)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/kanjivg.xml"

if [ -f "$OUT" ]; then
    echo "kanjivg.xml already exists, skipping download."
    exit 0
fi

URL=$(curl -s "https://api.github.com/repos/KanjiVG/kanjivg/releases/latest" \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print(next(a['browser_download_url'] for a in r['assets'] if a['name'].endswith('.xml.gz')))")

echo "Downloading $URL ..."
curl -L "$URL" -o "$DIR/kanjivg.xml.gz"
gunzip "$DIR/kanjivg.xml.gz"
echo "Saved to $OUT"
