#!/bin/bash
# Download KanjiVG stroke-order data (required for Stage 3 automatic trajectory generation)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/kanjivg.xml"
SHA256="293dc9741cf6b6cb47aabb6235dca05da0ae4210590e139b5b1aebe37ddf72a4"

if [ -f "$OUT" ]; then
    echo "$SHA256  $OUT" | sha256sum --check
    exit 0
fi

URL="https://github.com/KanjiVG/kanjivg/releases/download/r20260714/kanjivg-20260714.xml.gz"

echo "Downloading $URL ..."
curl --fail --location --retry 3 "$URL" -o "$DIR/kanjivg.xml.gz"
gzip -dc "$DIR/kanjivg.xml.gz" > "$OUT.part"
echo "$SHA256  $OUT.part" | sha256sum --check
mv "$OUT.part" "$OUT"
rm "$DIR/kanjivg.xml.gz"
echo "Saved to $OUT"
