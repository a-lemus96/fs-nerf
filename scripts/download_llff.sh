#!/usr/bin/env bash
# Downloads the LLFF dataset (nerf_llff_data, ~1.8 GB) into data/llff/.
# Mirror of the original Google Drive release:
# https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
set -euo pipefail

URL="https://huggingface.co/donydchen/matchnerf/resolve/main/nerf_llff_data.zip"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/llff"
TMP="$ROOT/data/.llff_download"
ZIP="$TMP/nerf_llff_data.zip"

mkdir -p "$DEST" "$TMP"

# -C - resumes a partial download if the script is re-run
echo "Downloading LLFF dataset to $ZIP"
curl -L -C - --fail -o "$ZIP" "$URL"

echo "Extracting"
unzip -q -o "$ZIP" -d "$TMP"

for scene in "$TMP"/nerf_llff_data/*/; do
    name="$(basename "$scene")"
    if [ -e "$DEST/$name" ]; then
        echo "Skipping $name (already in $DEST)"
    else
        mv "$scene" "$DEST/$name"
        echo "Added $name"
    fi
done

rm -rf "$TMP"
echo "Done: $(ls "$DEST" | tr '\n' ' ')"
