#!/usr/bin/env bash
# Download a small public COLMAP scene to verify training without the camera.
# Fetches the Tanks&Temples + Deep Blending set used by the reference 3DGS paper;
# each scene already has images/ + sparse/0/ (the layout gs3d expects).
set -euo pipefail
cd "$(dirname "$0")/.."   # → project root

URL="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
mkdir -p samples
cd samples

if [ ! -f tandt_db.zip ]; then
  echo "[sample] downloading $URL ..."
  curl -L -o tandt_db.zip "$URL"
fi
echo "[sample] extracting ..."
unzip -n -q tandt_db.zip

echo "[sample] scenes available:"
find . -maxdepth 2 -name sparse -type d | sed 's#/sparse##; s#^\./##'
echo
echo "Try:  uv run gs3d train ./samples/tandt/truck -o outputs/truck --max-steps 1000"
