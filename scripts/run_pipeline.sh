#!/usr/bin/env bash
# End-to-end: SfM → train → render for one captured scene.
# Usage: bash scripts/run_pipeline.sh data/<scene> [max_steps]
set -euo pipefail
cd "$(dirname "$0")/.."   # → project root

SCENE="${1:?usage: run_pipeline.sh <scene_dir> [max_steps]}"
STEPS="${2:-7000}"
NAME="$(basename "$SCENE")"
OUT="outputs/$NAME"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

uv run gs3d sfm "$SCENE"
uv run gs3d train "$SCENE" -o "$OUT" --max-steps "$STEPS"
uv run gs3d render "$OUT"
echo "[pipeline] outputs in $OUT (point_cloud.ply, orbit.mp4, eval/)"
