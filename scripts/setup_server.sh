#!/usr/bin/env bash
# One-time setup for the recon subpart on the Ubuntu/H20 server.
# Installs uv, configures CUDA, syncs the `recon` extra, warms up gsplat's CUDA build.
set -euo pipefail

cd "$(dirname "$0")/.."   # → project root

# 1. uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv $(uv --version)"

# 2. CUDA toolkit (needed for gsplat's CUDA extension build) -----------------
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "[setup] WARNING: nvcc not found at $CUDA_HOME/bin/nvcc."
  echo "        Set CUDA_HOME to your CUDA toolkit before re-running."
else
  echo "[setup] nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1)"
fi

# 3. Python env + deps -------------------------------------------------------
uv python pin 3.11

# Optional fast package mirror (e.g. in China). torch still comes from the
# explicit pytorch-cu128 index; other wheels (nvidia-*, etc.) use this default.
#   GS3D_INDEX=https://mirrors.aliyun.com/pypi/simple/ bash scripts/setup_server.sh
INDEX_ARG=""
if [ -n "${GS3D_INDEX:-}" ]; then
  INDEX_ARG="--default-index ${GS3D_INDEX}"
  echo "[setup] package index: ${GS3D_INDEX}"
fi

echo "[setup] syncing the recon extra (torch cu128, gsplat, pycolmap, ...) ..."
uv sync --extra recon ${INDEX_ARG}

# 4. Warm up gsplat (compiles its CUDA kernels on first import) --------------
echo "[setup] verifying torch CUDA + compiling gsplat (first import is slow) ..."
uv run python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not visible to torch"
print("device:", torch.cuda.get_device_name(0))
import gsplat
from gsplat import rasterization, DefaultStrategy  # noqa: F401
# Actually invoke a kernel so gsplat's CUDA extension compiles now (not on the
# first train). `uv run` puts ninja (and nvcc via CUDA_HOME/PATH) on PATH.
N = 100
means = torch.randn(N, 3, device="cuda")
quats = torch.zeros(N, 4, device="cuda"); quats[:, 0] = 1
scales = torch.full((N, 3), 0.1, device="cuda")
opac = torch.ones(N, device="cuda")
colors = torch.rand(N, 3, device="cuda")
viewmats = torch.eye(4, device="cuda")[None]
Ks = torch.tensor([[[300., 0, 128], [0, 300, 128], [0, 0, 1]]], device="cuda")
img, _, _ = rasterization(means, quats, scales, opac, colors, viewmats, Ks, 256, 256)
print("gsplat", gsplat.__version__, "kernel compiled + ran:", tuple(img.shape))
PY

echo "[setup] done. Try:  bash scripts/get_sample.sh && uv run gs3d train ./samples/tandt/truck -o outputs/sample --max-steps 1000"
