#!/usr/bin/env bash
# Vendor a self-contained CUDA 12.8 *compiler* for gsplat's JIT build, no root.
#
# Why: gsplat ships no cu128 prebuilt wheel, so it compiles its CUDA kernels with
# nvcc on first use. Blackwell / RTX 50-series GPUs are sm_120 and need CUDA >=
# 12.8, which the NVIDIA pip nvcc wheel does NOT fully provide (ptxas only). This
# fetches a complete nvcc toolchain from NVIDIA's conda channel via micromamba
# (a single static binary — no conda install, no sudo) into ./.cuda-jit/cuda128,
# then assembles a standard CUDA_HOME layout the recon code auto-detects
# (see src/gs3d/recon/_cuda.py). Idempotent.
#
# Machines that already have a CUDA >= 12.8 toolkit on PATH/CUDA_HOME don't need
# this — the recon code honors an existing nvcc first (e.g. the H20 server).
set -euo pipefail

cd "$(dirname "$0")/.."                 # → project root
JIT_DIR="$PWD/.cuda-jit"
PREFIX="$JIT_DIR/cuda128"
CUDA_VER="12.8.1"                       # NVIDIA conda channel label
mkdir -p "$JIT_DIR"

# 1. micromamba (static binary, ~18 MB) -------------------------------------
MM="$JIT_DIR/bin/micromamba"
if [ ! -x "$MM" ]; then
  echo "[cuda-jit] downloading micromamba ..."
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$JIT_DIR/micromamba.tar.bz2"
  tar -xjf "$JIT_DIR/micromamba.tar.bz2" -C "$JIT_DIR" bin/micromamba
fi
echo "[cuda-jit] micromamba $("$MM" --version)"

# 2. CUDA 12.8 nvcc + cudart headers + cccl (cub/thrust) --------------------
if [ ! -x "$PREFIX/bin/nvcc" ]; then
  echo "[cuda-jit] fetching CUDA $CUDA_VER compiler into $PREFIX ..."
  MAMBA_ROOT_PREFIX="$JIT_DIR/mamba-root" "$MM" create -y -p "$PREFIX" \
    -c "nvidia/label/cuda-$CUDA_VER" -c conda-forge \
    cuda-nvcc cuda-cudart-dev cuda-cccl
fi

# 3. Standard CUDA_HOME layout (torch's cpp_extension wants include/ + lib64/) -
#    Conda places CUDA headers/libs under targets/x86_64-linux/{include,lib}.
for item in "$PREFIX"/targets/x86_64-linux/include/*; do
  name="$(basename "$item")"
  [ -e "$PREFIX/include/$name" ] || ln -s "../targets/x86_64-linux/include/$name" "$PREFIX/include/$name"
done
[ -e "$PREFIX/lib64" ] || ln -s lib "$PREFIX/lib64"

# 4. Sanity check ------------------------------------------------------------
echo "[cuda-jit] nvcc: $("$PREFIX/bin/nvcc" --version | tail -1)"
"$PREFIX/bin/nvcc" --list-gpu-arch | grep -q compute_120 \
  && echo "[cuda-jit] OK: nvcc supports sm_120 (Blackwell)." \
  || echo "[cuda-jit] WARNING: nvcc lacks compute_120 — check CUDA version."
echo "[cuda-jit] done. 'uv run gs3d view-seg ...' will auto-use $PREFIX."
