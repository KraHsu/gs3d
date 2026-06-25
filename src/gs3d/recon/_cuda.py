"""Make a CUDA compiler discoverable for gsplat's just-in-time kernel build.

gsplat ships no cu128 prebuilt wheel, so on the first rasterization it compiles
its CUDA kernels with `nvcc` via `torch.utils.cpp_extension`. That needs a CUDA
toolkit whose `nvcc` matches the GPU's architecture (sm_120 for Blackwell / RTX
50-series, which requires CUDA >= 12.8).

For machines without a system toolkit we vendor a self-contained CUDA 12.8
compiler under ``<repo>/.cuda-jit/cuda128`` (created by
``scripts/setup_cuda_jit.sh``). `ensure_cuda_toolkit()` points `CUDA_HOME` at it
— unless the caller already has a working `nvcc` — and must run *before* the
first gsplat rasterization (i.e. at the start of each recon entry point).
"""

from __future__ import annotations

import os
from pathlib import Path

_BUNDLED_SUBPATH = Path(".cuda-jit") / "cuda128"


def _has_nvcc(root: str | os.PathLike) -> bool:
    return (Path(root) / "bin" / "nvcc").exists()


def ensure_cuda_toolkit() -> str | None:
    """Ensure CUDA_HOME points at a usable nvcc; return its path (or None).

    Honors an existing working CUDA_HOME/CUDA_PATH first; otherwise searches for
    the vendored ``.cuda-jit/cuda128`` prefix walking up from the CWD and this
    file. Idempotent and silent when nothing is needed.
    """
    existing = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if existing and _has_nvcc(existing):
        return existing

    here = Path(__file__).resolve()
    seen: set[Path] = set()
    for base in (Path.cwd().resolve(), *here.parents):
        prefix = (base / _BUNDLED_SUBPATH).resolve()
        if prefix in seen:
            continue
        seen.add(prefix)
        if _has_nvcc(prefix):
            os.environ["CUDA_HOME"] = str(prefix)
            os.environ.setdefault("CUDA_PATH", str(prefix))
            bin_dir = str(prefix / "bin")
            if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            print(f"[cuda] using bundled CUDA toolkit for gsplat JIT: {prefix}")
            return str(prefix)

    return None  # no toolkit found; gsplat JIT will fail with a clear nvcc error
