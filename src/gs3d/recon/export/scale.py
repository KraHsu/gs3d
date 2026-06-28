"""Estimate the metric scale of a 3DGS scene from the D435i depth frames.

3DGS/COLMAP geometry is scale-free; every downstream physical quantity (object
size, mass, robot reach, contact dynamics) is wrong without a metres-per-unit
factor. We recover it directly from the capture: render the Gaussians' expected
depth from each training camera (gsplat), and compare to the aligned D435i sensor
depth at the same pixels. Their ratio is COLMAP-units-per-metre; we take a robust
median over many cameras and pixels.

This needs no COLMAP sparse model — only the per-view poses in ``cameras.json``
(written by the reference-3DGS exporter), the sibling ``depth/`` frames, and the
colour intrinsics (for cx/cy, which ``cameras.json`` omits).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _c2w_from_cam(cam) -> np.ndarray:
    """INRIA cameras.json entry -> 4x4 camera-to-world (rotation cols = cam axes)."""
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(cam["rotation"], dtype=np.float32)
    c2w[:3, 3] = np.asarray(cam["position"], dtype=np.float32)
    return c2w


def estimate_metric_scale(
    checkpoint: str | Path,
    cameras_json: str | Path,
    depth_dir: str | Path,
    *,
    intrinsics_json: str | Path | None = None,
    n_samples: int = 24,
    depth_min_m: float = 0.2,
    depth_max_m: float = 5.0,
    invalid_mm: int = 65535,
    device: str = "cuda",
) -> float:
    """Return COLMAP-units-per-metre (divide coords by this to get metres).

    Renders expected depth for ``n_samples`` evenly-spaced cameras and compares to
    the matching ``depth/<img_name>.png`` (uint16 mm). Robust to outliers via a
    per-camera then global median of the render/sensor depth ratio.
    """
    import imageio.v2 as imageio

    from .._cuda import ensure_cuda_toolkit
    from ..inria import load_inria_checkpoint
    from ..model import rasterize_splats

    ensure_cuda_toolkit()
    cams = json.loads(Path(cameras_json).read_text())
    depth_dir = Path(depth_dir)

    # cx/cy live in the colour intrinsics, not cameras.json.
    if intrinsics_json is None:
        guess = Path(cameras_json).resolve().parent.parent / "table_new" / "intrinsics.json"
        intrinsics_json = guess if guess.exists() else None
    if intrinsics_json is not None and Path(intrinsics_json).exists():
        intr = json.loads(Path(intrinsics_json).read_text())
        cx, cy = float(intr["cx"]), float(intr["cy"])
    else:
        cx = cy = None  # fall back to image centre per camera

    model = load_inria_checkpoint(checkpoint, device=device)
    splats, sh_degree = model.splats, model.sh_degree

    idx = np.linspace(0, len(cams) - 1, min(n_samples, len(cams))).round().astype(int)
    per_cam = []
    used = 0
    for i in idx:
        cam = cams[int(i)]
        dpath = depth_dir / f"{cam['img_name']}.png"
        if not dpath.exists():
            continue
        sensor_mm = imageio.imread(dpath).astype(np.float32)
        H, W = sensor_mm.shape[:2]
        sensor_m = sensor_mm / 1000.0

        c2w = torch.as_tensor(_c2w_from_cam(cam), device=device)
        viewmat = torch.linalg.inv(c2w)[None]
        fx, fy = float(cam["fx"]), float(cam["fy"])
        kcx = cx if cx is not None else W / 2
        kcy = cy if cy is not None else H / 2
        K = torch.tensor([[fx, 0, kcx], [0, fy, kcy], [0, 0, 1]],
                         dtype=torch.float32, device=device)[None]

        with torch.no_grad():
            depth, alpha, _ = rasterize_splats(
                splats, viewmat, K, W, H, sh_degree, render_mode="ED"
            )
        render = depth[0, ..., 0].cpu().numpy()       # expected depth, COLMAP units
        a = alpha[0, ..., 0].cpu().numpy()

        valid = (
            (sensor_mm > 0) & (sensor_mm < invalid_mm)
            & (sensor_m > depth_min_m) & (sensor_m < depth_max_m)
            & (a > 0.5) & (render > 1e-4)
        )
        if valid.sum() < 500:
            continue
        ratios = render[valid] / sensor_m[valid]      # COLMAP units per metre
        per_cam.append(float(np.median(ratios)))
        used += 1

    if not per_cam:
        raise RuntimeError("metric scale: no camera had enough overlapping valid depth")
    scale = float(np.median(per_cam))
    spread = float(np.std(per_cam))
    print(f"[scale] {used} cameras used; COLMAP units per metre = {scale:.4f} "
          f"(per-camera std {spread:.4f}); 1 unit = {1.0/scale*100:.1f} cm")
    return scale
