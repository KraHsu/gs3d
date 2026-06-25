"""Render a trained model: test-view comparisons + an orbit fly-through video.

Loads a checkpoint, rebuilds the dataset (for camera poses/intrinsics), reports
PSNR/SSIM on held-out views, and writes an orbit mp4.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .dataset import ColmapDataset
from .model import load_checkpoint, psnr, rasterize_splats, ssim


def _look_at(eye: np.ndarray, target: np.ndarray, world_down: np.ndarray) -> np.ndarray:
    """Build an OpenCV-convention world-to-camera matrix."""
    z = target - eye
    z /= np.linalg.norm(z) + 1e-9
    x = np.cross(world_down, z)
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return np.linalg.inv(c2w)


def _orbit_path(dataset: ColmapDataset, n_frames: int) -> list[np.ndarray]:
    centers, downs = [], []
    for v in dataset.views:
        c2w = np.linalg.inv(v.w2c)
        centers.append(c2w[:3, 3])
        downs.append(c2w[:3, 1])
    centers = np.stack(centers)
    target = dataset.points.mean(axis=0) if len(dataset.points) else centers.mean(0)
    world_down = np.mean(downs, axis=0)
    world_down /= np.linalg.norm(world_down) + 1e-9
    axis = -world_down
    radius = float(np.median(np.linalg.norm(centers - target, axis=1)))
    mean_eye_offset = centers.mean(0) - target

    a = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, a)
    e1 /= np.linalg.norm(e1) + 1e-9
    e2 = np.cross(axis, e1)
    height = float(np.dot(mean_eye_offset, axis))  # keep the cameras' average elevation

    poses = []
    for t in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        eye = target + radius * (np.cos(t) * e1 + np.sin(t) * e2) + height * axis
        poses.append(_look_at(eye, target, world_down))
    return poses


@torch.no_grad()
def render(
    out_dir: str | Path,
    n_frames: int = 120,
    fps: int = 30,
    device: str = "cuda",
) -> None:
    from ._cuda import ensure_cuda_toolkit

    ensure_cuda_toolkit()  # let gsplat JIT-compile its kernels (Blackwell/cu128)
    out_dir = Path(out_dir)
    splats, config = load_checkpoint(out_dir / "ckpt.pt", device=device)
    sh_degree = config["sh_degree"]
    dataset = ColmapDataset(config["scene_dir"], downscale=config.get("downscale", 1))

    # Held-out evaluation with side-by-side comparison images.
    cmp_dir = out_dir / "eval"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    psnrs, ssims = [], []
    for view in dataset.test_views():
        viewmats = torch.from_numpy(view.w2c).float().to(device).unsqueeze(0)
        Ks = torch.from_numpy(view.K).float().to(device).unsqueeze(0)
        renders, _, _ = rasterize_splats(splats, viewmats, Ks, view.width, view.height, sh_degree)
        pred = renders[..., :3].clamp(0, 1)
        gt = torch.from_numpy(view.image).float().to(device).unsqueeze(0) / 255.0
        psnrs.append(psnr(pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).item())
        ssims.append(ssim(pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)).item())
        side = torch.cat([gt[0], pred[0]], dim=1).cpu().numpy()
        imageio.imwrite(cmp_dir / f"{Path(view.name).stem}.png", (side * 255).astype(np.uint8))
    if psnrs:
        print(f"[render] test PSNR={np.mean(psnrs):.2f} SSIM={np.mean(ssims):.4f} "
              f"({len(psnrs)} views) -> {cmp_dir}")

    # Orbit fly-through video.
    use_view = dataset.views[0]
    Ks = torch.from_numpy(use_view.K).float().to(device).unsqueeze(0)
    w, h = use_view.width, use_view.height
    video_path = out_dir / "orbit.mp4"
    writer = imageio.get_writer(video_path, fps=fps)
    for w2c in _orbit_path(dataset, n_frames):
        viewmats = torch.from_numpy(w2c).float().to(device).unsqueeze(0)
        renders, _, _ = rasterize_splats(splats, viewmats, Ks, w, h, sh_degree)
        frame = (renders[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()
    print(f"[render] wrote {video_path}")
