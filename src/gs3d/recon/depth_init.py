"""Dense Gaussian initialization from RealSense depth.

3DGS normally seeds Gaussians from the sparse COLMAP point cloud. The D435i's
aligned depth lets us seed a *dense* colored cloud instead — better geometry,
faster convergence, fewer floaters (esp. on low-texture surfaces).

COLMAP reconstructions are scale-free, so we first estimate the scale between
COLMAP units and metres (median of z_colmap / z_sensor over sparse observations),
then back-project the depth maps into the COLMAP world frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .dataset import _camera_K, _image_w2c, _is_registered


def _load_depth_m(path: Path) -> np.ndarray | None:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return d.astype(np.float32) / 1000.0  # uint16 mm -> metres


def estimate_colmap_scale(rec, depth_dir: Path, z_min: float, z_max: float) -> float:
    """COLMAP units per metre = median(z_colmap / z_sensor) over sparse points."""
    depth_dir = Path(depth_dir)
    ratios: list[float] = []
    cache: dict[str, np.ndarray | None] = {}
    for img in rec.images.values():
        if not _is_registered(img):
            continue
        stem = Path(img.name).stem
        if stem not in cache:
            cache[stem] = _load_depth_m(depth_dir / f"{stem}.png")
        depth = cache[stem]
        if depth is None:
            continue
        w2c = _image_w2c(img)
        R, t = w2c[:3, :3], w2c[:3, 3]
        H, W = depth.shape
        for p2 in img.get_valid_points2D():
            Pw = np.asarray(rec.points3D[p2.point3D_id].xyz)
            z = float((R @ Pw + t)[2])
            x, y = int(round(p2.xy[0])), int(round(p2.xy[1]))
            if 0 <= y < H and 0 <= x < W:
                d = float(depth[y, x])
                if z_min < d < z_max and z > 0:
                    ratios.append(z / d)
        if len(ratios) > 80_000:
            break
    if not ratios:
        return 1.0
    return float(np.median(ratios))


def build_dense_cloud(
    scene_dir: str | Path,
    rec,
    scale: float,
    image_stride: int = 3,
    pixel_stride: int = 4,
    z_min: float = 0.2,
    z_max: float = 3.0,
    max_points: int = 1_500_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project depth maps into a dense colored point cloud (COLMAP frame)."""
    scene_dir = Path(scene_dir)
    depth_dir, image_dir = scene_dir / "depth", scene_dir / "images"
    imgs = [im for im in sorted(rec.images.values(), key=lambda i: i.name) if _is_registered(im)]
    imgs = imgs[::image_stride]

    pts_all, col_all = [], []
    for img in imgs:
        stem = Path(img.name).stem
        depth = _load_depth_m(depth_dir / f"{stem}.png")
        if depth is None:
            continue
        bgr = cv2.imread(str(image_dir / img.name), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        K = _camera_K(rec.cameras[img.camera_id])
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        H, W = depth.shape
        xs = np.arange(0, W, pixel_stride)
        ys = np.arange(0, H, pixel_stride)
        gx, gy = np.meshgrid(xs, ys)
        d = depth[gy, gx]
        m = (d > z_min) & (d < z_max)
        if not m.any():
            continue
        z = d[m] * scale
        X = (gx[m] - cx) / fx * z
        Y = (gy[m] - cy) / fy * z
        cam_pts = np.stack([X, Y, z], axis=1)  # Nx3, COLMAP units

        c2w = np.linalg.inv(_image_w2c(img))
        world = cam_pts @ c2w[:3, :3].T + c2w[:3, 3]
        pts_all.append(world.astype(np.float32))
        col_all.append(rgb[gy[m], gx[m]].astype(np.float32))

    if not pts_all:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    pts = np.concatenate(pts_all)
    cols = np.concatenate(col_all)
    if len(pts) > max_points:
        idx = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols
