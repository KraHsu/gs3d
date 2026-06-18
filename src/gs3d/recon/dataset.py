"""Load a COLMAP reconstruction into camera views + an SfM point cloud.

Uses `pycolmap` to read the model, avoiding a bespoke binary parser. Camera
poses are returned as OpenCV-convention world-to-camera matrices (the format
gsplat's rasterizer expects via `viewmats`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap


@dataclass
class CameraView:
    name: str
    image: np.ndarray  # HxWx3 uint8 RGB
    K: np.ndarray  # 3x3 intrinsics (already scaled for downscale)
    w2c: np.ndarray  # 4x4 world-to-camera (OpenCV)
    width: int
    height: int


def _camera_K(cam: "pycolmap.Camera") -> np.ndarray:
    # Prefer the intrinsic matrix directly when available (most robust).
    if hasattr(cam, "calibration_matrix"):
        try:
            return np.asarray(cam.calibration_matrix(), dtype=np.float64)
        except Exception:
            pass
    model = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
    p = list(cam.params)
    if "SIMPLE_PINHOLE" in model or "SIMPLE_RADIAL" in model:
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif "PINHOLE" in model or "OPENCV" in model:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:  # fallback
        fx = fy = p[0]
        cx, cy = cam.width / 2.0, cam.height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _image_w2c(img: "pycolmap.Image") -> np.ndarray:
    w2c = np.eye(4, dtype=np.float64)
    if hasattr(img, "cam_from_world"):
        cfw = img.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        w2c[:3, :4] = np.asarray(cfw.matrix())  # Rigid3d -> 3x4 [R|t]
    else:  # older pycolmap API
        w2c[:3, :3] = img.rotmat()
        w2c[:3, 3] = np.asarray(img.tvec)
    return w2c


def _is_registered(img: "pycolmap.Image") -> bool:
    reg = getattr(img, "registered", None)  # pycolmap 0.6.x
    if reg is None:
        reg = getattr(img, "has_pose", True)  # newer pycolmap
    return bool(reg)


class ColmapDataset:
    """In-memory dataset over a `<scene>/sparse/0` reconstruction."""

    def __init__(self, scene_dir: str | Path, downscale: int = 1, test_every: int = 8):
        scene_dir = Path(scene_dir)
        sparse = scene_dir / "sparse" / "0"
        if not sparse.exists():
            raise FileNotFoundError(
                f"No reconstruction at {sparse}. Run `gs3d sfm {scene_dir}` first."
            )
        rec = pycolmap.Reconstruction(str(sparse))
        image_root = scene_dir / "images"

        # SfM point cloud for Gaussian initialization.
        xyz, rgb = [], []
        for pt in rec.points3D.values():
            xyz.append(pt.xyz)
            rgb.append(pt.color)
        self.points = np.asarray(xyz, dtype=np.float32)
        self.points_rgb = np.asarray(rgb, dtype=np.float32)

        # Camera views, ordered by image name for stable train/test splits.
        self.views: list[CameraView] = []
        for img in sorted(rec.images.values(), key=lambda im: im.name):
            if not _is_registered(img):
                continue  # unregistered image
            cam = rec.cameras[img.camera_id]
            K = _camera_K(cam)
            w2c = _image_w2c(img)

            bgr = cv2.imread(str(image_root / img.name), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            if downscale > 1:
                bgr = cv2.resize(
                    bgr,
                    (bgr.shape[1] // downscale, bgr.shape[0] // downscale),
                    interpolation=cv2.INTER_AREA,
                )
                K = K.copy()
                K[:2, :] /= downscale
            rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb_img.shape[:2]
            self.views.append(
                CameraView(name=img.name, image=rgb_img, K=K, w2c=w2c, width=w, height=h)
            )

        if not self.views:
            raise RuntimeError("Reconstruction has no posed images.")

        idx = np.arange(len(self.views))
        self.test_idx = idx[idx % test_every == 0]
        self.train_idx = idx[idx % test_every != 0]
        self.scene_scale = self._compute_scene_scale()

    def _compute_scene_scale(self) -> float:
        centers = []
        for v in self.views:
            c2w = np.linalg.inv(v.w2c)
            centers.append(c2w[:3, 3])
        centers = np.stack(centers)
        center = centers.mean(axis=0)
        return float(np.linalg.norm(centers - center, axis=1).max() * 1.1 + 1e-6)

    def train_views(self) -> list[CameraView]:
        return [self.views[i] for i in self.train_idx]

    def test_views(self) -> list[CameraView]:
        return [self.views[i] for i in self.test_idx]
