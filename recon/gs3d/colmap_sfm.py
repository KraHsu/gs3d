"""Structure-from-Motion front-end using pycolmap (no system COLMAP needed).

Takes ``<scene>/images/`` and produces a COLMAP sparse model at
``<scene>/sparse/0/`` (cameras, images, points3D) for Gaussian Splatting.

We force a single PINHOLE camera (all frames come from the same physical
camera). This skips lens-distortion estimation / image undistortion, keeping
the pipeline simple; the RealSense color stream is close to pinhole.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pycolmap


def run_sfm(
    scene_dir: str | Path,
    matching: str = "exhaustive",
    device: str = "auto",
    overwrite: bool = False,
) -> Path:
    scene_dir = Path(scene_dir)
    image_dir = scene_dir / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"No images/ folder in {scene_dir}")

    db_path = scene_dir / "database.db"
    sparse_dir = scene_dir / "sparse"

    if overwrite:
        db_path.unlink(missing_ok=True)
        shutil.rmtree(sparse_dir, ignore_errors=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    dev = {
        "auto": pycolmap.Device.auto,
        "cpu": pycolmap.Device.cpu,
        "cuda": pycolmap.Device.cuda,
    }[device]

    print(f"[sfm] feature extraction ({image_dir}) ...")
    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="PINHOLE",
        device=dev,
    )

    print(f"[sfm] feature matching ({matching}) ...")
    if matching == "sequential":
        pycolmap.match_sequential(database_path=db_path, device=dev)
    else:
        pycolmap.match_exhaustive(database_path=db_path, device=dev)

    print("[sfm] incremental mapping ...")
    recs = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=sparse_dir,
    )
    if not recs:
        raise RuntimeError(
            "COLMAP failed to register any images. Capture more overlapping, "
            "textured, sharp views and retry."
        )

    # Keep the largest reconstruction as sparse/0.
    best_id = max(recs, key=lambda i: recs[i].num_reg_images())
    best = recs[best_id]
    print(
        f"[sfm] done: {best.num_reg_images()} images, "
        f"{best.num_points3D()} points -> {sparse_dir / '0'}"
    )
    if best_id != 0:
        best.write(str(sparse_dir / "0"))
    return sparse_dir / "0"
