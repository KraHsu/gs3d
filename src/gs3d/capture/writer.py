"""Write captured frames into a COLMAP/gsplat-friendly dataset layout.

    data/<scene>/
        images/000001.jpg ...     # color frames (input to COLMAP SfM)
        depth/000001.png  ...      # 16-bit depth in millimetres, aligned to color
        intrinsics.json            # RealSense color intrinsics
        meta.json                  # capture config + per-frame log
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .camera import Frame, Intrinsics


class DatasetWriter:
    def __init__(self, root: Path, scene: str, intrinsics: Intrinsics) -> None:
        self.scene_dir = Path(root) / scene
        self.images_dir = self.scene_dir / "images"
        self.depth_dir = self.scene_dir / "depth"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        self.intrinsics = intrinsics
        self._count = self._highest_existing_index()
        self._log: list[dict] = []
        self._imu_count = 0
        self._imu_path = self.scene_dir / "imu.jsonl"

        # Persist intrinsics immediately so the scene is self-describing.
        (self.scene_dir / "intrinsics.json").write_text(
            json.dumps(intrinsics.to_dict(), indent=2)
        )

    def _highest_existing_index(self) -> int:
        existing = sorted(self.images_dir.glob("*.jpg"))
        if not existing:
            return 0
        try:
            return int(existing[-1].stem)
        except ValueError:
            return len(existing)

    @property
    def count(self) -> int:
        return self._count

    def save(self, frame: Frame, jpg_quality: int = 95) -> str:
        """Persist one frame; returns the stem (e.g. ``000007``)."""
        self._count += 1
        stem = f"{self._count:06d}"

        img_path = self.images_dir / f"{stem}.jpg"
        cv2.imwrite(str(img_path), frame.color_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        cv2.imwrite(str(self.depth_dir / f"{stem}.png"), frame.depth_mm)

        self._log.append(
            {"stem": stem, "ts": datetime.now(timezone.utc).isoformat()}
        )
        return stem

    def append_imu(self, samples: list[dict]) -> None:
        """Append IMU samples (one JSON object per line) to imu.jsonl."""
        if not samples:
            return
        with open(self._imu_path, "a") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        self._imu_count += len(samples)

    def flush_meta(self) -> None:
        meta = {
            "scene": self.scene_dir.name,
            "num_frames": self._count,
            "depth_units": "uint16 millimetres (aligned to color)",
            "imu": {
                "recorded": self._imu_path.exists(),
                "samples_this_session": self._imu_count,
                "units": "accel m/s^2, gyro rad/s, t = device ms; streams not hw-synced to frames",
            },
            "intrinsics": self.intrinsics.to_dict(),
            "frames": self._log,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.scene_dir / "meta.json").write_text(json.dumps(meta, indent=2))
