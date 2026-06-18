"""Quick smoke test: detect the D435i, grab one aligned frame, print stats.

Run:  uv run python scripts/check_camera.py
"""

from __future__ import annotations

import numpy as np

from realsense_capture.camera import RealSenseCamera, list_devices


def main() -> int:
    devices = list_devices()
    print("Devices:", devices or "NONE")
    if not devices:
        print("No RealSense device found.")
        return 1

    with RealSenseCamera() as cam:
        print("Active profile:", cam.active_profile)
        frame = cam.wait_for_frame()
        intr = cam.color_intrinsics
        depth_m = frame.depth_mm.astype(np.float32) * frame.depth_scale
        valid = depth_m[depth_m > 0]
        print(f"color shape : {frame.color_bgr.shape} dtype={frame.color_bgr.dtype}")
        print(f"depth shape : {frame.depth_mm.shape} dtype={frame.depth_mm.dtype}")
        print(f"depth scale : {frame.depth_scale} m/unit")
        print(f"depth range : {valid.min():.3f}..{valid.max():.3f} m ({valid.size} valid px)")
        print(f"intrinsics  : {intr.width}x{intr.height} fx={intr.fx:.1f} fy={intr.fy:.1f} "
              f"cx={intr.cx:.1f} cy={intr.cy:.1f} model={intr.model}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
