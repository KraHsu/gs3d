# capture/ — RealSense D435i capture GUI (Windows)

A PySide6 desktop app that records an RGB(+depth) image set from an Intel RealSense D435i,
written in the layout the `recon/` pipeline expects.

## Setup

```powershell
cd capture
uv sync
```

This creates a Python 3.11 virtualenv (pinned for `pyrealsense2` wheel compatibility) with
`pyrealsense2`, `PySide6`, `numpy`, and `opencv-python`.

> Requires the Intel RealSense D435i connected and its USB drivers installed. Verify the device
> is seen with: `uv run python -c "from realsense_capture.camera import list_devices; print(list_devices())"`

## Run

```powershell
uv run realsense-capture
# or
uv run python -m realsense_capture
```

### Using the GUI

1. **Output** folder (defaults to the repo `data/`) and a **Scene** name (e.g. `mug01`).
2. **Start stream** — opens the camera and begins live preview.
3. Capture frames while slowly orbiting the subject:
   - **Snap** button or **Spacebar** for a manual shot, or
   - tick **Auto-capture every N frames** for hands-free interval capture.
4. **Show depth** toggles a colormapped depth preview (depth is always saved regardless).
5. **Stop stream** — flushes `meta.json`.

### Capture tips

Orbit slowly, keep ~70% overlap between consecutive shots, avoid motion blur, and prefer a
textured, well-lit, static scene. 80–200 images is a good target for COLMAP to register reliably.

## Output layout

```
data/<scene>/
  images/000001.jpg ...   # color frames → COLMAP SfM input
  depth/000001.png  ...   # 16-bit depth (mm), aligned to color
  intrinsics.json         # RealSense color intrinsics (fx, fy, cx, cy, w, h, distortion)
  meta.json               # capture config + per-frame timestamps
```

Then copy the scene to the server and reconstruct (see the repo root `README.md`):

```powershell
scp -r data/<scene> H20:/mnt/cpfs/zch/3dgs/data/
```
