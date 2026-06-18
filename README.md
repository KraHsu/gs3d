# 3DGS: RealSense D435i → 3D Gaussian Splatting

A basic but complete 3D Gaussian Splatting (3DGS) pipeline, split across two machines by
hardware reality:

| Stage | Where | What |
|-------|-------|------|
| **1. Capture** | Windows (this PC, D435i attached) | PySide6 GUI records an RGB(+depth) image set of a scene |
| **2. Reconstruct** | Ubuntu server (`ssh H20`, 8× H20 GPU) | COLMAP SfM (via `pycolmap`) + [`gsplat`](https://github.com/nerfstudio-project/gsplat) training |

The two stages are **separate `uv` projects** (`capture/` and `recon/`) because they target
different OSes and have disjoint dependencies. Code moves to the server via **git**; the captured
dataset moves via **scp**.

```
capture (Windows)  ──scp data/<scene>──►  recon (Ubuntu/H20)
   PySide6 GUI                              pycolmap SfM → gsplat train → render
```

---

## Quick start

### Stage 1 — Capture (Windows)

```powershell
cd capture
uv sync
uv run realsense-capture        # or: uv run python -m realsense_capture
```

In the GUI: pick an output folder, set a **scene name**, click **Start stream**, then orbit the
object slowly while using **Snap** (Spacebar) or **Auto-capture every N frames**. Aim for
80–200 images with ~70% overlap. Output lands in `data/<scene>/` with `images/`, `depth/`,
`intrinsics.json`, `meta.json`.

See [`capture/README.md`](capture/README.md) for details.

### Stage 2 — Reconstruct (Ubuntu / H20)

First, sync code and data to the server (see **Sync** below), then:

```bash
cd /mnt/cpfs/zch/3dgs/recon
bash scripts/setup_server.sh                 # installs uv, sets CUDA_HOME, builds gsplat
uv run gs3d sfm   ../data/<scene>            # pycolmap → sparse/0
uv run gs3d train ../data/<scene> -o ../outputs/<scene>
uv run gs3d render ../outputs/<scene>        # orbit mp4 + point_cloud.ply + PSNR
```

To verify the pipeline **without the camera**, download a small public scene first:

```bash
bash scripts/get_sample.sh                   # → recon/samples/<scene>
uv run gs3d train ./samples/<scene> -o ../outputs/sample --max-steps 1000
```

See [`recon/README.md`](recon/README.md) for details.

---

## Sync between machines

**Code (git).** A bare repo on the server avoids needing a third-party host:

```powershell
# one-time, from Windows repo root
ssh H20 "git init --bare /mnt/cpfs/zch/3dgs.git"
git remote add origin "ssh://H20/mnt/cpfs/zch/3dgs.git"
git push -u origin main
# on the server, one-time
ssh H20 "git clone /mnt/cpfs/zch/3dgs.git /mnt/cpfs/zch/3dgs"
```

Thereafter: `git push` from Windows, `git -C /mnt/cpfs/zch/3dgs pull` on the server.
(Alternatively use GitHub/GitLab as `origin`.)

**Dataset (scp).** Captured scenes are gitignored; copy them directly:

```powershell
scp -r data/<scene> H20:/mnt/cpfs/zch/3dgs/data/
```

---

## Requirements

- **Windows side:** [uv](https://docs.astral.sh/uv/), an Intel RealSense D435i, RealSense USB drivers.
- **Server side:** NVIDIA GPU + driver, CUDA toolkit at `/usr/local/cuda` (12.x), git. `uv` and the
  rest are installed by `recon/scripts/setup_server.sh`.

## Layout

```
capture/   Windows PySide6 capture app (uv project)
recon/     Ubuntu gsplat reconstruction (uv project)
data/      captured datasets (gitignored; scp'd to server)
```
