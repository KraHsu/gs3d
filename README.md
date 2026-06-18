# gs3d — RealSense D435i → 3D Gaussian Splatting

A single Python project (`uv` + `pyproject.toml`) with **two subparts**:

| Subpart | Package | Where it runs | What it does |
|---------|---------|---------------|--------------|
| **Data acquisition** | `gs3d.capture` | Windows (D435i attached) | PySide6 GUI records an RGB(+depth) image set |
| **3DGS reconstruction** | `gs3d.recon` | Ubuntu + NVIDIA GPU | COLMAP SfM (`pycolmap`) + [`gsplat`](https://github.com/nerfstudio-project/gsplat) training/render |

One codebase, one CLI (`gs3d`). The two subparts target different OSes with disjoint
heavy dependencies, so they are split into **optional-dependency extras** with platform
markers — each machine installs only the subset it needs:

```
Windows:  uv sync --extra capture      # pyrealsense2 + PySide6
Ubuntu :  uv sync --extra recon        # torch (cu124) + gsplat + pycolmap
```

Code moves between machines via **git**; captured datasets move via **scp**.

```
capture (Windows)  ──scp data/<scene>──►  recon (Ubuntu/H20)
  gs3d capture                              gs3d sfm → gs3d train → gs3d render
```

## Layout

```
src/gs3d/
  capture/   camera.py  writer.py  gui.py  app.py  check.py     # subpart 1
  recon/     colmap_sfm.py  dataset.py  model.py  trainer.py  render.py   # subpart 2
  cli.py     # unified entry point: gs3d capture | check-camera | sfm | train | render
scripts/     setup_server.sh  get_sample.sh  run_pipeline.sh
data/        captured datasets (gitignored; scp'd to the server)
```

---

## Subpart 1 — Capture (Windows)

```powershell
uv sync --extra capture
uv run gs3d check-camera          # verify the D435i is detected
uv run gs3d capture               # launch the GUI
```

In the GUI: pick an output folder + **scene name**, **Start stream**, then orbit the subject
slowly using **Snap** (Spacebar) or **Auto-capture every N frames**. Aim for 80–200 images
with ~70% overlap. Output → `data/<scene>/` with `images/`, `depth/`, `intrinsics.json`,
`meta.json`.

> Needs the D435i + Intel RealSense USB drivers. Stream resolution auto-falls-back to the USB
> link's capability (USB 3 → 1280×720@30; USB 2.1 → 640×480@15). Use a USB 3 port/cable for best quality.
>
> **Troubleshooting "Frame didn't arrive":** the camera streams nothing in every mode → the
> D435i is wedged (often after an unclean exit) or on a marginal USB 2.1 link. **Physically
> unplug and replug** it, preferably into a **USB 3 port** with a good cable, then retry
> `gs3d check-camera`. Verify the link with `check-camera`'s `USB3.x`/`USB2.1` prefix.

## Subpart 2 — Reconstruct (Ubuntu / H20)

After syncing code + data to the server (see **Sync** below):

```bash
bash scripts/setup_server.sh                      # uv, CUDA_HOME, uv sync --extra recon, build gsplat
uv run gs3d sfm    data/<scene>                   # pycolmap → sparse/0
uv run gs3d train  data/<scene> -o outputs/<scene>
uv run gs3d render outputs/<scene>                # eval PSNR/SSIM + orbit.mp4 + point_cloud.ply
# or all three:
bash scripts/run_pipeline.sh data/<scene>
```

Verify the pipeline **without the camera** first:

```bash
bash scripts/get_sample.sh                        # downloads tandt/truck etc.
uv run gs3d train ./samples/tandt/truck -o outputs/truck --max-steps 1000
uv run gs3d render outputs/truck
```

---

## Sync between machines

**Code (git).** A bare repo on the server (no third-party host needed):

```powershell
ssh H20 "git init --bare /mnt/cpfs/zch/3dgs.git"
git remote add origin "ssh://H20/mnt/cpfs/zch/3dgs.git"
git push -u origin main
ssh H20 "git clone /mnt/cpfs/zch/3dgs.git /mnt/cpfs/zch/3dgs"
```

Thereafter `git push` from Windows, `git -C /mnt/cpfs/zch/3dgs pull` on the server.

**Dataset (scp):**

```powershell
scp -r data/<scene> H20:/mnt/cpfs/zch/3dgs/data/
```

## Outputs (`outputs/<scene>/`)

```
ckpt.pt            # Gaussian params + config (for re-rendering)
point_cloud.ply    # opens in standard 3DGS viewers (SuperSplat, etc.)
orbit.mp4          # fly-through video
eval/*.png         # held-out gt | prediction comparisons
```
