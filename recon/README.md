# recon/ — gsplat 3D Gaussian Splatting (Ubuntu / H20)

COLMAP Structure-from-Motion (via `pycolmap`) + [`gsplat`](https://github.com/nerfstudio-project/gsplat)
training and rendering. Targets the Ubuntu H20 server; not intended for Windows.

## Setup (one-time, on the server)

```bash
cd /mnt/cpfs/zch/3dgs/recon
bash scripts/setup_server.sh
```

This installs `uv`, pins Python 3.11, sets `CUDA_HOME=/usr/local/cuda`, runs `uv sync`
(PyTorch cu124, gsplat, pycolmap, …), and compiles gsplat's CUDA kernels on first import.

> gsplat builds CUDA extensions on first import, which needs `CUDA_HOME` + `nvcc` (toolkit 12.x).
> If `nvcc` lives elsewhere, run `export CUDA_HOME=/path/to/cuda` before the script.

## Verify without a camera

```bash
bash scripts/get_sample.sh                       # downloads tandt/truck etc.
uv run gs3d train ./samples/tandt/truck -o ../outputs/truck --max-steps 1000
uv run gs3d render ../outputs/truck
```

You should see the loss fall and PSNR rise, then `outputs/truck/point_cloud.ply`,
`orbit.mp4`, and `eval/*.png` (gt | prediction).

## Full pipeline on a captured scene

After `scp`-ing `data/<scene>` from the Windows capture machine:

```bash
uv run gs3d sfm   ../data/<scene>                # images/ → sparse/0/  (pycolmap)
uv run gs3d train ../data/<scene> -o ../outputs/<scene> --max-steps 7000
uv run gs3d render ../outputs/<scene>            # PSNR/SSIM + orbit.mp4 + eval/
# or all three:
bash scripts/run_pipeline.sh ../data/<scene>
```

## CLI

| Command | Purpose |
|---------|---------|
| `gs3d sfm <scene> [--matching exhaustive\|sequential] [--device auto\|cpu\|cuda] [--overwrite]` | Recover camera poses + sparse points |
| `gs3d train <scene> -o <out> [--max-steps N] [--sh-degree 3] [--downscale K]` | Train Gaussians (init from SfM points) |
| `gs3d render <out> [--n-frames N] [--fps 30]` | Eval held-out views + orbit video |

`--matching exhaustive` (default) is most robust for ≤ a few hundred images;
use `sequential` for long ordered captures. `--downscale 2` halves resolution for speed/VRAM.

## Modules (`gs3d/`)

- `colmap_sfm.py` — pycolmap SfM front-end (single PINHOLE camera, no undistortion).
- `dataset.py` — reads `sparse/0` into camera views (OpenCV world-to-cam) + SfM point cloud.
- `model.py` — Gaussian init, gsplat `rasterization` wrapper, SSIM/PSNR, PLY/checkpoint I/O.
- `trainer.py` — training loop reusing gsplat `DefaultStrategy` for densification.
- `render.py` — held-out evaluation + orbit fly-through.
- `cli.py` — `gs3d` entry point.

## Outputs (`outputs/<scene>/`)

```
ckpt.pt            # Gaussian params + config (for re-rendering)
point_cloud.ply    # opens in standard 3DGS viewers (SuperSplat, etc.)
orbit.mp4          # fly-through video
eval/*.png         # held-out gt | prediction comparisons
```
