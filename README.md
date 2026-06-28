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
Ubuntu :  uv sync --extra recon        # torch (cu128) + gsplat + pycolmap
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
uv run gs3d capture               # launch the dataset recorder GUI
```

In the GUI: set an output folder + **Dataset** name, click **Start camera**, then press
**● Record** (or `R`) and orbit the subject slowly for a loop or two — frames are captured
continuously (one every *N* previewed frames; default 3). Press **■ Stop** to finish; **Snap**
(Spacebar) takes single shots. Aim for ~100–300 frames with ~70% overlap. Tick **Capture IMU**
(before Start camera) to also log the D435i accel+gyro. Output → `data/<dataset>/`:

To reduce **motion blur** (the main cause of soft frames): uncheck **Auto exposure** and lower the
**Exposure** slider (raise **Gain** / add light to compensate). Tick **Sharpness gate** to auto-skip
frames blurrier than the threshold while recording — the live `sharp NN` readout helps you pick one.


```
images/000001.jpg ...   # color frames → COLMAP SfM input
depth/000001.png  ...   # 16-bit depth (mm), aligned to color
intrinsics.json         # color intrinsics (fx, fy, cx, cy, w, h, distortion)
meta.json               # capture config + per-frame timestamps
imu.jsonl               # optional: {stream:accel|gyro, t, x, y, z} per line
```

> The COLMAP→gsplat pipeline uses only `images/` (poses come from SfM). `depth/` and `imu.jsonl`
> are recorded for later use (e.g. point-cloud init, gravity alignment / VIO) and are optional.

> Needs the D435i + Intel RealSense USB drivers. Stream resolution auto-falls-back to the USB
> link's capability (USB 3 → 1280×720@30; USB 2.1 → 640×480@15). Use a USB 3 port/cable for best quality.
>
> **Troubleshooting "Frame didn't arrive":** the camera streams nothing in every mode → the
> D435i is wedged (often after an unclean exit) or on a marginal USB 2.1 link. **Physically
> unplug and replug** it, preferably into a **USB 3 port** with a good cable, then retry
> `gs3d check-camera`. Verify the link with `check-camera`'s `USB3.x`/`USB2.1` prefix.

### Optional — curate the capture first (recommended for long/blurry recordings)

Drop blurry frames and cap the count for faster, more robust SfM (sequence is split into
`--max-frames` windows; the sharpest frame per window is kept):

```powershell
uv run python scripts/curate_dataset.py data/<scene> -o data/<scene>_sharp --max-frames 280
```

Then sync/reconstruct the `_sharp` scene.

## Subpart 2 — Reconstruct (Ubuntu / H20)

After syncing code + data to the server (see **Sync** below):

```bash
bash scripts/setup_server.sh                      # uv, CUDA_HOME, uv sync --extra recon, build gsplat
uv run gs3d sfm    data/<scene> --matching sequential --device cpu   # pycolmap → sparse/0
uv run gs3d train  data/<scene> -o outputs/<scene>
uv run gs3d render outputs/<scene>                # eval PSNR/SSIM + orbit.mp4 + point_cloud.ply
# or all three:
bash scripts/run_pipeline.sh data/<scene>
```

> On a headless server use `--matching sequential --device cpu` for SfM (GPU SIFT needs a
> display). A real 280-frame capture reconstructs in ~1–2 min SfM + ~90 s training on an H20.

Verify the pipeline **without the camera** first:

```bash
bash scripts/get_sample.sh                        # downloads tandt/truck etc.
uv run gs3d train ./samples/tandt/truck -o outputs/truck --max-steps 1000
uv run gs3d render outputs/truck
```

### Interactive viewers

```bash
uv run gs3d view     outputs/<scene>              # a gs3d ckpt.pt (learned RGB)
uv run gs3d view-seg <checkpoint.pth>             # a segmented reference-3DGS model
```

Both render on the server GPU and stream to a browser (tunnel with
`ssh -L 8080:localhost:8080 <server>`, then open `http://localhost:8080`).

`view-seg` loads a **reference-3DGS / INRIA gaussian-splatting** checkpoint that
carries a per-Gaussian instance id (`gaussians._cluster_indices`) — e.g. the
output of our semantic-segmentation stage — and adds a GUI panel to toggle
between learned **RGB** and per-instance **Segmentation** colours, **isolate** a
single instance id (others dimmed), and gray-out the background cluster. Geometry
and instance ids come from the *same* checkpoint tensors, so they always align
(the exported `point_cloud.ply` may have a different Gaussian count and is not
used for segmentation). It accepts a `.pth` checkpoint or a reference `.ply`
(RGB only — a PLY has no instance ids).

### Export to a physics simulator (segmentation → Genesis)

Turn a **segmented** reference-3DGS checkpoint (one carrying per-Gaussian instance
ids, `gaussians._cluster_indices`) into per-object collision assets you can drop
into a physics engine. The pipeline is three inspectable stages
(`src/gs3d/recon/export/`): split by instance → convex-hull mesh + collision
proxy → URDF + `scene.json`. See `docs/3dgs-to-physics-sim-survey.md` for the
method landscape and why this is the pragmatic "mesh handles physics" route.

```bash
uv sync --extra recon --extra sim        # adds trimesh + coacd (light, lock-stable)

# 1. inspect the instance split first (per-cluster point PLYs + manifest.json)
uv run gs3d export-clusters <checkpoint.pth> -o export/<scene>

# 2. full export: split → mesh → URDF → scene.json (start with a few objects)
uv run gs3d export-sim <checkpoint.pth> -o export/<scene> --ids 26 4
#   --scale <colmap-units-per-metre>   metric coords (mass/friction depend on it)
#   --density <kg/m^3>                 per-object density (default 300)
#   --max-objects N / --include-static / --no-coacd

# 3. simulate: load the URDFs in Genesis and drop them onto a ground plane
uv pip install genesis-world             # heavy, version-strict — not in the lock
uv run gs3d sim-genesis export/<scene> --backend cpu --record drop.mp4
#   --layout layout   reproduce captured world poses (else 'drop' = grid above plane)

# 4. photorealistic: physics on the hulls, but render the REAL Gaussians (survey §4 / C)
uv run gs3d sim-render export/<scene> --checkpoint <checkpoint.pth> --record photoreal.mp4
#   --aspect-max / --scale-quantile / --opacity-min  tune the anti-spike filter
```

`export-clusters` filters floater clusters (`--min-points`) and flags the
table/background as static. `export-sim` writes a self-contained `urdf/` (meshes
copied alongside) plus `scene.json` with each object's world pose, mass and
static flag. The URDFs are standard, so the same scene also loads in
**PyBullet / Isaac**; `sim-genesis` is just the reference loader.

> **Scale matters.** 3DGS geometry is scale-free. Without `--scale`, coordinates
> stay in COLMAP units and masses are not real-world (physics still *runs*, sizes
> are just arbitrary). Recover the factor from the D435i depth
> (`depth_init.estimate_colmap_scale`) for metric assets.
>
> **Convex hulls are unrecognisable — use `sim-render` to see the real objects.**
> The hull is fine as a *collision* proxy but a poor *visual*. `sim-render` is the
> survey's family-C route: physics still runs on the hulls in Genesis, but each
> object's per-frame rigid pose drives *its own Gaussians*, which are rasterised
> with gsplat — so the video shows the captured appearance moving under physics,
> no retraining needed. It filters needle/floater Gaussians (which streak into
> spikes once rotated) via `--aspect-max` / `--scale-quantile` / `--opacity-min`.
>
> **Still approximate.** Collision is a convex hull (CoACD splits non-convex
> objects when it helps). A textured *visual mesh* (2DGS/PGSR surface extraction)
> and VLM-estimated physical parameters are the orthogonal next upgrades — see the
> survey's §4 and Appendix B.

### Blackwell / RTX 50-series GPUs (sm_120)

The recon extra pins **torch cu128**, which ships sm_120 kernels (it still runs
on the H20). gsplat has no cu128 prebuilt wheel, so it JIT-compiles its CUDA
kernels and needs a CUDA **≥ 12.8** toolkit. If you don't have one (no root, no
system CUDA), vendor a self-contained compiler once — no sudo:

```bash
bash scripts/setup_cuda_jit.sh        # → ./.cuda-jit/cuda128 (nvcc 12.8, via micromamba)
```

The recon commands auto-detect this prefix (an existing `nvcc` on `CUDA_HOME`/
`PATH` takes precedence, e.g. on the H20). Note: gsplat's JIT also needs Python
headers — use a **uv-managed** Python (`uv python pin 3.11`), not a bare system
Python without `python3.x-dev`.

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
