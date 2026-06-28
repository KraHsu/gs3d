"""Photorealistic render of the *real* Gaussians, driven by physics (survey §4, family C).

The exported collision hulls are unrecognisable on their own. This is the
"GS-as-renderer" route (SplatSim-style): physics runs in Genesis on the cheap
convex proxies, and every frame each object's rigid pose drives *its own
Gaussians*, which are then rasterised with gsplat. You get the captured
appearance — you can tell what each object is — moving under real physics.

Per object `o` with instance id, COM ``c`` (the URDF link frame) and export scale
``s`` (COLMAP units per metre), a Gaussian at COLMAP position ``x`` renders at::

    x_world = p_t + R(q_t) @ (x / s - c)          # p_t, q_t = Genesis base pose
    quat    = q_t ⊗ q_g                            # rotate the Gaussian too
    scale   = exp(raw) / s                         # geometry shrinks with the frame

Only the exported (dynamic) objects are rendered, on a neutral background, so the
recognisable objects are isolated from the unaligned static scene. A textured
visual surface (2DGS/PGSR) is the orthogonal upgrade; this one needs no retraining.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(...,4) wxyz unit quaternion -> (...,3,3) rotation matrix."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions; broadcasts a (4,) over b (N,4)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV camera-to-world (x right, y down, +z forward) looking at target."""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)  # y points down in OpenCV
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, d, f, eye
    return c2w


def _simulate_poses(export_dir: Path, objects, steps, drop_height, spacing, backend):
    """Run Genesis on the dynamic objects, return per-frame (pos, quat) arrays.

    Returns ``(positions, quats)`` shaped ``[T, O, 3]`` and ``[T, O, 4]`` (wxyz),
    aligned with ``objects`` order.
    """
    import genesis as gs

    gs.init(backend=getattr(gs, backend))
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())

    cols = max(1, int(math.ceil(math.sqrt(len(objects)))))
    ents = []
    for i, obj in enumerate(objects):
        urdf = str((export_dir / obj["urdf"]).resolve())
        r, c = divmod(i, cols)
        pos = (c * spacing, r * spacing, drop_height + i * 0.02)
        ents.append(scene.add_entity(gs.morphs.URDF(file=urdf, pos=pos, fixed=False)))
    scene.build()

    pos_t, quat_t = [], []
    for _ in range(steps):
        scene.step()
        pos_t.append([np.asarray(e.get_pos().cpu(), dtype=np.float32) for e in ents])
        quat_t.append([np.asarray(e.get_quat().cpu(), dtype=np.float32) for e in ents])
    return np.asarray(pos_t), np.asarray(quat_t)


def sim_render(
    export_dir: str | Path,
    checkpoint: str | Path,
    record: str | Path,
    *,
    steps: int = 250,
    fps: int = 60,
    width: int = 960,
    height: int = 720,
    drop_height: float = 0.3,
    spacing: float = 0.4,
    backend: str = "gpu",
    bg: float = 0.0,
    opacity_min: float = 0.1,
    radius_quantile: float = 0.98,
    scale_quantile: float = 0.97,
    aspect_max: float = 18.0,
    device: str = "cuda",
) -> None:
    """Simulate the exported objects and render their real Gaussians to ``record`` (mp4)."""
    import imageio.v2 as imageio

    from .._cuda import ensure_cuda_toolkit
    from ..inria import load_inria_checkpoint
    from ..model import rasterize_splats

    ensure_cuda_toolkit()
    export_dir = Path(export_dir)
    scene_spec = json.loads((export_dir / "scene.json").read_text())
    objects = scene_spec["objects"]
    s = float(scene_spec.get("scale_colmap_per_metre", 1.0))

    model = load_inria_checkpoint(checkpoint, device=device)
    if model.cluster_indices is None:
        raise ValueError(f"{checkpoint}: no _cluster_indices; cannot map physics to Gaussians")
    ids = model.cluster_indices
    sp = model.splats
    sh_degree = model.sh_degree

    # Pre-slice each object's Gaussians into its COM-local, metric frame (matches
    # the URDF link frame the physics poses are expressed in). Filter the floaters
    # and oversized/elongated Gaussians that otherwise streak into long spikes.
    log_inv_s = math.log(1.0 / s)
    per_obj = []
    for obj in objects:
        m = ids == int(obj["id"])
        com = torch.tensor(obj["world_pos"], dtype=torch.float32, device=device)
        local = sp["means"][m] / s - com                 # [n,3]
        scales = sp["scales"][m] + log_inv_s             # raw log, shrunk by 1/s
        opac = sp["opacities"][m]
        # keep: opaque enough, near the COM, not a giant blob, not a needle.
        # Needle (high-aspect) Gaussians look thin head-on at the training view but
        # streak into long spikes once the object is rotated, so cap the aspect ratio.
        dist = local.norm(dim=-1)
        sz = torch.exp(scales)
        biggest = sz.max(dim=-1).values
        aspect = biggest / sz.min(dim=-1).values.clamp_min(1e-8)
        keep = torch.sigmoid(opac) >= opacity_min
        keep &= dist <= torch.quantile(dist, radius_quantile)
        keep &= biggest <= torch.quantile(biggest, scale_quantile)
        keep &= aspect <= aspect_max
        per_obj.append({
            "local": local[keep],
            "quats": sp["quats"][m][keep],
            "scales": scales[keep],
            "opac": opac[keep],
            "sh0": sp["sh0"][m][keep],
            "shN": sp["shN"][m][keep],
        })
    print(f"[sim-render] {len(objects)} objects, "
          f"{sum(o['local'].shape[0] for o in per_obj)} gaussians; simulating {steps} steps...")

    positions, quats = _simulate_poses(export_dir, objects, steps, drop_height, spacing, backend)

    # Fixed camera framing the union of first/last object positions.
    sample = positions[[0, -1]].reshape(-1, 3)
    center = sample.mean(axis=0)
    radius = float(np.linalg.norm(sample - center, axis=1).max()) + 0.3
    eye = center + np.array([1.0, 1.0, 0.7]) / np.linalg.norm([1.0, 1.0, 0.7]) * radius * 2.6
    c2w = torch.tensor(_look_at(eye, center, np.array([0.0, 0.0, 1.0])), device=device)
    viewmat = torch.linalg.inv(c2w)[None]
    fx = fy = 0.5 * width / math.tan(0.5 * math.radians(50.0))
    K = torch.tensor([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]],
                     dtype=torch.float32, device=device)[None]

    writer = imageio.get_writer(str(record), fps=fps, macro_block_size=1)
    with torch.no_grad():
        for t in range(steps):
            means, qs, scales, opac, sh0, shN = [], [], [], [], [], []
            for o, obj in enumerate(per_obj):
                p = torch.tensor(positions[t, o], dtype=torch.float32, device=device)
                q = torch.tensor(quats[t, o], dtype=torch.float32, device=device)
                R = _quat_to_rotmat(q)
                means.append(obj["local"] @ R.T + p)
                qs.append(_quat_mul(q, obj["quats"]))
                scales.append(obj["scales"]); opac.append(obj["opac"])
                sh0.append(obj["sh0"]); shN.append(obj["shN"])
            splats = {
                "means": torch.cat(means), "quats": torch.cat(qs),
                "scales": torch.cat(scales), "opacities": torch.cat(opac),
                "sh0": torch.cat(sh0), "shN": torch.cat(shN),
            }
            colors, alphas, _ = rasterize_splats(splats, viewmat, K, width, height, sh_degree)
            rgb = colors[0, ..., :3] + (1.0 - alphas[0]) * bg  # composite over grey bg
            img = (rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            writer.append_data(img)
    writer.close()
    print(f"[sim-render] wrote {record} ({steps} frames @ {fps} fps)")
