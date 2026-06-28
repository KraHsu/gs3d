"""Split a segmented reference-3DGS checkpoint into per-instance point subsets.

This is the first, simulator-agnostic stage of the scene -> sim pipeline. It reads
the per-Gaussian instance ids (`gaussians._cluster_indices`) and writes, per kept
instance, a small coloured point-cloud PLY plus a `manifest.json` describing every
cluster (count, bounding box, centroid, extent, static/kept flags). You inspect the
PLYs (e.g. in `gs3d view-seg` or MeshLab) before paying for meshing downstream.

Why points and not the exported `point_cloud.ply`: the instance ids live only in
the checkpoint and index its Gaussians 1:1, while the exported PLY may have a
different Gaussian count. So we colour each Gaussian by its degree-0 SH and split
*the checkpoint's* means — geometry and ids always align.

Scale: COLMAP/3DGS geometry is scale-free. Pass ``--scale`` (COLMAP units per
metre, e.g. from `depth_init.estimate_colmap_scale`) to emit metric coordinates;
mass/friction/grasping downstream all depend on getting this right.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Degree-0 spherical-harmonics constant: rgb = C0 * sh0 + 0.5  (reference-3DGS).
_SH_C0 = 0.28209479177387814


@dataclass
class ClusterInfo:
    id: int
    num_gaussians: int
    kept: bool                  # passed the min-point filter (else floater noise)
    static: bool                # heuristic: table / background -> fixed base in sim
    centroid: list[float]       # median xyz (metric if --scale given)
    bbox_min: list[float]
    bbox_max: list[float]
    extent: float               # bbox diagonal length
    ply: str | None             # per-cluster PLY path relative to out_dir, or None


def _sh0_to_rgb(sh0: np.ndarray) -> np.ndarray:
    """(N,1,3) or (N,3) degree-0 SH -> (N,3) uint8 RGB."""
    dc = sh0.reshape(sh0.shape[0], -1)[:, :3]
    rgb = np.clip(_SH_C0 * dc + 0.5, 0.0, 1.0)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    verts = np.empty(
        xyz.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")]).write(str(path))


def export_clusters(
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    scale: float = 1.0,
    min_points: int = 200,
    min_opacity: float = 0.0,
    background_id: int = 0,
    write_ply: bool = True,
    device: str = "cpu",
) -> list[ClusterInfo]:
    """Split ``checkpoint`` by instance id into ``out_dir/clusters/<id>.ply`` + manifest.

    Args:
        scale: COLMAP units per metre; coordinates are divided by it (1.0 = leave
            as-is). Get it from `depth_init.estimate_colmap_scale`.
        min_points: clusters with fewer Gaussians are dropped as floater noise
            (``kept=False``); still listed in the manifest.
        min_opacity: drop Gaussians whose (sigmoid) opacity is below this before
            counting/splitting (0.0 = keep all).
        background_id: this instance id is flagged ``static`` (usually the
            table/floor). The single largest-extent kept cluster is also flagged.
        write_ply: write per-cluster PLYs (set False to only refresh the manifest).
    """
    import torch

    from ..inria import load_inria_checkpoint

    out_dir = Path(out_dir)
    clusters_dir = out_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    model = load_inria_checkpoint(checkpoint, device=device)
    if model.cluster_indices is None:
        raise ValueError(
            f"{checkpoint}: no per-Gaussian instance ids (_cluster_indices); "
            "cannot split by instance. Re-export from the segmentation stage."
        )

    means = model.splats["means"].detach().cpu().numpy().astype(np.float32)
    rgb = _sh0_to_rgb(model.splats["sh0"].detach().cpu().numpy().astype(np.float32))
    ids = model.cluster_indices.detach().cpu().numpy().astype(np.int64)

    if min_opacity > 0.0:
        op = torch.sigmoid(model.splats["opacities"].detach().cpu()).numpy()
        keep = op >= min_opacity
        means, rgb, ids = means[keep], rgb[keep], ids[keep]

    if scale != 1.0:
        means = means / float(scale)  # COLMAP units -> metres

    infos: list[ClusterInfo] = []
    extents: dict[int, float] = {}
    for cid in np.unique(ids):
        m = ids == cid
        pts = means[m]
        n = int(pts.shape[0])
        bmin, bmax = pts.min(axis=0), pts.max(axis=0)
        extent = float(np.linalg.norm(bmax - bmin))
        kept = n >= min_points
        if kept:
            extents[int(cid)] = extent
        ply_rel: str | None = None
        if kept and write_ply:
            ply_path = clusters_dir / f"{int(cid):04d}.ply"
            _write_points_ply(ply_path, pts, rgb[m])
            ply_rel = str(ply_path.relative_to(out_dir))
        infos.append(
            ClusterInfo(
                id=int(cid),
                num_gaussians=n,
                kept=kept,
                static=False,  # set below once we know the largest kept cluster
                centroid=np.median(pts, axis=0).tolist(),
                bbox_min=bmin.tolist(),
                bbox_max=bmax.tolist(),
                extent=extent,
                ply=ply_rel,
            )
        )

    # Static heuristic: the background id + the single largest kept cluster
    # (table/floor span the scene). User can edit `static` in the manifest.
    largest = max(extents, key=lambda k: extents[k]) if extents else None
    for info in infos:
        if info.kept and (info.id == background_id or info.id == largest):
            info.static = True

    manifest = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "scale_colmap_per_metre": float(scale),
        "metric": scale != 1.0,
        "num_clusters": len(infos),
        "num_kept": sum(i.kept for i in infos),
        "min_points": min_points,
        "background_id": background_id,
        "clusters": [asdict(i) for i in infos],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    kept = [i for i in infos if i.kept]
    static_ids = [i.id for i in kept if i.static]
    print(
        f"[export-clusters] {len(infos)} clusters, {len(kept)} kept "
        f"(>= {min_points} pts), {len(infos) - len(kept)} dropped as floaters"
    )
    print(f"[export-clusters] static (table/background): {static_ids}")
    if scale == 1.0:
        print("[export-clusters] WARNING: scale=1.0 (non-metric). Pass --scale for "
              "real-world units (mass/friction depend on it).")
    print(f"[export-clusters] wrote {out_dir/'manifest.json'} and clusters/*.ply")
    return infos
