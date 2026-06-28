"""Per-cluster point cloud -> a watertight visual + collision mesh.

Stage 2 of the scene -> sim pipeline. Takes one instance's coloured points (a
`clusters/<id>.ply` from `clusters.py`) and produces meshes a physics engine can
ingest. v1 uses a **convex hull**: always watertight, always a valid collision
body, and a fair proxy for the compact rigid tabletop objects we start with. The
hull doubles as the visual mesh so the slice needs only `trimesh` (no open3d).

Two refinements are wired but optional, so the first slice runs with what's
installed:

  * statistical outlier trim (drop stray in-cluster floaters before hulling) —
    pure numpy, always on;
  * CoACD convex decomposition for a multi-hull collision proxy that keeps real
    concavities (handles/spouts) — used only if `coacd` is importable and the
    object is meaningfully non-convex.

A photorealistic *visual* mesh (2DGS/PGSR surface extraction, see the survey) is a
later upgrade; the URDF stage accepts whatever visual mesh this writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ObjectMesh:
    id: int
    visual_obj: str          # path to the visual mesh (.obj)
    collision_objs: list[str]  # one or more convex collision pieces (.obj)
    volume: float            # solid volume of the (visual) mesh, in input units^3
    centroid: list[float]    # mesh centre of mass, input units
    bbox_min: list[float]
    bbox_max: list[float]
    convex: bool             # collision is a single convex hull (vs CoACD pieces)


def _load_points(ply_path: Path) -> np.ndarray:
    from plyfile import PlyData

    v = PlyData.read(str(ply_path))["vertex"]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(
        np.float64
    )


def _trim_outliers(pts: np.ndarray, quantile: float) -> np.ndarray:
    """Drop points whose distance from the median centre exceeds ``quantile``.

    Cluster boundaries bleed at object contacts, leaving a few stray Gaussians
    that would inflate the convex hull. A simple radial quantile is robust and
    dependency-free.
    """
    if quantile >= 1.0 or pts.shape[0] < 50:
        return pts
    c = np.median(pts, axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    return pts[d <= np.quantile(d, quantile)]


def _coacd_pieces(mesh, out_dir: Path, cid: int, threshold: float) -> list[str] | None:
    """Convex decomposition via CoACD, or None if unavailable/not worth it."""
    try:
        import coacd
    except ImportError:
        return None
    cmesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(cmesh, threshold=threshold)
    if len(parts) <= 1:
        return None  # essentially convex; caller keeps the single hull
    import trimesh

    paths: list[str] = []
    for i, (vs, fs) in enumerate(parts):
        piece = trimesh.Trimesh(vertices=vs, faces=fs)
        p = out_dir / f"{cid:04d}_collision_{i:02d}.obj"
        piece.export(p)
        paths.append(p.name)
    return paths


def mesh_cluster(
    ply_path: str | Path,
    out_dir: str | Path,
    cluster_id: int,
    *,
    outlier_quantile: float = 0.98,
    use_coacd: bool = True,
    coacd_threshold: float = 0.05,
) -> ObjectMesh:
    """Turn one cluster PLY into a watertight visual hull + collision proxy.

    Writes ``<id>_visual.obj`` and one or more ``<id>_collision*.obj`` under
    ``out_dir/meshes/`` and returns their paths + mass-relevant geometry.
    """
    import trimesh

    out_dir = Path(out_dir)
    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    pts = _trim_outliers(_load_points(Path(ply_path)), outlier_quantile)
    hull = trimesh.points.PointCloud(pts).convex_hull
    if not hull.is_volume:  # ensure a closed, oriented solid for inertia/physics
        hull.fix_normals()

    visual_path = meshes_dir / f"{cluster_id:04d}_visual.obj"
    hull.export(visual_path)

    collision_names: list[str] | None = None
    convex = True
    if use_coacd:
        collision_names = _coacd_pieces(hull, meshes_dir, cluster_id, coacd_threshold)
        convex = collision_names is None
    if collision_names is None:  # single convex hull collider
        coll = meshes_dir / f"{cluster_id:04d}_collision.obj"
        hull.export(coll)
        collision_names = [coll.name]

    return ObjectMesh(
        id=cluster_id,
        visual_obj=str(visual_path.relative_to(out_dir)),
        collision_objs=[f"meshes/{n}" for n in collision_names],
        volume=float(hull.volume),
        centroid=hull.center_mass.tolist(),
        bbox_min=hull.bounds[0].tolist(),
        bbox_max=hull.bounds[1].tolist(),
        convex=convex,
    )
