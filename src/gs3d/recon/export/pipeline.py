"""End-to-end: segmented checkpoint -> per-object URDFs + scene.json.

Ties the three stages together (`clusters` -> `meshing` -> `urdf`) so a single
command turns a segmented reference-3DGS checkpoint into a folder of physics-ready
objects you can load in Genesis/PyBullet. Run on a subset first (``--ids`` or
``--max-objects``) to validate one vertical slice before fanning out to all ~470.
"""

from __future__ import annotations

from pathlib import Path

from .clusters import export_clusters
from .meshing import mesh_cluster
from .urdf import build_scene, write_object_urdf


def export_sim(
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    ids: list[int] | None = None,
    max_objects: int | None = None,
    scale: float = 1.0,
    density: float = 300.0,
    min_points: int = 200,
    use_coacd: bool = True,
    include_static: bool = False,
) -> Path:
    """Split -> mesh -> URDF for selected clusters; write ``scene.json``.

    Args:
        ids: explicit instance ids to export. Default: all kept, non-static
            clusters (the manipulable objects), largest first.
        max_objects: cap the number of objects (after id selection / ordering).
        scale: COLMAP units per metre (see `clusters.export_clusters`).
        density: per-object density (kg/m^3) for mass/inertia.
        use_coacd: allow CoACD multi-hull collision when available.
        include_static: also mesh clusters flagged static (table/background).
    """
    out_dir = Path(out_dir)

    infos = export_clusters(
        checkpoint, out_dir, scale=scale, min_points=min_points, write_ply=True
    )
    by_id = {i.id: i for i in infos}

    if ids is None:
        cand = [i for i in infos if i.kept and (include_static or not i.static)]
        cand.sort(key=lambda i: -i.num_gaussians)
        ids = [i.id for i in cand]
    else:
        missing = [c for c in ids if c not in by_id or not by_id[c].kept]
        if missing:
            print(f"[export-sim] WARNING: requested ids absent/filtered, skipping: {missing}")
        ids = [c for c in ids if c in by_id and by_id[c].kept]
    if max_objects is not None:
        ids = ids[:max_objects]

    print(f"[export-sim] meshing {len(ids)} objects: {ids}")
    entries = []
    for cid in ids:
        ply = out_dir / "clusters" / f"{cid:04d}.ply"
        om = mesh_cluster(ply, out_dir, cid, use_coacd=use_coacd)
        entry = write_object_urdf(om, out_dir, density=density)
        entry["static"] = by_id[cid].static
        entries.append(entry)
        print(f"[export-sim]   id {cid}: mass={entry['mass']:.4g} "
              f"vol={om.volume:.4g} {'(CoACD)' if not om.convex else '(convex)'}")

    scene_path = build_scene(entries, out_dir, density=density, scale=scale)
    print(f"[export-sim] wrote {scene_path} ({len(entries)} objects)")
    return scene_path
