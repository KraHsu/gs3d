"""Per-object mesh + physics -> URDF, and a scene manifest tying them together.

Stage 3 of the scene -> sim pipeline. For each object mesh from `meshing.py` we
write a single-link URDF whose:

  * **visual** is the full mesh (later: the photorealistic 2DGS/PGSR surface),
  * **collision** is the convex hull (or CoACD pieces),
  * **inertial** is mass = density x volume with the inertia tensor from the mesh,
    placed at the centre of mass.

Meshes are re-exported **centred on their centre of mass** so the URDF link frame
is the COM (clean inertia, clean spawn pose). The object's world location is
recorded in ``scene.json`` as the spawn pose, so the loader can either reproduce
the captured layout or just drop the objects onto a ground plane.

Density is a single per-object scalar (default a light-plastic ~300 kg/m^3); this
is the "reasonable, not calibrated" value the survey flags — refine per object
with GaussianProperty/VLM later. Mass is only meaningful if the meshes are metric
(export with the right `--scale`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6g}"/>
      <inertia ixx="{ixx:.6g}" ixy="{ixy:.6g}" ixz="{ixz:.6g}" iyy="{iyy:.6g}" iyz="{iyz:.6g}" izz="{izz:.6g}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{visual}" scale="1 1 1"/></geometry>
    </visual>
{collisions}
  </link>
</robot>
"""

_COLLISION_BLOCK = """    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{path}" scale="1 1 1"/></geometry>
    </collision>"""


def _center_and_export(src: Path, dst: Path, com: np.ndarray):
    import trimesh

    m = trimesh.load(src, process=False)
    m.apply_translation(-com)
    m.export(dst)
    return m


def write_object_urdf(
    object_mesh,
    out_dir: str | Path,
    *,
    density: float = 300.0,
) -> dict:
    """Write ``<id>.urdf`` (+ centred meshes) for one object; return its scene entry.

    ``object_mesh`` is an `ObjectMesh` from `meshing.mesh_cluster`. Returns a dict
    with the URDF path, world spawn pose (the object's COM), mass and static flag
    placeholder — collected by `build_scene` into ``scene.json``.
    """
    out_dir = Path(out_dir)
    urdf_dir = out_dir / "urdf"
    cmesh_dir = urdf_dir / "meshes"
    cmesh_dir.mkdir(parents=True, exist_ok=True)

    cid = object_mesh.id
    com = np.asarray(object_mesh.centroid, dtype=np.float64)  # mesh COM in world units

    # Re-export every mesh centred on the COM so the link frame == COM.
    vis_src = out_dir / object_mesh.visual_obj
    vis_dst = cmesh_dir / f"{cid:04d}_visual.obj"
    vmesh = _center_and_export(vis_src, vis_dst, com)
    vmesh.density = density  # type: ignore[attr-defined]
    mass = float(vmesh.mass)
    it = vmesh.moment_inertia  # 3x3 about COM, density-scaled

    coll_rel = []
    for c in object_mesh.collision_objs:
        src = out_dir / c
        dst = cmesh_dir / Path(c).name
        _center_and_export(src, dst, com)
        coll_rel.append(f"meshes/{dst.name}")

    collisions = "\n".join(_COLLISION_BLOCK.format(path=p) for p in coll_rel)
    urdf = _URDF_TEMPLATE.format(
        name=f"obj_{cid:04d}",
        mass=max(mass, 1e-4),
        ixx=it[0, 0], ixy=it[0, 1], ixz=it[0, 2],
        iyy=it[1, 1], iyz=it[1, 2], izz=it[2, 2],
        visual=f"meshes/{vis_dst.name}",
        collisions=collisions,
    )
    urdf_path = urdf_dir / f"{cid:04d}.urdf"
    urdf_path.write_text(urdf)

    return {
        "id": cid,
        "urdf": str(urdf_path.relative_to(out_dir)),
        "world_pos": com.tolist(),  # spawn at captured location (COLMAP/world frame)
        "mass": mass,
        "volume": object_mesh.volume,
        "convex": object_mesh.convex,
    }


def build_scene(
    entries: list[dict],
    out_dir: str | Path,
    *,
    density: float,
    scale: float,
) -> Path:
    """Collect per-object entries into ``scene.json`` and return its path."""
    out_dir = Path(out_dir)
    scene = {
        "density_kg_m3": density,
        "scale_colmap_per_metre": scale,
        "metric": scale != 1.0,
        "up_axis_world": "-y",  # COLMAP/OpenCV world is y-down; sim loader re-aligns
        "objects": entries,
    }
    p = out_dir / "scene.json"
    p.write_text(json.dumps(scene, indent=2))
    return p
