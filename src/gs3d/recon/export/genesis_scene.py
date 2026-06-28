"""Load an exported scene (``scene.json`` + URDFs) into Genesis and simulate.

Stage 4 — the first end-to-end check that the exported geometry behaves as a
physics scene. Adds a ground plane and each object as a rigid body, then steps
gravity. Two layouts:

  * ``drop`` (default): spawn objects in a grid a little above the plane and let
    them fall — proves collision/contact/inertia without needing the real table
    pose. Best first slice.
  * ``layout``: place each object at its captured world position, rotated from the
    COLMAP (y-down) frame into Genesis' z-up frame — reproduces the scene (the GS
    <-> sim alignment the survey flags as a recurring footgun).

Genesis is an optional, heavy dependency (``uv pip install genesis-world``); it is
imported lazily so the rest of the export pipeline runs without it. The URDFs are
standard, so the same scene also loads in PyBullet/Isaac if Genesis is unavailable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_scene(
    export_dir: str | Path,
    *,
    layout: str = "drop",
    drop_height: float = 0.3,
    spacing: float = 0.4,
    steps: int = 240,
    show_viewer: bool = False,
    record: str | Path | None = None,
    backend: str = "gpu",
) -> None:
    """Load ``export_dir/scene.json`` into Genesis and step the simulation.

    Args:
        layout: ``drop`` (grid above the plane) or ``layout`` (captured poses).
        drop_height / spacing: grid drop placement (metres).
        steps: simulation steps to run.
        record: write an mp4 of a fixed camera to this path (headless friendly).
        backend: ``gpu`` or ``cpu``.
    """
    import genesis as gs

    export_dir = Path(export_dir)
    scene_spec = json.loads((export_dir / "scene.json").read_text())
    objects = scene_spec["objects"]
    if not scene_spec.get("metric", False):
        print("[sim] NOTE: scene is non-metric (scale=1.0). Sizes/masses are not "
              "real-world; pass --scale at export for metric physics.")

    gs.init(backend=getattr(gs, backend))
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())

    # COLMAP/OpenCV world is y-down; rotate -90deg about X so world -Y -> sim +Z.
    align_euler = (-90.0, 0.0, 0.0)

    n = len(objects)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    ents = []
    for i, obj in enumerate(objects):
        urdf = str((export_dir / obj["urdf"]).resolve())
        static = obj.get("static", False)
        if layout == "layout":
            pos = tuple(obj["world_pos"])
            euler = align_euler
        else:  # drop grid
            r, c = divmod(i, cols)
            pos = (c * spacing, r * spacing, drop_height + i * 0.02)
            euler = (0.0, 0.0, 0.0)
        ent = scene.add_entity(
            gs.morphs.URDF(file=urdf, pos=pos, euler=euler, fixed=static),
        )
        ents.append((obj["id"], ent))

    cam = None
    if record is not None:
        cam = scene.add_camera(res=(960, 720), pos=(1.5, 1.5, 1.2),
                               lookat=(0.3, 0.3, 0.0), fov=45, GUI=False)

    scene.build()
    if cam is not None:
        cam.start_recording()
    for _ in range(steps):
        scene.step()
        if cam is not None:
            cam.render()
    if cam is not None and record is not None:
        cam.stop_recording(save_to_filename=str(record), fps=60)
        print(f"[sim] wrote {record}")

    # Report resting heights so we can confirm objects landed (didn't fall through).
    for cid, ent in ents:
        z = float(ent.get_pos()[2])
        print(f"[sim] object {cid}: rest z = {z:+.3f} m")
    print(f"[sim] stepped {steps} frames over {n} objects on a ground plane.")
