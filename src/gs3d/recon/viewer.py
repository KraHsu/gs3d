"""Interactive viewer for a trained model.

Renders the Gaussians on the server GPU (gsplat) and streams frames to a browser
via viser/nerfview — far faster than client-side web viewers for big scenes, and
no need to download the .ply. Access over an SSH tunnel:

    ssh -L 8080:localhost:8080 H20      # on your machine
    # then open http://localhost:8080

Two entry points:
    view(out_dir)           — a gs3d gsplat checkpoint (outputs/<scene>/ckpt.pt)
    view_seg(checkpoint)    — a reference-3DGS / segmentation checkpoint, with a
                              GUI toggle between learned RGB and per-instance
                              segmentation colours (see `inria` + `seg`).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .model import load_checkpoint, rasterize_splats


def view(out_dir: str | Path, port: int = 8080, device: str = "cuda") -> None:
    import nerfview
    import viser

    from ._cuda import ensure_cuda_toolkit

    ensure_cuda_toolkit()  # let gsplat JIT-compile its kernels (Blackwell/cu128)
    splats, config = load_checkpoint(Path(out_dir) / "ckpt.pt", device=device)
    sh_degree = config.get("sh_degree", 3)
    n = splats["means"].shape[0]
    print(f"[view] loaded {n} gaussians from {out_dir}")

    @torch.no_grad()
    def render_fn(camera_state, arg2):
        # nerfview passes either an (width, height) tuple (older API) or a
        # RenderTabState (newer API); support both.
        if isinstance(arg2, (tuple, list)):
            width, height = arg2
        elif getattr(arg2, "preview_render", False):
            width, height = arg2.render_width, arg2.render_height
        else:
            width, height = arg2.viewer_width, arg2.viewer_height
        width, height = int(width), int(height)

        c2w = torch.as_tensor(camera_state.c2w, dtype=torch.float32, device=device)
        K = torch.as_tensor(camera_state.get_K((width, height)), dtype=torch.float32, device=device)
        viewmat = torch.linalg.inv(c2w)
        colors, _, _ = rasterize_splats(
            splats, viewmat[None], K[None], width, height, sh_degree
        )
        return colors[0, ..., :3].clamp(0, 1).cpu().numpy()

    server = viser.ViserServer(port=port, verbose=False)
    _frame_scene_camera(server, splats["means"])
    nerfview.Viewer(server=server, render_fn=render_fn, mode="rendering")
    print(f"[view] serving on the server at http://localhost:{port}")
    print(f"[view] on your machine:  ssh -L {port}:localhost:{port} H20   then open http://localhost:{port}")
    print("[view] Ctrl+C to stop.")
    while True:
        time.sleep(1.0)


def _wh(arg2) -> tuple[int, int]:
    """Resolve (width, height) from nerfview's old tuple or new RenderTabState arg."""
    if isinstance(arg2, (tuple, list)):
        w, h = arg2
    elif getattr(arg2, "preview_render", False):
        w, h = arg2.render_width, arg2.render_height
    else:
        w, h = arg2.viewer_width, arg2.viewer_height
    return int(w), int(h)


def _inria_camera_pose(checkpoint: str | Path):
    """Initial (eye, up) from a sibling INRIA ``cameras.json`` (a real training
    viewpoint), or None. Geometry shares the COLMAP frame, so a training camera
    frames the scene exactly as captured."""
    import json

    import numpy as np

    p = Path(checkpoint).resolve().parent / "cameras.json"
    if not p.exists():
        return None
    try:
        cams = json.loads(p.read_text())
        if not cams:
            return None
        c = cams[len(cams) // 2]  # a mid-sequence view (cameras orbit the subject)
        rot = np.asarray(c["rotation"], dtype=float)  # camera-to-world rotation
        eye = np.asarray(c["position"], dtype=float)  # camera centre in world
        up = -rot[:, 1]  # camera +Y is down in OpenCV → world up is -Y axis
        return eye, up
    except Exception:
        return None


def _frame_scene_camera(server, means: torch.Tensor, checkpoint: str | Path | None = None) -> None:
    """Point each connecting client's camera at the Gaussians.

    nerfview's default camera sits at the origin, but COLMAP scenes live at an
    arbitrary centre/scale — so without this the browser opens on empty space.
    We always look at the point-cloud centroid; the eye is a real training camera
    (from a sibling ``cameras.json``) when available, else a robust pull-back
    along a fixed diagonal.
    """
    import numpy as np

    pts = means.detach().cpu().numpy()
    center = np.median(pts, axis=0)
    radius = float(np.median(np.linalg.norm(pts - center, axis=1))) or 1.0

    pose = _inria_camera_pose(checkpoint) if checkpoint is not None else None
    if pose is not None:
        eye, up = pose
        print(f"[view] framing camera from cameras.json: eye={np.round(eye, 3).tolist()}")
    else:
        up = np.array([0.0, -1.0, 0.0])  # COLMAP/OpenCV world is y-down
        offset = np.array([0.7, -0.7, -1.0])
        eye = center + offset / np.linalg.norm(offset) * radius * 3.0
        print(f"[view] framing camera: center={center.round(3).tolist()} radius={radius:.3f}")

    @server.on_client_connect
    def _(client) -> None:
        client.camera.up_direction = up.astype(float)
        client.camera.position = eye.astype(float)
        client.camera.look_at = center.astype(float)


def view_seg(checkpoint: str | Path, port: int = 8080, device: str = "cuda") -> None:
    """View a segmented reference-3DGS checkpoint with an RGB/segmentation toggle.

    The checkpoint must carry per-Gaussian instance ids (`_cluster_indices`); the
    geometry and the ids are loaded from the *same* tensor set, so they always
    align (the exported PLY may have a different Gaussian count and is not used).
    """
    import nerfview
    import viser

    from ._cuda import ensure_cuda_toolkit
    from .inria import load_inria_checkpoint
    from .seg import cluster_colors, instance_palette

    ensure_cuda_toolkit()  # let gsplat JIT-compile its kernels (Blackwell/cu128)
    model = load_inria_checkpoint(checkpoint, device=device)
    splats, sh_degree = model.splats, model.sh_degree
    ids = model.cluster_indices
    n = model.num_gaussians
    if ids is None:
        print(f"[view-seg] WARNING: no instance ids in {checkpoint}; only RGB available")
        n_instances = 0
        palette = None
    else:
        n_instances = int(ids.max().item()) + 1
        palette = instance_palette(n_instances, device=device)  # built once, reused per frame
    print(f"[view-seg] loaded {n} gaussians, {n_instances} instances from {checkpoint}")

    server = viser.ViserServer(port=port, verbose=False)
    _frame_scene_camera(server, splats["means"], checkpoint=checkpoint)
    with server.gui.add_folder("Segmentation"):
        gui_mode = server.gui.add_dropdown(
            "Render", options=["RGB", "Segmentation"],
            initial_value="Segmentation" if ids is not None else "RGB",
        )
        gui_isolate = server.gui.add_text(
            "Isolate id", initial_value="all",
            hint="instance id to highlight (others dimmed), or 'all'",
        )
        gui_gray_bg = server.gui.add_checkbox(
            "Gray background cluster", initial_value=True,
            hint="paint instance 0 (usually floor/table) neutral gray",
        )

    def current_colors() -> torch.Tensor | None:
        if ids is None or gui_mode.value == "RGB":
            return None
        bg = 0 if gui_gray_bg.value else None
        sel_txt = gui_isolate.value.strip().lower()
        sel: int | None = None
        if sel_txt not in ("", "all"):
            try:
                sel = int(sel_txt)
            except ValueError:
                sel = None
        return cluster_colors(ids, selected=sel, background_id=bg, palette=palette)

    @torch.no_grad()
    def render_fn(camera_state, arg2):
        width, height = _wh(arg2)
        c2w = torch.as_tensor(camera_state.c2w, dtype=torch.float32, device=device)
        K = torch.as_tensor(camera_state.get_K((width, height)), dtype=torch.float32, device=device)
        viewmat = torch.linalg.inv(c2w)
        colors, _, _ = rasterize_splats(
            splats, viewmat[None], K[None], width, height, sh_degree,
            override_colors=current_colors(),
        )
        return colors[0, ..., :3].clamp(0, 1).cpu().numpy()

    viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode="rendering")

    def _rerender(_=None) -> None:
        # Force a re-render when a GUI control (not the camera) changes.
        if hasattr(viewer, "rerender"):
            viewer.rerender(None)

    gui_mode.on_update(_rerender)
    gui_isolate.on_update(_rerender)
    gui_gray_bg.on_update(_rerender)

    print(f"[view-seg] serving on the server at http://localhost:{port}")
    print(f"[view-seg] on your machine:  ssh -L {port}:localhost:{port} <server>   then open http://localhost:{port}")
    print("[view-seg] Ctrl+C to stop.")
    while True:
        time.sleep(1.0)


def view_sim(
    export_dir: str | Path,
    checkpoint: str | Path,
    port: int = 8080,
    device: str = "cuda",
    *,
    steps: int = 250,
    drop_height: float = 0.3,
    backend: str = "gpu",
    opacity_min: float = 0.1,
    aspect_max: float = 18.0,
    scale_quantile: float = 0.97,
) -> None:
    """Interactive viewer of the *real* Gaussians driven by physics (survey §4 / C).

    Unlike `sim-render` (a baked mp4 from one camera), this runs the physics once,
    then serves the photorealistic, physics-driven Gaussians in an orbitable viewer
    with a time slider and play toggle — you control the camera, zoom, and time.
    """
    import json

    import nerfview
    import viser

    from ._cuda import ensure_cuda_toolkit
    from .export.sim_render import (
        assemble_frame,
        build_per_object,
        compute_spawns,
        simulate_poses,
    )
    from .inria import load_inria_checkpoint

    ensure_cuda_toolkit()
    export_dir = Path(export_dir)
    scene_spec = json.loads((export_dir / "scene.json").read_text())
    objects = scene_spec["objects"]
    s = float(scene_spec.get("scale_colmap_per_metre", 1.0))

    model = load_inria_checkpoint(checkpoint, device=device)
    if model.cluster_indices is None:
        raise ValueError(f"{checkpoint}: no _cluster_indices; cannot map physics to Gaussians")
    sh_degree = model.sh_degree
    per_obj = build_per_object(
        model, objects, s, opacity_min=opacity_min, aspect_max=aspect_max,
        scale_quantile=scale_quantile, device=device,
    )
    n_gauss = sum(o["local"].shape[0] for o in per_obj)
    print(f"[view-sim] {len(objects)} objects, {n_gauss} gaussians; simulating {steps} steps...")
    spawns = compute_spawns(per_obj, drop_height=drop_height)
    positions, quats = simulate_poses(export_dir, objects, steps, spawns, backend)
    print("[view-sim] physics done; serving viewer.")

    state = {"frame": steps - 1}  # start settled (clearest view)

    @torch.no_grad()
    def render_fn(camera_state, arg2):
        width, height = _wh(arg2)
        c2w = torch.as_tensor(camera_state.c2w, dtype=torch.float32, device=device)
        K = torch.as_tensor(camera_state.get_K((width, height)), dtype=torch.float32, device=device)
        viewmat = torch.linalg.inv(c2w)
        splats = assemble_frame(per_obj, positions[state["frame"]], quats[state["frame"]], device=device)
        colors, _, _ = rasterize_splats(splats, viewmat[None], K[None], width, height, sh_degree)
        return colors[0, ..., :3].clamp(0, 1).cpu().numpy()

    server = viser.ViserServer(port=port, verbose=False)
    rest = assemble_frame(per_obj, positions[-1], quats[-1], device=device)  # settled
    _frame_scene_camera(server, rest["means"])
    with server.gui.add_folder("Physics"):
        gui_frame = server.gui.add_slider("Frame", min=0, max=steps - 1, step=1, initial_value=steps - 1)
        gui_play = server.gui.add_checkbox("Play", initial_value=False)

    viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode="rendering")

    @gui_frame.on_update
    def _(_=None) -> None:
        state["frame"] = int(gui_frame.value)
        if hasattr(viewer, "rerender"):
            viewer.rerender(None)

    print(f"[view-sim] serving at http://localhost:{port}  (tunnel: ssh -L {port}:localhost:{port} <server>)")
    print("[view-sim] drag to orbit; Frame slider / Play scrubs the physics. Ctrl+C to stop.")
    while True:
        if gui_play.value:
            state["frame"] = (state["frame"] + 1) % steps
            gui_frame.value = state["frame"]  # updates slider + triggers rerender
            time.sleep(1.0 / 60.0)
        else:
            time.sleep(0.05)


def view_env(
    checkpoint: str | Path,
    data_dir: str | Path,
    port: int = 8080,
    device: str = "cuda",
    *,
    scale: float | None = None,
    backend: str = "cpu",
    max_object_size: float = 0.45,
) -> None:
    """Interactive viewer of the metric, gravity-aligned GS3DSimScene: the real
    captured scene rendered photorealistically, with live Genesis physics you can
    play/pause. This is the training env (minus your robot) — orbit it to verify
    sizing/orientation/appearance are correct."""
    import nerfview
    import numpy as np
    import viser

    from .export.env import GS3DSimScene

    env = GS3DSimScene(checkpoint, data_dir, scale=scale, backend=backend,
                       max_object_size=max_object_size, device=device)
    env.build()  # no robot; user's robot would be added before build in code

    @torch.no_grad()
    def render_fn(camera_state, arg2):
        width, height = _wh(arg2)
        c2w = torch.as_tensor(camera_state.c2w, dtype=torch.float32, device=device)
        K = torch.as_tensor(camera_state.get_K((width, height)), dtype=torch.float32, device=device)
        img = env.render(c2w, K, width, height)
        return img.astype(np.float32) / 255.0

    server = viser.ViserServer(port=port, verbose=False)

    # Frame on the table/objects (z-up metric world).
    if env.objects:
        center = np.mean([o["rest_pos"] for o in env.objects], axis=0)
    else:
        center = env.bg["means"].mean(0).cpu().numpy()
    eye = center + np.array([0.6, 0.6, 0.5])

    @server.on_client_connect
    def _(client) -> None:
        client.camera.up_direction = np.array([0.0, 0.0, 1.0])
        client.camera.position = eye.astype(float)
        client.camera.look_at = center.astype(float)

    with server.gui.add_folder("Sim"):
        gui_play = server.gui.add_checkbox("Play physics", initial_value=False)

    viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode="rendering")

    print(f"[view-env] serving at http://localhost:{port}  (tunnel: ssh -L {port}:localhost:{port} <server>)")
    print("[view-env] orbit the scene; 'Play physics' steps Genesis. Ctrl+C to stop.")
    while True:
        if gui_play.value:
            env.step()
            if hasattr(viewer, "rerender"):
                viewer.rerender(None)
            time.sleep(1.0 / 60.0)
        else:
            time.sleep(0.05)
