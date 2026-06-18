"""Interactive viewer for a trained model.

Renders the Gaussians on the server GPU (gsplat) and streams frames to a browser
via viser/nerfview — far faster than client-side web viewers for big scenes, and
no need to download the .ply. Access over an SSH tunnel:

    ssh -L 8080:localhost:8080 H20      # on your machine
    # then open http://localhost:8080
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .model import load_checkpoint, rasterize_splats


def view(out_dir: str | Path, port: int = 8080, device: str = "cuda") -> None:
    import nerfview
    import viser

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
    nerfview.Viewer(server=server, render_fn=render_fn, mode="rendering")
    print(f"[view] serving on the server at http://localhost:{port}")
    print(f"[view] on your machine:  ssh -L {port}:localhost:{port} H20   then open http://localhost:{port}")
    print("[view] Ctrl+C to stop.")
    while True:
        time.sleep(1.0)
