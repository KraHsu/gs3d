"""Smoke test for the viewer stack: load ckpt, GPU render, construct viser+nerfview."""
import sys

import numpy as np
import torch

from gs3d.recon.model import load_checkpoint, rasterize_splats

out = sys.argv[1] if len(sys.argv) > 1 else "outputs/table"
splats, cfg = load_checkpoint(f"{out}/ckpt.pt")
print("loaded gaussians:", splats["means"].shape[0])

H, W = 256, 256
c2w = torch.eye(4, device="cuda")
c2w[2, 3] = -3.0
K = torch.tensor([[200.0, 0, 128], [0, 200.0, 128], [0, 0, 1]], device="cuda")
viewmat = torch.linalg.inv(c2w)
colors, _, _ = rasterize_splats(splats, viewmat[None], K[None], W, H, cfg["sh_degree"])
print("render ok:", tuple(colors.shape), "range", float(colors.min()), float(colors.max()))

import nerfview
import viser

server = viser.ViserServer(port=8088, verbose=False)
nerfview.Viewer(
    server=server,
    render_fn=lambda cs, a: np.zeros((64, 64, 3), dtype=np.float32),
    mode="rendering",
)
print("viser", viser.__version__, "+ nerfview viewer constructed OK")
