"""Gaussian model utilities shared by training and rendering.

Wraps gsplat's `rasterization` and stores the Gaussian parameters in the same
convention as the reference 3DGS implementation, so exported PLYs open in
standard splat viewers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

# Zeroth-order SH constant (DC term), as in the reference implementation.
SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * SH_C0 + 0.5


# ---------------------------------------------------------------------------
# Splat construction
# ---------------------------------------------------------------------------
def create_splats_and_optimizers(
    points: np.ndarray,
    points_rgb: np.ndarray,
    scene_scale: float,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    device: str = "cuda",
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Adam]]:
    """Initialize Gaussians from an SfM point cloud + per-attribute optimizers."""
    from sklearn.neighbors import NearestNeighbors

    pts = torch.from_numpy(points).float()
    rgb = torch.from_numpy(points_rgb).float() / 255.0
    n = pts.shape[0]

    # Initialize each Gaussian's size to the mean distance to its 3 NN.
    nn = NearestNeighbors(n_neighbors=4).fit(points)
    dists, _ = nn.kneighbors(points)
    dist_avg = torch.from_numpy(dists[:, 1:].mean(axis=1)).float().clamp_min(1e-7)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)

    quats = torch.zeros((n, 4))
    quats[:, 0] = 1.0  # identity rotation (w, x, y, z)
    opacities = torch.logit(torch.full((n,), init_opacity))

    num_sh = (sh_degree + 1) ** 2
    colors = torch.zeros((n, num_sh, 3))
    colors[:, 0, :] = rgb_to_sh(rgb)

    # learning rates (reference 3DGS values; means lr scaled by scene extent)
    params = [
        ("means", torch.nn.Parameter(pts), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20),
    ]
    splats = torch.nn.ParameterDict({name: p for name, p, _ in params}).to(device)
    optimizers = {
        name: torch.optim.Adam([{"params": splats[name], "lr": lr}], eps=1e-15)
        for name, _, lr in params
    }
    return splats, optimizers


def rasterize_splats(
    splats: torch.nn.ParameterDict,
    viewmats: torch.Tensor,  # [C,4,4] world-to-camera
    Ks: torch.Tensor,  # [C,3,3]
    width: int,
    height: int,
    sh_degree: int | None,
    override_colors: torch.Tensor | None = None,  # [N,3] flat RGB; bypasses SH
    **kwargs,
):
    if override_colors is not None:
        # Per-Gaussian flat colours (e.g. instance ids) — composite directly, no SH.
        colors = override_colors
        sh_degree = None
    else:
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, K, 3]
    render_colors, render_alphas, info = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=True,
        **kwargs,
    )
    return render_colors, render_alphas, info


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _gaussian_window(window_size: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)
    kernel_2d = (g @ g.t()).unsqueeze(0).unsqueeze(0)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """SSIM for images shaped [B, C, H, W] in [0, 1]."""
    channels = img1.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, img1.device)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(img1, img2)
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, splats: torch.nn.ParameterDict, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"splats": {k: v.detach().cpu() for k, v in splats.items()}, "config": config}, path)


def load_checkpoint(path: Path, device: str = "cuda") -> tuple[torch.nn.ParameterDict, dict]:
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    splats = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.to(device)) for k, v in blob["splats"].items()}
    )
    return splats, blob["config"]


def save_ply(path: Path, splats: torch.nn.ParameterDict) -> None:
    """Export Gaussians as a PLY compatible with standard 3DGS viewers."""
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    means = splats["means"].detach().cpu().numpy()
    n = means.shape[0]
    normals = np.zeros_like(means)
    f_dc = splats["sh0"].detach().cpu().numpy().reshape(n, -1)  # [N, 3]
    # channel-major flatten of higher-order SH, matching the reference format
    f_rest = (
        splats["shN"].detach().transpose(1, 2).flatten(1).cpu().numpy()
        if splats["shN"].shape[1] > 0
        else np.zeros((n, 0), dtype=np.float32)
    )
    opacities = splats["opacities"].detach().cpu().numpy().reshape(n, 1)
    scales = splats["scales"].detach().cpu().numpy()
    rots = splats["quats"].detach().cpu().numpy()

    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    fields += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    fields += ["opacity"]
    fields += [f"scale_{i}" for i in range(scales.shape[1])]
    fields += [f"rot_{i}" for i in range(rots.shape[1])]

    data = np.concatenate([means, normals, f_dc, f_rest, opacities, scales, rots], axis=1)
    dtype = [(f, "f4") for f in fields]
    elements = np.empty(n, dtype=dtype)
    elements[:] = list(map(tuple, data))
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))
