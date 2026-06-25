"""Per-instance colouring for segmented Gaussians.

`inria.load_inria_checkpoint` returns a `cluster_indices` tensor (one instance id
per Gaussian). Here we turn those ids into stable, visually distinct RGB colours
so the viewer/renderer can paint each instance a flat colour instead of its
learned appearance.
"""

from __future__ import annotations

import colorsys

import torch

# Golden-ratio conjugate — successive multiples spread hues maximally around the
# wheel, so adjacent instance ids never get near-identical colours.
_GOLDEN = 0.61803398875


def instance_palette(num_ids: int, device: str = "cuda", saturation: float = 0.65, value: float = 0.95) -> torch.Tensor:
    """Deterministic (num_ids, 3) RGB palette in [0, 1], indexable by instance id."""
    cols = []
    for i in range(max(num_ids, 1)):
        h = (i * _GOLDEN) % 1.0
        cols.append(colorsys.hsv_to_rgb(h, saturation, value))
    return torch.tensor(cols, dtype=torch.float32, device=device)


def cluster_colors(
    cluster_indices: torch.Tensor,
    *,
    selected: int | None = None,
    dim_factor: float = 0.05,
    background_id: int | None = 0,
    background_gray: float = 0.35,
    palette: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map per-Gaussian instance ids to (N, 3) flat RGB colours in [0, 1].

    selected: if given, only that instance keeps its colour; all others are dimmed
              to ``dim_factor`` brightness so a single object stands out.
    background_id: painted a neutral gray (the largest cluster — id 0 — is usually
              floor/table backdrop); pass None to colour it like any other instance.
    palette: optional precomputed (>=max_id+1, 3) palette to avoid rebuilding it
              every call (the interactive viewer passes one in per frame).
    """
    ids = cluster_indices.long()
    device = ids.device
    max_id = int(ids.max().item()) if ids.numel() else 0
    if palette is None:
        palette = instance_palette(max_id + 1, device=device)
    colors = palette[ids.clamp_min(0)]  # (N,3)

    if background_id is not None:
        colors = torch.where(
            (ids == background_id).unsqueeze(-1),
            torch.full_like(colors, background_gray),
            colors,
        )

    if selected is not None:
        keep = (ids == selected).unsqueeze(-1)
        colors = torch.where(keep, colors, colors * dim_factor)

    return colors
