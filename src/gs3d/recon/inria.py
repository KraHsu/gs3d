"""Load a reference-3DGS ("INRIA gaussian-splatting") model into gs3d's splat format.

Our semantic-segmentation stage is built on the reference implementation
(https://github.com/graphdeco-inria/gaussian-splatting) plus a per-Gaussian
instance clustering, so its artifacts are *not* gs3d's gsplat-native `ckpt.pt`:

  * `chkpnt<ITER>.pth` / `chkpnt<ITER>_langfeat_*.pth` — a pickled
    `GaussianModel.capture()` tuple, and
  * `point_cloud/iteration_<ITER>/point_cloud.ply` — the reference PLY layout.

Both encode the *same* parameter conventions, which line up 1:1 with gsplat once
re-keyed into gs3d's `splats` dict (activations applied later by
`model.rasterize_splats`):

  reference                gs3d `splats`        activation in rasterize_splats
  _xyz           (N,3)     means      (N,3)     (none)
  _features_dc   (N,1,3)   sh0        (N,1,3)   spherical harmonics
  _features_rest (N,15,3)  shN        (N,15,3)  spherical harmonics
  _scaling       (N,3)     scales     (N,3)     exp()
  _rotation      (N,4)     quats      (N,4)     normalize (gsplat-internal)
  _opacity       (N,1)     opacities  (N,)      sigmoid()

The segmentation fork additionally captures **`_cluster_indices`** — a 1-D Long
tensor of per-Gaussian instance ids — inside the same `capture()` tuple. We find
it by dtype+length rather than a fixed position, so this also loads vanilla
reference checkpoints (where it is simply absent).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class InriaModel:
    splats: torch.nn.ParameterDict  # gs3d-keyed Gaussian params (see module docstring)
    sh_degree: int                  # active SH degree to evaluate
    cluster_indices: torch.Tensor | None  # (N,) Long instance ids, or None if unsegmented

    @property
    def num_gaussians(self) -> int:
        return self.splats["means"].shape[0]


def _as_splats(
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    device: str,
) -> torch.nn.ParameterDict:
    splats = {
        "means": xyz.reshape(-1, 3),
        "sh0": features_dc.reshape(xyz.shape[0], -1, 3),     # (N,1,3)
        "shN": features_rest.reshape(xyz.shape[0], -1, 3),   # (N,15,3)
        "scales": scaling.reshape(-1, 3),                    # raw log-scale (exp applied later)
        "quats": rotation.reshape(-1, 4),                    # raw quat (normalized by gsplat)
        "opacities": opacity.reshape(-1),                    # raw logit (sigmoid applied later)
    }
    return torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.to(device).float(), requires_grad=False) for k, v in splats.items()}
    )


def load_inria_checkpoint(path: str | Path, device: str = "cuda") -> InriaModel:
    """Load a reference-3DGS `GaussianModel.capture()` checkpoint (`.pth`).

    The reference saver writes either the capture tuple directly or
    ``(capture_tuple, iteration)``; we unwrap both. `_cluster_indices` is located
    by being the lone 1-D integer tensor of length N.
    """
    blob = torch.load(Path(path), map_location=device, weights_only=False)

    # Unwrap (capture_tuple, iteration) -> capture_tuple.
    model = blob
    if isinstance(model, (list, tuple)) and len(model) and isinstance(model[0], (list, tuple)):
        model = model[0]
    if not isinstance(model, (list, tuple)):
        raise ValueError(f"{path}: not a reference-3DGS capture tuple (got {type(model).__name__})")

    # Positional capture() layout:
    #   (active_sh_degree, _xyz, _features_dc, _features_rest,
    #    _scaling, _rotation, _opacity, [_cluster_indices,] max_radii2D, ...)
    active_sh_degree = int(model[0])
    xyz, features_dc, features_rest, scaling, rotation, opacity = model[1:7]
    n = xyz.shape[0]

    # Find the per-Gaussian instance ids: the only 1-D integer tensor of length N.
    cluster_indices = None
    for t in model[7:]:
        if (
            torch.is_tensor(t)
            and t.dtype in (torch.int64, torch.int32, torch.int16, torch.int8)
            and t.ndim == 1
            and t.shape[0] == n
        ):
            cluster_indices = t.to(device).long()
            break

    splats = _as_splats(xyz, features_dc, features_rest, scaling, rotation, opacity, device)
    return InriaModel(splats=splats, sh_degree=active_sh_degree, cluster_indices=cluster_indices)


def load_inria_ply(path: str | Path, device: str = "cuda", sh_degree: int = 3) -> InriaModel:
    """Load a reference-3DGS `point_cloud.ply`.

    The PLY has no instance ids (those live only in the checkpoint), so
    `cluster_indices` is None. Useful for plain RGB viewing of the dense export.
    """
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = set(v.data.dtype.names)

    def col(name: str) -> np.ndarray:
        return np.asarray(v[name])

    xyz = np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float32)
    n = xyz.shape[0]

    f_dc = np.stack([col(f"f_dc_{i}") for i in range(3)], axis=1).astype(np.float32)  # (N,3)
    rest_names = sorted(
        (nm for nm in names if nm.startswith("f_rest_")), key=lambda s: int(s.split("_")[-1])
    )
    if rest_names:
        # Reference stores f_rest channel-major: [c0*K, c1*K, c2*K]; invert to (N,K,3).
        f_rest = np.stack([col(nm) for nm in rest_names], axis=1).astype(np.float32)
        k = f_rest.shape[1] // 3
        f_rest = f_rest.reshape(n, 3, k).transpose(0, 2, 1)  # (N,K,3)
    else:
        f_rest = np.zeros((n, 0, 3), dtype=np.float32)

    opacity = col("opacity").astype(np.float32).reshape(n, 1)
    scaling = np.stack([col(f"scale_{i}") for i in range(3)], axis=1).astype(np.float32)
    rot_names = sorted(
        (nm for nm in names if nm.startswith("rot_")), key=lambda s: int(s.split("_")[-1])
    )
    rotation = np.stack([col(nm) for nm in rot_names], axis=1).astype(np.float32)

    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    splats = _as_splats(
        to_t(xyz), to_t(f_dc), to_t(f_rest), to_t(scaling), to_t(rotation), to_t(opacity), device
    )
    return InriaModel(splats=splats, sh_degree=sh_degree, cluster_indices=None)


def load_inria(path: str | Path, device: str = "cuda") -> InriaModel:
    """Dispatch on extension: `.pth`/`.pt` -> checkpoint, `.ply` -> point cloud."""
    suffix = Path(path).suffix.lower()
    if suffix in (".pth", ".pt"):
        return load_inria_checkpoint(path, device)
    if suffix == ".ply":
        return load_inria_ply(path, device)
    raise ValueError(f"{path}: expected a .pth checkpoint or .ply, got '{suffix}'")
