"""Train a 3D Gaussian Splatting model with gsplat.

Reuses gsplat's `rasterization` (forward render) and `DefaultStrategy`
(adaptive densification / pruning). A compact equivalent of gsplat's
`examples/simple_trainer.py`, trimmed to the essentials.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gsplat import DefaultStrategy
from tqdm import tqdm

from .dataset import CameraView, ColmapDataset
from .model import (
    create_splats_and_optimizers,
    psnr,
    rasterize_splats,
    save_checkpoint,
    save_ply,
    ssim,
)


def _view_tensors(view: CameraView, device: str):
    gt = torch.from_numpy(view.image).float().to(device) / 255.0  # [H,W,3]
    gt = gt.unsqueeze(0)  # [1,H,W,3]
    viewmats = torch.from_numpy(view.w2c).float().to(device).unsqueeze(0)  # [1,4,4]
    Ks = torch.from_numpy(view.K).float().to(device).unsqueeze(0)  # [1,3,3]
    return gt, viewmats, Ks


@torch.no_grad()
def evaluate(splats, dataset: ColmapDataset, sh_degree: int, device: str) -> dict:
    psnrs, ssims = [], []
    for view in dataset.test_views():
        gt, viewmats, Ks = _view_tensors(view, device)
        renders, _, _ = rasterize_splats(
            splats, viewmats, Ks, view.width, view.height, sh_degree
        )
        pred = renders[..., :3].clamp(0, 1)
        pred_chw = pred.permute(0, 3, 1, 2)
        gt_chw = gt.permute(0, 3, 1, 2)
        psnrs.append(psnr(pred_chw, gt_chw).item())
        ssims.append(ssim(pred_chw, gt_chw).item())
    if not psnrs:
        return {"psnr": float("nan"), "ssim": float("nan")}
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}


def train(
    scene_dir: str | Path,
    out_dir: str | Path,
    max_steps: int = 7000,
    sh_degree: int = 3,
    downscale: int = 1,
    ssim_lambda: float = 0.2,
    sh_increase_every: int = 1000,
    eval_every: int = 2000,
    seed: int = 0,
    device: str = "cuda",
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat training but is not available.")
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading dataset from {scene_dir} (downscale={downscale}) ...")
    dataset = ColmapDataset(scene_dir, downscale=downscale)
    print(
        f"[train] {len(dataset.train_idx)} train / {len(dataset.test_idx)} test views, "
        f"{dataset.points.shape[0]} init points, scene_scale={dataset.scene_scale:.3f}"
    )

    splats, optimizers = create_splats_and_optimizers(
        dataset.points, dataset.points_rgb, dataset.scene_scale, sh_degree, device=device
    )

    # Decay the means learning rate to 1% over training (reference schedule).
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
    )

    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=dataset.scene_scale)

    train_views = dataset.train_views()
    pbar = tqdm(range(max_steps), desc="train")
    for step in pbar:
        view = train_views[torch.randint(0, len(train_views), (1,)).item()]
        gt, viewmats, Ks = _view_tensors(view, device)
        active_sh = min(step // sh_increase_every, sh_degree)

        renders, _, info = rasterize_splats(
            splats, viewmats, Ks, view.width, view.height, active_sh
        )
        strategy.step_pre_backward(
            params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info
        )

        pred = renders[..., :3].clamp(0, 1)
        pred_chw = pred.permute(0, 3, 1, 2)
        gt_chw = gt.permute(0, 3, 1, 2)
        l1 = (pred - gt).abs().mean()
        loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * (1.0 - ssim(pred_chw, gt_chw))
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()

        strategy.step_post_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
            packed=True,
        )

        if step % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", n=splats["means"].shape[0])
        if eval_every and step > 0 and step % eval_every == 0:
            m = evaluate(splats, dataset, sh_degree, device)
            tqdm.write(f"[eval @ {step}] PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f}")

    metrics = evaluate(splats, dataset, sh_degree, device)
    print(f"[train] final PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}")

    config = {
        "scene_dir": str(Path(scene_dir).resolve()),
        "sh_degree": sh_degree,
        "downscale": downscale,
        "max_steps": max_steps,
        "final_metrics": metrics,
    }
    save_checkpoint(out_dir / "ckpt.pt", splats, config)
    save_ply(out_dir / "point_cloud.ply", splats)
    print(f"[train] saved {out_dir / 'ckpt.pt'} and {out_dir / 'point_cloud.ply'}")
    return out_dir / "ckpt.pt"
