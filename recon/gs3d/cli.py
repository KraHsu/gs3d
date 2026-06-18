"""Command-line interface: `gs3d sfm | train | render`."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gs3d", description="RealSense → 3D Gaussian Splatting")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sfm = sub.add_parser("sfm", help="Run COLMAP SfM (pycolmap) on a captured scene")
    p_sfm.add_argument("scene", help="Path to data/<scene> (must contain images/)")
    p_sfm.add_argument("--matching", choices=["exhaustive", "sequential"], default="exhaustive")
    p_sfm.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p_sfm.add_argument("--overwrite", action="store_true", help="Rebuild database/sparse")

    p_tr = sub.add_parser("train", help="Train a 3DGS model on a scene")
    p_tr.add_argument("scene", help="Path to data/<scene> (with sparse/0 from `sfm`)")
    p_tr.add_argument("-o", "--out", required=True, help="Output directory")
    p_tr.add_argument("--max-steps", type=int, default=7000)
    p_tr.add_argument("--sh-degree", type=int, default=3)
    p_tr.add_argument("--downscale", type=int, default=1)
    p_tr.add_argument("--eval-every", type=int, default=2000)

    p_rd = sub.add_parser("render", help="Render eval views + orbit video from a checkpoint")
    p_rd.add_argument("out", help="Output directory containing ckpt.pt")
    p_rd.add_argument("--n-frames", type=int, default=120)
    p_rd.add_argument("--fps", type=int, default=30)

    args = parser.parse_args(argv)

    if args.cmd == "sfm":
        from .colmap_sfm import run_sfm

        run_sfm(args.scene, matching=args.matching, device=args.device, overwrite=args.overwrite)
    elif args.cmd == "train":
        from .trainer import train

        train(
            args.scene,
            args.out,
            max_steps=args.max_steps,
            sh_degree=args.sh_degree,
            downscale=args.downscale,
            eval_every=args.eval_every,
        )
    elif args.cmd == "render":
        from .render import render

        render(args.out, n_frames=args.n_frames, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
