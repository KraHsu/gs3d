"""Unified CLI for both subparts.

  gs3d capture                 # launch the RealSense capture GUI (Windows)
  gs3d check-camera            # detect the D435i and grab one test frame
  gs3d sfm    <scene>          # COLMAP SfM via pycolmap            (Ubuntu)
  gs3d train  <scene> -o <out> # train a 3DGS model with gsplat     (Ubuntu)
  gs3d render <out>            # eval views + orbit video           (Ubuntu)

Subcommand dependencies are imported lazily, so the capture commands work on a
``--extra capture`` install (no torch) and the recon commands work on a
``--extra recon`` install (no PySide6).
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gs3d", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- capture subpart --
    sub.add_parser("capture", help="Launch the RealSense capture GUI (Windows)")
    sub.add_parser("check-camera", help="Detect the D435i and grab one test frame")

    # -- recon subpart --
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
    p_tr.add_argument("--init", choices=["sfm", "depth"], default="sfm",
                      help="Gaussian init: sparse SfM points, or dense RealSense depth back-projection")

    p_rd = sub.add_parser("render", help="Render eval views + orbit video from a checkpoint")
    p_rd.add_argument("out", help="Output directory containing ckpt.pt")
    p_rd.add_argument("--n-frames", type=int, default=120)
    p_rd.add_argument("--fps", type=int, default=30)

    p_vw = sub.add_parser("view", help="Interactive GPU viewer (viser) for a trained model")
    p_vw.add_argument("out", help="Output directory containing ckpt.pt")
    p_vw.add_argument("--port", type=int, default=8080)

    args = parser.parse_args(argv)

    if args.cmd == "capture":
        from .capture.app import main as gui_main

        return gui_main()
    if args.cmd == "check-camera":
        from .capture.check import main as check_main

        return check_main()
    if args.cmd == "sfm":
        from .recon.colmap_sfm import run_sfm

        run_sfm(args.scene, matching=args.matching, device=args.device, overwrite=args.overwrite)
    elif args.cmd == "train":
        from .recon.trainer import train

        train(
            args.scene,
            args.out,
            max_steps=args.max_steps,
            sh_degree=args.sh_degree,
            downscale=args.downscale,
            eval_every=args.eval_every,
            init=args.init,
        )
    elif args.cmd == "render":
        from .recon.render import render

        render(args.out, n_frames=args.n_frames, fps=args.fps)
    elif args.cmd == "view":
        from .recon.viewer import view

        view(args.out, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
