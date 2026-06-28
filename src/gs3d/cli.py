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

    p_vs = sub.add_parser(
        "view-seg",
        help="Interactive GPU viewer for a segmented reference-3DGS checkpoint "
        "(RGB / per-instance segmentation toggle)",
    )
    p_vs.add_argument("checkpoint", help="reference-3DGS .pth (with _cluster_indices) or .ply")
    p_vs.add_argument("--port", type=int, default=8080)

    p_ec = sub.add_parser(
        "export-clusters",
        help="Split a segmented reference-3DGS checkpoint into per-instance point "
        "subsets + manifest (first stage of scene->sim export)",
    )
    p_ec.add_argument("checkpoint", help="reference-3DGS .pth with _cluster_indices")
    p_ec.add_argument("-o", "--out", required=True, help="Output directory for clusters/ + manifest.json")
    p_ec.add_argument("--scale", type=float, default=1.0,
                      help="COLMAP units per metre (divide coords to get metres); 1.0 = non-metric")
    p_ec.add_argument("--min-points", type=int, default=200,
                      help="Drop clusters with fewer Gaussians as floater noise")
    p_ec.add_argument("--min-opacity", type=float, default=0.0,
                      help="Drop Gaussians below this (sigmoid) opacity before splitting")
    p_ec.add_argument("--background-id", type=int, default=0,
                      help="Instance id to flag as static (table/floor)")

    p_es = sub.add_parser(
        "export-sim",
        help="Full scene->sim export: split checkpoint -> per-object meshes -> URDFs "
        "+ scene.json (loadable in Genesis/PyBullet)",
    )
    p_es.add_argument("checkpoint", help="reference-3DGS .pth with _cluster_indices")
    p_es.add_argument("-o", "--out", required=True, help="Output directory")
    p_es.add_argument("--ids", type=int, nargs="+", default=None,
                      help="Instance ids to export (default: all kept non-static objects)")
    p_es.add_argument("--max-objects", type=int, default=None, help="Cap number of objects")
    p_es.add_argument("--scale", type=float, default=1.0,
                      help="COLMAP units per metre (1.0 = non-metric)")
    p_es.add_argument("--density", type=float, default=300.0, help="Object density kg/m^3")
    p_es.add_argument("--min-points", type=int, default=200, help="Floater filter threshold")
    p_es.add_argument("--no-coacd", action="store_true", help="Force single convex-hull collision")
    p_es.add_argument("--include-static", action="store_true", help="Also mesh table/background")

    p_sg = sub.add_parser(
        "sim-genesis",
        help="Load an exported scene (scene.json + URDFs) into Genesis and simulate",
    )
    p_sg.add_argument("export_dir", help="Directory produced by export-sim (has scene.json)")
    p_sg.add_argument("--layout", choices=["drop", "layout"], default="drop",
                      help="'drop' = grid above plane; 'layout' = captured world poses")
    p_sg.add_argument("--steps", type=int, default=240, help="Simulation steps")
    p_sg.add_argument("--viewer", action="store_true", help="Open the interactive viewer")
    p_sg.add_argument("--record", default=None, help="Write an mp4 of a fixed camera")
    p_sg.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")

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
    elif args.cmd == "view-seg":
        from .recon.viewer import view_seg

        view_seg(args.checkpoint, port=args.port)
    elif args.cmd == "export-clusters":
        from .recon.export.clusters import export_clusters

        export_clusters(
            args.checkpoint,
            args.out,
            scale=args.scale,
            min_points=args.min_points,
            min_opacity=args.min_opacity,
            background_id=args.background_id,
        )
    elif args.cmd == "export-sim":
        from .recon.export.pipeline import export_sim

        export_sim(
            args.checkpoint,
            args.out,
            ids=args.ids,
            max_objects=args.max_objects,
            scale=args.scale,
            density=args.density,
            min_points=args.min_points,
            use_coacd=not args.no_coacd,
            include_static=args.include_static,
        )
    elif args.cmd == "sim-genesis":
        from .recon.export.genesis_scene import load_scene

        load_scene(
            args.export_dir,
            layout=args.layout,
            steps=args.steps,
            show_viewer=args.viewer,
            record=args.record,
            backend=args.backend,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
