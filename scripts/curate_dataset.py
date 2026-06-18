"""Curate a captured dataset for 3DGS: drop blurry frames, keep even coverage.

Sharpness = variance of the Laplacian. To preserve orbit coverage *and* prefer
sharp frames, the sequence is split into `--max-frames` equal windows and the
single sharpest frame in each window is kept (frames below `--min-sharpness`
are dropped first). Selected frames are renumbered 000001.. in the output.

    uv run python scripts/curate_dataset.py data/table -o data/table_sharp --max-frames 280
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def _sharpness(path: Path) -> float:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="source scene dir (with images/)")
    ap.add_argument("-o", "--out", required=True, help="output scene dir")
    ap.add_argument("--max-frames", type=int, default=280)
    ap.add_argument("--min-sharpness", type=float, default=0.0)
    ap.add_argument("--copy-depth", action="store_true", help="also copy aligned depth/")
    args = ap.parse_args(argv)

    src, out = Path(args.src), Path(args.out)
    imgs = sorted((src / "images").glob("*.jpg"))
    if not imgs:
        ap.error(f"no images in {src/'images'}")

    sh = [_sharpness(p) for p in imgs]
    keep = [i for i, v in enumerate(sh) if v >= args.min_sharpness]
    print(f"{len(imgs)} frames; {len(keep)} above min-sharpness {args.min_sharpness}")

    if len(keep) > args.max_frames:
        edges = np.linspace(0, len(keep), args.max_frames + 1).astype(int)
        sel = [max(keep[a:b], key=lambda i: sh[i]) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    else:
        sel = keep

    out_img = out / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    out_depth = out / "depth"
    if args.copy_depth:
        out_depth.mkdir(parents=True, exist_ok=True)

    for rank, i in enumerate(sel, 1):
        shutil.copy(imgs[i], out_img / f"{rank:06d}.jpg")
        if args.copy_depth:
            d = src / "depth" / f"{imgs[i].stem}.png"
            if d.exists():
                shutil.copy(d, out_depth / f"{rank:06d}.png")

    for extra in ("intrinsics.json", "meta.json"):
        if (src / extra).exists():
            shutil.copy(src / extra, out / extra)

    kept_sh = np.array([sh[i] for i in sel])
    print(f"kept {len(sel)} frames -> {out_img}")
    print(f"sharpness of kept: min/median/max = "
          f"{kept_sh.min():.0f} / {np.median(kept_sh):.0f} / {kept_sh.max():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
