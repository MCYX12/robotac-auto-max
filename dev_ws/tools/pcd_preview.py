#!/usr/bin/env python3
"""Render a deterministic top/side PNG preview from a native binary PCD."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pcd_quality_report import load_binary_pcd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcd", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-points", type=int, default=250_000)
    parser.add_argument("--title")
    return parser.parse_args()


def main():
    args = parse_args()
    points, _ = load_binary_pcd(args.pcd)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise SystemExit("PCD contains no finite XYZ points")
    if args.max_points <= 0:
        raise SystemExit("--max-points must be positive")

    stride = max(1, int(np.ceil(len(points) / args.max_points)))
    sample = points[::stride]
    output = args.output or args.pcd.with_name(f"{args.pcd.stem}_preview.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or args.pcd.stem

    figure, (top, side) = plt.subplots(
        1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )
    top.scatter(sample[:, 0], sample[:, 1], s=0.35, alpha=0.55, linewidths=0)
    top.set_aspect("equal", adjustable="box")
    top.set_xlabel("x (m)")
    top.set_ylabel("y (m)")
    top.set_title(f"{title} - top")
    top.grid(True, alpha=0.45)

    side.scatter(sample[:, 0], sample[:, 2], s=0.35, alpha=0.45, linewidths=0)
    side.set_xlabel("x (m)")
    side.set_ylabel("z (m)")
    side.set_title(f"{title} - side")
    side.grid(True, alpha=0.45)

    figure.suptitle(
        f"Native PCD: {len(points):,} finite points (preview: {len(sample):,})",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(output.resolve())


if __name__ == "__main__":
    main()
