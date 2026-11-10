#!/usr/bin/env python3
"""Fail-closed, read-only checks for a task-one native PCD quality report."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("quality_json", type=Path)
    parser.add_argument("--minimum-points", type=int, default=1000)
    parser.add_argument("--ground-maximum-z", type=float, default=0.18)
    parser.add_argument("--minimum-vertical-span", type=float, default=0.50)
    parser.add_argument("--minimum-major-horizontal-span", type=float, default=5.0)
    parser.add_argument("--minimum-minor-horizontal-span", type=float, default=2.0)
    parser.add_argument("--minimum-z", type=float, default=-0.30)
    parser.add_argument("--maximum-z", type=float, default=3.35)
    args = parser.parse_args()

    payload = json.loads(args.quality_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise SystemExit("[ERROR] quality report must contain exactly one PCD result")
    item = payload[0]
    declared = int(item["pcd_declared_points"])
    decoded = int(item["decoded_points"])
    finite = int(item["finite_points"])
    invalid_ratio = float(item["invalid_ratio"])
    bbox_min = [float(value) for value in item["bbox_min_xyz_m"]]
    bbox_max = [float(value) for value in item["bbox_max_xyz_m"]]
    p01 = [float(value) for value in item["percentile_01_xyz_m"]]
    errors = []
    if item.get("pcd_structure_valid") is not True:
        errors.append("PCD structure was not verified by the strict reader")
    if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
        errors.append("native PCD checksum is missing or invalid")
    numeric_values = [invalid_ratio, *bbox_min, *bbox_max, *p01]
    if len(bbox_min) != 3 or len(bbox_max) != 3 or len(p01) != 3:
        errors.append("quality report XYZ vectors must each have three elements")
    elif not all(math.isfinite(value) for value in numeric_values):
        errors.append("quality report contains non-finite summary values")
    if declared < args.minimum_points:
        errors.append(f"only {declared} points (minimum {args.minimum_points})")
    if not (declared == decoded == finite):
        errors.append(
            f"point count mismatch declared={declared} decoded={decoded} finite={finite}"
        )
    if invalid_ratio != 0.0:
        errors.append(f"invalid_ratio is {invalid_ratio}, expected 0")
    if len(bbox_min) == 3 and len(p01) == 3 and (
        bbox_min[2] > args.ground_maximum_z
        or p01[2] > args.ground_maximum_z
    ):
        errors.append(
            f"ground evidence missing: min_z={bbox_min[2]:.3f}, p01_z={p01[2]:.3f}"
        )
    vertical_span = bbox_max[2] - bbox_min[2] if len(bbox_min) == len(bbox_max) == 3 else 0.0
    if vertical_span < args.minimum_vertical_span:
        errors.append(
            f"vertical structure span {vertical_span:.3f} m is below "
            f"{args.minimum_vertical_span:.3f} m"
        )
    if len(bbox_min) == len(bbox_max) == 3:
        horizontal_spans = sorted(
            (bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1])
        )
        if horizontal_spans[1] < args.minimum_major_horizontal_span:
            errors.append(
                f"major horizontal span {horizontal_spans[1]:.3f} m is below "
                f"{args.minimum_major_horizontal_span:.3f} m"
            )
        if horizontal_spans[0] < args.minimum_minor_horizontal_span:
            errors.append(
                f"minor horizontal span {horizontal_spans[0]:.3f} m is below "
                f"{args.minimum_minor_horizontal_span:.3f} m"
            )
        if bbox_min[2] < args.minimum_z or bbox_max[2] > args.maximum_z:
            errors.append(
                f"z bounds [{bbox_min[2]:.3f}, {bbox_max[2]:.3f}] m exceed "
                f"configured [{args.minimum_z:.3f}, {args.maximum_z:.3f}] m"
            )
    if errors:
        print(json.dumps({"state": "FAILED", "errors": errors}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "state": "PASS",
                "points": declared,
                "invalid_ratio": invalid_ratio,
                "ground_p01_z_m": p01[2],
                "vertical_span_m": vertical_span,
                "horizontal_span_xy_m": [
                    bbox_max[0] - bbox_min[0],
                    bbox_max[1] - bbox_min[1],
                ],
                "z_bounds_m": [bbox_min[2], bbox_max[2]],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
