#!/usr/bin/env python3
"""Generate a reproducible local-only five-box scenario from a sealed Task-1 PCD."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from .pcd_map_server import load_xyz_pcd, sha256_file, validate_task1_evidence
from .task2_sim_core import project_points_to_grid, sample_reachable_targets


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1-pcd", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--output-metadata", required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    options = parser.parse_args(args)

    pcd = Path(options.task1_pcd).expanduser().resolve(strict=True)
    output_yaml = Path(options.output_yaml).expanduser()
    output_metadata = Path(options.output_metadata).expanduser()
    for output in (output_yaml, output_metadata):
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing scenario output: {output}")

    points, pcd_metadata = load_xyz_pcd(pcd, maximum_points=1_500_000)
    digest = sha256_file(pcd)
    evidence = validate_task1_evidence(
        pcd, digest, int(pcd_metadata["finite_points"])
    )
    grid, origin = project_points_to_grid(points)
    targets, sample_metadata = sample_reachable_targets(
        grid,
        origin,
        resolution=0.10,
        start=(options.start_x, options.start_y),
        seed=options.seed,
    )
    names = [f"official_cashbox_sim_{index}" for index in range(1, 6)]
    coordinates = [round(value, 3) for target in targets for value in target]
    provenance = (
        f"LOCAL_SIMULATION_ONLY seed={options.seed} task1_sha256={digest}"
    )
    def target_parameters() -> dict:
        # ROS 2's parameter parser rejects YAML aliases. Return independent
        # list objects for each node so PyYAML cannot emit anchors/references.
        return {
            "target_profile_kind": "local_random_simulation",
            "target_names": list(names),
            "target_coordinates": list(coordinates),
            "target_coordinate_provenance": provenance,
            "targets_verified": True,
        }

    configuration = {
        "robotac_task2_mission_manager": {
            "ros__parameters": target_parameters()
        },
        "robotac_task2_box_simulator": {
            "ros__parameters": {
                **target_parameters(),
                "scenario_seed": int(options.seed),
                "task1_map_sha256": digest,
            }
        },
    }
    metadata = {
        "state": "READY",
        "profile_kind": "local_random_simulation",
        "official_evidence": False,
        "warning": (
            "Official cashbox mesh copies at local random coordinates; "
            "not an organizer USB scene or score."
        ),
        "visual_asset": "package://robotac_finals/meshes/official_cashbox.glb",
        "task1_pcd": str(pcd),
        "task1_pcd_sha256": digest,
        "task1_pcd_metadata": pcd_metadata,
        "task1_evidence": evidence,
        "grid": {
            "width": int(grid.shape[1]),
            "height": int(grid.shape[0]),
            "resolution": 0.10,
            "origin": list(origin),
            "free_cells": int(np.count_nonzero(grid == 0)),
            "occupied_cells": int(np.count_nonzero(grid == 100)),
        },
        "targets": [
            {"name": name, "x": target[0], "y": target[1]}
            for name, target in zip(names, targets)
        ],
        "sampling": sample_metadata,
    }
    atomic_write(output_yaml, yaml.safe_dump(configuration, sort_keys=False))
    atomic_write(
        output_metadata,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        f"TASK2_SIM_SCENARIO_READY seed={options.seed} "
        f"targets={len(targets)} task1_sha256={digest}"
    )
    for item in metadata["targets"]:
        print(f"  {item['name']}: x={item['x']:.3f} y={item['y']:.3f}")


if __name__ == "__main__":
    main()
