"""Pure helpers for reproducible, simulation-only Task-2 treasure placement."""

from __future__ import annotations

import math

import numpy as np

from .navigation_core import (
    cell_to_world,
    inflate_obstacles,
    nearest_free,
    reachable_mask,
    world_to_cell,
)


def project_points_to_grid(
    points: np.ndarray,
    *,
    resolution: float = 0.10,
    margin: float = 2.0,
    ground_min: float = -0.20,
    ground_max: float = 0.18,
    obstacle_min: float = 0.22,
    obstacle_max: float = 1.60,
    free_fill_radius: int = 2,
    max_grid_cells: int = 4_000_000,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Project XYZ points with the same height bands used by Task 2."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if resolution <= 0.0 or ground_max > obstacle_min:
        raise ValueError("invalid grid resolution or overlapping height bands")
    usable = values[
        np.all(np.isfinite(values), axis=1)
        & (values[:, 2] >= ground_min)
        & (values[:, 2] <= obstacle_max)
    ]
    if not len(usable):
        raise ValueError("task-one PCD has no points in navigation height bands")
    xy_min = usable[:, :2].min(axis=0) - margin
    xy_max = usable[:, :2].max(axis=0) + margin
    origin = np.floor(xy_min / resolution) * resolution
    dimensions = np.ceil((xy_max - origin) / resolution).astype(int) + 1
    width, height = int(dimensions[0]), int(dimensions[1])
    if width <= 0 or height <= 0 or width * height > max_grid_cells:
        raise ValueError(f"refusing projected grid {width}x{height}")

    grid = np.full((height, width), -1, dtype=np.int8)
    indices = np.floor((usable[:, :2] - origin) / resolution).astype(int)
    valid = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < width)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < height)
    )
    usable = usable[valid]
    indices = indices[valid]
    ground = usable[:, 2] <= ground_max
    obstacle = usable[:, 2] >= obstacle_min
    ground_cells = np.zeros_like(grid, dtype=bool)
    ground_cells[indices[ground, 1], indices[ground, 0]] = True
    free_cells = ground_cells.copy()
    radius = max(0, int(free_fill_radius))
    for delta_row in range(-radius, radius + 1):
        for delta_column in range(-radius, radius + 1):
            if delta_row * delta_row + delta_column * delta_column > radius * radius:
                continue
            source_r0 = max(0, -delta_row)
            source_r1 = min(grid.shape[0], grid.shape[0] - delta_row)
            source_c0 = max(0, -delta_column)
            source_c1 = min(grid.shape[1], grid.shape[1] - delta_column)
            free_cells[
                source_r0 + delta_row : source_r1 + delta_row,
                source_c0 + delta_column : source_c1 + delta_column,
            ] |= ground_cells[source_r0:source_r1, source_c0:source_c1]
    grid[free_cells] = 0
    grid[indices[obstacle, 1], indices[obstacle, 0]] = 100
    return grid, (float(origin[0]), float(origin[1]))


def sample_reachable_targets(
    grid: np.ndarray,
    origin: tuple[float, float],
    *,
    resolution: float,
    start: tuple[float, float],
    seed: int,
    count: int = 5,
    clearance: float = 0.44,
    minimum_separation: float = 3.0,
    minimum_start_distance: float = 1.5,
) -> tuple[list[tuple[float, float]], dict]:
    """Choose seeded random, pairwise-separated cells reachable from start."""
    if grid.ndim != 2 or grid.size == 0 or resolution <= 0.0 or count <= 0:
        raise ValueError("invalid grid or target sampling parameters")
    clearance_cells = int(math.ceil(clearance / resolution))
    blocked = inflate_obstacles(grid, clearance_cells)
    requested_start = world_to_cell(start[0], start[1], origin[0], origin[1], resolution)
    start_cell = nearest_free(blocked, requested_start, maximum_radius=100)
    if start_cell is None:
        raise ValueError("no safe cell near the configured simulation start")
    reachable = reachable_mask(blocked, start_cell)
    candidates = np.argwhere(reachable & (grid == 0) & ~blocked)
    if not len(candidates):
        raise ValueError("no reachable known-free cells for simulated boxes")
    start_distance_cells = minimum_start_distance / resolution
    distance_squared = np.sum((candidates - np.asarray(start_cell)) ** 2, axis=1)
    candidates = candidates[distance_squared >= start_distance_cells**2]
    if len(candidates) < count:
        raise ValueError("too few reachable cells after start-distance filtering")

    rng = np.random.default_rng(int(seed))
    separation_cells_squared = (minimum_separation / resolution) ** 2
    selected: list[np.ndarray] | None = None
    for _ in range(128):
        trial: list[np.ndarray] = []
        for index in rng.permutation(len(candidates)):
            candidate = candidates[int(index)]
            if all(
                float(np.sum((candidate - existing) ** 2))
                >= separation_cells_squared
                for existing in trial
            ):
                trial.append(candidate)
                if len(trial) == count:
                    selected = trial
                    break
        if selected is not None:
            break
    if selected is None:
        raise ValueError(
            "cannot place pairwise-separated simulated boxes on the reachable map"
        )

    cells = [(int(cell[0]), int(cell[1])) for cell in selected]
    targets = [
        cell_to_world(cell, origin[0], origin[1], resolution) for cell in cells
    ]
    return targets, {
        "seed": int(seed),
        "start_world": [float(start[0]), float(start[1])],
        "start_cell": [int(start_cell[0]), int(start_cell[1])],
        "reachable_cells": int(np.count_nonzero(reachable)),
        "candidate_cells": int(len(candidates)),
        "clearance_metres": float(clearance),
        "minimum_separation_metres": float(minimum_separation),
        "target_cells": [[row, column] for row, column in cells],
    }
