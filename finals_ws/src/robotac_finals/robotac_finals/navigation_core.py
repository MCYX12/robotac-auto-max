"""Pure grid-planning helpers used by the task-one exploration controller."""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np


Cell = tuple[int, int]


def world_to_cell(x: float, y: float, origin_x: float, origin_y: float, resolution: float) -> Cell:
    """Convert world coordinates to a (row, column) grid cell."""
    return (
        int(math.floor((y - origin_y) / resolution)),
        int(math.floor((x - origin_x) / resolution)),
    )


def cell_to_world(cell: Cell, origin_x: float, origin_y: float, resolution: float) -> tuple[float, float]:
    row, column = cell
    return (
        origin_x + (column + 0.5) * resolution,
        origin_y + (row + 0.5) * resolution,
    )


def inflate_obstacles(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    """Return a blocked mask containing occupied/unknown cells and inflation."""
    occupied = grid > 50
    inflated = occupied.copy()
    radius_cells = max(0, int(radius_cells))
    for delta_row in range(-radius_cells, radius_cells + 1):
        for delta_column in range(-radius_cells, radius_cells + 1):
            if delta_row * delta_row + delta_column * delta_column > radius_cells * radius_cells:
                continue
            source_r0 = max(0, -delta_row)
            source_r1 = min(grid.shape[0], grid.shape[0] - delta_row)
            source_c0 = max(0, -delta_column)
            source_c1 = min(grid.shape[1], grid.shape[1] - delta_column)
            target_r0 = source_r0 + delta_row
            target_r1 = source_r1 + delta_row
            target_c0 = source_c0 + delta_column
            target_c1 = source_c1 + delta_column
            inflated[target_r0:target_r1, target_c0:target_c1] |= occupied[
                source_r0:source_r1, source_c0:source_c1
            ]
    # Unknown space is never traversed. It is added after inflation so an unknown
    # frontier does not unnecessarily erode already observed free corridors.
    return inflated | (grid < 0)


def nearest_free(blocked: np.ndarray, requested: Cell, maximum_radius: int = 12) -> Cell | None:
    """Find the nearest unblocked cell around a possibly inflated goal/start."""
    row, column = requested
    if 0 <= row < blocked.shape[0] and 0 <= column < blocked.shape[1] and not blocked[row, column]:
        return requested
    best = None
    best_distance = float("inf")
    for radius in range(1, maximum_radius + 1):
        for candidate_row in range(row - radius, row + radius + 1):
            for candidate_column in range(column - radius, column + radius + 1):
                if not (0 <= candidate_row < blocked.shape[0] and 0 <= candidate_column < blocked.shape[1]):
                    continue
                if blocked[candidate_row, candidate_column]:
                    continue
                distance = (candidate_row - row) ** 2 + (candidate_column - column) ** 2
                if distance < best_distance:
                    best = (candidate_row, candidate_column)
                    best_distance = distance
        if best is not None:
            return best
    return None


def reachable_mask(blocked: np.ndarray, start: Cell) -> np.ndarray:
    """Return cells connected to start with the planner's no-corner-cutting rule."""
    reachable = np.zeros_like(blocked, dtype=bool)
    if (
        blocked.ndim != 2
        or not (0 <= start[0] < blocked.shape[0] and 0 <= start[1] < blocked.shape[1])
        or blocked[start]
    ):
        return reachable
    queue = deque([start])
    reachable[start] = True
    while queue:
        row, column = queue.popleft()
        for delta_row in (-1, 0, 1):
            for delta_column in (-1, 0, 1):
                if delta_row == 0 and delta_column == 0:
                    continue
                neighbour = (row + delta_row, column + delta_column)
                if not (
                    0 <= neighbour[0] < blocked.shape[0]
                    and 0 <= neighbour[1] < blocked.shape[1]
                ):
                    continue
                if blocked[neighbour] or reachable[neighbour]:
                    continue
                if delta_row and delta_column:
                    if blocked[row + delta_row, column] or blocked[row, column + delta_column]:
                        continue
                reachable[neighbour] = True
                queue.append(neighbour)
    return reachable


def select_patrol_cell(
    grid: np.ndarray,
    blocked: np.ndarray,
    reachable: np.ndarray,
    start: Cell,
    visited: list[Cell] | tuple[Cell, ...] = (),
    minimum_distance_cells: int = 1,
    visited_radius_cells: int = 1,
    prefer_nearby: bool = False,
) -> Cell | None:
    """Select the farthest safe, reachable, *known-free* patrol cell.

    Patrol is deliberately stricter than reachability.  The controller may
    temporarily clear a local sensor blind zone in ``blocked``; requiring the
    original occupancy value to be exactly zero prevents one of those unknown
    cells from ever becoming a patrol goal.  ``visited`` cells are excluded by
    a circular radius so a completed patrol cannot immediately oscillate to a
    neighbouring cell.
    """
    if (
        grid.ndim != 2
        or grid.shape != blocked.shape
        or grid.shape != reachable.shape
        or not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1])
    ):
        return None

    eligible = (grid == 0) & reachable & ~blocked
    rows, columns = np.ogrid[: grid.shape[0], : grid.shape[1]]
    minimum_distance_cells = max(0, int(minimum_distance_cells))
    if minimum_distance_cells:
        from_start = (rows - start[0]) ** 2 + (columns - start[1]) ** 2
        eligible &= from_start >= minimum_distance_cells * minimum_distance_cells

    visited_radius_cells = max(0, int(visited_radius_cells))
    radius_squared = visited_radius_cells * visited_radius_cells
    for visited_row, visited_column in visited:
        row0 = max(0, int(visited_row) - visited_radius_cells)
        row1 = min(grid.shape[0], int(visited_row) + visited_radius_cells + 1)
        column0 = max(0, int(visited_column) - visited_radius_cells)
        column1 = min(grid.shape[1], int(visited_column) + visited_radius_cells + 1)
        if row0 >= row1 or column0 >= column1:
            continue
        local_rows, local_columns = np.ogrid[row0:row1, column0:column1]
        local_disk = (
            (local_rows - int(visited_row)) ** 2
            + (local_columns - int(visited_column)) ** 2
            <= radius_squared
        )
        eligible[row0:row1, column0:column1][local_disk] = False

    candidates = np.argwhere(eligible)
    if candidates.size == 0:
        return None
    distances_squared = np.sum((candidates - np.asarray(start)) ** 2, axis=1)
    index = np.argmin(distances_squared) if prefer_nearby else np.argmax(distances_squared)
    row, column = candidates[int(index)]
    return int(row), int(column)


def astar(blocked: np.ndarray, start: Cell, goal: Cell) -> list[Cell]:
    """Plan an eight-connected path without corner cutting."""
    if blocked.ndim != 2 or blocked.size == 0:
        return []
    height, width = blocked.shape
    if not (
        0 <= start[0] < height
        and 0 <= start[1] < width
        and 0 <= goal[0] < height
        and 0 <= goal[1] < width
    ):
        return []
    if blocked[start] or blocked[goal]:
        return []

    steps = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    queue: list[tuple[float, float, Cell]] = [(0.0, 0.0, start)]
    parent: dict[Cell, Cell] = {}
    cost = {start: 0.0}
    closed: set[Cell] = set()

    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current != start:
                current = parent[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)
        row, column = current
        for delta_row, delta_column, step_cost in steps:
            neighbour = (row + delta_row, column + delta_column)
            if not (0 <= neighbour[0] < height and 0 <= neighbour[1] < width):
                continue
            if blocked[neighbour] or neighbour in closed:
                continue
            if delta_row and delta_column:
                if blocked[row + delta_row, column] or blocked[row, column + delta_column]:
                    continue
            tentative = current_cost + step_cost
            if tentative >= cost.get(neighbour, float("inf")):
                continue
            cost[neighbour] = tentative
            parent[neighbour] = current
            heuristic = math.hypot(goal[0] - neighbour[0], goal[1] - neighbour[1])
            heapq.heappush(queue, (tentative + heuristic, tentative, neighbour))
    return []


def line_is_free(blocked: np.ndarray, first: Cell, second: Cell) -> bool:
    """Check a grid segment using dense interpolation."""
    delta_row = second[0] - first[0]
    delta_column = second[1] - first[1]
    steps = max(abs(delta_row), abs(delta_column))
    if steps == 0:
        return not blocked[first]
    previous = first
    for index in range(steps + 1):
        alpha = index / steps
        current = (
            int(round(first[0] + alpha * delta_row)),
            int(round(first[1] + alpha * delta_column)),
        )
        if blocked[current]:
            return False
        if current[0] != previous[0] and current[1] != previous[1]:
            if blocked[current[0], previous[1]] or blocked[previous[0], current[1]]:
                return False
        previous = current
    return True


def simplify_path(path: list[Cell], blocked: np.ndarray) -> list[Cell]:
    """Greedily remove grid zig-zags while preserving collision clearance."""
    if len(path) <= 2:
        return path.copy()
    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1 and not line_is_free(blocked, path[anchor], path[candidate]):
            candidate -= 1
        simplified.append(path[candidate])
        anchor = candidate
    return simplified
