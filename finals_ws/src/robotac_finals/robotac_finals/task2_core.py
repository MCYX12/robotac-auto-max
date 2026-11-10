"""Pure helpers for ROBOTAC finals task-two routing and scoring."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence


Point = tuple[float, float]


def validate_targets(names: Sequence[str], coordinates: Sequence[float], expected: int) -> list[Point]:
    if len(names) != expected:
        raise ValueError(f"expected {expected} target names, got {len(names)}")
    if len(set(names)) != len(names) or any(not str(name).strip() for name in names):
        raise ValueError("target names must be non-empty and unique")
    if len(coordinates) != expected * 2:
        raise ValueError(f"expected {expected * 2} target coordinates, got {len(coordinates)}")
    values = [float(value) for value in coordinates]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("target coordinates must be finite")
    targets = [(values[index], values[index + 1]) for index in range(0, len(values), 2)]
    if len(set(targets)) != len(targets):
        raise ValueError("target coordinates must be pairwise distinct")
    return targets


def euclidean(first: Point, second: Point) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def best_visit_order(
    start: Point,
    targets: Sequence[Point],
    distance: Callable[[Point, Point], float] = euclidean,
) -> tuple[list[int], float]:
    """Return the deterministic minimum-cost open route through all targets."""
    if not targets:
        return [], 0.0
    best_order: tuple[int, ...] | None = None
    best_cost = float("inf")
    for order in itertools.permutations(range(len(targets))):
        previous = start
        cost = 0.0
        feasible = True
        for index in order:
            step = float(distance(previous, targets[index]))
            if not math.isfinite(step) or step < 0.0:
                feasible = False
                break
            cost += step
            previous = targets[index]
        if feasible and (cost < best_cost - 1.0e-9 or (
            abs(cost - best_cost) <= 1.0e-9 and (best_order is None or order < best_order)
        )):
            best_order = order
            best_cost = cost
    if best_order is None:
        raise ValueError("no route can reach every target")
    return list(best_order), best_cost


def task2_score(contact_count: int, elapsed_seconds: float, target_count: int = 5) -> dict:
    """Compute the v2.0 task-two score estimate from contact count and time."""
    contact_count = max(0, min(int(contact_count), int(target_count)))
    elapsed_seconds = max(0.0, float(elapsed_seconds))
    box_score = min(20.0, contact_count * 4.0)
    remaining_whole_minutes = max(0, int((1800.0 - elapsed_seconds) // 60.0))
    time_score = (
        min(10.0, remaining_whole_minutes * 0.8)
        if contact_count == target_count
        else 0.0
    )
    return {
        "box_score": box_score,
        "time_score": time_score,
        "total_score": box_score + time_score,
        "remaining_whole_minutes": remaining_whole_minutes,
    }
