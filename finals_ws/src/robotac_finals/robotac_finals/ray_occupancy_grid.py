#!/usr/bin/env python3
"""Build a navigation-only 2-D occupancy grid from live LiDAR rays.

This node never reads, filters, or republishes the score point cloud.  Its map
is a disposable navigation aid: ray interiors are observed free space and
returns inside the configured obstacle-height band are occupied endpoints.
The LiDAR-to-base and base-to-world transforms deliberately mirror the native
``odom_voxel_mapper`` used for the score PCD.

The pure helpers and :class:`DynamicLogOddsGrid` intentionally do not depend on
ROS, which keeps the safety-critical ray integration deterministic and easy to
unit test outside a ROS installation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def rpy_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return ``Rz(yaw) * Ry(pitch) * Rx(roll)``, matching Eigen mapper code."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def normalise_quaternion_xyzw(quaternion: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(quaternion), dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(values))
    if norm < 1.0e-12:
        raise ValueError("quaternion norm is zero")
    return values / norm


def quaternion_rotation_matrix(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    """Convert a normalised ROS-order quaternion (x, y, z, w) to a matrix."""
    x, y, z, w = normalise_quaternion_xyzw(quaternion_xyzw)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp_xyzw(first: Iterable[float], second: Iterable[float], alpha: float) -> np.ndarray:
    """Spherical interpolation equivalent to Eigen's Quaternion::slerp."""
    q0 = normalise_quaternion_xyzw(first)
    q1 = normalise_quaternion_xyzw(second)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalise_quaternion_xyzw(q0 + alpha * (q1 - q0))
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    scale = math.sin(theta)
    return (
        math.sin((1.0 - alpha) * theta) / scale * q0
        + math.sin(alpha * theta) / scale * q1
    )


def select_azimuth_endpoints(
    ray_directions_world: np.ndarray,
    planar_ranges: np.ndarray,
    ground_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    bin_degrees: float,
) -> np.ndarray:
    """Choose at most one occlusion-safe endpoint per horizontal azimuth bin.

    The closest obstacle takes precedence over every ground return in its bin.
    Only when a bin has no obstacle is its farthest ground return selected.
    With 0.5-degree bins the result is therefore bounded to 720 rays per scan.
    """
    directions = np.asarray(ray_directions_world, dtype=np.float64)
    distances = np.asarray(planar_ranges, dtype=np.float64)
    ground = np.asarray(ground_mask, dtype=bool)
    obstacles = np.asarray(obstacle_mask, dtype=bool)
    if directions.ndim != 2 or directions.shape[1:] != (3,):
        raise ValueError("ray_directions_world must have shape (N, 3)")
    count = len(directions)
    if distances.shape != (count,) or ground.shape != (count,) or obstacles.shape != (count,):
        raise ValueError("range and classification arrays must match ray count")
    if not (0.0 < float(bin_degrees) <= 180.0):
        raise ValueError("bin_degrees must be in (0, 180]")
    eligible = (
        np.all(np.isfinite(directions), axis=1)
        & np.isfinite(distances)
        & (distances > 0.0)
        & (ground | obstacles)
    )
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        return np.empty(0, dtype=np.int64)

    bin_count = int(math.ceil(360.0 / float(bin_degrees)))
    angles = np.arctan2(directions[candidates, 1], directions[candidates, 0])
    bin_width = 2.0 * math.pi / bin_count
    bins = np.floor((angles + math.pi) / bin_width).astype(np.int64)
    bins = np.clip(bins, 0, bin_count - 1)
    selected: list[int] = []
    for bin_id in np.unique(bins):
        members = candidates[bins == bin_id]
        obstacle_members = members[obstacles[members]]
        if len(obstacle_members):
            selected.append(
                int(obstacle_members[int(np.argmin(distances[obstacle_members]))])
            )
            continue
        ground_members = members[ground[members]]
        if len(ground_members):
            selected.append(
                int(ground_members[int(np.argmax(distances[ground_members]))])
            )
    return np.asarray(selected, dtype=np.int64)


def clip_planar_ray_endpoints(
    sensor_world_xy: Iterable[float],
    ray_vectors_world: np.ndarray,
    maximum_planar_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip ray endpoints in the navigation plane without inventing a hit.

    A physical return farther than the navigation horizon still proves that
    the segment up to that horizon was visible.  The clipped endpoint is
    therefore returned together with a boolean mask so callers can mark it as
    free rather than turning the artificial range boundary into an obstacle.
    """
    sensor = np.asarray(tuple(sensor_world_xy), dtype=np.float64)
    vectors = np.asarray(ray_vectors_world, dtype=np.float64)
    maximum = float(maximum_planar_range)
    if sensor.shape != (2,) or not np.all(np.isfinite(sensor)):
        raise ValueError("sensor_world_xy must contain two finite values")
    if vectors.ndim != 2 or vectors.shape[1:] != (3,):
        raise ValueError("ray_vectors_world must have shape (N, 3)")
    if not np.all(np.isfinite(vectors)) or not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("ray vectors and maximum_planar_range must be finite")
    planar_ranges = np.linalg.norm(vectors[:, :2], axis=1)
    if np.any(planar_ranges <= 0.0):
        raise ValueError("ray vectors must have non-zero planar length")
    truncated = planar_ranges > maximum
    scales = np.minimum(1.0, maximum / planar_ranges)
    endpoints = sensor + vectors[:, :2] * scales[:, None]
    return endpoints, truncated


def select_and_clip_planar_rays(
    sensor_world_xy: Iterable[float],
    ray_vectors_world: np.ndarray,
    ground_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    bin_degrees: float,
    maximum_planar_range: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the exact navigation endpoint selection and clipping pipeline."""
    vectors = np.asarray(ray_vectors_world, dtype=np.float64)
    planar_ranges = np.linalg.norm(vectors[:, :2], axis=1)
    selected = select_azimuth_endpoints(
        vectors, planar_ranges, ground_mask, obstacle_mask, bin_degrees
    )
    if not len(selected):
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=bool),
            selected,
        )
    endpoints, truncated = clip_planar_ray_endpoints(
        sensor_world_xy, vectors[selected], maximum_planar_range
    )
    obstacles = np.asarray(obstacle_mask, dtype=bool)[selected] & ~truncated
    return endpoints, obstacles, selected


def bresenham_cells(start: tuple[int, int], end: tuple[int, int]):
    """Yield an inclusive integer line in (row, column) order."""
    row0, column0 = int(start[0]), int(start[1])
    row1, column1 = int(end[0]), int(end[1])
    delta_column = abs(column1 - column0)
    step_column = 1 if column0 < column1 else -1
    delta_row = -abs(row1 - row0)
    step_row = 1 if row0 < row1 else -1
    error = delta_column + delta_row
    while True:
        yield row0, column0
        if row0 == row1 and column0 == column1:
            return
        doubled = 2 * error
        if doubled >= delta_row:
            error += delta_row
            column0 += step_column
        if doubled <= delta_column:
            error += delta_column
            row0 += step_row


def integrate_ray_cells(
    log_odds: np.ndarray,
    observed: np.ndarray,
    free_observations: np.ndarray,
    occupied_observations: np.ndarray,
    sensor_cell: tuple[int, int],
    endpoint_cells: np.ndarray,
    obstacle_endpoints: np.ndarray,
    *,
    free_delta: int,
    occupied_delta: int,
    minimum_log_odds: int,
    maximum_log_odds: int,
    occupied_threshold: int,
    occupied_confirmation_scans: int,
) -> tuple[int, int]:
    """Integrate one scan and return unique free/occupied update counts.

    A cell that is both traversed by one ray and an occupied endpoint of
    another ray is treated as occupied for that scan.  This density-invariant
    update prevents thousands of vertically stacked returns from dominating a
    single scan solely because of sensor layout.
    """
    if (
        log_odds.ndim != 2
        or observed.shape != log_odds.shape
        or free_observations.shape != log_odds.shape
        or occupied_observations.shape != log_odds.shape
    ):
        raise ValueError("all grid state arrays must have the same 2-D shape")
    endpoints = np.asarray(endpoint_cells, dtype=np.int64)
    obstacles = np.asarray(obstacle_endpoints, dtype=bool)
    if endpoints.ndim != 2 or endpoints.shape[1:] != (2,):
        raise ValueError("endpoint_cells must have shape (N, 2)")
    if obstacles.shape != (len(endpoints),):
        raise ValueError("obstacle_endpoints must have shape (N,)")
    height, width = log_odds.shape
    if not (0 <= sensor_cell[0] < height and 0 <= sensor_cell[1] < width):
        raise ValueError("sensor cell is outside the grid")

    free_indices: set[int] = set()
    occupied_indices: set[int] = set()
    confirmed_occupied = (
        (occupied_observations >= int(occupied_confirmation_scans))
        & (log_odds >= int(occupied_threshold))
    )
    for endpoint, is_obstacle in zip(endpoints, obstacles):
        end = int(endpoint[0]), int(endpoint[1])
        if not (0 <= end[0] < height and 0 <= end[1] < width):
            continue
        line = list(bresenham_cells(sensor_cell, end))
        blocking_index = next(
            (
                index
                for index, cell in enumerate(line[1:], start=1)
                if confirmed_occupied[cell]
            ),
            None,
        )
        if blocking_index is not None:
            # A confirmed wall is an absolute ray terminator.  Do not clear it
            # and, crucially, do not update any cell geometrically behind it.
            # The return behind this cell is legitimate counter-evidence for
            # the blocker itself, so update the blocker once for this scan but
            # keep every geometrically later cell protected.  A same-cell
            # occupied endpoint still wins via the conflict rule below.
            for row, column in line[: blocking_index + 1]:
                free_indices.add(row * width + column)
            if is_obstacle and blocking_index == len(line) - 1:
                occupied_indices.add(end[0] * width + end[1])
            continue
        free_line = line[:-1] if is_obstacle else line
        for row, column in free_line:
            free_indices.add(row * width + column)
        if is_obstacle:
            occupied_indices.add(end[0] * width + end[1])

    # Occupied evidence wins conflicts within this scan.  Across scans the
    # bounded scores behave like conventional log odds and may recover from an
    # isolated false return.
    free_indices.difference_update(occupied_indices)
    flat_scores = log_odds.reshape(-1)
    flat_observed = observed.reshape(-1)
    free_update_count = 0
    if free_indices:
        free_array = np.fromiter(free_indices, dtype=np.int64)
        free_update_count = int(len(free_array))
        updated = flat_scores[free_array].astype(np.int32) + int(free_delta)
        flat_scores[free_array] = np.clip(
            updated, int(minimum_log_odds), int(maximum_log_odds)
        ).astype(log_odds.dtype)
        flat_observed[free_array] = True
        flat_free_observations = free_observations.reshape(-1)
        flat_free_observations[free_array] = np.minimum(
            flat_free_observations[free_array].astype(np.uint32) + 1,
            np.iinfo(free_observations.dtype).max,
        ).astype(free_observations.dtype)
    if occupied_indices:
        occupied_array = np.fromiter(occupied_indices, dtype=np.int64)
        updated = flat_scores[occupied_array].astype(np.int32) + int(occupied_delta)
        flat_scores[occupied_array] = np.clip(
            updated, int(minimum_log_odds), int(maximum_log_odds)
        ).astype(log_odds.dtype)
        flat_observed[occupied_array] = True
        flat_occupied_observations = occupied_observations.reshape(-1)
        flat_occupied_observations[occupied_array] = np.minimum(
            flat_occupied_observations[occupied_array].astype(np.uint32) + 1,
            np.iinfo(occupied_observations.dtype).max,
        ).astype(occupied_observations.dtype)
    return free_update_count, len(occupied_indices)


class DynamicLogOddsGrid:
    """Dynamically expanding world-aligned log-odds grid."""

    def __init__(
        self,
        resolution: float,
        margin: float,
        maximum_cells: int,
        free_delta: int = -1,
        occupied_delta: int = 4,
        minimum_log_odds: int = -30,
        maximum_log_odds: int = 30,
        free_threshold: int = -1,
        occupied_threshold: int = 2,
        free_confirmation_scans: int = 2,
        occupied_confirmation_scans: int = 3,
    ) -> None:
        self.resolution = float(resolution)
        self.margin = float(margin)
        self.maximum_cells = int(maximum_cells)
        self.free_delta = int(free_delta)
        self.occupied_delta = int(occupied_delta)
        self.minimum_log_odds = int(minimum_log_odds)
        self.maximum_log_odds = int(maximum_log_odds)
        self.free_threshold = int(free_threshold)
        self.occupied_threshold = int(occupied_threshold)
        self.free_confirmation_scans = int(free_confirmation_scans)
        self.occupied_confirmation_scans = int(occupied_confirmation_scans)
        if self.resolution <= 0.0 or self.margin < 0.0 or self.maximum_cells <= 0:
            raise ValueError("resolution, margin, and maximum_cells are invalid")
        if not (
            self.minimum_log_odds < self.free_threshold <= 0
            and 0 < self.occupied_threshold < self.maximum_log_odds
            and self.free_delta < 0 < self.occupied_delta
            and self.free_confirmation_scans >= 2
            and self.occupied_confirmation_scans >= 3
        ):
            raise ValueError("invalid log-odds parameters")
        self.origin_xy: np.ndarray | None = None
        self.log_odds = np.zeros((0, 0), dtype=np.int16)
        self.observed = np.zeros((0, 0), dtype=bool)
        self.free_observations = np.zeros((0, 0), dtype=np.uint16)
        self.occupied_observations = np.zeros((0, 0), dtype=np.uint16)

    def _aligned_lower(self, values: np.ndarray) -> np.ndarray:
        return np.floor(values / self.resolution) * self.resolution

    def _aligned_upper(self, values: np.ndarray) -> np.ndarray:
        return np.ceil(values / self.resolution) * self.resolution

    def ensure_bounds(self, world_xy: np.ndarray) -> None:
        points = np.asarray(world_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
            raise ValueError("world_xy must have shape (N, 2), N > 0")
        if not np.all(np.isfinite(points)):
            raise ValueError("world_xy contains non-finite values")
        requested_lower = self._aligned_lower(points.min(axis=0) - self.margin)
        requested_upper = self._aligned_upper(points.max(axis=0) + self.margin)
        # Include the upper boundary cell rather than relying on exact floating
        # point equality at a cell edge.
        requested_upper += self.resolution

        if self.origin_xy is None:
            new_origin = requested_lower
            new_upper = requested_upper
            old_offset = None
        else:
            current_upper = self.origin_xy + np.asarray(
                [self.log_odds.shape[1], self.log_odds.shape[0]], dtype=np.float64
            ) * self.resolution
            new_origin = np.minimum(self.origin_xy, requested_lower)
            new_upper = np.maximum(current_upper, requested_upper)
            old_offset = np.rint((self.origin_xy - new_origin) / self.resolution).astype(int)

        dimensions = np.rint((new_upper - new_origin) / self.resolution).astype(int)
        width, height = int(dimensions[0]), int(dimensions[1])
        if width <= 0 or height <= 0 or width * height > self.maximum_cells:
            raise ValueError(
                f"requested navigation grid {width}x{height} exceeds "
                f"maximum_cells={self.maximum_cells}"
            )
        if self.origin_xy is not None and (height, width) == self.log_odds.shape:
            return

        new_scores = np.zeros((height, width), dtype=np.int16)
        new_observed = np.zeros((height, width), dtype=bool)
        new_free_observations = np.zeros((height, width), dtype=np.uint16)
        new_occupied_observations = np.zeros((height, width), dtype=np.uint16)
        if old_offset is not None:
            column_offset, row_offset = int(old_offset[0]), int(old_offset[1])
            old_height, old_width = self.log_odds.shape
            new_scores[
                row_offset : row_offset + old_height,
                column_offset : column_offset + old_width,
            ] = self.log_odds
            new_observed[
                row_offset : row_offset + old_height,
                column_offset : column_offset + old_width,
            ] = self.observed
            new_free_observations[
                row_offset : row_offset + old_height,
                column_offset : column_offset + old_width,
            ] = self.free_observations
            new_occupied_observations[
                row_offset : row_offset + old_height,
                column_offset : column_offset + old_width,
            ] = self.occupied_observations
        self.origin_xy = new_origin
        self.log_odds = new_scores
        self.observed = new_observed
        self.free_observations = new_free_observations
        self.occupied_observations = new_occupied_observations

    def world_to_cells(self, world_xy: np.ndarray) -> np.ndarray:
        if self.origin_xy is None:
            raise RuntimeError("grid bounds have not been initialised")
        points = np.asarray(world_xy, dtype=np.float64)
        columns_rows = np.floor(
            (points - self.origin_xy) / self.resolution + 1.0e-9
        ).astype(np.int64)
        return columns_rows[:, ::-1]

    def integrate(
        self,
        sensor_world_xy: Iterable[float],
        endpoints_world_xy: np.ndarray,
        obstacle_endpoints: np.ndarray,
    ) -> tuple[int, int]:
        sensor = np.asarray(tuple(sensor_world_xy), dtype=np.float64)
        endpoints = np.asarray(endpoints_world_xy, dtype=np.float64)
        obstacles = np.asarray(obstacle_endpoints, dtype=bool)
        if sensor.shape != (2,) or not np.all(np.isfinite(sensor)):
            raise ValueError("sensor_world_xy must contain two finite values")
        if endpoints.ndim != 2 or endpoints.shape[1:] != (2,):
            raise ValueError("endpoints_world_xy must have shape (N, 2)")
        if obstacles.shape != (len(endpoints),):
            raise ValueError("obstacle_endpoints must have shape (N,)")
        if not len(endpoints):
            return 0, 0
        self.ensure_bounds(np.vstack((sensor, endpoints)))
        sensor_cell = tuple(int(value) for value in self.world_to_cells(sensor[None, :])[0])
        endpoint_cells = self.world_to_cells(endpoints)
        return integrate_ray_cells(
            self.log_odds,
            self.observed,
            self.free_observations,
            self.occupied_observations,
            sensor_cell,
            endpoint_cells,
            obstacles,
            free_delta=self.free_delta,
            occupied_delta=self.occupied_delta,
            minimum_log_odds=self.minimum_log_odds,
            maximum_log_odds=self.maximum_log_odds,
            occupied_threshold=self.occupied_threshold,
            occupied_confirmation_scans=self.occupied_confirmation_scans,
        )

    def occupancy_values(self) -> np.ndarray:
        values = np.full(self.log_odds.shape, -1, dtype=np.int8)
        confirmed_free = (
            self.observed
            & (self.free_observations >= self.free_confirmation_scans)
            & (self.log_odds <= self.free_threshold)
        )
        confirmed_occupied = (
            self.observed
            & (self.occupied_observations >= self.occupied_confirmation_scans)
            & (self.log_odds >= self.occupied_threshold)
        )
        values[confirmed_free] = 0
        values[confirmed_occupied] = 100
        # Observed but numerically undecided cells remain unknown, which is the
        # conservative choice for a planner that never traverses unknown space.
        return values


@dataclass(frozen=True)
class OdomPose:
    stamp: float
    position: np.ndarray
    orientation_xyzw: np.ndarray
    yaw_rate: float


# ROS imports are optional so the algorithm tests remain truly ROS-independent.
try:
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
except ModuleNotFoundError:  # pragma: no cover - exercised by pure host tests
    rclpy = None
    Node = object


class RayOccupancyGrid(Node):
    def __init__(self) -> None:
        if rclpy is None:  # pragma: no cover - defensive runtime message
            raise RuntimeError("ROS 2 Python modules are required to run this node")
        super().__init__("robotac_ray_occupancy_grid")
        self.declare_parameter("lidar_topic", "/front_lidar")
        self.declare_parameter("odom_topic", "/odom/mujoco_odom")
        self.declare_parameter("grid_topic", "/robotac/nav_map")
        self.declare_parameter("map_frame", "world")
        self.declare_parameter("resolution", 0.10)
        self.declare_parameter("margin", 2.0)
        self.declare_parameter("max_grid_cells", 4000000)
        self.declare_parameter("min_range", 0.20)
        self.declare_parameter("input_max_range", 40.0)
        self.declare_parameter("max_range", 12.0)
        self.declare_parameter("world_z_min", -0.20)
        self.declare_parameter("world_z_max", 1.60)
        self.declare_parameter("ground_max_z", 0.15)
        self.declare_parameter("obstacle_min_z", 0.22)
        self.declare_parameter("obstacle_max_z", 1.60)
        self.declare_parameter("process_every_n_scans", 2)
        self.declare_parameter("azimuth_bin_degrees", 0.5)
        self.declare_parameter("max_rays_per_scan", 720)
        self.declare_parameter("publish_hz", 2.0)
        self.declare_parameter("max_odom_time_offset", 0.02)
        self.declare_parameter("maximum_fusion_yaw_rate", 0.15)
        self.declare_parameter("lidar_translation", [0.0, 0.0, 0.30])
        self.declare_parameter("lidar_rpy", [0.0, 0.0, 0.0])
        self.declare_parameter("self_filter_enabled", True)
        self.declare_parameter("self_min", [-0.45, -0.30, 0.00])
        self.declare_parameter("self_max", [0.45, 0.30, 0.55])
        self.declare_parameter("free_log_odds_delta", -1)
        self.declare_parameter("occupied_log_odds_delta", 4)
        self.declare_parameter("minimum_log_odds", -30)
        self.declare_parameter("maximum_log_odds", 30)
        self.declare_parameter("free_threshold", -1)
        self.declare_parameter("occupied_threshold", 2)
        self.declare_parameter("free_confirmation_scans", 2)
        self.declare_parameter("occupied_confirmation_scans", 3)

        self.lidar_topic = str(self.get_parameter("lidar_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.grid_topic = str(self.get_parameter("grid_topic").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.min_range = float(self.get_parameter("min_range").value)
        self.input_max_range = float(self.get_parameter("input_max_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.world_z_min = float(self.get_parameter("world_z_min").value)
        self.world_z_max = float(self.get_parameter("world_z_max").value)
        self.ground_max_z = float(self.get_parameter("ground_max_z").value)
        self.obstacle_min_z = float(self.get_parameter("obstacle_min_z").value)
        self.obstacle_max_z = float(self.get_parameter("obstacle_max_z").value)
        self.process_every_n_scans = int(
            self.get_parameter("process_every_n_scans").value
        )
        self.azimuth_bin_degrees = float(
            self.get_parameter("azimuth_bin_degrees").value
        )
        self.max_rays_per_scan = int(self.get_parameter("max_rays_per_scan").value)
        self.max_odom_time_offset = float(
            self.get_parameter("max_odom_time_offset").value
        )
        self.maximum_fusion_yaw_rate = float(
            self.get_parameter("maximum_fusion_yaw_rate").value
        )
        publish_hz = float(self.get_parameter("publish_hz").value)
        self.self_filter_enabled = bool(
            self.get_parameter("self_filter_enabled").value
        )
        self.lidar_translation = self._vector3_parameter("lidar_translation")
        lidar_rpy = self._vector3_parameter("lidar_rpy")
        self.lidar_rotation = rpy_rotation_matrix(*lidar_rpy)
        self.self_min = self._vector3_parameter("self_min")
        self.self_max = self._vector3_parameter("self_max")

        if (
            self.min_range < 0.0
            or self.max_range <= self.min_range
            or self.input_max_range < self.max_range
            or self.world_z_max <= self.world_z_min
            or not self.world_z_min <= self.ground_max_z < self.obstacle_min_z
            or self.obstacle_min_z < self.world_z_min
            or self.obstacle_max_z > self.world_z_max
            or self.obstacle_max_z <= self.obstacle_min_z
            or self.process_every_n_scans <= 0
            or not 0.0 < self.azimuth_bin_degrees <= 180.0
            or self.max_rays_per_scan <= 0
            or self.max_rays_per_scan > int(math.ceil(360.0 / self.azimuth_bin_degrees))
            or self.max_odom_time_offset <= 0.0
            or self.maximum_fusion_yaw_rate <= 0.0
            or publish_hz <= 0.0
        ):
            raise ValueError("invalid LiDAR, height, timing, or sampling parameters")

        self.grid = DynamicLogOddsGrid(
            resolution=float(self.get_parameter("resolution").value),
            margin=float(self.get_parameter("margin").value),
            maximum_cells=int(self.get_parameter("max_grid_cells").value),
            free_delta=int(self.get_parameter("free_log_odds_delta").value),
            occupied_delta=int(
                self.get_parameter("occupied_log_odds_delta").value
            ),
            minimum_log_odds=int(self.get_parameter("minimum_log_odds").value),
            maximum_log_odds=int(self.get_parameter("maximum_log_odds").value),
            free_threshold=int(self.get_parameter("free_threshold").value),
            occupied_threshold=int(self.get_parameter("occupied_threshold").value),
            free_confirmation_scans=int(
                self.get_parameter("free_confirmation_scans").value
            ),
            occupied_confirmation_scans=int(
                self.get_parameter("occupied_confirmation_scans").value
            ),
        )
        self.odom_poses: deque[OdomPose] = deque()
        self.received_scans = 0
        self.processed_scans = 0
        self.dropped_scans = 0
        self.motion_rejected_scans = 0
        self.last_stamp = None
        self.last_integrated_scan_stamp: float | None = None
        self.last_published_scan_stamp: float | None = None

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(
            OccupancyGrid, self.grid_topic, latched_qos
        )
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, sensor_qos)
        self.create_subscription(
            PointCloud2, self.lidar_topic, self._lidar_callback, sensor_qos
        )
        self.create_timer(1.0 / publish_hz, self._publish)
        self.get_logger().info(
            f"navigation-only ray grid: lidar={self.lidar_topic} "
            f"odom={self.odom_topic} grid={self.grid_topic} "
            f"resolution={self.grid.resolution:.3f}m "
            f"azimuth_bin={self.azimuth_bin_degrees:.2f}deg "
            f"max_rays={self.max_rays_per_scan}"
        )

    def _vector3_parameter(self, name: str) -> np.ndarray:
        values = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain exactly three finite values")
        return values

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

    def _odom_callback(self, message: "Odometry") -> None:
        pose = message.pose.pose
        try:
            orientation = normalise_quaternion_xyzw(
                (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
            )
        except ValueError:
            self.get_logger().warning("discarding odometry with invalid quaternion")
            return
        sample = OdomPose(
            stamp=self._stamp_seconds(message.header.stamp),
            position=np.asarray(
                [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
            ),
            orientation_xyzw=orientation,
            yaw_rate=float(message.twist.twist.angular.z),
        )
        if not np.all(np.isfinite(sample.position)) or not math.isfinite(sample.yaw_rate):
            self.get_logger().warning("discarding non-finite odometry sample")
            return
        self.odom_poses.append(sample)
        while self.odom_poses and sample.stamp - self.odom_poses[0].stamp > 5.0:
            self.odom_poses.popleft()

    def _odom_at_time(self, stamp: float) -> OdomPose | None:
        if not self.odom_poses:
            return None
        samples = list(self.odom_poses)
        next_index = next(
            (index for index, sample in enumerate(samples) if sample.stamp >= stamp),
            len(samples),
        )
        previous = samples[next_index - 1] if next_index > 0 else None
        following = samples[next_index] if next_index < len(samples) else None
        if previous is not None and following is not None and following.stamp > previous.stamp:
            before = stamp - previous.stamp
            after = following.stamp - stamp
            if (
                0.0 <= before <= self.max_odom_time_offset
                and 0.0 <= after <= self.max_odom_time_offset
            ):
                alpha = before / (following.stamp - previous.stamp)
                return OdomPose(
                    stamp=stamp,
                    position=(1.0 - alpha) * previous.position + alpha * following.position,
                    orientation_xyzw=slerp_xyzw(
                        previous.orientation_xyzw, following.orientation_xyzw, alpha
                    ),
                    yaw_rate=(1.0 - alpha) * previous.yaw_rate + alpha * following.yaw_rate,
                )
        candidates = [sample for sample in (previous, following) if sample is not None]
        best = min(candidates, key=lambda sample: abs(sample.stamp - stamp))
        if abs(best.stamp - stamp) > self.max_odom_time_offset:
            return None
        return best

    def _decode_points(self, message: "PointCloud2") -> np.ndarray:
        records = point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        try:
            points = np.column_stack(
                (records["x"], records["y"], records["z"])
            ).astype(np.float64, copy=False)
        except (IndexError, KeyError, TypeError, ValueError):
            # Compatibility with sensor_msgs_py versions that return an
            # iterable of tuples instead of a structured NumPy array.
            points = np.asarray(list(records), dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("PointCloud2 x/y/z fields do not form an Nx3 array")
        return points

    def _lidar_callback(self, message: "PointCloud2") -> None:
        self.received_scans += 1
        if (self.received_scans - 1) % self.process_every_n_scans:
            return
        stamp = self._stamp_seconds(message.header.stamp)
        if (
            self.last_integrated_scan_stamp is not None
            and stamp <= self.last_integrated_scan_stamp
        ):
            self.dropped_scans += 1
            self.get_logger().warning(
                "rejecting duplicate or out-of-order LiDAR timestamp",
                throttle_duration_sec=2.0,
            )
            return
        odom = self._odom_at_time(stamp)
        if odom is None:
            self.dropped_scans += 1
            self.get_logger().warning(
                f"no odometry within {self.max_odom_time_offset:.3f}s for LiDAR "
                f"(dropped={self.dropped_scans})",
                throttle_duration_sec=2.0,
            )
            return
        if abs(odom.yaw_rate) > self.maximum_fusion_yaw_rate:
            # The competition cloud does not carry per-point time offsets, so
            # deskew is impossible here.  Reject fast-turning frames instead of
            # painting a rotationally smeared wall into the navigation grid.
            self.motion_rejected_scans += 1
            self.get_logger().warning(
                f"rejecting LiDAR while |yaw_rate|={abs(odom.yaw_rate):.3f} "
                f"> {self.maximum_fusion_yaw_rate:.3f}rad/s "
                f"(motion_rejected={self.motion_rejected_scans})",
                throttle_duration_sec=2.0,
            )
            return
        try:
            points_lidar = self._decode_points(message)
        except (AssertionError, KeyError, RuntimeError, TypeError, ValueError) as error:
            self.dropped_scans += 1
            self.get_logger().error(f"cannot decode LiDAR PointCloud2: {error}")
            return
        if not len(points_lidar):
            return
        finite = np.all(np.isfinite(points_lidar), axis=1)
        ranges = np.linalg.norm(points_lidar, axis=1)
        range_mask = (
            finite
            & (ranges >= self.min_range)
            & (ranges <= self.input_max_range)
        )
        points_lidar = points_lidar[range_mask]
        ranges = ranges[range_mask]
        if not len(points_lidar):
            return

        ray_directions_base = points_lidar @ self.lidar_rotation.T
        points_base = ray_directions_base + self.lidar_translation
        if self.self_filter_enabled:
            inside_self = np.all(
                (points_base >= self.self_min) & (points_base <= self.self_max), axis=1
            )
            points_base = points_base[~inside_self]
            ray_directions_base = ray_directions_base[~inside_self]
            ranges = ranges[~inside_self]
        if not len(points_base):
            return

        odom_rotation = quaternion_rotation_matrix(odom.orientation_xyzw)
        endpoints_world = points_base @ odom_rotation.T + odom.position
        sensor_world = odom_rotation @ self.lidar_translation + odom.position
        ray_vectors_world = endpoints_world - sensor_world
        ground_mask = (
            (endpoints_world[:, 2] >= self.world_z_min)
            & (endpoints_world[:, 2] <= self.ground_max_z)
        )
        obstacle_mask = (
            (endpoints_world[:, 2] >= self.obstacle_min_z)
            & (endpoints_world[:, 2] <= self.obstacle_max_z)
        )
        endpoints_world_xy, obstacle_endpoints, selected = select_and_clip_planar_rays(
            sensor_world[:2],
            ray_vectors_world,
            ground_mask,
            obstacle_mask,
            self.azimuth_bin_degrees,
            self.max_range,
        )
        if not len(selected):
            return
        if len(selected) > self.max_rays_per_scan:
            # The configured 0.5-degree bins already cap this at 720.  Keep a
            # deterministic secondary cap for defensive parameter overrides.
            selection_stride = int(math.ceil(len(selected) / self.max_rays_per_scan))
            selected = selected[::selection_stride][: self.max_rays_per_scan]
        try:
            free_count, occupied_count = self.grid.integrate(
                sensor_world[:2], endpoints_world_xy, obstacle_endpoints
            )
        except ValueError as error:
            self.dropped_scans += 1
            self.get_logger().error(f"ray grid rejected scan: {error}")
            return
        self.processed_scans += 1
        self.last_integrated_scan_stamp = stamp
        self.last_stamp = message.header.stamp
        self.get_logger().info(
            f"ray_grid scans={self.processed_scans} rays={len(endpoints_world_xy)} "
            f"free_updates={free_count} occupied_updates={occupied_count}",
            throttle_duration_sec=5.0,
        )

    def _publish(self) -> None:
        if self.grid.origin_xy is None or self.last_stamp is None:
            return
        if (
            self.last_integrated_scan_stamp is None
            or self.last_integrated_scan_stamp == self.last_published_scan_stamp
        ):
            # Never refresh a stale map merely because the publication timer
            # fired.  Downstream motion safety keys freshness to progressing
            # LiDAR timestamps, not to repeated delivery of old state.
            return
        cells = self.grid.occupancy_values()
        output = OccupancyGrid()
        output.header.stamp = self.last_stamp
        output.header.frame_id = self.map_frame
        output.info.resolution = self.grid.resolution
        output.info.width = cells.shape[1]
        output.info.height = cells.shape[0]
        output.info.origin.position.x = float(self.grid.origin_xy[0])
        output.info.origin.position.y = float(self.grid.origin_xy[1])
        output.info.origin.orientation.w = 1.0
        output.data = cells.reshape(-1).tolist()
        self.publisher.publish(output)
        self.last_published_scan_stamp = self.last_integrated_scan_stamp


def main(args=None) -> None:
    if rclpy is None:  # pragma: no cover
        raise RuntimeError("ROS 2 Python modules are required to run this node")
    rclpy.init(args=args)
    node = RayOccupancyGrid()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass


if __name__ == "__main__":
    main()
