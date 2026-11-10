#!/usr/bin/env python3
"""Project the live 3D score cloud into a navigation-only occupancy grid."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class CloudToGrid(Node):
    def __init__(self) -> None:
        super().__init__("robotac_cloud_to_grid")
        self.declare_parameter("cloud_topic", "/robotac/finals/score_cloud")
        self.declare_parameter("grid_topic", "/robotac/nav_map")
        self.declare_parameter("resolution", 0.10)
        self.declare_parameter("margin", 2.0)
        self.declare_parameter("ground_min_z", -0.20)
        self.declare_parameter("ground_max_z", 0.18)
        self.declare_parameter("obstacle_min_z", 0.18)
        self.declare_parameter("obstacle_max_z", 1.60)
        self.declare_parameter("max_input_points", 1500000)
        self.declare_parameter("max_grid_cells", 4000000)
        self.declare_parameter("free_fill_radius_cells", 2)

        self.cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.grid_topic = str(self.get_parameter("grid_topic").value)
        self.resolution = float(self.get_parameter("resolution").value)
        self.margin = float(self.get_parameter("margin").value)
        self.ground_min = float(self.get_parameter("ground_min_z").value)
        self.ground_max = float(self.get_parameter("ground_max_z").value)
        self.obstacle_min = float(self.get_parameter("obstacle_min_z").value)
        self.obstacle_max = float(self.get_parameter("obstacle_max_z").value)
        self.max_input = int(self.get_parameter("max_input_points").value)
        self.max_cells = int(self.get_parameter("max_grid_cells").value)
        self.free_fill_radius = int(
            self.get_parameter("free_fill_radius_cells").value
        )
        if self.resolution <= 0.0 or self.ground_max > self.obstacle_min:
            raise ValueError("invalid grid resolution or overlapping height bands")

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(
            OccupancyGrid, self.grid_topic, latched_qos
        )
        self.subscription = self.create_subscription(
            PointCloud2, self.cloud_topic, self._cloud_callback, latched_qos
        )
        self.get_logger().info(
            f"cloud={self.cloud_topic} grid={self.grid_topic} "
            f"resolution={self.resolution:.3f}m"
        )

    def _cloud_callback(self, message: PointCloud2) -> None:
        try:
            records = point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
            points = np.column_stack(
                (records["x"], records["y"], records["z"])
            ).astype(np.float64, copy=False)
        except (AssertionError, KeyError, ValueError) as error:
            self.get_logger().error(f"cannot decode score cloud: {error}")
            return
        if not len(points):
            return
        if len(points) > self.max_input:
            stride = int(math.ceil(len(points) / self.max_input))
            points = points[::stride]

        usable = points[
            (points[:, 2] >= self.ground_min)
            & (points[:, 2] <= self.obstacle_max)
        ]
        if not len(usable):
            return
        xy_min = usable[:, :2].min(axis=0) - self.margin
        xy_max = usable[:, :2].max(axis=0) + self.margin
        origin = np.floor(xy_min / self.resolution) * self.resolution
        dimensions = np.ceil((xy_max - origin) / self.resolution).astype(int) + 1
        width, height = int(dimensions[0]), int(dimensions[1])
        if width <= 0 or height <= 0 or width * height > self.max_cells:
            self.get_logger().error(
                f"refusing grid {width}x{height}; check cloud units/outliers"
            )
            return

        cells = np.full((height, width), -1, dtype=np.int8)
        indices = np.floor((usable[:, :2] - origin) / self.resolution).astype(int)
        valid = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < width)
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < height)
        )
        usable = usable[valid]
        indices = indices[valid]
        ground = usable[:, 2] <= self.ground_max
        obstacle = usable[:, 2] >= self.obstacle_min
        ground_cells = np.zeros_like(cells, dtype=bool)
        ground_cells[indices[ground, 1], indices[ground, 0]] = True
        # Ground returns are sampled rather than raster-complete. Fill only a
        # small neighbourhood around measured ground, then let obstacles win.
        # The planner's larger obstacle inflation prevents this fill crossing a wall.
        free_cells = ground_cells.copy()
        radius = max(0, self.free_fill_radius)
        for delta_row in range(-radius, radius + 1):
            for delta_column in range(-radius, radius + 1):
                if delta_row * delta_row + delta_column * delta_column > radius * radius:
                    continue
                source_r0 = max(0, -delta_row)
                source_r1 = min(cells.shape[0], cells.shape[0] - delta_row)
                source_c0 = max(0, -delta_column)
                source_c1 = min(cells.shape[1], cells.shape[1] - delta_column)
                free_cells[
                    source_r0 + delta_row : source_r1 + delta_row,
                    source_c0 + delta_column : source_c1 + delta_column,
                ] |= ground_cells[source_r0:source_r1, source_c0:source_c1]
        cells[free_cells] = 0
        cells[indices[obstacle, 1], indices[obstacle, 0]] = 100

        output = OccupancyGrid()
        output.header = message.header
        output.header.frame_id = "world"
        output.info.resolution = self.resolution
        output.info.width = width
        output.info.height = height
        output.info.origin.position.x = float(origin[0])
        output.info.origin.position.y = float(origin[1])
        output.info.origin.orientation.w = 1.0
        output.data = cells.reshape(-1).tolist()
        self.publisher.publish(output)
        free_count = int(np.count_nonzero(cells == 0))
        occupied_count = int(np.count_nonzero(cells == 100))
        self.get_logger().info(
            f"nav_map={width}x{height} free={free_count} occupied={occupied_count}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CloudToGrid()
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
