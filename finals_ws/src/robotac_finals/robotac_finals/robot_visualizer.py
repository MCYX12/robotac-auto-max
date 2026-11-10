#!/usr/bin/env python3
"""Publish a responsive robot arrow, trail, safety state, and mission status."""

from __future__ import annotations

from collections import deque
import json
import math

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker, MarkerArray


class RobotVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("robotac_robot_visualizer")
        self.declare_parameter("odom_topic", "/odom/mujoco_odom")
        self.declare_parameter("marker_topic", "/robotac/robot_markers")
        self.declare_parameter("safety_state_topic", "/robotac/control_safety_state")
        self.declare_parameter("task_status_topic", "/robotac/task1_status")
        self.declare_parameter("front_clearance_topic", "/robotac/front_clearance")
        self.declare_parameter("navigation_map_topic", "/robotac/nav_map")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("trail_spacing", 0.10)
        self.declare_parameter("trail_max_points", 3000)
        self.declare_parameter("robot_length", 0.90)
        self.declare_parameter("robot_width", 0.60)
        self.declare_parameter("safety_radius", 0.50)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.trail_spacing = float(self.get_parameter("trail_spacing").value)
        trail_max = int(self.get_parameter("trail_max_points").value)
        self.robot_length = float(self.get_parameter("robot_length").value)
        self.robot_width = float(self.get_parameter("robot_width").value)
        self.safety_radius = float(self.get_parameter("safety_radius").value)
        if (
            self.publish_hz <= 0.0
            or self.trail_spacing <= 0.0
            or trail_max <= 1
            or min(self.robot_length, self.robot_width, self.safety_radius) <= 0.0
            or self.safety_radius < self.robot_width / 2.0
        ):
            raise ValueError("invalid RViz visualization rates or trail limits")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_topic").value), qos
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("safety_state_topic").value),
            self._safety_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("task_status_topic").value),
            self._task_status_callback,
            qos,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("front_clearance_topic").value),
            self._clearance_callback,
            10,
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("navigation_map_topic").value),
            self._map_callback,
            qos,
        )
        self.pose = None
        self.safety_state = "WAITING"
        self.task_state = "WAITING"
        self.elapsed_seconds = 0.0
        self.remaining_seconds = 0.0
        self.map_points = 0
        self.contact_count = 0
        self.expected_targets = 0
        self.current_target = ""
        self.front_clearance = 0.0
        self.map_message: OccupancyGrid | None = None
        self.wall_clearance = float("inf")
        self.trail: deque[Point] = deque(maxlen=trail_max)
        self.create_timer(1.0 / self.publish_hz, self._publish)

    def _odom_callback(self, message: Odometry) -> None:
        pose = message.pose.pose
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return
        self.pose = pose
        point = Point(x=float(pose.position.x), y=float(pose.position.y), z=0.05)
        if not self.trail or math.hypot(
            point.x - self.trail[-1].x, point.y - self.trail[-1].y
        ) >= self.trail_spacing:
            self.trail.append(point)

    def _safety_callback(self, message: String) -> None:
        self.safety_state = str(message.data).strip() or "UNKNOWN"

    def _clearance_callback(self, message: Float32) -> None:
        value = float(message.data)
        if math.isfinite(value) and value >= 0.0:
            self.front_clearance = value

    def _map_callback(self, message: OccupancyGrid) -> None:
        expected = int(message.info.width * message.info.height)
        if (
            message.info.resolution > 0.0
            and expected > 0
            and len(message.data) == expected
        ):
            self.map_message = message

    def _nearest_wall_clearance(self) -> float:
        if self.pose is None or self.map_message is None:
            return float("inf")
        message = self.map_message
        resolution = float(message.info.resolution)
        width = int(message.info.width)
        height = int(message.info.height)
        origin_x = float(message.info.origin.position.x)
        origin_y = float(message.info.origin.position.y)
        robot_x = float(self.pose.position.x)
        robot_y = float(self.pose.position.y)
        column = int(math.floor((robot_x - origin_x) / resolution))
        row = int(math.floor((robot_y - origin_y) / resolution))
        search_cells = int(math.ceil(2.0 / resolution))
        nearest = float("inf")
        for candidate_row in range(
            max(0, row - search_cells), min(height, row + search_cells + 1)
        ):
            offset = candidate_row * width
            for candidate_column in range(
                max(0, column - search_cells),
                min(width, column + search_cells + 1),
            ):
                if int(message.data[offset + candidate_column]) < 50:
                    continue
                wall_x = origin_x + (candidate_column + 0.5) * resolution
                wall_y = origin_y + (candidate_row + 0.5) * resolution
                nearest = min(nearest, math.hypot(wall_x - robot_x, wall_y - robot_y))
        return nearest

    def _task_status_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            self.task_state = str(payload.get("state", "UNKNOWN"))
            self.elapsed_seconds = float(payload.get("elapsed_seconds", 0.0))
            hard_runtime = float(payload.get("hard_runtime_seconds", 0.0))
            self.remaining_seconds = float(
                payload.get(
                    "remaining_seconds",
                    max(0.0, hard_runtime - self.elapsed_seconds),
                )
            )
            task1_map = payload.get("task1_map")
            task1_map = task1_map if isinstance(task1_map, dict) else {}
            self.map_points = int(
                payload.get("map_points", task1_map.get("finite_points", 0))
            )
            self.contact_count = int(payload.get("contact_inferred_count", 0))
            self.expected_targets = int(payload.get("expected_target_count", 0))
            self.current_target = str(payload.get("current_target") or "")
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    @staticmethod
    def _colour(state: str) -> tuple[float, float, float]:
        if state in ("ACTIVE", "CONTACT_ACTIVE"):
            return 0.10, 0.95, 0.20
        if state == "FRONT_OBSTACLE_STOP":
            return 1.00, 0.10, 0.05
        return 1.00, 0.70, 0.05

    def _publish(self) -> MarkerArray | None:
        if self.pose is None:
            return
        stamp = self.get_clock().now().to_msg()
        red, green, blue = self._colour(self.safety_state)
        self.wall_clearance = self._nearest_wall_clearance()
        clearance_red, clearance_green, clearance_blue = red, green, blue
        if self.safety_state == "ACTIVE" and self.wall_clearance < self.safety_radius:
            clearance_red, clearance_green, clearance_blue = 1.0, 0.05, 0.02
        output = MarkerArray()

        safety_disc = Marker()
        safety_disc.header.frame_id = self.frame_id
        safety_disc.header.stamp = stamp
        safety_disc.ns = "safety_clearance"
        safety_disc.id = 0
        safety_disc.type = Marker.CYLINDER
        safety_disc.action = Marker.ADD
        safety_disc.pose.position.x = float(self.pose.position.x)
        safety_disc.pose.position.y = float(self.pose.position.y)
        safety_disc.pose.position.z = 0.015
        safety_disc.pose.orientation.w = 1.0
        safety_disc.scale.x = 2.0 * self.safety_radius
        safety_disc.scale.y = 2.0 * self.safety_radius
        safety_disc.scale.z = 0.025
        safety_disc.color.r = clearance_red
        safety_disc.color.g = clearance_green
        safety_disc.color.b = clearance_blue
        safety_disc.color.a = 0.16
        output.markers.append(safety_disc)

        footprint = Marker()
        footprint.header = safety_disc.header
        footprint.ns = "robot_footprint"
        footprint.id = 1
        footprint.type = Marker.CUBE
        footprint.action = Marker.ADD
        footprint.pose.position.x = float(self.pose.position.x)
        footprint.pose.position.y = float(self.pose.position.y)
        footprint.pose.position.z = 0.055
        footprint.pose.orientation = self.pose.orientation
        footprint.scale.x = self.robot_length
        footprint.scale.y = self.robot_width
        footprint.scale.z = 0.08
        footprint.color.r = red
        footprint.color.g = green
        footprint.color.b = blue
        footprint.color.a = 0.38
        output.markers.append(footprint)

        arrow = Marker()
        arrow.header.frame_id = self.frame_id
        arrow.header.stamp = stamp
        arrow.ns = "robot_pose"
        arrow.id = 2
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose = self.pose
        arrow.scale.x = 1.0
        arrow.scale.y = 0.20
        arrow.scale.z = 0.20
        arrow.color.r = red
        arrow.color.g = green
        arrow.color.b = blue
        arrow.color.a = 1.0
        output.markers.append(arrow)

        trail = Marker()
        trail.header = arrow.header
        trail.ns = "robot_trail"
        trail.id = 3
        trail.type = Marker.LINE_STRIP
        trail.action = Marker.ADD
        trail.scale.x = 0.06
        trail.color.r = 0.10
        trail.color.g = 0.65
        trail.color.b = 1.00
        trail.color.a = 0.90
        trail.points = list(self.trail)
        output.markers.append(trail)

        label = Marker()
        label.header = arrow.header
        label.ns = "robot_status"
        label.id = 4
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(self.pose.position.x)
        label.pose.position.y = float(self.pose.position.y)
        label.pose.position.z = float(self.pose.position.z) + 0.80
        label.pose.orientation.w = 1.0
        label.scale.z = 0.32
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0
        elapsed = max(0, int(self.elapsed_seconds))
        remaining = max(0, int(self.remaining_seconds))
        progress = f"points {self.map_points:,}"
        if self.expected_targets > 0:
            progress = f"boxes {self.contact_count}/{self.expected_targets}"
            if self.current_target:
                progress += f"  target {self.current_target}"
        clearance_text = (
            f"front {self.front_clearance:.2f}m"
            if self.front_clearance < 98.0
            else "front clear"
        )
        wall_text = (
            f"wall {self.wall_clearance:.2f}m"
            if math.isfinite(self.wall_clearance)
            else "wall unknown"
        )
        label.text = (
            f"{self.task_state} | {self.safety_state}\n"
            f"elapsed {elapsed // 60:02d}:{elapsed % 60:02d}  "
            f"left {remaining // 60:02d}:{remaining % 60:02d}  {progress}\n"
            f"{clearance_text}  {wall_text}  safety {self.safety_radius:.2f}m"
        )
        output.markers.append(label)
        self.publisher.publish(output)
        return output


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotVisualizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
