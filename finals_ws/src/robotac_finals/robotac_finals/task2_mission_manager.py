#!/usr/bin/env python3
"""Run the five-target task-two route and expose referee-safe mission state."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from .navigation_core import astar, inflate_obstacles, nearest_free, world_to_cell
from .task2_core import best_visit_order, task2_score, validate_targets


class Task2MissionManager(Node):
    def __init__(self, *, parameter_overrides=None) -> None:
        super().__init__(
            "robotac_task2_mission_manager",
            parameter_overrides=parameter_overrides,
        )
        self.declare_parameter("map_topic", "/robotac/task2/known_map")
        self.declare_parameter("map_status_topic", "/robotac/task2/map_status")
        self.declare_parameter("odom_topic", "/odom/mujoco_odom")
        self.declare_parameter("goal_topic", "/robotac/task2/goal")
        self.declare_parameter("controller_result_topic", "/robotac/task2/result")
        self.declare_parameter("mission_enable_topic", "/robotac/mission_enable")
        self.declare_parameter("status_topic", "/robotac/task2/status")
        self.declare_parameter("status_path", "/workspace/finals_ws/output/task2_status.json")
        self.declare_parameter("start_service", "/robotac/task2/start")
        self.declare_parameter("stop_service", "/robotac/task2/stop")
        self.declare_parameter("target_names", ["box_1", "box_2", "box_3", "box_4", "box_5"])
        self.declare_parameter("target_coordinates", [0.0] * 10)
        self.declare_parameter("target_profile_kind", "official_measured")
        self.declare_parameter("target_coordinate_provenance", "UNSET")
        self.declare_parameter("targets_verified", False)
        self.declare_parameter("expected_target_count", 5)
        self.declare_parameter("robot_radius", 0.32)
        self.declare_parameter("inflation_margin", 0.12)
        self.declare_parameter("route_snap_radius", 20)
        self.declare_parameter("result_coordinate_tolerance", 0.05)
        self.declare_parameter("maximum_target_retries", 1)
        self.declare_parameter("hard_runtime", 1800.0)
        self.declare_parameter("auto_start", False)

        self.map_topic = str(self.get_parameter("map_topic").value)
        self.map_status_topic = str(self.get_parameter("map_status_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.result_topic = str(self.get_parameter("controller_result_topic").value)
        self.enable_topic = str(self.get_parameter("mission_enable_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.status_path = Path(str(self.get_parameter("status_path").value))
        self.start_service = str(self.get_parameter("start_service").value)
        self.stop_service = str(self.get_parameter("stop_service").value)
        self.expected_targets = int(self.get_parameter("expected_target_count").value)
        self.names = [str(value) for value in self.get_parameter("target_names").value]
        coordinates = [float(value) for value in self.get_parameter("target_coordinates").value]
        self.targets = validate_targets(self.names, coordinates, self.expected_targets)
        self.profile_kind = str(self.get_parameter("target_profile_kind").value)
        self.provenance = str(self.get_parameter("target_coordinate_provenance").value)
        self.targets_verified = bool(self.get_parameter("targets_verified").value)
        self.clearance = float(self.get_parameter("robot_radius").value) + float(
            self.get_parameter("inflation_margin").value
        )
        self.route_snap_radius = int(self.get_parameter("route_snap_radius").value)
        self.result_coordinate_tolerance = float(
            self.get_parameter("result_coordinate_tolerance").value
        )
        self.maximum_retries = int(self.get_parameter("maximum_target_retries").value)
        self.hard_runtime = float(self.get_parameter("hard_runtime").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)
        if self.hard_runtime <= 0.0 or self.hard_runtime > 1800.0:
            raise ValueError("hard_runtime must be in (0, 1800]")
        if self.provenance.strip() in ("", "UNSET") and self.targets_verified:
            raise ValueError("verified targets require coordinate provenance")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.goal_publisher = self.create_publisher(PoseStamped, self.goal_topic, qos)
        self.enable_publisher = self.create_publisher(Bool, self.enable_topic, qos)
        self.status_publisher = self.create_publisher(String, self.status_topic, qos)
        self.create_subscription(OccupancyGrid, self.map_topic, self._map_callback, qos)
        self.create_subscription(String, self.map_status_topic, self._map_status_callback, qos)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 20)
        self.create_subscription(String, self.result_topic, self._result_callback, qos)
        self.create_service(Trigger, self.start_service, self._start_callback)
        self.create_service(Trigger, self.stop_service, self._stop_callback)

        self.map_message: OccupancyGrid | None = None
        self.task1_map_status: dict | None = None
        self.pose: tuple[float, float] | None = None
        self.route: list[int] = []
        self.route_cost = 0.0
        self.route_position = 0
        self.current_target: int | None = None
        self.retries = 0
        self.contacted: list[int] = []
        self.state = "WAITING_FOR_INPUTS"
        self.started_at: float | None = None
        self.last_result_key = ""
        self.auto_start_attempted = False
        self.enable_publisher.publish(Bool(data=False))
        self._write_status()
        self.create_timer(0.50, self._tick)

    def _map_callback(self, message: OccupancyGrid) -> None:
        expected = int(message.info.width) * int(message.info.height)
        if expected > 0 and len(message.data) == expected:
            self.map_message = message

    def _map_status_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            state = str(payload["state"])
            source_kind = str(payload["source_kind"])
            path = str(payload["path"])
            digest = str(payload["sha256"])
            finite_points = int(payload["finite_points"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if (
            state == "READY"
            and source_kind == "task1_native_pcd"
            and path.endswith(".pcd")
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest.lower())
            and finite_points > 0
        ):
            self.task1_map_status = payload

    def _odom_callback(self, message: Odometry) -> None:
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        if math.isfinite(x) and math.isfinite(y):
            self.pose = (x, y)

    def _grid_distance_function(self):
        if self.map_message is None or self.pose is None:
            raise ValueError("known map and odometry are required")
        message = self.map_message
        grid = np.asarray(message.data, dtype=np.int8).reshape(
            int(message.info.height), int(message.info.width)
        )
        resolution = float(message.info.resolution)
        if resolution <= 0.0:
            raise ValueError("known map has invalid resolution")
        blocked = inflate_obstacles(grid, int(math.ceil(self.clearance / resolution)))
        origin_x = float(message.info.origin.position.x)
        origin_y = float(message.info.origin.position.y)
        points = [self.pose, *self.targets]
        cells = []
        for point in points:
            requested = world_to_cell(point[0], point[1], origin_x, origin_y, resolution)
            cell = nearest_free(blocked, requested, maximum_radius=self.route_snap_radius)
            if cell is None:
                raise ValueError(f"no safe map cell near target {point}")
            cells.append(cell)
        lookup = {point: cell for point, cell in zip(points, cells)}
        cache: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}

        def distance(first, second):
            first_cell = lookup[first]
            second_cell = lookup[second]
            key = (first_cell, second_cell)
            reverse = (second_cell, first_cell)
            if key in cache:
                return cache[key]
            if reverse in cache:
                return cache[reverse]
            path = astar(blocked, first_cell, second_cell)
            if not path:
                cost = float("inf")
            else:
                cost = sum(
                    math.hypot(b[0] - a[0], b[1] - a[1]) * resolution
                    for a, b in zip(path, path[1:])
                )
            cache[key] = cost
            return cost

        return distance

    def _start(self) -> tuple[bool, str]:
        if self.started_at is not None:
            return False, "task two has already started"
        if not self.targets_verified:
            return False, "target coordinates are not marked verified"
        if self.task1_map_status is None:
            return False, "validated task-one native PCD is not ready"
        if self.map_message is None:
            return False, "task-one occupancy map is not ready"
        if self.pose is None:
            return False, "odometry is not ready"
        try:
            self.route, self.route_cost = best_visit_order(
                self.pose, self.targets, self._grid_distance_function()
            )
        except ValueError as error:
            self.state = "ROUTE_INVALID"
            self._write_status(error=str(error))
            return False, str(error)
        self.started_at = time.monotonic()
        self.state = "RUNNING"
        self.route_position = 0
        self.contacted = []
        self.retries = 0
        self.enable_publisher.publish(Bool(data=True))
        self._publish_current_goal()
        self._write_status()
        return True, "task two started"

    def _start_callback(self, _, response):
        response.success, response.message = self._start()
        return response

    def _stop_callback(self, _, response):
        if self.started_at is None or self.state != "RUNNING":
            response.success = False
            response.message = "task two is not running"
            return response
        self._finish("STOPPED_BY_OPERATOR")
        response.success = True
        response.message = "task two stopped; this is not an all-box completion"
        return response

    def _publish_current_goal(self) -> None:
        if self.route_position >= len(self.route):
            self._finish("ALL_TARGETS_CONTACTED_RAISE_HAND")
            return
        self.current_target = self.route[self.route_position]
        x, y = self.targets[self.current_target]
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.orientation.w = 1.0
        self.goal_publisher.publish(message)
        self.get_logger().info(
            f"target={self.names[self.current_target]} x={x:.3f} y={y:.3f} "
            f"route={self.route_position + 1}/{len(self.route)}"
        )

    def _result_callback(self, message: String) -> None:
        if self.state != "RUNNING" or self.current_target is None:
            return
        try:
            payload = json.loads(message.data)
            result = str(payload["result"])
            reason = str(payload["reason"])
            x = float(payload["x"])
            y = float(payload["y"])
            source = str(payload.get("source", "controller_inference"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if (
            self.profile_kind == "local_random_simulation"
            and source != "local_random_box_simulator"
        ):
            return
        expected = self.targets[self.current_target]
        if math.hypot(x - expected[0], y - expected[1]) > self.result_coordinate_tolerance:
            return
        key = f"{result}:{reason}:{x:.6f}:{y:.6f}:{self.route_position}:{self.retries}"
        if key == self.last_result_key:
            return
        self.last_result_key = key
        if result == "REACHED" and reason == "contact_goal_tolerance":
            self.contacted.append(self.current_target)
            self.route_position += 1
            self.retries = 0
            self._write_status(last_contact=self.names[self.current_target])
            self._publish_current_goal()
        elif result == "FAILED":
            if self.retries < self.maximum_retries:
                self.retries += 1
                self._publish_current_goal()
                self._write_status(last_failure=reason)
            else:
                self._finish("FAILED", failure_reason=reason)

    def _finish(self, state: str, **fields) -> None:
        self.state = state
        self.current_target = None
        self.enable_publisher.publish(Bool(data=False))
        self._write_status(**fields)
        if state == "ALL_TARGETS_CONTACTED_RAISE_HAND":
            if self.profile_kind == "local_random_simulation":
                self.get_logger().warning(
                    "Local simulation completed five official-mesh box contacts; "
                    "this is not organizer score evidence."
                )
            else:
                self.get_logger().warning(
                    "All configured contacts inferred. Raise hand immediately; "
                    "referee score is authoritative."
                )

    def _status_payload(self, **fields) -> dict:
        elapsed = 0.0 if self.started_at is None else time.monotonic() - self.started_at
        score = task2_score(len(self.contacted), elapsed, self.expected_targets)
        current_name = (
            None if self.current_target is None else self.names[self.current_target]
        )
        return {
            "state": self.state,
            "elapsed_seconds": round(elapsed, 2),
            "hard_runtime_seconds": self.hard_runtime,
            "remaining_seconds": round(max(0.0, self.hard_runtime - elapsed), 2),
            "contact_inferred_count": len(self.contacted),
            "expected_target_count": self.expected_targets,
            "contacted_targets": [self.names[index] for index in self.contacted],
            "current_target": current_name,
            "route": [self.names[index] for index in self.route],
            "route_cost_metres": round(self.route_cost, 3),
            "target_coordinate_provenance": self.provenance,
            "target_profile_kind": self.profile_kind,
            "targets_verified": self.targets_verified,
            "task1_map": self.task1_map_status,
            "score_estimate_not_official": score,
            **fields,
        }

    def _write_status(self, **fields) -> None:
        payload = self._status_payload(**fields)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, self.status_path)
        self.status_publisher.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def _tick(self) -> None:
        if self.state == "RUNNING" and self.started_at is not None:
            if time.monotonic() - self.started_at >= self.hard_runtime:
                self._finish("TIME_EXPIRED")
            else:
                self._write_status()
            return
        if self.started_at is None:
            ready = (
                self.task1_map_status is not None
                and self.map_message is not None
                and self.pose is not None
            )
            wanted = "ARMED" if ready and self.targets_verified else "WAITING_FOR_INPUTS"
            if wanted != self.state:
                self.state = wanted
                self._write_status()
            if self.auto_start and ready and not self.auto_start_attempted:
                self.auto_start_attempted = True
                success, message = self._start()
                if not success:
                    self.get_logger().error(message)

    def destroy_node(self) -> bool:
        self.enable_publisher.publish(Bool(data=False))
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Task2MissionManager()
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
