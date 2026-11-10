#!/usr/bin/env python3
"""RViz-visible, local-only virtual treasure boxes with proximity triggers."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

from .task2_core import validate_targets


class Task2BoxSimulator(Node):
    def __init__(self, *, parameter_overrides=None) -> None:
        super().__init__(
            "robotac_task2_box_simulator", parameter_overrides=parameter_overrides
        )
        self.declare_parameter("target_names", ["box_1", "box_2", "box_3", "box_4", "box_5"])
        self.declare_parameter("target_coordinates", [0.0] * 10)
        self.declare_parameter("target_profile_kind", "UNSET")
        self.declare_parameter("targets_verified", False)
        self.declare_parameter("expected_target_count", 5)
        self.declare_parameter("scenario_seed", 0)
        self.declare_parameter("task1_map_sha256", "")
        self.declare_parameter("odom_topic", "/odom/mujoco_odom")
        self.declare_parameter("goal_topic", "/robotac/task2/goal")
        self.declare_parameter("mission_enable_topic", "/robotac/mission_enable")
        self.declare_parameter("result_topic", "/robotac/task2/result")
        self.declare_parameter("event_topic", "/robotac/task2/sim_box_events")
        self.declare_parameter("marker_topic", "/robotac/task2/sim_box_markers")
        self.declare_parameter(
            "official_mesh_resource",
            "package://robotac_finals/meshes/official_cashbox.glb",
        )
        self.declare_parameter("event_path", "/workspace/finals_ws/output/task2_sim_events.json")
        self.declare_parameter("trigger_radius", 0.22)

        names = [str(value) for value in self.get_parameter("target_names").value]
        coordinates = [float(value) for value in self.get_parameter("target_coordinates").value]
        expected = int(self.get_parameter("expected_target_count").value)
        self.targets = validate_targets(names, coordinates, expected)
        self.names = names
        profile_kind = str(self.get_parameter("target_profile_kind").value)
        verified = bool(self.get_parameter("targets_verified").value)
        if profile_kind != "local_random_simulation" or not verified:
            raise ValueError("box simulator requires a verified local_random_simulation profile")
        self.seed = int(self.get_parameter("scenario_seed").value)
        self.map_digest = str(self.get_parameter("task1_map_sha256").value)
        if len(self.map_digest) != 64:
            raise ValueError("box simulator requires the sealed Task-1 map SHA-256")
        self.trigger_radius = float(self.get_parameter("trigger_radius").value)
        if not (0.05 <= self.trigger_radius <= 0.50):
            raise ValueError("simulation trigger_radius must be in [0.05, 0.50]")
        self.event_path = Path(str(self.get_parameter("event_path").value))
        self.official_mesh_resource = str(
            self.get_parameter("official_mesh_resource").value
        )
        if not self.official_mesh_resource.startswith(
            "package://robotac_finals/meshes/"
        ):
            raise ValueError("official_mesh_resource must use the packaged robotac_finals mesh")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, str(self.get_parameter("marker_topic").value), qos
        )
        self.event_publisher = self.create_publisher(
            String, str(self.get_parameter("event_topic").value), qos
        )
        self.result_publisher = self.create_publisher(
            String, str(self.get_parameter("result_topic").value), qos
        )
        self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self._odom_callback, 20
        )
        self.create_subscription(
            PoseStamped, str(self.get_parameter("goal_topic").value), self._goal_callback, qos
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("mission_enable_topic").value),
            self._enable_callback,
            qos,
        )
        self.enabled = False
        self.active_index: int | None = None
        self.triggered: list[int] = []
        self.events: list[dict] = []
        self.started_at = time.monotonic()
        self.initial_markers_published = False
        self._write_state()
        self.create_timer(0.25, self._initial_publish)

    def _initial_publish(self) -> None:
        if not self.initial_markers_published:
            self._publish_markers()
            self._publish_event("SCENARIO_READY")
            self.initial_markers_published = True

    def _enable_callback(self, message: Bool) -> None:
        self.enabled = bool(message.data)

    def _goal_callback(self, message: PoseStamped) -> None:
        candidate = (float(message.pose.position.x), float(message.pose.position.y))
        self.active_index = None
        for index, target in enumerate(self.targets):
            if index in self.triggered:
                continue
            if math.hypot(candidate[0] - target[0], candidate[1] - target[1]) <= 0.05:
                self.active_index = index
                self._publish_event("TARGET_ACTIVE", target=index)
                break

    def _odom_callback(self, message: Odometry) -> None:
        if not self.enabled or self.active_index is None:
            return
        index = self.active_index
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        target = self.targets[index]
        distance = math.hypot(x - target[0], y - target[1])
        if not math.isfinite(distance) or distance > self.trigger_radius:
            return
        self.triggered.append(index)
        self.active_index = None
        event = {
            "event": "BOX_TRIGGERED",
            "target": self.names[index],
            "x": target[0],
            "y": target[1],
            "robot_x": x,
            "robot_y": y,
            "distance": distance,
            "elapsed_seconds": time.monotonic() - self.started_at,
        }
        self.events.append(event)
        self._publish_markers()
        self._write_state()
        self.event_publisher.publish(
            String(data=json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        )
        result = {
            "result": "REACHED",
            "reason": "contact_goal_tolerance",
            "x": target[0],
            "y": target[1],
            "source": "local_random_box_simulator",
            "target": self.names[index],
        }
        self.result_publisher.publish(
            String(data=json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        )
        self.get_logger().warning(
            f"LOCAL SIMULATION box disappeared: {self.names[index]} "
            f"({len(self.triggered)}/{len(self.targets)})"
        )

    def _publish_event(self, event: str, *, target: int | None = None) -> None:
        payload = {
            "event": event,
            "profile_kind": "local_random_simulation",
            "official_evidence": False,
            "seed": self.seed,
        }
        if target is not None:
            payload["target"] = self.names[target]
        self.event_publisher.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def _publish_markers(self) -> None:
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = "world"
        clear.header.stamp = now
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for index, ((x, y), name) in enumerate(zip(self.targets, self.names)):
            if index in self.triggered:
                continue
            box = Marker()
            box.header.frame_id = "world"
            box.header.stamp = now
            box.ns = "official_cashbox_mesh_simulation"
            box.id = index * 2
            box.type = Marker.MESH_RESOURCE
            box.action = Marker.ADD
            box.pose.position.x = x
            box.pose.position.y = y
            box.pose.position.z = 0.0
            # The official glTF is Y-up; rotate it into ROS's Z-up frame.
            box.pose.orientation.x = math.sqrt(0.5)
            box.pose.orientation.w = math.sqrt(0.5)
            # Cooked mesh vertices are in Unreal centimetres.
            box.scale.x = 0.01
            box.scale.y = 0.01
            box.scale.z = 0.01
            box.mesh_resource = self.official_mesh_resource
            box.mesh_use_embedded_materials = True
            box.color.r = 1.0
            box.color.g = 1.0
            box.color.b = 1.0
            box.color.a = 1.0
            markers.markers.append(box)
            label = Marker()
            label.header = box.header
            label.ns = "local_sim_treasure_labels"
            label.id = index * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = 0.75
            label.pose.orientation.w = 1.0
            label.scale.z = 0.28
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0
            label.text = name
            markers.markers.append(label)
        self.marker_publisher.publish(markers)

    def _write_state(self) -> None:
        payload = {
            "state": (
                "ALL_SIMULATED_BOXES_TRIGGERED"
                if len(self.triggered) == len(self.targets)
                else "RUNNING_LOCAL_SIMULATION"
            ),
            "profile_kind": "local_random_simulation",
            "official_evidence": False,
            "warning": (
                "Official cashbox mesh at local random coordinates; "
                "local trigger only, not an organizer score."
            ),
            "visual_asset": self.official_mesh_resource,
            "seed": self.seed,
            "task1_map_sha256": self.map_digest,
            "triggered_count": len(self.triggered),
            "triggered_targets": [self.names[index] for index in self.triggered],
            "remaining_targets": [
                name for index, name in enumerate(self.names) if index not in self.triggered
            ],
            "events": self.events,
        }
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.event_path.with_suffix(self.event_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.event_path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Task2BoxSimulator()
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
