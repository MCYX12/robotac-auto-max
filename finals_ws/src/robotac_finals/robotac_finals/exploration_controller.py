#!/usr/bin/env python3
"""A deterministic A* frontier follower for ROBOTAC finals task one."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from .navigation_core import (
    astar,
    cell_to_world,
    inflate_obstacles,
    nearest_free,
    simplify_path,
    world_to_cell,
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class ExplorationController(Node):
    def __init__(self) -> None:
        super().__init__("robotac_exploration_controller")
        self.declare_parameter("map_topic", "/robotac/nav_map")
        self.declare_parameter("goal_topic", "/robotac/frontier_goal")
        self.declare_parameter("path_topic", "/robotac/exploration_path")
        self.declare_parameter("result_topic", "/robotac/frontier_result")
        self.declare_parameter("status_topic", "/robotac/controller_status")
        self.declare_parameter("mission_enable_topic", "/robotac/mission_enable")
        self.declare_parameter("contact_mode_topic", "/robotac/contact_mode")
        self.declare_parameter("odom_topic", "/odom/mujoco_odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("robot_radius", 0.32)
        self.declare_parameter("inflation_margin", 0.12)
        self.declare_parameter("local_observation_radius", 6.00)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("maximum_drive_heading_error", 0.65)
        self.declare_parameter("align_goal_orientation", False)
        self.align_goal_orientation = bool(self.get_parameter('align_goal_orientation').value)
        self.declare_parameter("align_only_observation_goals", False)
        self.align_only_observation_goals = bool(self.get_parameter('align_only_observation_goals').value)
        self.observation_metadata = None
        self.goal_yaw = None
        self.alignment_started = None
        self.alignment_anchor = None
        self.declare_parameter("waypoint_tolerance", 0.22)
        self.declare_parameter("maximum_linear_speed", 0.32)
        self.declare_parameter("maximum_yaw_rate", 0.45)
        self.declare_parameter("heading_kp", 1.2)
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("replan_period", 3.0)
        self.declare_parameter("maximum_map_age", 1.5)
        self.declare_parameter("stuck_timeout", 18.0)
        self.declare_parameter("minimum_progress", 0.12)
        self.declare_parameter("maximum_recoveries", 3)
        self.declare_parameter("recovery_duration", 4.0)
        self.declare_parameter("contact_goal_mode", False)
        self.declare_parameter("contact_goal_tolerance", 0.18)
        self.declare_parameter("maximum_contact_approach", 1.00)
        self.declare_parameter("contact_maximum_linear_speed", 0.08)
        self.declare_parameter("contact_heading_threshold", 0.35)
        self.declare_parameter("contact_timeout", 10.0)
        self.declare_parameter("safety_state_topic", "/robotac/control_safety_state")
        self.declare_parameter("front_obstacle_recovery_delay", 1.5)
        self.declare_parameter("scan_pause_period", 0.0)
        self.declare_parameter("scan_pause_duration", 3.0)
        self.scan_pause_period = float(self.get_parameter("scan_pause_period").value)
        self.scan_pause_duration = float(self.get_parameter("scan_pause_duration").value)
        if (not math.isfinite(self.scan_pause_period) or self.scan_pause_period < 0 or
                not math.isfinite(self.scan_pause_duration) or self.scan_pause_duration <= 0):
            raise ValueError("invalid stationary scan pause limits")
        self.next_scan_pause = time.monotonic() + self.scan_pause_period
        self.scan_pause_until = 0.0
        self.declare_parameter("initial_panorama", False)
        self.initial_panorama_done = not bool(self.get_parameter("initial_panorama").value)
        self.panorama_started = None
        self.panorama_previous_yaw = None
        self.panorama_angle = 0.0

        self.map_topic = str(self.get_parameter("map_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.result_topic = str(self.get_parameter("result_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.mission_enable_topic = str(
            self.get_parameter("mission_enable_topic").value
        )
        self.contact_mode_topic = str(
            self.get_parameter("contact_mode_topic").value
        )
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.clearance = float(self.get_parameter("robot_radius").value) + float(
            self.get_parameter("inflation_margin").value
        )
        self.local_observation_radius = float(
            self.get_parameter("local_observation_radius").value
        )
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.maximum_drive_heading_error = float(self.get_parameter("maximum_drive_heading_error").value)
        if not (0.0 < self.maximum_drive_heading_error <= math.pi / 2):
            raise ValueError('maximum_drive_heading_error must be in (0, pi/2]')
        self.waypoint_tolerance = float(self.get_parameter("waypoint_tolerance").value)
        self.maximum_speed = float(self.get_parameter("maximum_linear_speed").value)
        self.maximum_yaw = float(self.get_parameter("maximum_yaw_rate").value)
        self.heading_kp = float(self.get_parameter("heading_kp").value)
        self.replan_period = float(self.get_parameter("replan_period").value)
        self.maximum_map_age = float(self.get_parameter("maximum_map_age").value)
        self.stuck_timeout = float(self.get_parameter("stuck_timeout").value)
        self.minimum_progress = float(self.get_parameter("minimum_progress").value)
        self.maximum_recoveries = int(self.get_parameter("maximum_recoveries").value)
        self.recovery_duration = float(self.get_parameter("recovery_duration").value)
        self.contact_goal_mode = bool(self.get_parameter("contact_goal_mode").value)
        self.contact_goal_tolerance = float(
            self.get_parameter("contact_goal_tolerance").value
        )
        self.maximum_contact_approach = float(
            self.get_parameter("maximum_contact_approach").value
        )
        self.contact_maximum_speed = float(
            self.get_parameter("contact_maximum_linear_speed").value
        )
        self.contact_heading_threshold = float(
            self.get_parameter("contact_heading_threshold").value
        )
        self.contact_timeout = float(self.get_parameter("contact_timeout").value)
        self.safety_state_topic = str(
            self.get_parameter("safety_state_topic").value
        )
        self.front_obstacle_recovery_delay = float(
            self.get_parameter("front_obstacle_recovery_delay").value
        )
        control_hz = float(self.get_parameter("control_hz").value)
        if self.maximum_map_age < 0.0:
            raise ValueError("maximum_map_age cannot be negative")
        if self.contact_goal_mode and min(
            self.contact_goal_tolerance,
            self.maximum_contact_approach,
            self.contact_maximum_speed,
            self.contact_timeout,
        ) <= 0.0:
            raise ValueError("contact-goal limits must be positive")
        if self.front_obstacle_recovery_delay <= 0.0:
            raise ValueError("front_obstacle_recovery_delay must be positive")

        self.map_message: OccupancyGrid | None = None
        self.last_map_stamp: float | None = None
        self.last_map_progress_time: float | None = None
        self.pose: tuple[float, float, float] | None = None
        self.goal_id_ns = 0
        self.cancelled_goal_id_ns = 0
        self.goal: tuple[float, float] | None = None
        self.planned_goal: tuple[float, float] | None = None
        self.path: list[tuple[float, float]] = []
        self.path_index = 0
        self.last_plan = 0.0
        self.plan_requested = False
        self.enabled = True
        self.recovery_until = 0.0
        self.recovery_direction = 1.0
        self.recovery_count = 0
        self.contact_started_at: float | None = None
        self.contact_mode_active = False
        self.progress_pose: tuple[float, float, float] | None = None
        self.progress_time = time.monotonic()
        self.progress_pause_started: float | None = None
        self.last_status = ""
        self.safety_state = "UNKNOWN"
        self.front_obstacle_stop_since: float | None = None

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.command_publisher = self.create_publisher(Twist, self.cmd_topic, 10)
        self.path_publisher = self.create_publisher(Path, self.path_topic, latched_qos)
        self.result_publisher = self.create_publisher(String, self.result_topic, latched_qos)
        self.status_publisher = self.create_publisher(String, self.status_topic, latched_qos)
        self.contact_mode_publisher = self.create_publisher(
            Bool, self.contact_mode_topic, latched_qos
        )
        self.create_subscription(OccupancyGrid, self.map_topic, self._map_callback, latched_qos)
        self.create_subscription(PoseStamped, self.goal_topic, self._goal_callback, latched_qos)
        self.create_subscription(String, '/robotac/frontier_cancel', self._cancel_callback, latched_qos)
        self.create_subscription(String, '/robotac/exploration_status', self._goal_status_callback, latched_qos)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 20)
        self.create_subscription(
            Bool, self.mission_enable_topic, self._enable_callback, latched_qos
        )
        self.create_subscription(
            String,
            self.safety_state_topic,
            self._safety_state_callback,
            10,
        )
        self.create_timer(1.0 / control_hz, self._control_tick)
        self._set_contact_mode(False, force=True)
        self.get_logger().info(
            f"map={self.map_topic} goal={self.goal_topic} cmd={self.cmd_topic} clearance={self.clearance:.2f}m"
        )

    def _map_callback(self, message: OccupancyGrid) -> None:
        info = message.info
        expected = int(info.width) * int(info.height)
        if (expected <= 0 or len(message.data) != expected or
                not math.isfinite(info.resolution) or info.resolution <= 0.0 or
                not all(math.isfinite(v) for v in
                        (info.origin.position.x, info.origin.position.y))):
            self.get_logger().warning("discarding invalid navigation map")
            return
        stamp = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )
        if self.last_map_stamp is not None and stamp < self.last_map_stamp:
            self.get_logger().warning("discarding out-of-order navigation map")
            return
        if self.last_map_stamp is None or stamp > self.last_map_stamp:
            self.last_map_stamp = stamp
            self.last_map_progress_time = time.monotonic()
        self.map_message = message
        if time.monotonic() - self.last_plan >= self.replan_period:
            self.plan_requested = True

    def _matched_observation_yaw(self, candidate):
        meta = self.observation_metadata
        if (meta is not None and time.monotonic()-meta[4] <= 3.0 and
                math.hypot(candidate[0]-meta[0], candidate[1]-meta[1]) < .05 and
                meta[3] in ('FLOOR_REOBSERVE','STRUCTURE_REOBSERVE') and
                (len(meta) < 6 or meta[5] is None or meta[5] == self.goal_id_ns)):
            return meta[2]
        return None

    def _goal_status_callback(self, message: String) -> None:
        try:
            data=json.loads(message.data)
            if not isinstance(data,dict):return
            if data.get('state') != 'GOAL_SELECTED':return
            x,y=float(data['x']),float(data['y']);yaw=data.get('observation_yaw')
            yaw=float(yaw) if yaw is not None else None
            if not math.isfinite(x) or not math.isfinite(y) or (yaw is not None and not math.isfinite(yaw)):return
        except (ValueError,TypeError,KeyError):return
        self.observation_metadata=(x,y,yaw,str(data.get('goal_type')),time.monotonic(),data.get('goal_id_ns'))
        if self.align_only_observation_goals and self.goal is not None:
            if (math.hypot(self.goal[0]-x,self.goal[1]-y)<.05 and
                    (data.get('goal_id_ns') is None or data.get('goal_id_ns') == self.goal_id_ns)):
                self.goal_yaw=self._matched_observation_yaw(self.goal)
                self.alignment_started=None
                self.alignment_anchor=None

    def _goal_callback(self, message: PoseStamped) -> None:
        candidate = (float(message.pose.position.x), float(message.pose.position.y))
        if not all(math.isfinite(v) for v in candidate):
            self.get_logger().warning("discarding nonfinite navigation goal")
            return
        goal_id_ns = int(message.header.stamp.sec) * 1000000000 + int(message.header.stamp.nanosec)
        if goal_id_ns > 0 and goal_id_ns <= max(self.goal_id_ns, self.cancelled_goal_id_ns):
            return  # Duplicate or superseded token cannot restart navigation.
        if goal_id_ns == 0 and self.goal is not None and math.hypot(candidate[0] - self.goal[0], candidate[1] - self.goal[1]) < .05:
            return  # Preserve unstamped legacy callers.
        self.goal_id_ns = goal_id_ns
        self.goal = candidate
        q = message.pose.orientation
        self.goal_yaw = (math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                         if self.align_goal_orientation and all(math.isfinite(v) for v in (q.x,q.y,q.z,q.w))
                         and abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1) < .01 else None)
        if self.align_only_observation_goals:
            self.goal_yaw=self._matched_observation_yaw(candidate)
        self.alignment_started = None
        self.alignment_anchor = None
        self.planned_goal = None
        self.path = []
        self.path_index = 0
        self.recovery_count = 0
        self.recovery_until = 0.0
        self.contact_started_at = None
        self._set_contact_mode(False)
        self.plan_requested = True
        self._reset_progress()
        self._publish_status("GOAL_RECEIVED")

    def _cancel_callback(self, message: String) -> None:
        """Stop only the cancelled/superseded goal, even if topics arrive reordered."""
        try:
            data = json.loads(message.data)
            token = data['goal_id_ns']
            if type(token) is not int or token <= 0:
                return
        except (ValueError, TypeError, KeyError):
            return
        self.cancelled_goal_id_ns = max(self.cancelled_goal_id_ns, token)
        if self.goal is None or not 0 < self.goal_id_ns <= token:
            return
        self.command_publisher.publish(Twist())
        self._publish_result('CANCELLED', 'selector_timeout')
        self.goal = None
        self.goal_yaw = None
        self.alignment_started = None
        self.alignment_anchor = None
        self.planned_goal = None
        self.path = []
        self.path_index = 0
        self.plan_requested = False
        self.recovery_until = 0.0
        self.scan_pause_until = 0.0
        self.contact_started_at = None
        self._set_contact_mode(False)
        self._publish_status('GOAL_CANCELLED', goal_id_ns=token)

    def _odom_callback(self, message: Odometry) -> None:
        q = message.pose.pose.orientation
        position = message.pose.pose.position
        values = (position.x, position.y, position.z, q.x, q.y, q.z, q.w)
        norm_squared = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        if not all(math.isfinite(v) for v in values) or not .5 <= norm_squared <= 1.5:
            self.get_logger().warning("discarding invalid navigation odometry")
            return
        norm = math.sqrt(norm_squared)
        qx, qy, qz, qw = q.x/norm, q.y/norm, q.z/norm, q.w/norm
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        self.pose = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            yaw,
        )

    def _enable_callback(self, message: Bool) -> None:
        self.enabled = bool(message.data)
        if not self.enabled:
            self.command_publisher.publish(Twist())
            self.contact_started_at = None
            self._set_contact_mode(False)
            self._publish_status("MISSION_STOPPED")

    def _safety_state_callback(self, message: String) -> None:
        state = str(message.data).strip()
        self.safety_state = state
        if state == "FRONT_OBSTACLE_STOP":
            if self.front_obstacle_stop_since is None:
                self.front_obstacle_stop_since = time.monotonic()
        else:
            self.front_obstacle_stop_since = None

    def _set_contact_mode(self, active: bool, *, force: bool = False) -> None:
        active = bool(active)
        if force or active != self.contact_mode_active or active:
            self.contact_mode_publisher.publish(Bool(data=active))
        self.contact_mode_active = active

    def _reset_progress(self) -> None:
        if self.pose is not None:
            self.progress_pose = self.pose
        self.progress_time = time.monotonic()
        self.progress_pause_started = None

    def _pause_progress(self, now: float) -> None:
        # A deliberate observation stop is not lack of commanded progress.
        # Preserve accrued motion time: resetting it every 8 seconds otherwise
        # makes an 18-second stuck deadline impossible to reach.
        if self.progress_pause_started is None:
            self.progress_pause_started = now

    def _resume_progress(self, now: float) -> None:
        if self.progress_pause_started is not None:
            self.progress_time += max(0.0, now - self.progress_pause_started)
            self.progress_pause_started = None

    def _publish_status(self, state: str, **fields) -> None:
        payload = {"state": state, **fields}
        encoded = json.dumps(payload, separators=(",", ":"))
        if encoded != self.last_status:
            self.status_publisher.publish(String(data=encoded))
            self.last_status = encoded

    def _publish_result(self, result: str, reason: str) -> None:
        if self.goal is None:
            return
        payload = {
            "result": result,
            "goal_id_ns": self.goal_id_ns,
            "reason": reason,
            "x": self.goal[0],
            "y": self.goal[1],
            "recoveries": self.recovery_count,
        }
        self.result_publisher.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self.get_logger().info(json.dumps(payload))

    def _plan(self) -> bool:
        self.plan_requested = False
        self.last_plan = time.monotonic()
        self.planned_goal = None
        if self.map_message is None or self.pose is None or self.goal is None:
            return False
        message = self.map_message
        expected = int(message.info.width * message.info.height)
        if expected <= 0 or len(message.data) != expected:
            return False
        grid = np.asarray(message.data, dtype=np.int8).reshape(
            int(message.info.height), int(message.info.width)
        )
        radius_cells = int(math.ceil(self.clearance / message.info.resolution))
        blocked = inflate_obstacles(grid, radius_cells)
        origin_x = float(message.info.origin.position.x)
        origin_y = float(message.info.origin.position.y)
        resolution = float(message.info.resolution)
        requested_start = world_to_cell(self.pose[0], self.pose[1], origin_x, origin_y, resolution)
        requested_goal = world_to_cell(self.goal[0], self.goal[1], origin_x, origin_y, resolution)
        # A missing ground return under the robot must not make its occupied
        # footprint "unknown". Clear unknown-only cells under the measured body,
        # while preserving the planner's inflated obstacle mask.
        known_only = grid.copy()
        known_only[known_only < 0] = 0
        inflated_known_obstacles = inflate_obstacles(known_only, radius_cells)
        start_clear_radius = max(
            1, int(math.ceil(self.local_observation_radius / resolution))
        )
        for row in range(requested_start[0] - start_clear_radius, requested_start[0] + start_clear_radius + 1):
            for column in range(requested_start[1] - start_clear_radius, requested_start[1] + start_clear_radius + 1):
                if not (0 <= row < blocked.shape[0] and 0 <= column < blocked.shape[1]):
                    continue
                if (row - requested_start[0]) ** 2 + (column - requested_start[1]) ** 2 > start_clear_radius ** 2:
                    continue
                if not inflated_known_obstacles[row, column]:
                    blocked[row, column] = False
        start = nearest_free(blocked, requested_start, maximum_radius=max(4, radius_cells + 2))
        goal = nearest_free(blocked, requested_goal, maximum_radius=max(12, radius_cells * 3))
        if start is None or goal is None:
            self._publish_status("PLAN_WAITING_FOR_FREE_CELL")
            return False
        cells = astar(blocked, start, goal)
        if not cells:
            self._publish_status("NO_PATH")
            return False
        cells = simplify_path(cells, blocked)
        self.path = [cell_to_world(cell, origin_x, origin_y, resolution) for cell in cells]
        self.planned_goal = self.path[-1]
        self.path_index = 1 if len(self.path) > 1 else 0
        self._publish_path(message.header.frame_id or "world")
        self._publish_status("PATH_READY", waypoints=len(self.path))
        return True

    def _publish_path(self, frame_id: str) -> None:
        output = Path()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = frame_id
        for x, y in self.path:
            pose = PoseStamped()
            pose.header = output.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            output.poses.append(pose)
        self.path_publisher.publish(output)

    def _fail_or_recover(self, reason: str) -> None:
        self.contact_started_at = None
        self._set_contact_mode(False)
        if self.recovery_count >= self.maximum_recoveries:
            self.command_publisher.publish(Twist())
            self._publish_result("FAILED", reason)
            self._publish_status("GOAL_FAILED", reason=reason)
            self.goal = None
            self.planned_goal = None
            self.path = []
            return
        self.recovery_count += 1
        self.recovery_direction *= -1.0
        self.recovery_until = time.monotonic() + self.recovery_duration
        self.planned_goal = None
        self.path = []
        self.plan_requested = True
        self._reset_progress()
        self._publish_status("RECOVERY", attempt=self.recovery_count, reason=reason)

    def _align_at_goal(self, now):
        if self.goal_yaw is None:
            return False
        error = wrap_angle(self.goal_yaw-self.pose[2])
        if abs(error) <= .12:
            self.alignment_started = None
            self.alignment_anchor = None
            return False
        if self.alignment_started is None:
            self.alignment_started = now
            self.alignment_anchor = self.pose[:2]
        if now-self.alignment_started > 45.0:
            self.command_publisher.publish(Twist())
            self._publish_result('FAILED', 'observation_heading_timeout')
            self.goal = None
            self.path = []
            self.alignment_started = None
            self.alignment_anchor = None
            return True
        command = Twist()
        command.angular.z = clamp(.6*error, -min(.15,self.maximum_yaw), min(.15,self.maximum_yaw))
        self.command_publisher.publish(command)
        self._publish_status('ALIGNING_OBSERVATION', heading_error=error)
        return True

    def _control_tick(self) -> None:
        if not self.enabled:
            return
        if not self.initial_panorama_done:
            now = time.monotonic()
            if self.panorama_started is not None and now-self.panorama_started > 70.0:
                self.initial_panorama_done = True
                self.get_logger().warning("initial panorama deadline reached; final coverage audit remains required")
            elif self.pose is None or self.safety_state != "ACTIVE":
                self.command_publisher.publish(Twist())
                self._publish_status("INITIAL_PANORAMA_WAITING_FOR_SAFE_CONTROL")
                return
            else:
                if self.panorama_started is None:
                    self.panorama_started = now
                    self.get_logger().info("开始原地环视采集地面，随后自动探索。")
                if self.panorama_previous_yaw is not None:
                    self.panorama_angle += wrap_angle(self.pose[2]-self.panorama_previous_yaw)
                self.panorama_previous_yaw = self.pose[2]
                self._reset_progress()
                if self.panorama_angle >= 2*math.pi-.05:
                    self.initial_panorama_done = True
                    self.command_publisher.publish(Twist())
                    self.scan_pause_until = now+self.scan_pause_duration
                    self.next_scan_pause = self.scan_pause_until+self.scan_pause_period
                    self.get_logger().info("原地环视结束，开始自主探索。")
                else:
                    command = Twist();command.angular.z=min(.15,self.maximum_yaw)
                    self.command_publisher.publish(command)
                self._publish_status("INITIAL_PANORAMA", observed_degrees=round(math.degrees(self.panorama_angle),1))
                return
        if self.pose is None or self.goal is None:
            self.command_publisher.publish(Twist())
            self.contact_started_at = None
            self._set_contact_mode(False)
            return
        now = time.monotonic()
        map_age = (
            float("inf")
            if self.last_map_progress_time is None
            else now - self.last_map_progress_time
        )
        if self.maximum_map_age > 0.0 and map_age > self.maximum_map_age:
            self.command_publisher.publish(Twist())
            self.contact_started_at = None
            self._set_contact_mode(False)
            self.planned_goal = None
            self.path = []
            self.plan_requested = True
            self._publish_status(
                "NAV_MAP_STALE",
                map_age_seconds=None if not math.isfinite(map_age) else round(map_age, 3),
            )
            return
        if self.scan_pause_period > 0.0:
            if now >= self.next_scan_pause and now >= self.scan_pause_until:
                self.scan_pause_until = now + self.scan_pause_duration
                self.next_scan_pause = self.scan_pause_until + self.scan_pause_period
            if now < self.scan_pause_until:
                self.command_publisher.publish(Twist())
                self._set_contact_mode(False)
                self._pause_progress(now)
                self._publish_status("SCANNING_STATIONARY")
                return
        self._resume_progress(now)
        if (
            self.front_obstacle_stop_since is not None
            and now - self.front_obstacle_stop_since
            >= self.front_obstacle_recovery_delay
            and now >= self.recovery_until
        ):
            blocked_for = now - self.front_obstacle_stop_since
            self.get_logger().warning(
                f"front obstacle stop persisted for {blocked_for:.2f}s; "
                "starting bounded turn recovery"
            )
            self.front_obstacle_stop_since = None
            self._fail_or_recover("front_obstacle_stop")
            return
        if self.alignment_started is not None and self.alignment_anchor is not None:
            # Turning a legged body can shift XY slightly across the arrival
            # tolerance. Keep the attained observation pose while aligning,
            # rather than steering back toward the waypoint on alternate ticks.
            drift = math.hypot(self.pose[0]-self.alignment_anchor[0],
                               self.pose[1]-self.alignment_anchor[1])
            if drift > .25:
                self.command_publisher.publish(Twist())
                self._publish_result('FAILED', 'observation_alignment_drift')
                self.goal = None
                self.planned_goal = None
                self.path = []
                self.alignment_started = None
                self.alignment_anchor = None
                return
            if self._align_at_goal(now):
                return
            self.command_publisher.publish(Twist())
            self.contact_started_at = None
            self._set_contact_mode(False)
            self._publish_result('REACHED', 'observation_heading_aligned')
            self._publish_status('GOAL_REACHED', alignment_drift=drift)
            self.goal = None
            self.planned_goal = None
            self.path = []
            return
        goal_distance = math.hypot(self.goal[0] - self.pose[0], self.goal[1] - self.pose[1])
        completion_tolerance = (
            self.contact_goal_tolerance
            if self.contact_goal_mode
            else self.goal_tolerance
        )
        if goal_distance <= completion_tolerance:
            if self._align_at_goal(now):
                return
            self.command_publisher.publish(Twist())
            self.contact_started_at = None
            self._set_contact_mode(False)
            reason = (
                "contact_goal_tolerance"
                if self.contact_goal_mode
                else "goal_tolerance"
            )
            self._publish_result("REACHED", reason)
            self._publish_status("GOAL_REACHED", distance=goal_distance)
            self.goal = None
            self.planned_goal = None
            self.path = []
            return
        if self.contact_started_at is not None:
            self._set_contact_mode(True)
            contact_elapsed = now - self.contact_started_at
            if contact_elapsed >= self.contact_timeout:
                self.command_publisher.publish(Twist())
                self._fail_or_recover("contact_timeout")
                return
            target_heading = math.atan2(
                self.goal[1] - self.pose[1], self.goal[0] - self.pose[0]
            )
            heading_error = wrap_angle(target_heading - self.pose[2])
            command = Twist()
            command.angular.z = clamp(
                self.heading_kp * heading_error,
                -min(0.25, self.maximum_yaw),
                min(0.25, self.maximum_yaw),
            )
            if abs(heading_error) <= self.contact_heading_threshold:
                command.linear.x = min(
                    self.contact_maximum_speed,
                    max(0.055, 0.35 * goal_distance),
                )
            self.command_publisher.publish(command)
            self._publish_status(
                "CONTACT_APPROACH",
                distance=round(goal_distance, 3),
                elapsed=round(contact_elapsed, 2),
            )
            return
        if self.planned_goal is not None:
            planned_distance = math.hypot(
                self.planned_goal[0] - self.pose[0],
                self.planned_goal[1] - self.pose[1],
            )
            if planned_distance <= self.waypoint_tolerance:
                if self.contact_goal_mode:
                    if goal_distance > self.maximum_contact_approach:
                        self.command_publisher.publish(Twist())
                        self._fail_or_recover("contact_target_too_far")
                        return
                    self.contact_started_at = now
                    self._set_contact_mode(True)
                    self.command_publisher.publish(Twist())
                    self._publish_status(
                        "CONTACT_APPROACH",
                        distance=round(goal_distance, 3),
                        elapsed=0.0,
                    )
                    return
                # Obstacle inflation can move the requested frontier to the
                # nearest safe free cell. Reaching that A* endpoint is a valid
                # observation pose; waiting for the unsafe original cell can
                # otherwise waste the full goal timeout while already stopped.
                if self._align_at_goal(now):
                    return
                self.command_publisher.publish(Twist())
                self._publish_result("REACHED", "planned_safe_goal_tolerance")
                self._publish_status(
                    "GOAL_REACHED",
                    distance=goal_distance,
                    planned_distance=planned_distance,
                )
                self.goal = None
                self.planned_goal = None
                self.path = []
                return
        if now < self.recovery_until:
            command = Twist()
            command.angular.z = self.recovery_direction * min(0.30, self.maximum_yaw)
            self.command_publisher.publish(command)
            return
        if self.plan_requested or not self.path:
            if not self._plan():
                if now - self.progress_time >= self.stuck_timeout:
                    self._fail_or_recover("no_path")
                self.command_publisher.publish(Twist())
                return

        while self.path_index < len(self.path) - 1:
            waypoint = self.path[self.path_index]
            if math.hypot(waypoint[0] - self.pose[0], waypoint[1] - self.pose[1]) > self.waypoint_tolerance:
                break
            self.path_index += 1
        target = self.path[min(self.path_index, len(self.path) - 1)]
        target_heading = math.atan2(target[1] - self.pose[1], target[0] - self.pose[0])
        heading_error = wrap_angle(target_heading - self.pose[2])
        target_distance = math.hypot(target[0] - self.pose[0], target[1] - self.pose[1])
        command = Twist()
        command.angular.z = clamp(self.heading_kp * heading_error, -self.maximum_yaw, self.maximum_yaw)
        if abs(heading_error) < self.maximum_drive_heading_error:
            heading_scale = max(0.15, math.cos(heading_error))
            command.linear.x = min(self.maximum_speed, max(0.08, 0.7 * target_distance)) * heading_scale
        self.command_publisher.publish(command)

        if self.progress_pose is None:
            self._reset_progress()
        elif (
            math.hypot(
                self.pose[0] - self.progress_pose[0],
                self.pose[1] - self.progress_pose[1],
            )
            >= self.minimum_progress
            or abs(wrap_angle(self.pose[2] - self.progress_pose[2])) >= 0.20
        ):
            self._reset_progress()
        elif (
            command.linear.x > 0.07 or abs(command.angular.z) > 0.10
        ) and now - self.progress_time >= self.stuck_timeout:
            self._fail_or_recover("stuck")
        self._publish_status(
            "TRACKING",
            waypoint=self.path_index,
            waypoints=len(self.path),
            goal_distance=round(goal_distance, 3),
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExplorationController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node.command_publisher.publish(Twist())
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass


if __name__ == "__main__":
    main()
