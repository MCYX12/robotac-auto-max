#!/usr/bin/env python3
"""Safety-gated ROS Twist to ROBOTAC high-level SDK bridge."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32, String


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def slew(current: float, target: float, maximum_change: float) -> float:
    return current + clamp(target - current, -maximum_change, maximum_change)


def sdk_deadband(value: float, minimum: float) -> float:
    return 0.0 if abs(value) < minimum else value


def finite_twist(message: Twist) -> bool:
    """Return false when any Twist component is NaN or infinite."""
    return all(
        math.isfinite(float(value))
        for value in (
            message.linear.x,
            message.linear.y,
            message.linear.z,
            message.angular.x,
            message.angular.y,
            message.angular.z,
        )
    )


def expected_context_shutdown(error: BaseException, context_ok: bool) -> bool:
    """Identify Humble's narrow WaitSet race during coordinated shutdown."""
    return (not context_ok) and "context is not valid" in str(error).lower()


class SdkTwistBridge(Node):
    """Apply limits, posture/lidar gates, and a watchdog before SDK commands."""

    def __init__(self) -> None:
        super().__init__("robotac_sdk_twist_bridge")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("safe_cmd_topic", "/robotac/safe_cmd_vel")
        self.declare_parameter("front_clearance_topic", "/robotac/front_clearance")
        self.declare_parameter("lidar_topic", "/front_lidar")
        self.declare_parameter("odom_topic", "/odom/mujoco_odom")
        self.declare_parameter("local_ip", "127.0.0.1")
        self.declare_parameter("local_port", 43988)
        self.declare_parameter("robot_ip", "127.0.0.1")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("update_hz", 20.0)
        self.declare_parameter("command_timeout", 0.30)
        self.declare_parameter("sensor_timeout", 0.50)
        self.declare_parameter("max_vx", 0.45)
        self.declare_parameter("max_vy", 0.20)
        self.declare_parameter("max_wz", 0.60)
        self.declare_parameter("max_ax", 0.30)
        self.declare_parameter("max_ay", 0.25)
        self.declare_parameter("max_aw", 0.80)
        self.declare_parameter("front_stop_distance", 0.65)
        self.declare_parameter("contact_mode_topic", "/robotac/contact_mode")
        self.declare_parameter("contact_mode_timeout", 0.30)
        self.declare_parameter("contact_stop_distance", 0.08)
        self.declare_parameter("contact_max_forward_speed", 0.10)
        self.declare_parameter("require_upright", True)
        self.declare_parameter("auto_stand", True)
        self.declare_parameter("stand_retry_period", 3.0)
        self.declare_parameter("stand_settle_time", 2.0)

        self.cmd_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.safe_topic = str(self.get_parameter("safe_cmd_topic").value)
        self.front_clearance_topic = str(
            self.get_parameter("front_clearance_topic").value
        )
        self.lidar_topic = str(self.get_parameter("lidar_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.update_hz = float(self.get_parameter("update_hz").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        self.sensor_timeout = float(self.get_parameter("sensor_timeout").value)
        self.max_vx = float(self.get_parameter("max_vx").value)
        self.max_vy = float(self.get_parameter("max_vy").value)
        self.max_wz = float(self.get_parameter("max_wz").value)
        self.max_ax = float(self.get_parameter("max_ax").value)
        self.max_ay = float(self.get_parameter("max_ay").value)
        self.max_aw = float(self.get_parameter("max_aw").value)
        self.front_stop = float(self.get_parameter("front_stop_distance").value)
        self.contact_mode_topic = str(
            self.get_parameter("contact_mode_topic").value
        )
        self.contact_mode_timeout = float(
            self.get_parameter("contact_mode_timeout").value
        )
        self.contact_stop = float(
            self.get_parameter("contact_stop_distance").value
        )
        self.contact_max_forward_speed = float(
            self.get_parameter("contact_max_forward_speed").value
        )
        self.require_upright = bool(self.get_parameter("require_upright").value)
        self.auto_stand = bool(self.get_parameter("auto_stand").value)
        self.stand_retry_period = float(
            self.get_parameter("stand_retry_period").value
        )
        self.stand_settle_time = float(
            self.get_parameter("stand_settle_time").value
        )
        limits = [self.update_hz, self.command_timeout, self.sensor_timeout, self.max_vx, self.max_vy, self.max_wz, self.max_ax, self.max_ay, self.max_aw, self.front_stop, self.stand_retry_period]
        if not all(math.isfinite(v) and v > 0 for v in limits):
            raise ValueError("control rates, timeouts and limits must be finite and positive")
        if self.update_hz <= 0.0 or min(self.max_vx, self.max_vy, self.max_wz) <= 0.0:
            raise ValueError("update_hz and velocity limits must be positive")
        if min(
            self.contact_mode_timeout,
            self.contact_stop,
            self.contact_max_forward_speed,
        ) <= 0.0 or self.contact_stop >= self.front_stop:
            raise ValueError("invalid contact-mode safety limits")

        self.desired = Twist()
        self.current = Twist()
        self.command_valid = True
        self.last_command = 0.0
        self.last_lidar = 0.0
        self.last_odom = 0.0
        self.front_clearance = 0.0
        self.lidar_valid = False
        self.contact_mode = False
        self.last_contact_mode = 0.0
        self.upright = False
        self.last_state = ""
        self.last_stand_attempt = 0.0
        self.stand_ready_since = None
        self.sdk = None

        self.safe_publisher = self.create_publisher(Twist, self.safe_topic, 10)
        self.state_publisher = self.create_publisher(
            String, "/robotac/control_safety_state", 10
        )
        self.clearance_publisher = self.create_publisher(
            Float32, self.front_clearance_topic, 10
        )
        self.create_subscription(Twist, self.cmd_topic, self._command_callback, 10)
        self.create_subscription(
            PointCloud2, self.lidar_topic, self._lidar_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Bool, self.contact_mode_topic, self._contact_mode_callback, 10
        )

        if not self.dry_run:
            self.sdk = self._connect_sdk()
        self.create_timer(1.0 / self.update_hz, self._control_tick)
        self.get_logger().info(
            f"cmd={self.cmd_topic} safe={self.safe_topic} dry_run={self.dry_run} "
            f"limits=[{self.max_vx:.2f},{self.max_vy:.2f},{self.max_wz:.2f}]"
        )

    def _connect_sdk(self):
        architecture = platform.machine().replace("amd64", "x86_64").replace(
            "arm64", "aarch64"
        )
        sdk_root = os.environ.get("ROBOTAC_SDK_ROOT", "/workspace/deps/zsibot_sdk")
        sys.path.insert(0, os.path.join(sdk_root, "lib", "zsl-1", architecture))
        import mc_sdk_zsl_1_py

        app = mc_sdk_zsl_1_py.HighLevel()
        app.initRobot(
            str(self.get_parameter("local_ip").value),
            int(self.get_parameter("local_port").value),
            str(self.get_parameter("robot_ip").value),
        )
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if app.checkConnect():
                self.get_logger().info("ROBOTAC SDK connected")
                return app
            time.sleep(0.20)
        raise RuntimeError("ROBOTAC SDK did not connect within 8 seconds")

    def _command_callback(self, message: Twist) -> None:
        if not finite_twist(message):
            # Do not retain a prior walking request after a corrupt command.
            # The timer performs the SDK stop on its next bounded control tick.
            self.desired = Twist()
            self.command_valid = False
            self.last_command = time.monotonic()
            self.get_logger().warning("Rejected non-finite cmd_vel")
            return
        self.desired = message
        self.command_valid = True
        self.last_command = time.monotonic()

    def _invalidate_lidar(self, received_at: float) -> None:
        self.front_clearance = 0.0
        self.lidar_valid = False
        self.last_lidar = received_at
        self.clearance_publisher.publish(Float32(data=0.0))

    def _lidar_callback(self, message: PointCloud2) -> None:
        received_at = time.monotonic()
        try:
            records = point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
            xyz = np.column_stack((records["x"], records["y"], records["z"]))
            if xyz.shape[0] == 0:
                self._invalidate_lidar(received_at)
                return
            xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
            if xyz.shape[0] == 0:
                self._invalidate_lidar(received_at)
                return
            front = xyz[
                (xyz[:, 0] > 0.0)
                & (np.abs(xyz[:, 1]) < 0.40)
                & (xyz[:, 2] > -0.20)
                & (xyz[:, 2] < 1.50)
            ]
            self.front_clearance = (
                float(front[:, 0].min()) if len(front) else float("inf")
            )
            self.lidar_valid = True
            self.last_lidar = received_at
            displayed_clearance = (
                self.front_clearance if math.isfinite(self.front_clearance) else 99.0
            )
            self.clearance_publisher.publish(Float32(data=displayed_clearance))
        except (AssertionError, IndexError, KeyError, TypeError, ValueError):
            self._invalidate_lidar(received_at)

    def _odom_callback(self, message: Odometry) -> None:
        q = message.pose.pose.orientation
        values = [message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z, q.x, q.y, q.z, q.w]
        if not all(math.isfinite(v) for v in values) or not .5 < sum(v*v for v in values[3:]) < 1.5:
            self.upright = False
            return
        roll = math.atan2(
            2.0 * (q.w * q.x + q.y * q.z),
            1.0 - 2.0 * (q.x * q.x + q.y * q.y),
        )
        pitch = math.asin(
            clamp(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0)
        )
        self.upright = (
            message.pose.pose.position.z >= 0.22
            and abs(roll) <= 0.60
            and abs(pitch) <= 0.60
        )
        self.last_odom = time.monotonic()

    def _contact_mode_callback(self, message: Bool) -> None:
        self.contact_mode = bool(message.data)
        self.last_contact_mode = time.monotonic()

    @staticmethod
    def _zero_twist() -> Twist:
        return Twist()

    def _gated_target(self, now: float) -> tuple[Twist, str]:
        if not self.command_valid:
            return self._zero_twist(), "INVALID_COMMAND_STOP"
        if now - self.last_command > self.command_timeout:
            return self._zero_twist(), "WATCHDOG_STOP"
        if self.require_upright and (
            now - self.last_odom > self.sensor_timeout or not self.upright
        ):
            return self._zero_twist(), "POSTURE_STOP"
        if now - self.last_lidar > self.sensor_timeout:
            return self._zero_twist(), "LIDAR_STALE_STOP"
        if not self.lidar_valid:
            return self._zero_twist(), "LIDAR_INVALID_STOP"
        target = Twist()
        target.linear.x = clamp(self.desired.linear.x, -self.max_vx, self.max_vx)
        target.linear.y = clamp(self.desired.linear.y, -self.max_vy, self.max_vy)
        target.angular.z = clamp(self.desired.angular.z, -self.max_wz, self.max_wz)
        contact_active = (
            self.contact_mode
            and now - self.last_contact_mode <= self.contact_mode_timeout
        )
        stop_distance = self.contact_stop if contact_active else self.front_stop
        if contact_active:
            target.linear.x = min(target.linear.x, self.contact_max_forward_speed)
            target.linear.y = 0.0
        if target.linear.x > 0.0 and self.front_clearance < stop_distance:
            return self._zero_twist(), "FRONT_OBSTACLE_STOP"
        return target, "CONTACT_ACTIVE" if contact_active else "ACTIVE"

    def _control_tick(self) -> None:
        now = time.monotonic()
        if self.sdk is not None and (
            now - self.last_odom > self.sensor_timeout or
            now - self.last_lidar > self.sensor_timeout or not self.lidar_valid
        ):
            self.current = Twist()
            self.stand_ready_since = None
            self.safe_publisher.publish(self.current)
            if self.last_state != "SENSOR_STALE_STOP":
                self.last_state = "SENSOR_STALE_STOP"
                self.state_publisher.publish(String(data=self.last_state))
                self.get_logger().warning("safety_state=SENSOR_STALE_STOP; stand/walk commands disabled")
            # A missing simulator must never trigger repeated stand commands.
            if int(self.sdk.getCurrentCtrlmode()) in (1, 18):
                self.sdk.move(0.0, 0.0, 0.0)
            return
        if self.sdk is not None:
            mode = int(self.sdk.getCurrentCtrlmode())
            if mode not in (1, 18):
                self.stand_ready_since = None
                self.current = Twist()
                self.safe_publisher.publish(self.current)
                state = "WAITING_FOR_STAND"
                if self.auto_stand and now - self.last_stand_attempt >= self.stand_retry_period:
                    result = int(self.sdk.standUp())
                    self.last_stand_attempt = now
                    self.get_logger().info(
                        f"standUp result=0x{result:04x} mode_before={mode}"
                    )
                if state != self.last_state:
                    self.state_publisher.publish(String(data=state))
                    self.get_logger().info(f"safety_state={state}")
                    self.last_state = state
                return
            posture_ready = (
                now - self.last_odom <= self.sensor_timeout and self.upright
            )
            if self.require_upright and not posture_ready:
                self.stand_ready_since = None
                self.current = Twist()
                self.safe_publisher.publish(self.current)
                state = "WAITING_FOR_UPRIGHT"
                if state != self.last_state:
                    self.state_publisher.publish(String(data=state))
                    self.get_logger().info(f"safety_state={state}")
                    self.last_state = state
                return
            if self.stand_ready_since is None:
                self.stand_ready_since = now
            if now - self.stand_ready_since < self.stand_settle_time:
                self.current = Twist()
                self.safe_publisher.publish(self.current)
                state = "STAND_SETTLING"
                if state != self.last_state:
                    self.state_publisher.publish(String(data=state))
                    self.get_logger().info(f"safety_state={state}")
                    self.last_state = state
                return

        target, state = self._gated_target(now)
        period = 1.0 / self.update_hz
        if state not in ("ACTIVE", "CONTACT_ACTIVE"):
            # Safety gates are emergency stops. Decelerating over 1-2 seconds here
            # would let the robot keep walking after a stale sensor or obstacle.
            self.current = Twist()
        else:
            self.current.linear.x = slew(
                self.current.linear.x, target.linear.x, self.max_ax * period
            )
            self.current.linear.y = slew(
                self.current.linear.y, target.linear.y, self.max_ay * period
            )
            self.current.angular.z = slew(
                self.current.angular.z, target.angular.z, self.max_aw * period
            )
        self.safe_publisher.publish(self.current)
        if state != self.last_state:
            self.state_publisher.publish(String(data=state))
            self.get_logger().info(f"safety_state={state}")
            self.last_state = state
        if self.sdk is not None:
            result = int(
                self.sdk.move(
                    sdk_deadband(self.current.linear.x, 0.05),
                    sdk_deadband(self.current.linear.y, 0.05),
                    sdk_deadband(self.current.angular.z, 0.02),
                )
            )
            if result != 0:
                self.get_logger().error(f"SDK move failed: 0x{result:04x}")

    def destroy_node(self) -> bool:
        if self.sdk is not None and int(self.sdk.getCurrentCtrlmode()) in (1, 18):
            for _ in range(3):
                self.sdk.move(0.0, 0.0, 0.0)
                time.sleep(0.05)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SdkTwistBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError as error:
        # Launch tears the shared ROS context down when another critical node
        # exits.  Humble can surface that narrow WaitSet race as RCLError.
        # Every other RCLError remains a hard failure.
        if not expected_context_shutdown(error, rclpy.ok()):
            raise
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        except RCLError as error:
            if not expected_context_shutdown(error, rclpy.ok()):
                raise


if __name__ == "__main__":
    main()
