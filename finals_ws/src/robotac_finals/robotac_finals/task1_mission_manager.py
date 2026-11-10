#!/usr/bin/env python3
"""Finish task one safely, save the native PCD, and emit a verified status file."""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, CompressedImage
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
import math
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger


class Task1MissionManager(Node):
    def __init__(self, *, parameter_overrides=None) -> None:
        super().__init__(
            "robotac_task1_mission_manager",
            parameter_overrides=parameter_overrides,
        )
        self.declare_parameter("score_cloud_topic", "/robotac/finals/score_cloud")
        self.declare_parameter("save_service", "/robotac/finals/save_map")
        self.declare_parameter("finalize_service", "/robotac/finals/finalize")
        self.declare_parameter("output_pcd", "/workspace/finals_ws/output/task1_native.pcd")
        self.declare_parameter("status_path", "/workspace/finals_ws/output/task1_status.json")
        self.declare_parameter("minimum_runtime", 600.0)
        self.declare_parameter("hard_runtime", 1500.0)
        self.declare_parameter("enable_coverage_converged_stop", False)
        self.declare_parameter("no_frontier_duration", 30.0)
        self.declare_parameter("growth_window", 60.0)
        self.declare_parameter("maximum_growth_ratio", 0.005)
        self.declare_parameter("minimum_map_points", 500000)
        self.declare_parameter("minimum_saved_points", 1000)
        self.declare_parameter("save_retry_limit", 3)
        self.declare_parameter("save_retry_delay", 2.0)
        self.declare_parameter("save_service_wait_timeout", 1.0)
        self.declare_parameter("save_response_timeout", 10.0)

        self.output_pcd = Path(str(self.get_parameter("output_pcd").value))
        self.status_path = Path(str(self.get_parameter("status_path").value))
        self.minimum_runtime = float(self.get_parameter("minimum_runtime").value)
        self.hard_runtime = float(self.get_parameter("hard_runtime").value)
        self.enable_coverage_stop = bool(
            self.get_parameter("enable_coverage_converged_stop").value
        )
        self.no_frontier_duration = float(self.get_parameter("no_frontier_duration").value)
        self.growth_window = float(self.get_parameter("growth_window").value)
        self.maximum_growth = float(self.get_parameter("maximum_growth_ratio").value)
        self.minimum_points = int(self.get_parameter("minimum_map_points").value)
        self.minimum_saved_points = max(
            1, int(self.get_parameter("minimum_saved_points").value)
        )
        self.save_retry_limit = max(1, int(self.get_parameter("save_retry_limit").value))
        self.save_retry_delay = max(0.0, float(self.get_parameter("save_retry_delay").value))
        self.save_service_wait_timeout = max(
            0.0, float(self.get_parameter("save_service_wait_timeout").value)
        )
        self.save_response_timeout = max(
            0.0, float(self.get_parameter("save_response_timeout").value)
        )
        self.declare_parameter("sensor_failure_timeout", 5.0)
        self.declare_parameter("startup_sensor_timeout", 25.0)
        self.declare_parameter("runtime_health_required", False)
        self.declare_parameter("runtime_health_path", "/workspace/deployment/runtime_status.json")
        self.sensor_failure_timeout = float(self.get_parameter("sensor_failure_timeout").value)
        self.startup_sensor_timeout = float(self.get_parameter("startup_sensor_timeout").value)
        self.runtime_health_required = bool(self.get_parameter("runtime_health_required").value)
        self.runtime_health_path = Path(str(self.get_parameter("runtime_health_path").value))
        self.runtime_generation = None
        if self.runtime_health_required:
            try:self.runtime_generation = json.loads(self.runtime_health_path.read_text()).get("generation")
            except (OSError, ValueError):pass
        if not all(math.isfinite(v) for v in [self.sensor_failure_timeout, self.startup_sensor_timeout, self.hard_runtime, self.minimum_runtime, self.growth_window, self.save_response_timeout]):
            raise ValueError("mission timing parameters must be finite")
        if self.sensor_failure_timeout <= 0 or self.startup_sensor_timeout < 0 or self.hard_runtime <= 0 or self.minimum_runtime < 0 or self.minimum_runtime > self.hard_runtime:
            raise ValueError("invalid mission timing parameters")
        self.sensor_last = {"lidar": 0.0, "odom": 0.0, "depth": 0.0}
        self.failure_reason = ""
        self.terminal_state = "FAILED"
        self.control_wait_since = None
        self.control_state = "UNKNOWN"
        self.start_time = time.monotonic()
        self.no_frontier_since: float | None = None
        self.cloud_samples: deque[tuple[float, int]] = deque()
        self.latest_points = 0
        self.saving = False
        self.save_future = None
        self.save_future_deadline = 0.0
        self.next_save_retry_at = 0.0
        self.save_attempts = 0
        self.save_reason = ""
        self.completed = False

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.enable_publisher = self.create_publisher(Bool, "/robotac/mission_enable", latched_qos)
        self.status_publisher = self.create_publisher(String, "/robotac/task1_status", latched_qos)
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("score_cloud_topic").value),
            self._cloud_callback,
            latched_qos,
        )
        self.create_subscription(String, "/robotac/exploration_status", self._exploration_callback, latched_qos)
        for kind, msg_type, topic in [
            ("lidar", PointCloud2, "/front_lidar"),
            ("odom", Odometry, "/odom/mujoco_odom"),
            ("depth", CompressedImage, "/front_depth/image/compressed"),
        ]:
            self.create_subscription(msg_type, topic,
                lambda message, kind=kind: self._sensor_callback(kind, message), qos_profile_sensor_data)
        self.create_subscription(String, "/robotac/control_safety_state", self._safety_callback, 10)
        self.create_service(Trigger, "/robotac/finals/abort", self._abort_callback)
        self.save_client = self.create_client(Trigger, str(self.get_parameter("save_service").value))
        self.create_service(
            Trigger,
            str(self.get_parameter("finalize_service").value),
            self._finalize_callback,
        )
        self.create_timer(1.0, self._tick)
        self.enable_publisher.publish(Bool(data=True))
        self._write_status(
            "RUNNING",
            "mission_started",
            coverage_converged_stop_enabled=self.enable_coverage_stop,
            hard_runtime_seconds=self.hard_runtime,
        )

    def _sensor_callback(self, kind, message) -> None:
        if kind == "lidar" and (message.width * message.height == 0 or not message.data):
            return
        if kind == "depth" and not message.data:
            return
        if kind == "odom":
            pose = message.pose.pose
            values = [pose.position.x, pose.position.y, pose.position.z,
                      pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            if not all(math.isfinite(v) for v in values):
                return
            if sum(v * v for v in values[3:]) < 0.5:
                return
        self.sensor_last[kind] = time.monotonic()

    def _safety_callback(self, message) -> None:
        self.control_state = message.data
        if message.data in ("WAITING_FOR_STAND", "WAITING_FOR_UPRIGHT", "SENSOR_STALE_STOP"):
            if self.control_wait_since is None:
                self.control_wait_since = time.monotonic()
        else:
            self.control_wait_since = None

    def _health_failure(self, now) -> str:
        if self.runtime_health_required:
            try:
                runtime = json.loads(self.runtime_health_path.read_text())
                if runtime.get("state") not in ("RUNNING", "STOPPING"):
                    return "runtime_unavailable:" + str(runtime.get("reason", runtime.get("state")))
                if not self.runtime_generation or runtime.get("generation") != self.runtime_generation:
                    return "runtime_generation_changed"
                heartbeat = float(runtime["heartbeat_monotonic"])
                if not math.isfinite(heartbeat) or now - heartbeat > 4.0 or heartbeat > now + 1.0:
                    return "runtime_supervisor_stale"
            except (OSError, ValueError, KeyError, TypeError):
                return "runtime_health_missing"
        if now - self.start_time > self.startup_sensor_timeout or all(self.sensor_last.values()):
            stale = [k for k, t in self.sensor_last.items() if not t or now - t > self.sensor_failure_timeout]
            if stale:
                return "sensor_stream_stale:" + ",".join(stale)
        if self.control_wait_since is not None and now - self.control_wait_since > 30.0:
            return "standing_or_posture_failed_30s"
        return ""

    def _abort_callback(self, _, response):
        if self.completed:
            response.success = False
            response.message = "mission already terminal"
        else:
            self.failure_reason = self.failure_reason or "operator_cancelled"
            self.terminal_state = "CANCELLED" if self.failure_reason == "operator_cancelled" else "FAILED"
            self.enable_publisher.publish(Bool(data=False))
            if not self.saving and self.save_attempts == 0:
                self._request_save(self.failure_reason)
            response.success = True
            response.message = "stop accepted; saving partial native PCD"
        return response

    def _cloud_callback(self, message: PointCloud2) -> None:
        now = time.monotonic()
        self.latest_points = int(message.width * message.height)
        self.cloud_samples.append((now, self.latest_points))
        while self.cloud_samples and now - self.cloud_samples[0][0] > self.growth_window * 1.5:
            self.cloud_samples.popleft()

    def _exploration_callback(self, message: String) -> None:
        try:
            state = json.loads(message.data).get("state")
        except (json.JSONDecodeError, AttributeError):
            return
        now = time.monotonic()
        if state == "NO_FRONTIER":
            if self.no_frontier_since is None:
                self.no_frontier_since = now
        else:
            self.no_frontier_since = None

    def _growth_ratio(self) -> float | None:
        if len(self.cloud_samples) < 2:
            return None
        newest_time, newest_points = self.cloud_samples[-1]
        candidates = [sample for sample in self.cloud_samples if newest_time - sample[0] >= self.growth_window * 0.75]
        if not candidates:
            return None
        oldest_points = candidates[-1][1]
        return max(0.0, newest_points - oldest_points) / max(1, oldest_points)

    def _write_status(self, state: str, reason: str, **extra) -> None:
        elapsed = time.monotonic() - self.start_time
        payload = {
            "state": state,
            "reason": reason,
            "elapsed_seconds": round(elapsed, 3),
            "hard_runtime_seconds": self.hard_runtime,
            "remaining_seconds": round(max(0.0, self.hard_runtime - elapsed), 3),
            "map_points": self.latest_points,
            "output_pcd": str(self.output_pcd),
            "health_failure": self.failure_reason or None,
            "control_state": self.control_state,
            "sensor_age_seconds": {k: round(time.monotonic() - t, 3) if t else None for k, t in self.sensor_last.items()},
            **extra,
        }
        self.status_publisher.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.status_path)

    @staticmethod
    def _pcd_points(path: Path) -> int:
        try:
            with path.open("rb") as stream:
                for raw_line in stream:
                    line = raw_line.decode("ascii", errors="ignore").strip()
                    if line.startswith("POINTS "):
                        return int(line.split()[1])
                    if line.startswith("DATA "):
                        break
        except (OSError, ValueError):
            return 0
        return 0

    def _finalize_callback(self, _, response: Trigger.Response) -> Trigger.Response:
        """Route an operator finish request through the normal verified save path."""
        if self.completed:
            response.success = False
            response.message = "task one is already terminal"
            return response
        if self.saving or self.save_attempts > 0:
            response.success = False
            response.message = "native PCD save is already in progress"
            return response
        self._request_save("operator_finalize")
        response.success = True
        response.message = "operator finalize accepted; waiting for verified native PCD"
        return response

    def _request_save(self, reason: str) -> None:
        if self.saving or self.completed:
            return
        self.enable_publisher.publish(Bool(data=False))
        self.saving = True
        if not self.save_reason:
            self.save_reason = reason
        self.save_attempts += 1
        self._write_status(
            "SAVING",
            "save_attempt_started",
            trigger_reason=self.save_reason,
            save_attempt=self.save_attempts,
            save_retry_limit=self.save_retry_limit,
        )
        try:
            service_ready = self.save_client.wait_for_service(
                timeout_sec=self.save_service_wait_timeout
            )
        except Exception as error:
            self._record_save_failure("save_service_wait_exception", str(error))
            return
        if not service_ready:
            self._record_save_failure(
                "save_service_unavailable",
                f"service_not_ready_within_{self.save_service_wait_timeout:.3f}s",
            )
            return
        try:
            self.save_future = self.save_client.call_async(Trigger.Request())
        except Exception as error:
            self._record_save_failure("save_request_exception", str(error))
            return
        self.save_future_deadline = time.monotonic() + self.save_response_timeout

    def _record_save_failure(self, reason: str, service_message: str) -> None:
        """Record one bounded save failure and either retry later or terminate."""
        self.saving = False
        self.save_future = None
        self.save_future_deadline = 0.0
        points = self._pcd_points(self.output_pcd)
        details = {
            "trigger_reason": self.save_reason,
            "save_attempt": self.save_attempts,
            "save_retry_limit": self.save_retry_limit,
            "saved_points": points,
            "service_message": service_message,
        }
        if self.save_attempts >= self.save_retry_limit:
            self.enable_publisher.publish(Bool(data=False))
            self._write_status(self.terminal_state if self.failure_reason else "FAILED", self.failure_reason or reason, **details)
            self.completed = True
            self.get_logger().error(
                f"Task one save failed: reason={reason} attempts={self.save_attempts}"
            )
            return
        self.next_save_retry_at = time.monotonic() + self.save_retry_delay
        self._write_status(
            "RETRYING",
            reason,
            retry_in_seconds=self.save_retry_delay,
            **details,
        )

    def _poll_save_future(self, now: float) -> None:
        future = self.save_future
        if future is None:
            self._record_save_failure("save_future_missing", "no_pending_future")
            return
        if future.done():
            self.save_future = None
            self.save_future_deadline = 0.0
            self._save_complete(future)
            return
        if now < self.save_future_deadline:
            return
        try:
            future.cancel()
        except Exception:
            # Cancellation is best effort.  The late response is deliberately
            # ignored because this attempt has already consumed its retry slot.
            pass
        self._record_save_failure(
            "save_response_timeout",
            f"no_response_within_{self.save_response_timeout:.3f}s",
        )

    def _save_complete(self, future) -> None:
        self.saving = False
        try:
            response = future.result()
            service_success = bool(response.success)
            service_message = str(response.message)
        except Exception as error:  # rclpy transports service exceptions here.
            self._record_save_failure("save_response_exception", str(error))
            return
        if not service_success:
            self._record_save_failure("save_service_rejected", service_message)
            return

        # Validation is intentionally read-only: the mapper's native PCD is
        # never rewritten, filtered, converted, or otherwise post-processed.
        try:
            stat = self.output_pcd.stat()
        except FileNotFoundError:
            self._record_save_failure("saved_pcd_missing", service_message)
            return
        except OSError as error:
            self._record_save_failure("saved_pcd_stat_exception", str(error))
            return
        if not self.output_pcd.is_file():
            self._record_save_failure("saved_pcd_not_regular_file", service_message)
            return
        if stat.st_size <= 0:
            self._record_save_failure("saved_pcd_empty", service_message)
            return
        points = self._pcd_points(self.output_pcd)
        if points < self.minimum_saved_points:
            self._record_save_failure(
                "saved_pcd_point_count_below_minimum",
                f"{service_message}; points={points}; minimum={self.minimum_saved_points}",
            )
            return

        failure = self._health_failure(time.monotonic())
        if failure and not self.failure_reason:
            self.failure_reason = failure
        self.enable_publisher.publish(Bool(data=False))
        self._write_status(
            self.terminal_state if self.failure_reason else "MAP_SAVED",
            self.failure_reason or "native_pcd_saved_pending_quality_validation",
            trigger_reason=self.save_reason,
            save_attempt=self.save_attempts,
            saved_points=points,
            size_bytes=stat.st_size,
            partial_pcd_saved=bool(self.failure_reason),
            service_message=service_message,
        )
        # Publish the completion flag only after the atomic status file is
        # durable, so observers never see completed=True with stale status.
        self.completed = True
        self.get_logger().info(f"TASK1_TERMINAL path={self.output_pcd} points={points}")

    def _tick(self) -> None:
        if self.completed:
            return
        now = time.monotonic()
        elapsed = now - self.start_time
        if not self.failure_reason:
            failure = self._health_failure(now)
            if failure:
                self.failure_reason = failure
                self.enable_publisher.publish(Bool(data=False))
                self.get_logger().error("Mission stopped: " + failure + "; preserving partial map")
                if not self.saving and self.save_attempts == 0:
                    self._request_save(failure)
        if self.saving:
            self._poll_save_future(now)
            return
        if self.save_attempts > 0:
            if now >= self.next_save_retry_at:
                self._request_save(self.save_reason)
            return
        if elapsed >= self.hard_runtime:
            self._request_save("hard_deadline")
            return
        growth = self._growth_ratio()
        no_frontier_for = 0.0 if self.no_frontier_since is None else time.monotonic() - self.no_frontier_since
        ready = (
            elapsed >= self.minimum_runtime
            and self.latest_points >= self.minimum_points
            and no_frontier_for >= self.no_frontier_duration
            and growth is not None
            and growth <= self.maximum_growth
        )
        if ready and self.enable_coverage_stop:
            self._request_save("coverage_converged")
        elif int(elapsed) % 10 == 0:
            self._write_status(
                "RUNNING",
                "exploring",
                coverage_converged_stop_enabled=self.enable_coverage_stop,
                coverage_converged=ready,
                no_frontier_seconds=round(no_frontier_for, 1),
                growth_ratio=None if growth is None else round(growth, 6),
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Task1MissionManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node.enable_publisher.publish(Bool(data=False))
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass


if __name__ == "__main__":
    main()
