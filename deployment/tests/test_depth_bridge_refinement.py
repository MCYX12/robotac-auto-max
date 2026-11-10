"""Valid empty depth views must not become malformed derived sensor messages."""
import os
import signal
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, PointCloud2


def test_empty_depth_candidates_are_skipped_and_valid_views_still_publish(tmp_path):
    assert os.environ.get('ROS_DOMAIN_ID') == '178', 'Use isolated ROS_DOMAIN_ID=178'
    root = Path(__file__).resolve().parents[2]
    log_path = tmp_path / 'depth_bridge.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(
            ['python3', str(root / 'dev_ws/tools/depth_ground_bridge.py')],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rclpy.init()
        node = rclpy.create_node('depth_candidate_regression')
        publisher = node.create_publisher(
            CompressedImage, '/front_depth/image/compressed', qos_profile_sensor_data)
        received = []
        subscription = node.create_subscription(
            PointCloud2, '/robotac/depth_ground_candidates',
            lambda msg: received.append((msg.header.stamp.sec, msg.width)),
            qos_profile_sensor_data)

        def spin_until(predicate, timeout=5.):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return True
                rclpy.spin_once(node, timeout_sec=.03)
            return predicate()

        def publish_view(encoded_distance, stamp):
            image = np.zeros((480, 640, 3), np.uint8)
            image[:, :, 2] = encoded_distance
            ok, encoded = cv2.imencode('.png', image)
            assert ok
            msg = CompressedImage()
            msg.format = 'png'
            msg.header.stamp.sec = stamp
            msg.data = encoded.tobytes()
            publisher.publish(msg)
            for _ in range(4):
                rclpy.spin_once(node, timeout_sec=.03)

        try:
            assert spin_until(lambda: publisher.get_subscription_count() > 0)
            # Zero-range pixels contain no eligible samples, but PNG is valid.
            for _ in range(4):
                publish_view(0, 100)
            assert process.poll() is None, log_path.read_text()
            for _ in range(4):
                publish_view(2, 101)
            assert spin_until(lambda: any(stamp == 101 for stamp, _ in received)), log_path.read_text()
            assert all(stamp == 101 and width > 0 for stamp, width in received)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
            node.destroy_subscription(subscription)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
