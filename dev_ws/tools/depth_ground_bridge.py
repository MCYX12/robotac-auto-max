#!/usr/bin/env python3
"""Project real Matrix depth returns; mapper must apply the world-floor gate.

Encoding reference: liuxinxinbit/uesim_subscriber sensor_subscriber.py on_depth.
This configuration targets the current 120-degree, z=0.3 m Matrix depth camera.
"""
import time
import json
from pathlib import Path
import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CompressedImage, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from depth_decode import decode_ground_candidates


def main():
    config = json.loads((Path(__file__).resolve().parents[2] / 'config/config.json').read_text())
    sensor = config['robot']['sensors']['depth_sensor']
    expected = {'fov': 120, 'width': 640, 'height': 480,
                'position': {'x': 0, 'y': 0, 'z': .3},
                'rotation': {'roll': 0, 'pitch': 0, 'yaw': 0}}
    if any(sensor.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Depth configuration differs from validated calibration; refusing fusion')
    rclpy.init()
    node = rclpy.create_node('matrix_depth_ground_bridge')
    pub = node.create_publisher(PointCloud2, '/robotac/depth_ground_candidates', qos_profile_sensor_data)
    count, published, empty_candidates, last = 0, 0, 0, time.monotonic()
    def callback(msg):
        nonlocal count, published, empty_candidates, last
        count += 1
        if count % 2:
            return
        image = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        try:
            if image is None or image.shape[:2] != (480, 640):
                raise ValueError('Unexpected depth image; refuse uncalibrated projection')
            xyz = decode_ground_candidates(image)
        except ValueError as e:
            node.get_logger().error(str(e))
            return
        # A valid depth view may contain no eligible ground candidates.
        # Do not publish a zero-width derived cloud as malformed sensor input.
        if len(xyz):
            # Axes x forward, y left, z up; same extrinsic as LiDAR.
            msg.header.frame_id = 'matrix_depth_projected'
            pub.publish(create_cloud_xyz32(msg.header, xyz.astype(np.float32).tolist()))
            published += 1
        else:
            empty_candidates += 1
        if time.monotonic() - last > 5:
            node.get_logger().info(f'depth_frames={published} candidates={len(xyz)} empty_candidate_frames={empty_candidates} axial_range=[0.6,4.0]m')
            last = time.monotonic()
    node.create_subscription(CompressedImage, '/front_depth/image/compressed', callback, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
