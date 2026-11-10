"""Exercise the built mapper with one-frame floor ghosts and real sparse support."""
import csv
import json
import os
import signal
import struct
import subprocess
import time

import numpy as np
import pytest

from test_temporal_confidence import MAPPER, no_core, read_native_points


@pytest.mark.parametrize('invalid', [
    'native_preserve_ground:=false', 'native_temporal_patch_enabled:=false',
    'native_temporal_support_enabled:=false', 'native_confidence_filter_enabled:=false',
])
def test_ground_filter_rejects_missing_support_dependencies(invalid):
    assert os.environ.get('ROS_DOMAIN_ID') == '178'
    settings = {
        'native_ground_temporal_filter_enabled': 'true',
        'native_preserve_ground': 'true', 'native_temporal_patch_enabled': 'true',
        'native_temporal_support_enabled': 'true', 'native_confidence_filter_enabled': 'true',
        'voxel_size': '.025',
    }
    key, value = invalid.split(':='); settings[key] = value
    cmd = [str(MAPPER), '--ros-args']
    for key, value in settings.items(): cmd += ['-p', key + ':=' + value]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, preexec_fn=no_core)
    assert result.returncode != 0
    assert 'Invalid mapper parameters' in result.stdout + result.stderr


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('moving_source', ['lidar', 'depth'])
def test_floor_ghosts_need_independent_time_support_and_leave_source_evidence(tmp_path, enabled, moving_source):
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, PointField
    from std_srvs.srv import Trigger
    from rclpy.qos import qos_profile_sensor_data

    assert os.environ.get('ROS_DOMAIN_ID') == '178'
    namespace = ('/ground_hardened_' if enabled else '/ground_legacy_') + moving_source
    pcd = tmp_path / 'native.pcd'
    settings = {
        'lidar_topic': namespace + '/lidar', 'depth_ground_topic': namespace + '/depth',
        'odom_topic': namespace + '/odom', 'map_topic': namespace + '/map',
        'preview_topic': namespace + '/preview', 'save_service': namespace + '/save',
        'save_path': str(pcd), 'voxel_size': '.025', 'ground_snap_enabled': 'true',
        'native_preserve_ground': 'true', 'stationary_mapping': 'true',
        'stationary_settle_seconds': '0.', 'native_temporal_support_enabled': 'true',
        'native_temporal_patch_enabled': 'true', 'native_confidence_filter_enabled': 'true',
        'native_ground_temporal_filter_enabled': str(enabled).lower(),
        'native_rejected_evidence_enabled': 'true', 'publish_every_n_scans': '1',
        'preview_every_n_scans': '1', 'preview_z_min': '-.1',
        'preview_minimum_voxel_observations': '2', 'checkpoint_every_n_scans': '2',
    }
    cmd = [str(MAPPER), '--ros-args']
    for key, value in settings.items(): cmd += ['-p', key + ':=' + value]
    with (tmp_path / 'mapper.log').open('w') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, preexec_fn=no_core)
        rclpy.init(); node = rclpy.create_node('ground_injection')
        odom = node.create_publisher(Odometry, namespace + '/odom', qos_profile_sensor_data)
        lidar = node.create_publisher(PointCloud2, namespace + '/lidar', qos_profile_sensor_data)
        depth = node.create_publisher(PointCloud2, namespace + '/depth', qos_profile_sensor_data)
        maps = []; previews = []
        sub = node.create_subscription(PointCloud2, namespace + '/map', maps.append, 10)
        preview_sub = node.create_subscription(PointCloud2, namespace + '/preview', previews.append, 10)
        save = node.create_client(Trigger, namespace + '/save')

        def spin_until(check, timeout=5.):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if check(): return True
                rclpy.spin_once(node, timeout_sec=.02)
            return bool(check())

        def cloud(stamp, points):
            msg = PointCloud2(); msg.header.stamp.sec = 1793000000
            msg.header.stamp.nanosec = stamp; msg.width = len(points); msg.height = 1
            msg.point_step = 12; msg.row_step = 12 * len(points); msg.is_dense = True
            msg.fields = [PointField(name=n, offset=4*i, datatype=PointField.FLOAT32, count=1)
                          for i, n in enumerate('xyz')]
            msg.data = b''.join(struct.pack('<fff', *p) for p in points)
            return msg

        def send(stamp, points, source=lidar, moving=False):
            msg = Odometry(); msg.header.stamp.sec = 1793000000
            msg.header.stamp.nanosec = stamp; msg.pose.pose.orientation.w = 1.
            msg.twist.twist.linear.x = .3 if moving else 0.
            msg.twist.twist.angular.z = .15 if moving else 0.
            for _ in range(4):
                odom.publish(msg); spin_until(lambda: False, .03)
            before = len(maps); source.publish(cloud(stamp, points))
            assert spin_until(lambda: len(maps) > before), (tmp_path / 'mapper.log').read_text()

        def present(points, expected):
            return bool(np.min(np.linalg.norm(points - expected, axis=1)) < 1e-5)

        try:
            assert spin_until(lambda: lidar.get_subscription_count() and depth.get_subscription_count()
                              and odom.get_subscription_count() and save.service_is_ready())
            stable = (1.0875, .0125, -.3)
            sparse = [(x, .0125, -.3) for x in (2.0125, 2.0375, 2.0875)]
            wall = (5.0125, .0125, .7125)
            # One dense frame is not independent evidence, even across voxels.
            dense = [(6.01 + i*.0001, .0125, -.3) for i in range(150)]
            adjacent = (1.1125, .0125, -.3)  # next patch, next fine voxel
            isolated = (7.0125, .0125, -.3)
            rapid = (8.0125, .0125, -.3)
            send(10_000_000, [stable, sparse[0], wall] + dense)
            send(160_000_000, [stable, sparse[1], wall])
            send(310_000_000, [stable, sparse[2], wall, adjacent, rapid, isolated])
            # Source-specific evidence must be retained, but an equal timestamp
            # from another sensor must not promote a one-frame ghost to trust.
            send(310_000_000, [isolated], depth)
            for stamp in (310_000_000, 300_000_000):
                lidar.publish(cloud(stamp, [(9.0125, .0125, -.3)]))
                spin_until(lambda: False, .1)
            send(390_000_000, [rapid])
            send(510_000_000, [rapid])
            moving_ghost = (3.0125, .0125, -.3)
            for stamp in (610_000_000, 810_000_000, 960_000_000):
                send(stamp, [moving_ghost], source=lidar if moving_source == 'lidar' else depth, moving=True)
            future = save.call_async(Trigger.Request())
            assert spin_until(future.done) and future.result().success
            points = read_native_points(pcd)
            for point in [stable, *sparse, wall]:
                expected = np.array(point) + (0, 0, .3)
                assert present(points, expected), point
            for x in (1.1125, 6.01745, 7.0125, 8.0125):
                assert present(points, (x, .0125, 0.)) is (not enabled), x
            assert present(points, (3.0125, .0125, 0.)) is (not enabled or moving_source == 'depth')
            assert not present(points, (9.0125, .0125, 0.))
            evidence = list(csv.DictReader((tmp_path / 'native.pcd.ground_voxels.csv').open()))
            row = next(v for v in evidence if abs(float(v['x']) - 7.0125) < 1e-5)
            assert int(row['lidar_measurements']) == int(row['depth_measurements']) == 1
            assert int(row['distinct_frames']) == 1 and float(row['span_seconds']) == 0
            assert int(row['accepted']) == int(not enabled)
            stats = json.loads((tmp_path / 'native.pcd.acquisition.json').read_text())
            assert stats['native_ground_temporal_filter_enabled'] == enabled
            assert stats['ground_evidence_written'] is True
            assert (stats['ground_motion_rejected_measurements'] > 0) is (enabled and moving_source == 'lidar')
            assert stats['ground_motion_gate_scope'] == 'lidar_only_depth_geometry_and_temporal'
            assert stats['mapping_fault'] == ''
            # Both published maps and the saved map apply the same new gate.
            def xyz(msg):
                fields = {f.name: f.offset for f in msg.fields}
                dtype = np.dtype({'names': list('xyz'), 'formats': ['<f4']*3,
                                  'offsets': [fields[k] for k in 'xyz'], 'itemsize': msg.point_step})
                a = np.frombuffer(bytes(msg.data), dtype=dtype)
                return np.column_stack([a[k] for k in 'xyz'])
            assert spin_until(lambda: previews and previews[-1].header.stamp.nanosec == 960_000_000)
            assert {tuple(p) for p in xyz(maps[-1])} == {tuple(p) for p in points}
            assert {tuple(p) for p in xyz(previews[-1])} == {tuple(p) for p in points}
            assert not (tmp_path / 'native.pcd.checkpoint.pcd.ground_voxels.csv').exists()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=3)
            node.destroy_subscription(sub); node.destroy_subscription(preview_sub)
            node.destroy_node()
            if rclpy.ok(): rclpy.shutdown()
