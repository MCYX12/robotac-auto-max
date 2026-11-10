"""Real native mapper regressions; isolated ROS domain, no simulator or SDK."""
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import struct
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
MAPPER = ROOT / 'dev_ws/install_humble/robotac_mapping_baseline/lib/robotac_mapping_baseline/odom_voxel_mapper'


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def test_observation_support_counts_time_independent_frames(tmp_path):
    compiler = shutil.which('g++')
    assert compiler, 'g++ required for the exact deployed confidence helper'
    source = tmp_path / 'confidence.cpp'
    executable = tmp_path / 'confidence'
    source.write_text(r'''
#include "voxel_confidence.hpp"
#include <cassert>
#include <limits>
using namespace robotac_mapping_baseline;
int main() {
  ObservationSupport thin_pole;
  constexpr int64_t second = 1000000000LL;
  thin_pole.observe(1, 100 * second);
  for (int i=0; i<1000; ++i) thin_pole.observe(1, 100 * second);
  assert(thin_pole.distinct_frames == 1); // many rays are still one frame
  thin_pole.observe(2, 100 * second); // duplicate timestamp under another frame ID
  thin_pole.observe(3, 99 * second); // backward frame
  thin_pole.observe(4, -second);
  thin_pole.observe(5, 0);
  assert(thin_pole.distinct_frames == 1);
  thin_pole.observe(7, 100 * second + 150000000LL);
  assert(!thin_pole.trusted(3, .3));
  thin_pole.observe(8, 100 * second + 300000000LL);
  assert(thin_pole.trusted(3, .3)); // equality at boundary incl. FP rounding
  assert(!thin_pole.trusted(4, .3));
  assert(!thin_pole.trusted(3, .30001));
  ObservationSupport rapid;
  rapid.observe(1, 100 * second);
  rapid.observe(2, 100 * second + 10000000LL);
  rapid.observe(3, 100 * second + 20000000LL);
  assert(rapid.distinct_frames == 3 && !rapid.trusted(3, .3));
  ObservationSupport two_frames;
  two_frames.observe(1, 100 * second); two_frames.observe(2, 105 * second);
  assert(!two_frames.trusted(3, .3));
  ObservationSupport epoch;
  constexpr int64_t epoch_start = 1793000000LL * second;
  epoch.observe(1, epoch_start);
  epoch.observe(2, epoch_start + 150000000LL);
  epoch.observe(3, epoch_start + 299999999LL);
  assert(!epoch.trusted(3, .3)); // one nanosecond short cannot be rounded up
  epoch.observe(4, epoch_start + 300000000LL);
  assert(epoch.trusted(3, .3));
}
''')
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I',
                    str(ROOT / 'dev_ws/src/robotac_mapping_baseline/src'),
                    str(source), '-o', str(executable)], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('parameter', [
    'native_temporal_minimum_frames:=1', 'native_temporal_minimum_frames:=101',
    'native_temporal_minimum_span:=0.0', 'native_temporal_minimum_span:=5.01',
    'native_temporal_minimum_span:=.nan', 'native_confidence_filter_enabled:=false',
    'native_temporal_minimum_frames:=2', 'native_temporal_minimum_span:=0.29',
    'native_temporal_support_enabled:=false', 'voxel_size:=0.10001',
    'voxel_size:=0.035', 'voxel_size:=0.03',
])
def test_mapper_rejects_invalid_temporal_configuration(tmp_path, parameter):
    assert os.environ.get('ROS_DOMAIN_ID') == '178', 'Only isolated domain 178 is allowed'
    assert MAPPER.is_file(), 'Build the native mapper before running integration tests'
    parameters = ['native_confidence_filter_enabled:=true', 'native_temporal_support_enabled:=true',
                  'native_temporal_patch_enabled:=true', 'voxel_size:=0.025']
    name = parameter.split(':=', 1)[0]
    parameters = [p for p in parameters if not p.startswith(name + ':=')] + [parameter]
    command = [str(MAPPER), '--ros-args']
    for value in parameters:
        command.extend(['-p', value])
    result = subprocess.run(command, capture_output=True, text=True, timeout=8., preexec_fn=no_core)
    (tmp_path / 'invalid-parameters.log').write_text(result.stdout + result.stderr)
    assert result.returncode != 0
    assert 'Invalid mapper parameters' in result.stdout + result.stderr


@pytest.mark.parametrize('voxel_size', [.025, .05])
def test_mapper_accepts_nested_fine_and_temporal_grids(tmp_path, voxel_size):
    assert os.environ.get('ROS_DOMAIN_ID') == '178'
    command = [str(MAPPER), '--ros-args', '-p', 'native_confidence_filter_enabled:=true',
               '-p', 'native_temporal_support_enabled:=true', '-p', 'native_temporal_patch_enabled:=true',
               '-p', 'voxel_size:=' + str(voxel_size)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, preexec_fn=no_core)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            process.communicate(timeout=.5)
    finally:
        if process.poll() is None:
            process.terminate()
        output = process.communicate(timeout=5.)[0]
        (tmp_path / 'valid-parameters.log').write_text(output)
    assert 'Mapping ' in output
    assert 'Invalid mapper parameters' not in output


def read_native_points(path):
    """The mapper emits PCL binary XYZ with optional fixed padding per point."""
    import numpy as np
    header, data = path.read_bytes().split(b'DATA binary\n', 1)
    fields = {}
    for line in header.decode().splitlines():
        words = line.split()
        if words and not line.startswith('#'):
            fields[words[0]] = words[1:]
    sizes = [int(v) for v in fields['SIZE']]
    counts = [int(v) for v in fields['COUNT']]
    itemsize = sum(size * count for size, count in zip(sizes, counts))
    offsets = [sum(s * c for s, c in zip(sizes[:i], counts[:i]))
               for i in range(len(sizes))]
    dtype = np.dtype({'names': fields['FIELDS'], 'formats': ['<f4'] * len(sizes),
                      'offsets': offsets, 'itemsize': itemsize})
    points = np.frombuffer(data, dtype=dtype, count=int(fields['POINTS'][0]))
    return np.column_stack([points[name] for name in ('x', 'y', 'z')])


@pytest.mark.parametrize('mode', ['patch', 'temporal', 'legacy'])
def test_native_mapper_trust_and_final_evidence(tmp_path, mode):
    import numpy as np
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, PointField
    from std_srvs.srv import Trigger
    from rclpy.qos import qos_profile_sensor_data

    assert os.environ.get('ROS_DOMAIN_ID') == '178', 'Only isolated domain 178 is allowed'
    assert MAPPER.is_file(), 'Build the native mapper before running integration tests'
    temporal = mode != 'legacy'
    patch_enabled = mode == 'patch'
    namespace = '/confidence_' + mode
    pcd = tmp_path / 'native.pcd'
    log_path = tmp_path / 'mapper.log'
    parameters = {
        'lidar_topic': namespace + '/lidar', 'odom_topic': namespace + '/odom',
        'map_topic': namespace + '/map', 'preview_topic': namespace + '/preview',
        'save_service': namespace + '/save', 'save_path': str(pcd),
        'voxel_size': '.025', 'native_confidence_filter_enabled': 'true',
        'native_rejected_evidence_enabled': 'true', 'native_preserve_ground': 'true',
        'native_minimum_distinct_scans': '2', 'native_neighbor_radius_voxels': '1',
        'native_minimum_neighbor_voxels': '1', 'publish_every_n_scans': '1',
        'preview_every_n_scans': '1', 'checkpoint_every_n_scans': '3',
        # The actual local profile uses 2. Native confidence must supersede this
        # legacy count heuristic, so accepted single hits remain visible.
        'preview_minimum_voxel_observations': '2', 'preview_max_points': '10000',
    }
    # Omitting the switch is deliberate: old profiles must keep legacy behavior.
    if temporal:
        parameters['native_temporal_support_enabled'] = 'true'
    if patch_enabled:
        parameters['native_temporal_patch_enabled'] = 'true'
        parameters['native_ground_tolerance'] = '.05'
    command = [str(MAPPER), '--ros-args']
    for name, value in parameters.items():
        command.extend(['-p', name + ':=' + value])
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, preexec_fn=no_core)
        rclpy.init()
        node = rclpy.create_node('temporal_confidence_injection')
        odom_pub = node.create_publisher(Odometry, namespace + '/odom', qos_profile_sensor_data)
        cloud_pub = node.create_publisher(PointCloud2, namespace + '/lidar', qos_profile_sensor_data)
        maps = []
        previews = []
        subscription = node.create_subscription(PointCloud2, namespace + '/map', maps.append, 10)
        preview_subscription = node.create_subscription(PointCloud2, namespace + '/preview', previews.append, 10)
        client = node.create_client(Trigger, namespace + '/save')

        def spin_until(predicate, seconds=6.):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if predicate():
                    return True
                rclpy.spin_once(node, timeout_sec=.02)
            return bool(predicate())

        def cloud(nanosec, points):
            msg = PointCloud2()
            msg.header.stamp.sec = 1793000000
            msg.header.stamp.nanosec = nanosec
            msg.width = len(points)
            msg.height = 1
            msg.point_step = 12
            msg.row_step = msg.width * 12
            msg.is_dense = True
            msg.fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1)
                          for i, n in enumerate(('x', 'y', 'z'))]
            msg.data = b''.join(struct.pack('<fff', *point) for point in points)
            return msg

        def send(nanosec, points):
            msg = Odometry()
            msg.header.stamp.sec = 1793000000
            msg.header.stamp.nanosec = nanosec
            msg.pose.pose.orientation.w = 1.
            # Discovery counts can precede delivery of the first best-effort
            # sample. Repeat odometry only (never the tested sensor cloud).
            for _ in range(5):
                odom_pub.publish(msg)
                spin_until(lambda: False, seconds=.04)
            before = len(maps)
            cloud_pub.publish(cloud(nanosec, points))
            assert spin_until(lambda: len(maps) > before), log_path.read_text()

        try:
            assert spin_until(lambda: odom_pub.get_subscription_count() > 0 and
                              cloud_pub.get_subscription_count() > 0 and client.service_is_ready()), log_path.read_text()
            # World voxel centers: translation is +0.3m in z, size .025m.
            stable = [(1.0125, .0125, .7125)] + [
                (5.0125, .0125, z) for z in (.9125, 1.1125, 1.3125)]
            only_two = (9.0125, .0125, .7125)
            # A sparse positive/negative patch hits a different fine voxel per
            # frame. No constituent fine voxel becomes a temporal anchor.
            positive = [(x, .0125, .7125) for x in (10.0125, 10.0375, 10.0875)]
            negative = [(x, .0125, .7125) for x in (-10.0875, -10.0625, -10.0125)]
            extra_first = [positive[0], negative[0]] if patch_enabled else []
            extra_second = [positive[1], negative[1]] if patch_enabled else []
            send(10_000_000, stable + [only_two] + extra_first)
            send(160_000_000, stable + extra_second)
            pair = [(3.0125, .0125, .7125), (3.0375, .0125, .7125)]
            edge = (1.0375, .0125, .7125)
            # A complete one-frame 3x3 wall patch surrounds the trusted center.
            # Its eight new voxels may borrow support, but cannot be new anchors.
            surface = [(1.0125, .0125 + dy, .7125 + dz)
                       for dy in (-.025, 0., .025) for dz in (-.025, 0., .025)
                       if dy != 0. or dz != 0.]
            second_ring = (1.0625, .0125, .7125)
            duplicate = (7.0125, .0125, .7125)
            rapid = (8.0125, .0125, .7125)
            many_rays = [(6.0100 + i*.0001, .0125, .7125) for i in range(32)]
            across_boundary = [(10.1125, .0125, .7125), (-9.9875, .0125, .7125)]
            dense_single_frame = [(x, .0125, .7125) for x in (11.0125, 11.0375, 11.0625, 11.0875)]
            short_patch = [(x, .0125, .7125) for x in (12.0125, 12.0375, 12.0875)]
            duplicate_patch = (13.0125, .0125, .7125)
            exempt_ground = (14.0125, .0125, -.26)
            unsupported_above_ground = (14.0125, .0125, -.225)
            extra_third = ([positive[2], negative[2], short_patch[0], duplicate_patch,
                            exempt_ground, unsupported_above_ground] + across_boundary +
                           dense_single_frame) if patch_enabled else []
            send(310_000_000, stable + pair + surface + [edge, second_ring, duplicate, rapid,
                                              only_two, (4.0125, .0125, -.3)] + many_rays + extra_third)
            # Same timestamp messages and reordered messages never create trust.
            for stamp in [310_000_000, 310_000_000, 300_000_000]:
                duplicate_candidates = [(13.0375, .0125, .7125), (13.0875, .0125, .7125)] if patch_enabled else []
                cloud_pub.publish(cloud(stamp, [duplicate] + duplicate_candidates))
                for _ in range(3):
                    rclpy.spin_once(node, timeout_sec=.02)
            send(390_000_000, [rapid] + ([short_patch[1]] if patch_enabled else []))
            send(510_000_000, [rapid] + ([short_patch[2]] if patch_enabled else []))
            future = client.call_async(Trigger.Request())
            assert spin_until(future.done), log_path.read_text()
            assert future.result().success, log_path.read_text()
            stats = json.loads(Path(str(pcd) + '.acquisition.json').read_text())
            points = read_native_points(pcd)
            rejected_path = Path(str(pcd) + '.rejected_voxels.csv')
            with rejected_path.open() as stream:
                rejected = list(csv.DictReader(stream))
            assert stats['native_temporal_support_enabled'] == temporal
            assert stats['native_temporal_patch_enabled'] == patch_enabled
            assert stats['native_temporal_patch_size_m'] == .10
            assert stats['native_temporal_patch_origin_m'] == [0, 0, 0]
            assert stats['native_temporal_minimum_frames'] == 3
            assert stats['native_temporal_minimum_span_seconds'] == .3
            assert stats['nonadvancing_cloud_frames'] == 3
            assert stats['processed_frames'] == 5
            assert stats['invalid_cloud_frames'] == stats['missing_odom_frames'] == 0
            assert stats['rejected_evidence_written'] is True
            expected = {'disabled': 0, 'ground': 1, 'self': 4 if temporal else 6,
                        'neighbor': 9 if temporal else 12, 'patch': 0, 'rejected': 7 if temporal else 2}
            if patch_enabled:
                expected.update(ground=2, patch=7, rejected=17)
            assert stats['confidence_reason_counts'] == expected
            assert len(points) == stats['saved_points'] == sum(expected.values()) - expected['rejected']
            assert len(rejected) == expected['rejected']
            assert np.isfinite(points).all()
            assert spin_until(lambda: previews and previews[-1].header.stamp.nanosec == 510_000_000)
            def message_points(msg):
                fields = {field.name: field.offset for field in msg.fields}
                dtype = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                                  'offsets': [fields[axis] for axis in 'xyz'], 'itemsize': msg.point_step})
                data = np.frombuffer(bytes(msg.data), dtype=dtype)
                return np.column_stack([data[axis] for axis in 'xyz'])
            assert {tuple(point) for point in message_points(previews[-1])} == {tuple(point) for point in points}
            assert {tuple(point) for point in message_points(maps[-1])} == {tuple(point) for point in points}
            # Isolated persistent thin features can self-confirm without neighbors.
            for point in stable + surface + [edge, (4.0125, .0125, -.3)]:
                world = np.array(point) + (0., 0., .3)
                assert np.linalg.norm(points - world, axis=1).min() < 1e-5
            if patch_enabled:
                # Direct patch evidence is sufficient without moving any XYZ.
                for point in positive + negative + [second_ring, exempt_ground]:
                    world = np.array(point) + (0., 0., .3)
                    assert np.linalg.norm(points - world, axis=1).min() < 1e-5
                rejected_xyz = np.array([[float(row[axis]) for axis in 'xyz'] for row in rejected])
                for point in across_boundary + dense_single_frame + short_patch + [duplicate_patch, unsupported_above_ground]:
                    world = np.array(point) + (0., 0., .3)
                    assert np.linalg.norm(rejected_xyz - world, axis=1).min() < 1e-5
                    assert np.linalg.norm(points - world, axis=1).min() > .01
                # Duplicate message stamps never insert their new candidate voxels.
                assert not any(abs(float(row['x']) - 13.0375) < 1e-5 or
                               abs(float(row['x']) - 13.0875) < 1e-5 for row in rejected)
            if temporal and not patch_enabled:
                expected_x = [1.0625, 3.0125, 3.0375, 6.01155, 7.0125, 8.0125, 9.0125]
                assert sorted(float(row['x']) for row in rejected) == pytest.approx(expected_x, abs=1e-6)
                by_x = {x: next(row for row in rejected if abs(float(row['x']) - x) < 1e-6)
                        for x in expected_x}
                assert int(by_x[6.01155]['measurements']) == 32
                assert int(by_x[6.01155]['distinct_frames']) == 1
                assert int(by_x[7.0125]['distinct_frames']) == 1
                assert int(by_x[8.0125]['distinct_frames']) == 3
                assert float(by_x[8.0125]['span_seconds']) == pytest.approx(.2)
                assert int(by_x[9.0125]['distinct_frames']) == 2
                assert float(by_x[9.0125]['span_seconds']) == pytest.approx(.3)
                assert int(by_x[9.0125]['last_stamp_ns']) - int(by_x[9.0125]['first_stamp_ns']) == 300_000_000
            checkpoint = Path(str(pcd) + '.checkpoint.pcd')
            assert checkpoint.is_file()
            assert not Path(str(checkpoint) + '.rejected_voxels.csv').exists()
            checkpoint_stats = json.loads(Path(str(checkpoint) + '.acquisition.json').read_text())
            assert checkpoint_stats['rejected_evidence_written'] is False
            assert process.poll() is None, log_path.read_text()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3.)
            node.destroy_subscription(subscription)
            node.destroy_subscription(preview_subscription)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
