"""Floor arrival is navigation progress; fresh post-settle native data is required."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header, String
from sensor_msgs_py import point_cloud2
from robotac_finals.frontier_selector import FrontierSelector


@pytest.fixture
def selector():
    rclpy.init(args=['--ros-args', '-p', 'floor_feedback_enabled:=true',
                    '-p', 'quality_hard_runtime:=1800.0'])
    node = FrontierSelector()
    node.floor_frame = 'world'
    node.map_message = OccupancyGrid()
    node.map_message.header.frame_id = 'world'
    node.last_goal = np.array([2., 1.])
    node.last_goal_id_ns = 123
    node.last_goal_type = 'FLOOR_REOBSERVE'
    node.goal_active = True
    events = []
    node.status_publisher = SimpleNamespace(publish=lambda msg: events.append(json.loads(msg.data)))
    yield node, events
    node.destroy_node()
    rclpy.shutdown()


def reached(node, token=123, result='REACHED'):
    node._result_callback(String(data=json.dumps(dict(result=result, goal_id_ns=token, x=2., y=1.))))


def cloud(node, stamp, frame='world'):
    header = Header(frame_id=frame)
    header.stamp.sec, header.stamp.nanosec = divmod(stamp, 1000000000)
    x, y = np.meshgrid(np.linspace(0., 3., 20), np.linspace(0., 3., 20))
    return point_cloud2.create_cloud_xyz32(header, np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]))


def test_arrival_waits_and_prevents_another_goal_before_fresh_map(selector):
    node, events = selector
    reached(node)
    assert node.floor_review['required_source_ns'] >= node.floor_review['arrival_ns']+int(node.settle_seconds*1e9)
    assert events[-1]['observation_state'] == 'WAITING'
    assert node._wait_for_floor_review()
    node.robot_xy = None
    node._select()
    assert node.floor_review is not None and not node.quality_review_due


def test_only_post_settle_cloud_restarts_quality_review(selector):
    node, events = selector
    reached(node)
    now = node.get_clock().now().nanoseconds
    node.floor_review['required_source_ns'] = now-1000000000
    node._floor_callback(cloud(node, now-2000000000))
    assert node.floor_review is not None and not node.quality_review_due
    node._floor_callback(cloud(node, now-500000000))
    assert node.floor_review is None and node.quality_review_due
    assert events[-1]['observation_state'] == 'REVIEW_READY'
    assert events[-1]['fresh_source_ns'] >= events[-1]['required_source_ns']
    node._record_scheduled_goal('FRONTIER')
    assert not node.quality_review_due


@pytest.mark.parametrize('wrong_input', ['frame', 'duplicate'])
def test_mismatched_or_duplicate_cloud_does_not_end_wait(selector, wrong_input):
    node, _ = selector
    reached(node)
    now = node.get_clock().now().nanoseconds
    node.floor_review['required_source_ns'] = now-1000000000
    stamp = now-500000000
    if wrong_input == 'duplicate':
        node.floor_source_ns = stamp
    node._floor_callback(cloud(node, stamp, 'another_world' if wrong_input == 'frame' else 'world'))
    assert node.floor_review is not None and not node.quality_review_due


def test_wait_has_finite_timeout_without_false_ready(selector):
    node, events = selector
    reached(node)
    node.floor_review['deadline'] = time.monotonic()-1
    assert not node._wait_for_floor_review()
    assert node.floor_review is None and not node.quality_review_due
    assert events[-1]['observation_state'] == 'FRESH_MAP_TIMEOUT'


def test_late_callback_cannot_turn_expired_wait_into_success(selector):
    node, events = selector
    reached(node)
    now = node.get_clock().now().nanoseconds
    node.floor_review['required_source_ns'] = now-1000000000
    node.floor_review['deadline'] = time.monotonic()-1
    node._floor_callback(cloud(node, now-500000000))
    assert node.floor_review is None and not node.quality_review_due
    assert events[-1]['observation_state'] == 'FRESH_MAP_TIMEOUT'


@pytest.mark.parametrize('invalid_result', ['old_token', 'failed', 'disabled'])
def test_only_matching_enabled_reached_floor_goal_starts_wait(selector, invalid_result):
    node, _ = selector
    if invalid_result == 'disabled':
        node.floor_feedback_enabled = False
    reached(node, token=122 if invalid_result == 'old_token' else 123,
            result='FAILED' if invalid_result == 'failed' else 'REACHED')
    assert node.floor_review is None


def test_floor_wait_is_included_in_budget_and_hard_deadline(selector):
    node, _ = selector
    node.floor_feedback_enabled = False
    without_wait = node._floor_allocation(5., 'FLOOR_REOBSERVE')
    node.floor_feedback_enabled = True
    assert node._floor_allocation(5., 'FLOOR_REOBSERVE') == without_wait+node.floor_score_wait
    node.floor_detour_budget = 0.
    node.selection_started_at = time.monotonic()-1790
    assert not node._floor_within_budget(5., 'FLOOR_REOBSERVE')


@pytest.mark.parametrize('setting', ['floor_score_wait_seconds:=.nan',
                                     'floor_score_wait_seconds:=0.0',
                                     'floor_score_wait_seconds:=91.0',
                                     'quality_hard_runtime:=0.0'])
def test_unbounded_floor_feedback_parameters_are_rejected(setting):
    args = ['--ros-args', '-p', 'floor_feedback_enabled:=true',
            '-p', 'quality_hard_runtime:=1800.0', '-p', setting]
    rclpy.init(args=args)
    try:
        with pytest.raises(ValueError, match='Floor feedback'):
            FrontierSelector()
    finally:
        rclpy.shutdown()
