"""Measured structural feedback must fail closed on stale evidence and exhausted retries."""
import json
import math
import time

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header, String
from sensor_msgs_py import point_cloud2

from robotac_finals.frontier_selector import FrontierSelector
from robotac_finals.exploration_controller import ExplorationController
from robotac_finals.structure_observation import StructureObservation, distinct_viewpoint


def observation():
    return StructureObservation([[2., 0., 2.85], [2.1, 0., 2.85]], [.4, .32],
                                10_000_000_000, 'world', [2., 0.], threshold=.18)


def arrived(task):
    task.begin(100, [0., 0.])
    assert task.arrived(100, 20., 20_000_000_000, 5., 45.)


def test_frozen_targets_do_not_move_with_caller_or_cluster():
    points = np.array([[2., 0., 3.]])
    baseline = np.array([.4])
    task = StructureObservation(points, baseline, 1, 'world', [2., 0.])
    points[:] = 0.;baseline[:] = 0.
    assert np.array_equal(task.targets, [[2., 0., 3.]])
    assert task.baseline[0] == .4
    with pytest.raises(ValueError):task.targets[0, 0] = 9.


@pytest.mark.parametrize('stamp,frame,now', [
    (10_000_000_000, 'world', 30.), (24_000_000_000, 'world', 30.),
    (25_000_000_000, 'world', 30.), (26_000_000_000, 'odom', 30.),
    (26_000_000_000, 'world', 24.9), (80_000_000_000, 'world', 70.1),
])
def test_old_queued_wrong_frame_unsettled_or_late_cloud_cannot_close(stamp, frame, now):
    task = observation();arrived(task)
    assert not task.observe([0., 0.], stamp, frame, now)
    assert task.state == 'WAITING'


def test_all_fixed_targets_must_close_not_average_or_cluster_disappearance():
    task = observation();arrived(task)
    assert task.observe([.1, .30], 26_000_000_000, 'world', 26.)
    assert task.state == 'WAITING' and task.reason == 'PARTIAL_IMPROVEMENT'
    assert not task.observe([.1, .1], 26_000_000_000, 'world', 27.)
    assert not task.observe([.1, .1], 25_500_000_000, 'world', 27.)
    assert task.observe([.18, .1800005], 27_000_000_000, 'world', 27.)
    assert task.state == 'RESOLVED'
    assert task.payload()['geometric_acceptance'] is False


def test_missing_fresh_score_is_distinct_from_measured_no_improvement():
    task = observation();arrived(task);task.finish_attempt()
    assert task.state == 'RETRY' and task.reason == 'NO_FRESH_SCORE'
    task.begin(101, [2., -2.]);task.arrived(101, 80., 80_000_000_000, 5., 45.)
    task.observe([.4, .32], 86_000_000_000, 'world', 86.);task.finish_attempt()
    assert task.reason == 'NO_IMPROVEMENT'
    task.begin(102, [4., 0.]);task.finish_attempt('NAVIGATION_FAILED')
    assert task.state == 'UNRESOLVED' and task.attempt == 3
    with pytest.raises(ValueError):task.begin(103, [2., 2.])


def test_goal_identity_rejects_old_arrival_even_at_same_coordinates():
    task = observation();task.begin(100, [0., 0.])
    assert not task.arrived(99, 20., 20_000_000_000, 5., 45.)
    assert task.state == 'NAVIGATING'


def test_viewpoint_requires_translation_and_twenty_degree_angle():
    assert not distinct_viewpoint([0., 0.], [3., 0.], [[2., 0.]])
    assert not distinct_viewpoint([0., 0.], [.1, .1], [[.2, 0.]])
    assert not distinct_viewpoint([0., 0.], [2.*math.cos(.3), 2.*math.sin(.3)], [[2., 0.]])
    assert distinct_viewpoint([0., 0.], [0., 2.], [[2., 0.]])
    assert not distinct_viewpoint([0., 0.], [0., 2.], [[2., 0.], [0., 2.]])


@pytest.mark.parametrize('distances', [[], [0.], [float('nan'), 0.], [float('inf'), 0.]])
def test_invalid_measurements_never_close(distances):
    task = observation();arrived(task)
    assert not task.observe(distances, 26_000_000_000, 'world', 26.)
    assert task.state == 'WAITING'


@pytest.fixture
def selector():
    rclpy.init();node = FrontierSelector()
    node.structure_feedback_enabled = True
    node.structure_reobserve_enabled = True
    node.structure_support_tolerance = .18
    node.quality_hard_runtime = 1800.
    node.floor_detour_budget = 900.
    node.floor_maximum_path = 60.
    node.settle_seconds = 5.
    node.floor_frame = 'world'
    node.floor_stamp = node.get_clock().now().nanoseconds / 1e9
    node.floor_source_ns = node.get_clock().now().nanoseconds
    yield node
    node.destroy_node();rclpy.shutdown()


def wall(node, top=3., x_shift=0., y_shift=0., trim=0.):
    msg = OccupancyGrid();msg.info.resolution = .1
    msg.info.width = 70;msg.info.height = 50;msg.header.frame_id = 'world'
    msg.info.origin.position.x = x_shift;msg.info.origin.position.y = y_shift
    grid = np.full((50, 70), -1, dtype=np.int8)
    grid[6:45, 3:67] = 0;grid[5, 3:67] = 100
    xx, zz = np.meshgrid(np.arange(.30, 6.7, .025) + x_shift, np.arange(.25, top+.001, .025))
    node.structure_xyz = np.column_stack([xx.ravel(), np.full(xx.size, .55+y_shift+trim), zz.ravel()])
    return msg, grid


@pytest.mark.parametrize('top,x_shift,y_shift,trim', [
    (3., 0., 0., 0.), (2.8, 0., 0., 0.), (1., 0., 0., 0.),
    (3., .05, .05, 0.), (3., 0., 0., .10),
])
def test_complete_short_shifted_and_trimmed_walls_have_no_false_gap(selector, top, x_shift, y_shift, trim):
    msg, grid = wall(selector, top, x_shift, y_shift, trim)
    assert selector._structure_gaps(msg, grid) == []
    if top == 1.:
        selector.structure_xyz = np.vstack([selector.structure_xyz, [3.05, .55, 3.]])
        assert selector._structure_gaps(msg, grid) == []


def test_frozen_live_gap_and_world_history_survive_grid_origin_change(selector):
    msg, grid = wall(selector)
    xyz = selector.structure_xyz
    selector.structure_xyz = xyz[~((xyz[:, 0] > 2.5) & (xyz[:, 0] < 3.3) & (xyz[:, 2] > 2.5))]
    clusters = selector._structure_gaps(msg, grid)
    task = selector._new_structure_task(msg, clusters[0])
    assert np.all(task.baseline > .18) and np.all(task.targets[:, 2] > 2.5)
    selector.structure_observation = task
    selector._finish_structure_task('TEST_EXHAUSTED')
    # Same physical targets, entirely different grid indices and origin.
    msg.info.origin.position.x -= 1.
    shifted = clusters[0] + [0, 10]
    selector.structure_gap_targets = {tuple(np.asarray(cell)+[0,10]): value for cell,value in
                                     selector.structure_gap_targets.items()}
    assert selector._new_structure_task(msg, shifted) is None


def test_wait_counts_in_both_detour_and_monotonic_runtime_budget(selector, monkeypatch):
    allocation = selector._floor_allocation(2.4, 'STRUCTURE_REOBSERVE')
    assert allocation == pytest.approx(90.)
    selector.floor_reserved_seconds = 820.
    assert not selector._floor_within_budget(2.4, 'STRUCTURE_REOBSERVE')
    selector.floor_reserved_seconds = 0.
    monkeypatch.setattr(time, 'monotonic', lambda: selector.selection_started_at + 1710.)
    assert not selector._floor_within_budget(2.4, 'STRUCTURE_REOBSERVE')


def cloud(node, stamp, frame='world', points=None):
    header = Header();header.frame_id = frame
    header.stamp.sec, header.stamp.nanosec = divmod(stamp, 1000000000)
    if points is None:points = [[float(i)*.01, 0., 1.] for i in range(120)]
    return point_cloud2.create_cloud_xyz32(header, points)


def test_selector_discards_repeat_reverse_stale_future_and_wrong_frame_headers(selector):
    msg = OccupancyGrid();msg.header.frame_id = 'world';selector.map_message = msg
    now = selector.get_clock().now().nanoseconds
    selector._floor_callback(cloud(selector, now))
    saved = selector.structure_xyz.copy()
    for stamp, frame in [(now,'world'),(now-1,'world'),(now-46_000_000_000,'world'),
                         (now+2_000_000_000,'world'),(now+1,'odom')]:
        selector._floor_callback(cloud(selector, stamp, frame, [[99.,99.,1.]]))
        assert np.array_equal(selector.structure_xyz, saved)
        assert selector.floor_source_ns == now


def test_selector_reached_never_accepts_old_goal_id(selector):
    selector.last_goal = np.array([0.,0.]);selector.goal_active = True
    selector.last_goal_type = 'STRUCTURE_REOBSERVE';selector.last_goal_id_ns = 100
    task = observation();task.begin(100, [0.,0.]);selector.structure_observation = task
    selector._result_callback(String(data=json.dumps({'result':'REACHED','x':0.,'y':0.,'goal_id_ns':99})))
    assert selector.goal_active and task.state == 'NAVIGATING'
    selector._result_callback(String(data=json.dumps({'result':'REACHED','x':0.,'y':0.,'goal_id_ns':100})))
    assert not selector.goal_active and task.state == 'WAITING'


def test_controller_same_xy_new_token_is_new_goal_but_duplicate_preserves_progress():
    rclpy.init();node = ExplorationController()
    try:
        goal = PoseStamped();goal.pose.position.x = 1.;goal.pose.orientation.w = 1.
        goal.header.stamp.nanosec = 100;node._goal_callback(goal)
        node.recovery_count = 2
        node._goal_callback(goal)
        assert node.recovery_count == 2
        goal.header.stamp.nanosec = 101;node._goal_callback(goal)
        assert node.goal_id_ns == 101 and node.recovery_count == 0
        node.recovery_count = 2;goal.header.stamp.nanosec = 100;node._goal_callback(goal)
        assert node.goal_id_ns == 101 and node.recovery_count == 2
    finally:
        node.destroy_node();rclpy.shutdown()


def test_exhausted_patch_does_not_hide_adjacent_or_lower_new_deficits(selector):
    old = observation();old.state = 'UNRESOLVED';old.attempt = 3
    selector.finished_structure_regions.append(old)
    msg = OccupancyGrid();msg.header.frame_id = 'world'
    targets = np.array([[2., 0., 2.85], [2., 0., .4], [2.4, 0., 2.85]])
    selector.structure_gap_targets = {(1,1): (targets, np.full(3, .4))}
    task = selector._new_structure_task(msg, np.array([[1,1]]))
    assert task is not None and np.array_equal(task.targets, targets[1:])


def test_resolved_patch_losing_support_reopens_with_attempt_and_view_history(selector):
    old = observation();arrived(old)
    old.observe([.1,.1], 26_000_000_000, 'world', 26.)
    selector.finished_structure_regions.append(old)
    msg = OccupancyGrid();msg.header.frame_id = 'world'
    selector.structure_gap_targets = {(1,1): (old.targets.copy(), np.array([.4,.3]))}
    task = selector._new_structure_task(msg, np.array([[1,1]]))
    assert task.attempt == 1 and len(task.views) == 1
    with pytest.raises(ValueError, match='distinct'):task.begin(101, [0.,0.])
    task.begin(101, [2.,-2.])
    assert task.attempt == 2
    task.state = 'RESOLVED';task.attempt = 3
    selector.finished_structure_regions.append(task)
    assert selector._new_structure_task(msg, np.array([[1,1]])) is None


@pytest.mark.parametrize('cancel_before_goal', [False, True])
def test_cancel_stops_old_navigation_and_rejects_queued_goal(cancel_before_goal):
    rclpy.init();node = ExplorationController()
    try:
        sent = []
        node.command_publisher = type('Publisher', (), {'publish': lambda self,m: sent.append(m)})()
        goal = PoseStamped();goal.pose.position.x = 1.;goal.pose.orientation.w = 1.
        goal.header.stamp.nanosec = 100
        if not cancel_before_goal:
            node._goal_callback(goal);node.recovery_until = time.monotonic()+10.
        node._cancel_callback(String(data='{"goal_id_ns":100}'))
        node._goal_callback(goal)
        assert node.goal is None and not node.path and not node.plan_requested
        if not cancel_before_goal:
            assert sent[-1].linear.x == 0. and sent[-1].angular.z == 0.
            assert node.recovery_until == 0.
        goal.header.stamp.nanosec = 101;node._goal_callback(goal)
        node._cancel_callback(String(data='{"goal_id_ns":100}'))
        assert node.goal == (1.,0.) and node.goal_id_ns == 101
    finally:
        node.destroy_node();rclpy.shutdown()


@pytest.mark.parametrize('cancel', ['null', '[]', '{broken', '{"goal_id_ns":true}',
                                  '{"goal_id_ns":0}', '{"goal_id_ns":"100"}'])
def test_malformed_cancellation_cannot_erase_live_goal(cancel):
    rclpy.init();node = ExplorationController()
    try:
        goal = PoseStamped();goal.pose.position.x = 1.;goal.header.stamp.nanosec = 100
        node._goal_callback(goal);node._cancel_callback(String(data=cancel))
        assert node.goal == (1.,0.)
    finally:
        node.destroy_node();rclpy.shutdown()


def test_timeout_sends_token_cancellation_even_without_reachable_next_goal(selector):
    msg = OccupancyGrid();msg.header.frame_id = 'world'
    selector.map_message = msg;selector.robot_xy = np.array([0.,0.])
    selector.goal_active = True;selector.last_goal = np.array([1.,0.])
    selector.last_goal_id_ns = 123;selector.last_goal_type = 'STRUCTURE_REOBSERVE'
    selector.last_goal_monotonic = time.monotonic()-100.;selector.active_timeout = 10.
    task = observation();task.begin(123, [0.,0.]);selector.structure_observation = task
    messages = []
    selector.cancel_publisher = type('Publisher', (), {'publish': lambda self,m: messages.append(json.loads(m.data))})()
    selector._select()
    assert messages == [{'goal_id_ns':123,'reason':'selector_timeout'}]
    assert not selector.goal_active and task.state == 'RETRY'


def test_two_deficits_sharing_viewpoint_keep_selected_targets_and_yaw(selector):
    msg = OccupancyGrid();msg.header.frame_id = 'world';msg.info.resolution = .1
    msg.info.width = msg.info.height = 60;msg.data = [0]*3600
    selector.map_message = msg;selector.robot_xy = np.array([.5,.5])
    selector.floor_viewpoint_enabled = True
    big = StructureObservation([[3.,1.,2.85]], [.4], selector.floor_source_ns, 'world', [3.,1.], threshold=.18)
    small = StructureObservation([[3.,5.,2.85]], [.4], selector.floor_source_ns, 'world', [3.,5.], threshold=.18)
    selector._floor_gaps = lambda *_: []
    selector._structure_gaps = lambda *_: [np.zeros((4,2)), np.zeros((3,2))]
    selector._new_structure_task = lambda msg,c: big if len(c)==4 else small
    selector._structure_viewpoint = lambda msg,g,b,r,c,s,t: (4., (30,30), -1. if t is big else 1.)
    selector._select()
    assert selector.structure_observation is big and big.attempt == 1 and small.attempt == 0
