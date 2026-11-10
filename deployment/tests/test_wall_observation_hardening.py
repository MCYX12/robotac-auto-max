"""Reproduce missing navigation wall hits and legged turning across arrival XY."""
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from robotac_finals.exploration_controller import ExplorationController
from robotac_finals.frontier_selector import FrontierSelector


@pytest.fixture(autouse=True)
def ros_context():
    rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def selector():
    node = FrontierSelector()
    node.structure_reobserve_enabled = True
    node.floor_frame = 'world'
    node.floor_stamp = node.get_clock().now().nanoseconds / 1e9
    yield node
    node.destroy_node()


def measured_wall(node, top=3., gap=True):
    message = OccupancyGrid()
    message.header.frame_id = 'world'
    message.info.resolution = .1
    message.info.width = 70
    message.info.height = 50
    grid = np.full((50, 70), -1, dtype=np.int8)
    grid[5:45, 3:67] = 0  # Low navigation rays did not mark the wall occupied.
    x, z = np.meshgrid(np.arange(.35, 6.65, .025), np.arange(.3, top+.01, .025))
    points = np.column_stack([x.ravel(), np.full(x.size, .55), z.ravel()])
    if gap:
        points = points[~((points[:, 0] > 2.5) & (points[:, 0] < 3.3) &
                          (points[:, 2] > .7) & (points[:, 2] < 2.5))]
    node.structure_xyz = points
    return message, grid


def test_vertical_native_returns_find_hole_without_navigation_obstacle(selector):
    message, grid = measured_wall(selector)
    before = selector.structure_xyz.copy()
    groups = selector._structure_gaps(message, grid)
    assert any(25 < group[:, 1].mean() < 33 for group in groups)
    assert np.array_equal(selector.structure_xyz, before)
    assert not np.any(grid >= 50)


def test_complete_unmarked_wall_does_not_request_repair(selector):
    message, grid = measured_wall(selector, gap=False)
    assert selector._structure_gaps(message, grid) == []


def test_short_unmarked_wall_and_lone_high_return_do_not_invent_tall_wall(selector):
    message, grid = measured_wall(selector, top=1., gap=False)
    selector.structure_xyz = np.vstack([selector.structure_xyz, [3.05, .55, 3.]])
    assert selector._structure_gaps(message, grid) == []


def test_horizontal_native_surface_is_not_a_vertical_wall(selector):
    message, grid = measured_wall(selector)
    x, y = np.meshgrid(np.arange(.35, 6.65, .025), np.arange(.35, .75, .025))
    selector.structure_xyz = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 2.8)])
    assert selector._structure_gaps(message, grid) == []


@pytest.mark.parametrize('invalid_context', ['frame', 'unknown_grid'])
def test_native_wall_does_not_bypass_map_context(selector, invalid_context):
    message, grid = measured_wall(selector)
    if invalid_context == 'frame':
        message.header.frame_id = 'another_world'
    else:
        grid[:] = -1
    assert selector._structure_gaps(message, grid) == []


@pytest.mark.parametrize('terminal_state', ['RESOLVED', 'UNRESOLVED'])
def test_only_successful_structure_repair_requests_next_quality_review(selector, terminal_state):
    selector.frontiers_before_floor = 3
    selector.frontiers_since_floor = 0
    selector.floor_priority_after = 600.
    selector.selection_started_at = time.monotonic()-610
    selector.structure_observation = SimpleNamespace(state=terminal_state)
    selector._finish_structure_task()
    assert selector._floor_turn_due() == (terminal_state == 'RESOLVED')
    selector._record_scheduled_goal('FRONTIER')
    assert not selector._floor_turn_due()


def test_structure_review_cannot_preempt_initial_exploration(selector):
    selector.frontiers_before_floor = 3
    selector.frontiers_since_floor = 0
    selector.quality_review_due = True
    selector.floor_priority_after = 600.
    selector.selection_started_at = time.monotonic()-100
    assert not selector._floor_turn_due()


@pytest.fixture
def controller():
    node = ExplorationController()
    node.pose = (6.724, -11.538, 1.84)
    node.goal = (6.65, -11.35)
    node.goal_yaw = -.811
    node.maximum_map_age = 0.
    node.scan_pause_period = 0.
    node.planned_goal = node.goal
    commands, results, states = [], [], []
    node.command_publisher = SimpleNamespace(publish=commands.append)
    node._publish_result = lambda result, reason: results.append((result, reason))
    node._publish_status = lambda state, **kw: states.append(state)
    node._plan = lambda: pytest.fail('arrival alignment must not re-enter path tracking')
    yield node, commands, results, states
    node.destroy_node()


def test_observation_turn_survives_small_xy_shift_across_goal_tolerance(controller):
    node, commands, results, _ = controller
    node._control_tick()
    assert node.alignment_anchor == node.pose[:2]
    node.pose = (6.899, -11.605, -.459)
    assert np.linalg.norm(np.array(node.pose[:2])-node.goal) > node.goal_tolerance
    node._control_tick()
    assert commands[-1].linear.x == 0 and commands[-1].angular.z < 0
    assert not results
    node.pose = (6.899, -11.605, -.811)
    node._control_tick()
    assert results == [('REACHED', 'observation_heading_aligned')]
    assert node.goal is None and node.alignment_anchor is None
    assert commands[-1].linear.x == commands[-1].angular.z == 0


def test_excessive_alignment_displacement_stops_and_fails(controller):
    node, commands, results, _ = controller
    node._control_tick()
    node.pose = (7.024, -11.538, -.4)
    node._control_tick()
    assert results == [('FAILED', 'observation_alignment_drift')]
    assert node.goal is None and node.alignment_anchor is None
    assert commands[-1].linear.x == commands[-1].angular.z == 0


def test_stale_navigation_stops_even_during_latched_alignment(controller):
    node, commands, results, states = controller
    node._control_tick()
    node.maximum_map_age = 1.
    node.last_map_progress_time = time.monotonic()-20
    node._control_tick()
    assert states[-1] == 'NAV_MAP_STALE' and not results
    assert commands[-1].linear.x == commands[-1].angular.z == 0


def test_latched_alignment_still_has_finite_deadline(controller):
    node, commands, results, _ = controller
    node._control_tick()
    node.alignment_started = time.monotonic()-46
    node._control_tick()
    assert results == [('FAILED', 'observation_heading_timeout')]
    assert node.goal is None and node.alignment_anchor is None
    assert commands[-1].linear.x == commands[-1].angular.z == 0


def test_persistent_front_stop_retains_priority_over_alignment(controller):
    node, _, _, _ = controller
    node._control_tick()
    node.front_obstacle_stop_since = time.monotonic()-20
    recovery = []
    node._fail_or_recover = recovery.append
    node._control_tick()
    assert recovery == ['front_obstacle_stop']


def test_new_goal_discards_previous_arrival_anchor(controller):
    node, _, _, _ = controller
    node._control_tick()
    message = PoseStamped()
    message.header.stamp.sec = 1
    message.pose.position.x = 2.
    message.pose.orientation.w = 1.
    node._goal_callback(message)
    assert node.alignment_anchor is None and node.alignment_started is None


def test_cancellation_discards_arrival_anchor_and_stops(controller):
    node, commands, results, _ = controller
    node.goal_id_ns = 1
    node._control_tick()
    node._cancel_callback(String(data='{"goal_id_ns":1}'))
    assert node.goal is None and node.alignment_anchor is None
    assert results == [('CANCELLED', 'selector_timeout')]
    assert commands[-1].linear.x == commands[-1].angular.z == 0
