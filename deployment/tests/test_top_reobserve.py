from pathlib import Path
import sys

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import OccupancyGrid
from robotac_finals.frontier_selector import FrontierSelector


@pytest.fixture
def selector():
    rclpy.init()
    node = FrontierSelector()
    node.structure_reobserve_enabled = True
    node.floor_frame = 'world'
    node.floor_stamp = node.get_clock().now().nanoseconds / 1e9
    yield node
    node.destroy_node()
    rclpy.shutdown()


def wall(node, top=3.):
    message = OccupancyGrid()
    message.info.resolution = .1
    message.info.width = 70
    message.info.height = 50
    message.header.frame_id = 'world'
    grid = np.full((50, 70), -1, dtype=np.int8)
    grid[6:45, 3:67] = 0
    grid[5, 3:67] = 100
    xx, zz = np.meshgrid(np.arange(.35, 6.65, .025), np.arange(.3, top + .01, .025))
    node.structure_xyz = np.column_stack([xx.ravel(), np.full(xx.size, .55), zz.ravel()])
    return message, grid


def test_top_patch_inferred_from_neighboring_measured_height(selector):
    message, grid = wall(selector)
    assert selector._structure_gaps(message, grid) == []
    points = selector.structure_xyz
    selector.structure_xyz = points[~((points[:, 0] > 2.5) & (points[:, 0] < 3.3) & (points[:, 2] > 2.5))]
    gaps = selector._structure_gaps(message, grid)
    assert any(25 < group[:, 1].mean() < 33 for group in gaps)


def test_short_observed_wall_does_not_invent_missing_three_metre_wall(selector):
    message, grid = wall(selector, top=1.0)
    assert selector._structure_gaps(message, grid) == []
    # A lone high outlier must not fabricate a tall wall requiring observation.
    selector.structure_xyz = np.vstack([selector.structure_xyz, [3.05, .55, 3.]])
    assert selector._structure_gaps(message, grid) == []


def test_coherent_missing_cap_at_only_one_expected_height_is_reobserved(selector):
    # Adjacent real returns establish a 2.8m top. Only the 2.85m sample exceeds
    # 25cm in the missing patch; demanding two vertical levels misses it.
    message, grid = wall(selector, top=2.8)
    assert selector._structure_gaps(message, grid) == []
    points = selector.structure_xyz
    selector.structure_xyz = points[~((points[:, 0] > 2.5) &
                                     (points[:, 0] < 3.3) &
                                     (points[:, 2] > 2.55))]
    gaps = selector._structure_gaps(message, grid)
    assert any(25 < group[:, 1].mean() < 33 for group in gaps)
