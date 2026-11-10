"""Observation endpoint tolerance never weakens navigable/free-space masks."""
import numpy as np
import pytest
import rclpy
from nav_msgs.msg import OccupancyGrid
from robotac_finals.frontier_selector import FrontierSelector

@pytest.fixture
def scene():
    rclpy.init();node=FrontierSelector()
    message=OccupancyGrid();message.info.resolution=.1
    message.info.width=40;message.info.height=30
    message.header.frame_id='world'
    grid=np.zeros((30,40),dtype=np.int8)
    target=(10,30);viewpoint=(16,8)
    # One reachable and inflated-safe candidate, 2.28m from the wall target.
    grid[10,28:33]=100
    blocked=np.ones(grid.shape,bool);blocked[viewpoint]=False
    reachable=np.zeros(grid.shape,bool);reachable[viewpoint]=True
    node.robot_xy=np.array([.85,2.85]);node.minimum_distance=.75
    node._travel_distance=lambda *args:3.0
    cluster=np.array([target])
    yield node,message,grid,blocked,reachable,cluster,viewpoint
    node.destroy_node()
    if rclpy.ok():rclpy.shutdown()

def select(scene):
    node,message,grid,blocked,reachable,cluster,start=scene
    return node._structure_viewpoint(message,grid,blocked,reachable,cluster,start)

def test_target_wall_thickness_allows_observation_without_changing_collision_masks(scene):
    _,_,grid,blocked,reachable,_,viewpoint=scene
    originals=[array.copy() for array in (grid,blocked,reachable)]
    view=select(scene)
    assert view is not None and view[1]==viewpoint
    for array,original in zip((grid,blocked,reachable),originals):assert np.array_equal(array,original)

def test_unknown_endpoint_neighbor_remains_opaque(scene):
    _,_,grid,_,_,_,_=scene
    grid[10,29]=-1
    assert select(scene) is None

def test_known_obstacle_outside_endpoint_tolerance_remains_opaque(scene):
    _,_,grid,_,_,_,_=scene
    grid[11,26]=100
    assert select(scene) is None
