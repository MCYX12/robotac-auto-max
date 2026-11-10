#!/usr/bin/env python3
"""Select safe information-rich frontier goals from an OccupancyGrid."""

from __future__ import annotations

from collections import deque
import json
import math
import time

import numpy as np
from scipy.spatial import cKDTree
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from .structure_observation import StructureObservation, distinct_viewpoint

from .navigation_core import (
    astar,
    line_is_free,
    inflate_obstacles,
    nearest_free,
    reachable_mask,
    select_patrol_cell,
    world_to_cell,
)


class FrontierSelector(Node):
    def __init__(self) -> None:
        super().__init__("robotac_frontier_selector")
        self.declare_parameter("map_topic", "/robotac/nav_map")
        self.declare_parameter("odom_topic", "/odom/mujoco_odom")
        self.declare_parameter("goal_topic", "/robotac/frontier_goal")
        self.declare_parameter("minimum_cluster_cells", 8)
        self.declare_parameter("minimum_goal_distance", 0.75)
        self.declare_parameter("obstacle_clearance", 0.35)
        self.declare_parameter("local_observation_radius", 6.0)
        self.declare_parameter("gain_weight", 0.20)
        self.declare_parameter("distance_weight", 1.0)
        self.declare_parameter("selection_period", 2.0)
        self.declare_parameter("failed_goal_radius", 0.75)
        self.declare_parameter("blacklist_duration", 90.0)
        self.declare_parameter("reached_goal_radius", 0.75)
        self.declare_parameter("reached_blacklist_duration", 90.0)
        self.declare_parameter("goal_republish_period", 10.0)
        self.declare_parameter("goal_timeout", 120.0)
        self.declare_parameter("patrol_visit_radius", 1.25)
        self.declare_parameter("patrol_trace_spacing", 0.50)
        # Opt-in midterm quality-aware exploration; existing finals defaults stay unchanged.
        self.declare_parameter("path_cost_enabled", False)
        self.declare_parameter("expected_travel_speed", 0.24)
        self.declare_parameter("goal_settle_seconds", 0.0)
        self.declare_parameter("floor_cloud_topic", "")
        self.declare_parameter("frontiers_before_floor", 0)
        self.frontiers_before_floor = int(self.get_parameter('frontiers_before_floor').value)
        if self.frontiers_before_floor < 0:
            raise ValueError('frontiers_before_floor must be nonnegative')
        self.frontiers_since_floor = 0
        self.declare_parameter('floor_priority_after_seconds', 0.0)
        self.declare_parameter('floor_minimum_cluster_area', 0.0)
        self.floor_priority_after = float(self.get_parameter('floor_priority_after_seconds').value)
        self.floor_minimum_cluster_area = float(self.get_parameter('floor_minimum_cluster_area').value)
        if (not math.isfinite(self.floor_priority_after) or self.floor_priority_after < 0 or
                not math.isfinite(self.floor_minimum_cluster_area) or self.floor_minimum_cluster_area < 0):
            raise ValueError('Invalid floor scheduling limits')
        self.selection_started_at = time.monotonic()
        self.declare_parameter('floor_detour_budget_seconds', 0.0)
        self.declare_parameter('floor_maximum_path_m', 10.0)
        self.floor_detour_budget = float(self.get_parameter('floor_detour_budget_seconds').value)
        self.floor_maximum_path = float(self.get_parameter('floor_maximum_path_m').value)
        if (not math.isfinite(self.floor_detour_budget) or
                not math.isfinite(self.floor_maximum_path) or
                self.floor_detour_budget < 0 or self.floor_maximum_path <= 0):
            raise ValueError('Invalid floor detour limits')
        self.floor_reserved_seconds = 0.0
        self.declare_parameter("floor_minimum_points_per_cell", 2)
        self.floor_minimum_points_per_cell = int(self.get_parameter('floor_minimum_points_per_cell').value)
        if self.floor_minimum_points_per_cell < 1:
            raise ValueError('floor_minimum_points_per_cell must be positive')
        self.declare_parameter("floor_viewpoint_enabled", False)
        self.floor_viewpoint_enabled = bool(self.get_parameter('floor_viewpoint_enabled').value)
        self.path_cost_enabled = bool(self.get_parameter("path_cost_enabled").value)
        self.travel_speed = float(self.get_parameter("expected_travel_speed").value)
        self.settle_seconds = float(self.get_parameter("goal_settle_seconds").value)
        if (not math.isfinite(self.travel_speed) or not math.isfinite(self.settle_seconds)
                or self.travel_speed <= 0 or self.settle_seconds < 0):
            raise ValueError('Invalid travel speed or settling time')
        self.settle_until = 0.0
        self.active_timeout = 120.0
        self.floor_xy = None
        self.structure_xyz = None
        self.declare_parameter('structure_reobserve_enabled', False)
        self.structure_reobserve_enabled = bool(self.get_parameter('structure_reobserve_enabled').value)
        self.declare_parameter('structure_minimum_height', .3)
        self.declare_parameter('structure_maximum_height', 3.0)
        self.structure_minimum_height = float(self.get_parameter('structure_minimum_height').value)
        self.structure_maximum_height = float(self.get_parameter('structure_maximum_height').value)
        if (not math.isfinite(self.structure_minimum_height) or not math.isfinite(self.structure_maximum_height)
                or not .25 <= self.structure_minimum_height < self.structure_maximum_height <= 3.3):
            raise ValueError('Invalid structural observation height bounds')
        self.declare_parameter('structure_feedback_enabled', False)
        self.structure_feedback_enabled = bool(self.get_parameter('structure_feedback_enabled').value)
        self.declare_parameter('structure_support_tolerance', .25)
        self.structure_support_tolerance = float(self.get_parameter('structure_support_tolerance').value)
        self.declare_parameter('structure_score_wait_seconds', 45.0)
        self.structure_score_wait = float(self.get_parameter('structure_score_wait_seconds').value)
        self.declare_parameter('quality_hard_runtime', 0.0)
        self.quality_hard_runtime = float(self.get_parameter('quality_hard_runtime').value)
        if (not .10 <= self.structure_support_tolerance <= .25 or
                not 1.0 <= self.structure_score_wait <= 90.0 or
                not math.isfinite(self.quality_hard_runtime) or self.quality_hard_runtime < 0):
            raise ValueError('Invalid structure feedback limits')
        if self.structure_feedback_enabled and self.quality_hard_runtime <= 0:
            raise ValueError('Structure feedback requires a finite positive runtime budget')
        self.floor_source_ns = 0
        self.declare_parameter('floor_feedback_enabled', False)
        self.declare_parameter('floor_score_wait_seconds', 45.0)
        self.floor_feedback_enabled = bool(self.get_parameter('floor_feedback_enabled').value)
        self.floor_score_wait = float(self.get_parameter('floor_score_wait_seconds').value)
        if (not math.isfinite(self.floor_score_wait) or not 1.0 <= self.floor_score_wait <= 90.0 or
                (self.floor_feedback_enabled and self.quality_hard_runtime <= 0)):
            raise ValueError('Floor feedback requires a bounded wait and positive runtime')
        self.floor_review = None
        self.structure_observation = None
        self.quality_review_due = False
        self.finished_structure_regions = []
        self.structure_gap_targets = {}
        self.last_goal_id_ns = 0
        self.last_goal_monotonic = 0.0
        self.floor_stamp = None
        self.floor_frame = None

        self.map_topic = str(self.get_parameter("map_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.minimum_cluster = int(self.get_parameter("minimum_cluster_cells").value)
        self.minimum_distance = float(self.get_parameter("minimum_goal_distance").value)
        self.clearance = float(self.get_parameter("obstacle_clearance").value)
        self.local_observation_radius = float(
            self.get_parameter("local_observation_radius").value
        )
        self.gain_weight = float(self.get_parameter("gain_weight").value)
        self.distance_weight = float(self.get_parameter("distance_weight").value)
        self.failed_goal_radius = float(self.get_parameter("failed_goal_radius").value)
        self.blacklist_duration = float(self.get_parameter("blacklist_duration").value)
        self.reached_goal_radius = float(
            self.get_parameter("reached_goal_radius").value
        )
        self.reached_blacklist_duration = float(
            self.get_parameter("reached_blacklist_duration").value
        )
        self.goal_republish_period = float(
            self.get_parameter("goal_republish_period").value
        )
        self.goal_timeout = float(self.get_parameter("goal_timeout").value)
        self.active_timeout = self.goal_timeout
        self.patrol_visit_radius = float(
            self.get_parameter("patrol_visit_radius").value
        )
        self.patrol_trace_spacing = float(
            self.get_parameter("patrol_trace_spacing").value
        )
        period = float(self.get_parameter("selection_period").value)

        self.map_message = None
        self.robot_xy = None
        self.last_goal = None
        self.last_goal_time = 0.0
        self.goal_active = False
        self.last_goal_type = None
        self.blacklist = []
        # These histories intentionally have no timeout: they live for the
        # whole task run, unlike the temporary failed/reached goal blacklist.
        self.visited_world = []
        self.reached_patrol_world = []
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.goal_publisher = self.create_publisher(
            PoseStamped, self.goal_topic, latched_qos
        )
        self.cancel_publisher = self.create_publisher(
            String, "/robotac/frontier_cancel", latched_qos
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/robotac/frontier_markers", latched_qos
        )
        self.status_publisher = self.create_publisher(
            String, "/robotac/exploration_status", latched_qos
        )
        self.create_subscription(
            OccupancyGrid, self.map_topic, self._map_callback, latched_qos
        )
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, 10
        )
        self.create_subscription(
            String, "/robotac/frontier_result", self._result_callback, latched_qos
        )
        floor_topic = str(self.get_parameter('floor_cloud_topic').value)
        if floor_topic:
            self.create_subscription(PointCloud2, floor_topic, self._floor_callback, latched_qos)
        self.create_timer(period, self._select)
        self.get_logger().info(
            f"map={self.map_topic} odom={self.odom_topic} goal={self.goal_topic}"
        )

    def _map_callback(self, message: OccupancyGrid) -> None:
        self.map_message = message

    def _floor_callback(self, message: PointCloud2) -> None:
        source_ns = int(message.header.stamp.sec) * 1000000000 + int(message.header.stamp.nanosec)
        if self.structure_feedback_enabled or self.floor_feedback_enabled:
            now_ns = self.get_clock().now().nanoseconds
            if (source_ns <= self.floor_source_ns or source_ns <= 0
                    or not 0 <= message.header.stamp.nanosec < 1000000000
                    or now_ns - source_ns > 45000000000 or source_ns - now_ns > 500000000
                    or (self.map_message is not None and message.header.frame_id != self.map_message.header.frame_id)):
                return
        xyz = point_cloud2.read_points_numpy(message, field_names=('x', 'y', 'z'), skip_nans=True)
        xyz = np.asarray(xyz).reshape(-1, 3)
        floor = xyz[np.isfinite(xyz).all(axis=1) & (np.abs(xyz[:, 2]) <= .05), :2]
        # Only measured floor points. No ray-interior or scene-model points.
        self.floor_xy = floor
        self.structure_xyz = xyz[np.isfinite(xyz).all(axis=1) &
            (xyz[:,2]>=.25) & (xyz[:,2]<=3.3)]
        self.floor_stamp = (source_ns / 1e9 if self.structure_feedback_enabled or self.floor_feedback_enabled
                            else self.get_clock().now().nanoseconds / 1e9)
        self.floor_source_ns = source_ns
        self.floor_frame = message.header.frame_id
        if self.floor_review is not None and time.monotonic() > self.floor_review['deadline']:
            self._wait_for_floor_review()
        if (self.floor_review is not None and message.header.frame_id == self.floor_review['frame']
                and source_ns >= self.floor_review['required_source_ns']):
            self._publish_floor_review('REVIEW_READY', source_ns)
            self.floor_review = None
            self.quality_review_due = True
        task = self.structure_observation
        if (task is not None and task.state == 'WAITING' and len(self.structure_xyz) >= 100):
            distances = cKDTree(self.structure_xyz).query(task.targets, workers=2)[0]
            if task.observe(distances, source_ns, self.floor_frame, time.monotonic()):
                self._publish_structure_feedback()
                if task.state == 'RESOLVED':
                    self._finish_structure_task()

    def _floor_gaps(self, message, grid):
        now = self.get_clock().now().nanoseconds / 1e9
        if (self.floor_xy is None or self.floor_frame != message.header.frame_id
                or now - self.floor_stamp > 45.0):
            return []
        origin = np.array([message.info.origin.position.x, message.info.origin.position.y])
        if not len(self.floor_xy):return []
        cells=np.argwhere(grid==0)
        if not len(cells):return []
        xy=origin+(cells[:,::-1]+.5)*message.info.resolution
        distance=cKDTree(self.floor_xy).query(xy,workers=2)[0]
        gaps=np.zeros(grid.shape,dtype=bool)
        missing=cells[distance>.10+1e-6];gaps[missing[:,0],missing[:,1]]=True
        # A sparse cell is already covered if a nearby genuine return is within
        # 10 cm. Do not consume travel budget revisiting mere voxel boundaries.
        return [c for c in self._clusters(gaps)
                if len(c) >= max(2, math.ceil(self.floor_minimum_cluster_area / message.info.resolution**2))]

    def _structure_gaps(self, message, grid):
        """Find occupied boundary cells lacking measured vertical support.

        Uses only the live occupancy grid and native cloud, never scene geometry.
        The default 25 cm tolerance accommodates navigation cells and wall trim.
        The local deployment can use a tighter measured-support target; this
        schedules actual observations and never creates synthetic points.
        """
        self.structure_gap_targets = {}
        if (not self.structure_reobserve_enabled or self.structure_xyz is None or
                len(self.structure_xyz)<100 or self.floor_frame!=message.header.frame_id or
                self.floor_stamp is None or self.get_clock().now().nanoseconds/1e9-self.floor_stamp>45.):
            return []
        free=grid==0;adjacent=np.zeros(grid.shape,dtype=bool)
        adjacent[1:] |= free[:-1];adjacent[:-1] |= free[1:]
        adjacent[:,1:] |= free[:,:-1];adjacent[:,:-1] |= free[:,1:]
        origin=np.array([message.info.origin.position.x,message.info.origin.position.y])
        resolution=float(message.info.resolution)
        measured_cells=np.floor((self.structure_xyz[:,:2]-origin)/resolution).astype(np.int64)
        valid=((measured_cells>=0)&(measured_cells<np.array(grid.shape[::-1]))).all(axis=1)
        coordinates=np.column_stack([measured_cells[valid,1],measured_cells[valid,0],
                                     np.floor(self.structure_xyz[valid,2]/.1).astype(np.int64)])
        bins,counts=np.unique(coordinates,axis=0,return_counts=True)
        supported=bins[counts>=3]
        tops=np.full(grid.shape,-np.inf)
        bottoms=np.full(grid.shape,np.inf)
        levels=np.zeros(grid.shape,dtype=np.int32)
        if len(supported):
            np.maximum.at(tops,(supported[:,0],supported[:,1]),(supported[:,2]+1)*.1+.05)
            np.minimum.at(bottoms,(supported[:,0],supported[:,1]),supported[:,2]*.1)
            np.add.at(levels,(supported[:,0],supported[:,1]),1)
        # Native vertical columns remain observation evidence when sparse low
        # navigation rays omit the same wall. This only schedules real views.
        measured_vertical=(levels>=2)&((tops-bottoms)>=.35-1e-6)
        cells=np.argwhere(((grid>=50)|measured_vertical)&adjacent)
        if not len(cells):return []
        xy=origin+(cells[:,::-1]+.5)*resolution
        heights=np.arange(self.structure_minimum_height,self.structure_maximum_height+.001,.15)
        target=np.column_stack([np.repeat(xy,len(heights),axis=0),np.tile(heights,len(xy))])
        distances=cKDTree(self.structure_xyz).query(target,workers=2)[0].reshape(len(xy),-1)
        from scipy.ndimage import maximum_filter
        radius=max(1,int(math.ceil(1./resolution)))
        rows,columns=np.mgrid[-radius:radius+1,-radius:radius+1]
        nearby=maximum_filter(tops,footprint=rows**2+columns**2<=radius**2,mode='constant',cval=-np.inf)
        expected=heights[None,:]<=nearby[cells[:,0],cells[:,1],None]
        # A missing cap can occupy only one sampled height below the locally
        # observed top. Requiring two heights silently ignores such holes even
        # when the native cloud has a gap larger than the support tolerance.
        # Require spatial coherence below (at least three adjacent cells), not
        # two vertical layers, so an isolated sample still cannot create a goal.
        deficient_samples = (distances > self.structure_support_tolerance + 1e-6) & expected
        deficient=cells[np.any(deficient_samples,axis=1)]
        for row_index in np.flatnonzero(np.any(deficient_samples, axis=1)):
            mask = deficient_samples[row_index]
            xyz = np.column_stack([np.repeat(xy[row_index:row_index+1], int(mask.sum()), axis=0), heights[mask]])
            self.structure_gap_targets[tuple(cells[row_index])] = (xyz, distances[row_index, mask].copy())
        mask=np.zeros(grid.shape,bool);mask[deficient[:,0],deficient[:,1]]=True
        return [c for c in self._clusters(mask) if len(c)>=3]

    def _structure_viewpoint(self, message, grid, blocked, reachable, cluster, start, observation=None):
        centre=cluster.mean(axis=0)
        target=tuple(map(int,cluster[np.argmin(np.sum((cluster-centre)**2,axis=1))]))
        if observation is not None:
            target = world_to_cell(observation.anchor_xy[0], observation.anchor_xy[1],
                                   message.info.origin.position.x, message.info.origin.position.y,
                                   message.info.resolution)
            if not (0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1]):
                return None
        cells=np.argwhere((grid==0)&reachable&~blocked)
        distances=np.linalg.norm(cells-np.asarray(target),axis=1)*message.info.resolution
        # Keep some stand-off for upper-wall returns. The former 1.8m
        # viewpoints could repeat the same near-wall vertical field-of-view gap.
        cells=cells[(distances>=2.2)&(distances<=2.8)]
        if not len(cells):return None
        los=grid!=0
        # This is an observation ray ending on a wall, not a navigation path.
        # Thick/diagonal target-wall cells may occupy the final 20cm of that ray.
        # Only known occupied endpoint cells are transparent in this local copy;
        # unknown cells and all navigation collision masks remain unchanged.
        rows,columns=np.indices(grid.shape)
        endpoint=((rows-target[0])**2+(columns-target[1])**2)*message.info.resolution**2 <= .20**2+1e-12
        los[(grid>=50)&endpoint]=False
        trials=0;best=None
        for index in np.argsort(np.sum((cells-np.asarray(start))**2,axis=1)):
            cell=tuple(map(int,cells[index]));world=self._cell_world(message,*cell)
            if np.linalg.norm(world-self.robot_xy)<self.minimum_distance or self._is_blacklisted(world):continue
            if observation is not None and not distinct_viewpoint(observation.anchor_xy, world, observation.views):continue
            if not line_is_free(los,cell,target):continue
            trials+=1;distance=self._travel_distance(blocked,start,cell,message.info.resolution)
            if math.isfinite(distance) and (best is None or distance<best[0]):
                look=self._cell_world(message,*target)
                best=(distance,cell,math.atan2(look[1]-world[1],look[0]-world[0]))
            if trials>=8:break
        return best

    def _travel_distance(self, blocked, start, cell, resolution):
        path = astar(blocked, start, cell)
        if not path:
            return float('inf')
        return sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])) * resolution

    def _floor_viewpoint(self, message, grid, blocked, reachable, cluster, start):
        """Observe a live deficit from 0.8--2m away, never drive into the hole."""
        centre = cluster.mean(axis=0)
        target = tuple(map(int, cluster[np.argmin(np.sum((cluster-centre)**2, axis=1))]))
        cells = np.argwhere((grid == 0) & reachable & ~blocked)
        ranges = np.linalg.norm(cells-np.asarray(target), axis=1)*message.info.resolution
        cells = cells[(ranges >= .8) & (ranges <= 2.0)]
        if not len(cells):
            return None
        order = np.argsort(np.sum((cells-np.asarray(start))**2, axis=1))
        best = None
        # Bound A* calls. Occluded or blacklisted poses never become candidates.
        trials = 0
        for index in order:
            cell = tuple(map(int, cells[index]))
            world = self._cell_world(message, *cell)
            if np.linalg.norm(world-self.robot_xy) < self.minimum_distance or self._is_blacklisted(world):
                continue
            if not line_is_free(grid != 0, cell, target):
                continue
            trials += 1
            distance = self._travel_distance(blocked, start, cell, message.info.resolution)
            if math.isfinite(distance) and (best is None or distance < best[0]):
                look = self._cell_world(message, *target)
                best = (distance, cell, math.atan2(look[1]-world[1], look[0]-world[0]))
            if trials >= 8:
                break
        return best

    def _odom_callback(self, message: Odometry) -> None:
        self.robot_xy = np.array(
            [message.pose.pose.position.x, message.pose.pose.position.y], dtype=float
        )
        if not self.visited_world or all(
            np.linalg.norm(self.robot_xy - point) >= self.patrol_trace_spacing
            for point in self.visited_world
        ):
            self.visited_world.append(self.robot_xy.copy())

    def _result_callback(self, message: String) -> None:
        try:
            result = json.loads(message.data)
            point = np.array([float(result["x"]), float(result["y"])])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return
        # Keep both reached and failed positions out of the queue long enough to
        # span several (intentionally slow) score-cloud/nav-map publications.
        reached = str(result.get("result", "")).upper() == "REACHED"
        matches_active_goal = (
            self.goal_active
            and (not (self.structure_feedback_enabled or self.floor_feedback_enabled)
                 or result.get('goal_id_ns') == self.last_goal_id_ns)
            and self.last_goal is not None
            and np.linalg.norm(point - self.last_goal)
            <= max(0.50, self.reached_goal_radius)
        )
        if not matches_active_goal:
            return  # Ignore a late result from a superseded goal.
        if reached and self.last_goal_type == 'FLOOR_REOBSERVE' and self.floor_feedback_enabled:
            arrival_ns = self.get_clock().now().nanoseconds
            self.floor_review = dict(goal_id_ns=self.last_goal_id_ns,
                                     frame=self.floor_frame,
                                     baseline_source_ns=self.floor_source_ns,
                                     arrival_ns=arrival_ns,
                                     required_source_ns=arrival_ns+int(self.settle_seconds*1e9),
                                     deadline=time.monotonic()+self.settle_seconds+self.floor_score_wait)
            self._publish_floor_review('WAITING', self.floor_source_ns)
        task = self.structure_observation
        if task is not None and self.last_goal_type == 'STRUCTURE_REOBSERVE':
            if reached:
                task.arrived(self.last_goal_id_ns, time.monotonic(), self.get_clock().now().nanoseconds,
                             self.settle_seconds, self.structure_score_wait)
            else:
                task.finish_attempt('NAVIGATION_FAILED:' + str(result.get('reason', 'unknown')))
            self._publish_structure_feedback()
            if task.state == 'UNRESOLVED':
                self._finish_structure_task()
        if reached:
            self.settle_until = self.get_clock().now().nanoseconds / 1e9 + self.settle_seconds
        if reached and matches_active_goal and self.last_goal_type == "PATROL":
            if not any(
                np.linalg.norm(point - previous) < self.patrol_visit_radius
                for previous in self.reached_patrol_world
            ):
                self.reached_patrol_world.append(point.copy())
        duration = (
            self.reached_blacklist_duration if reached else self.blacklist_duration
        )
        radius = self.reached_goal_radius if reached else self.failed_goal_radius
        self.blacklist.append(
            (point, self.get_clock().now().nanoseconds / 1e9 + duration, radius)
        )
        self.last_goal = None
        self.last_goal_time = 0.0
        self.goal_active = False
        self.last_goal_type = None

    def _is_blacklisted(self, world: np.ndarray) -> bool:
        now = self.get_clock().now().nanoseconds / 1e9
        self.blacklist = [item for item in self.blacklist if item[1] > now]
        return any(np.linalg.norm(world - point) < radius for point, _, radius in self.blacklist)

    @staticmethod
    def _frontier_mask(grid: np.ndarray) -> np.ndarray:
        free = grid == 0
        unknown = grid < 0
        adjacent_unknown = np.zeros_like(unknown)
        adjacent_unknown[1:, :] |= unknown[:-1, :]
        adjacent_unknown[:-1, :] |= unknown[1:, :]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]
        return free & adjacent_unknown

    @staticmethod
    def _clusters(mask: np.ndarray) -> list[np.ndarray]:
        visited = np.zeros_like(mask, dtype=bool)
        clusters = []
        height, width = mask.shape
        for row, column in np.argwhere(mask):
            if visited[row, column]:
                continue
            queue = deque([(int(row), int(column))])
            visited[row, column] = True
            cells = []
            while queue:
                current_row, current_column = queue.popleft()
                cells.append((current_row, current_column))
                for delta_row in (-1, 0, 1):
                    for delta_column in (-1, 0, 1):
                        if delta_row == 0 and delta_column == 0:
                            continue
                        next_row = current_row + delta_row
                        next_column = current_column + delta_column
                        if (
                            0 <= next_row < height
                            and 0 <= next_column < width
                            and mask[next_row, next_column]
                            and not visited[next_row, next_column]
                        ):
                            visited[next_row, next_column] = True
                            queue.append((next_row, next_column))
            clusters.append(np.asarray(cells, dtype=int))
        return clusters

    @staticmethod
    def _cell_world(message: OccupancyGrid, row: int, column: int) -> np.ndarray:
        resolution = message.info.resolution
        return np.array(
            [
                message.info.origin.position.x + (column + 0.5) * resolution,
                message.info.origin.position.y + (row + 0.5) * resolution,
            ]
        )

    def _clear_of_obstacles(
        self, grid: np.ndarray, row: int, column: int, radius_cells: int
    ) -> bool:
        r0 = max(0, row - radius_cells)
        r1 = min(grid.shape[0], row + radius_cells + 1)
        c0 = max(0, column - radius_cells)
        c1 = min(grid.shape[1], column + radius_cells + 1)
        return not np.any(grid[r0:r1, c0:c1] > 50)

    def _candidate_for_cluster(
        self,
        message: OccupancyGrid,
        grid: np.ndarray,
        blocked: np.ndarray,
        reachable: np.ndarray,
        cluster: np.ndarray,
    ):
        """Return a safe reachable goal near a frontier cluster's centre.

        A cluster must not be rejected just because its centroid-side cell is
        inside the inflated wall band.  Search the whole cluster and project
        each promising frontier cell onto the same connected free component
        used by the controller.
        """
        centre = cluster.mean(axis=0)
        order = np.argsort(np.sum((cluster - centre) ** 2, axis=1))
        radius_cells = max(1, int(math.ceil(self.clearance / message.info.resolution)))
        # Connectivity may deliberately clear the LiDAR blind zone around the
        # robot, but a goal itself must still be backed by an observed free
        # cell.  This keeps both FRONTIER and PATROL targets out of unknown
        # space even when the planner has a short local connectivity bridge.
        goal_blocked = blocked | (grid != 0)
        for index in order:
            row, column = cluster[index]
            row = int(row)
            column = int(column)
            if not self._clear_of_obstacles(grid, row, column, radius_cells):
                continue
            goal_cell = nearest_free(
                goal_blocked,
                (row, column),
                maximum_radius=max(12, radius_cells * 3),
            )
            if goal_cell is not None and reachable[goal_cell]:
                return goal_cell
        return None

    def _floor_turn_due(self) -> bool:
        return (self.frontiers_before_floor > 0 and
                time.monotonic()-self.selection_started_at >= self.floor_priority_after and
                (self.frontiers_since_floor >= self.frontiers_before_floor or
                 self.quality_review_due))

    def _floor_allocation(self, distance: float, goal_type="FLOOR_REOBSERVE") -> float:
        # Conservative allowance includes travel back toward exploration, turning
        # and dwell. Reserve the full allowance even if the goal finishes early.
        wait = self.structure_score_wait if self.structure_feedback_enabled and goal_type == 'STRUCTURE_REOBSERVE' else 0.0
        if self.floor_feedback_enabled and goal_type == 'FLOOR_REOBSERVE':
            wait = self.floor_score_wait
        return 2.0 * distance / self.travel_speed + 20.0 + self.settle_seconds + wait

    def _floor_within_budget(self, distance: float, goal_type="FLOOR_REOBSERVE") -> bool:
        if self.floor_detour_budget == 0 and not (self.structure_feedback_enabled or self.floor_feedback_enabled):
            return True  # Preserve the opt-out policy of existing finals callers.
        if not math.isfinite(distance) or distance < 0 or distance > self.floor_maximum_path:
            return False
        if (self.structure_feedback_enabled or self.floor_feedback_enabled) and (self._floor_allocation(distance, goal_type) + 10.0 >
                self.quality_hard_runtime - (time.monotonic() - self.selection_started_at)):
            return False
        if self.floor_detour_budget == 0:
            return True  # legacy policy is opt-out
        return (math.isfinite(distance) and distance <= self.floor_maximum_path and
                self.floor_reserved_seconds + self._floor_allocation(distance, goal_type)
                <= self.floor_detour_budget)

    def _record_scheduled_goal(self, goal_type: str) -> None:
        self.quality_review_due = False
        if goal_type in ('FLOOR_REOBSERVE','STRUCTURE_REOBSERVE'):
            self.frontiers_since_floor = 0
        elif goal_type == 'FRONTIER':
            self.frontiers_since_floor += 1

    def _select(self) -> None:
        if self._wait_for_floor_review():
            return
        if self.map_message is None or self.robot_xy is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        task = self.structure_observation
        if task is not None and task.state == 'WAITING':
            if time.monotonic() <= task.deadline:
                return
            task.finish_attempt()
            self._publish_structure_feedback()
            if task.state == 'UNRESOLVED':
                self._finish_structure_task()
        if now < self.settle_until and self.structure_observation is None:
            return
        if self.goal_active and self.last_goal is not None:
            elapsed = (time.monotonic() - self.last_goal_monotonic
                       if self.structure_feedback_enabled else now - self.last_goal_time)
            if elapsed < self.active_timeout:
                return
            if self.structure_feedback_enabled:
                self.cancel_publisher.publish(String(data=json.dumps({
                    'goal_id_ns': self.last_goal_id_ns, 'reason': 'selector_timeout'})))
            if self.structure_observation is not None:
                self.structure_observation.finish_attempt('NAVIGATION_TIMEOUT')
                self._publish_structure_feedback()
                if self.structure_observation.state == 'UNRESOLVED':
                    self._finish_structure_task()
            self.blacklist.append(
                (self.last_goal, now + self.blacklist_duration, self.failed_goal_radius)
            )
            self.goal_active = False
            self.last_goal = None
            self.last_goal_type = None
        message = self.map_message
        expected = int(message.info.width * message.info.height)
        if expected <= 0 or len(message.data) != expected:
            return
        grid = np.asarray(message.data, dtype=np.int8).reshape(
            int(message.info.height), int(message.info.width)
        )
        resolution = float(message.info.resolution)
        origin_x = float(message.info.origin.position.x)
        origin_y = float(message.info.origin.position.y)
        clearance_cells = int(math.ceil(self.clearance / resolution))
        blocked = inflate_obstacles(grid, clearance_cells)
        requested_start = world_to_cell(
            self.robot_xy[0], self.robot_xy[1], origin_x, origin_y, resolution
        )
        # Match the controller's treatment of the Mid360 ground-return blind
        # zone, otherwise the selector and controller disagree on connectivity.
        known_only = grid.copy()
        known_only[known_only < 0] = 0
        inflated_known_obstacles = inflate_obstacles(known_only, clearance_cells)
        local_cells = max(1, int(math.ceil(self.local_observation_radius / resolution)))
        for row in range(requested_start[0] - local_cells, requested_start[0] + local_cells + 1):
            for column in range(requested_start[1] - local_cells, requested_start[1] + local_cells + 1):
                if not (0 <= row < blocked.shape[0] and 0 <= column < blocked.shape[1]):
                    continue
                if (row - requested_start[0]) ** 2 + (column - requested_start[1]) ** 2 > local_cells ** 2:
                    continue
                if not inflated_known_obstacles[row, column]:
                    blocked[row, column] = False
        start = nearest_free(
            blocked, requested_start, maximum_radius=max(4, clearance_cells + 2)
        )
        if start is None:
            self.status_publisher.publish(
                String(data=json.dumps({"state": "NO_REACHABLE_START"}))
            )
            return
        reachable = reachable_mask(blocked, start)
        task = self.structure_observation
        if task is not None and task.state == 'RETRY':
            view = self._structure_viewpoint(message, grid, blocked, reachable, np.zeros((1, 2)), start, task)
            if view is None:
                self._finish_structure_task('NO_SAFE_DISTINCT_VIEWPOINT')
            elif not self._floor_within_budget(view[0], 'STRUCTURE_REOBSERVE'):
                self._finish_structure_task('OBSERVATION_BUDGET_EXHAUSTED')
            else:
                distance, cell, yaw = view
                world = self._cell_world(message, *cell)
                self._dispatch_goal(message, [(0.0, world, len(task.targets), distance)], world,
                                    'STRUCTURE_REOBSERVE', {tuple(world): yaw}, blocked, start,
                                    distance, 0.0, len(task.targets))
                return
        clusters = [
            cluster
            for cluster in self._clusters(self._frontier_mask(grid))
            if len(cluster) >= self.minimum_cluster
        ]
        candidates = []
        for cluster in clusters:
            cell = self._candidate_for_cluster(
                message, grid, blocked, reachable, cluster
            )
            if cell is None:
                continue
            world = self._cell_world(message, *cell)
            distance = float(np.linalg.norm(world - self.robot_xy))
            if distance < self.minimum_distance:
                continue
            if self._is_blacklisted(world):
                continue
            if self.path_cost_enabled:
                distance = self._travel_distance(blocked, start, cell, resolution)
                if not math.isfinite(distance):
                    continue
            # Sub-linear gain avoids a giant far-away boundary dominating every
            # nearby reachable frontier simply because it contains more cells.
            score = self.gain_weight * math.sqrt(len(cluster)) - self.distance_weight * distance
            candidates.append((score, world, len(cluster), distance))
        candidates.sort(key=lambda item: item[0], reverse=True)
        goal_type = "FRONTIER"
        view_yaws = {}
        selected_structure_task = None
        if not candidates or self._floor_turn_due():
            # At goal boundaries give live floor deficits a bounded turn, rather
            # than waiting forever for every frontier to disappear. Keep the
            # frontier fallback if no safe, fresh floor candidate is available.
            frontier_candidates = candidates
            candidates = []
            quality_candidates=[]
            for cluster in self._floor_gaps(message, grid):
                view = (self._floor_viewpoint(message, grid, blocked, reachable, cluster, start)
                        if self.floor_viewpoint_enabled else None)
                cell = (view[1] if view is not None else None) if self.floor_viewpoint_enabled else self._candidate_for_cluster(message, grid, blocked, reachable, cluster)
                if cell is None:
                    continue
                world = self._cell_world(message, *cell)
                if np.linalg.norm(world-self.robot_xy) < self.minimum_distance or self._is_blacklisted(world):
                    continue
                distance = self._travel_distance(blocked, start, cell, resolution)
                if math.isfinite(distance) and self._floor_within_budget(distance):
                    score = math.sqrt(len(cluster)) / (1.0 + distance)
                    quality_candidates.append((score,world,len(cluster),distance,'FLOOR_REOBSERVE',view[2] if view is not None else None,None))
            for cluster in self._structure_gaps(message,grid):
                task = self._new_structure_task(message, cluster) if self.structure_feedback_enabled else None
                if self.structure_feedback_enabled and task is None:
                    continue
                view=self._structure_viewpoint(message,grid,blocked,reachable,cluster,start,task)
                if view is None or not self._floor_within_budget(view[0], 'STRUCTURE_REOBSERVE'):continue
                distance,cell,yaw=view;world=self._cell_world(message,*cell)
                quality_candidates.append((2.0*math.sqrt(len(cluster))/(1.0+distance),world,len(cluster),distance,'STRUCTURE_REOBSERVE',yaw,task))
            if quality_candidates:
                # Close coherent floor holes first, then missing wall structure.
                # Tiny nearby floor patches must not indefinitely defer a remote
                # vertical deficit that occupancy exploration already called seen.
                quality_candidates.sort(key=lambda item:(
                    2 if item[4]=='FLOOR_REOBSERVE' and item[2]*resolution**2>=.30
                    else 1 if item[4]=='STRUCTURE_REOBSERVE' else 0,
                    item[0]),reverse=True)
                best=quality_candidates[0];goal_type=best[4]
                selected_structure_task = best[6]
                if best[5] is not None:view_yaws[tuple(best[1])]=best[5]
                candidates=[item[:4] for item in quality_candidates]
            else:
                candidates = frontier_candidates
        if candidates:
            score, world, cells, distance = candidates[0]
        else:
            visited_cells = [
                world_to_cell(point[0], point[1], origin_x, origin_y, resolution)
                for point in self.visited_world + self.reached_patrol_world
            ]
            patrol_cell = select_patrol_cell(
                grid,
                blocked,
                reachable,
                start,
                visited=visited_cells,
                minimum_distance_cells=max(
                    1, int(math.ceil(self.minimum_distance / resolution))
                ),
                visited_radius_cells=max(
                    1, int(math.ceil(self.patrol_visit_radius / resolution))
                ),
                prefer_nearby=self.path_cost_enabled,
            )
            if patrol_cell is None:
                self.status_publisher.publish(
                    String(
                        data=json.dumps(
                            {
                                "state": "NO_FRONTIER",
                                "clusters": len(clusters),
                                "patrol_candidates": 0,
                            }
                        )
                    )
                )
                return
            goal_type = "PATROL"
            world = self._cell_world(message, *patrol_cell)
            distance = float(np.linalg.norm(world - self.robot_xy))
            if self.path_cost_enabled:
                distance = self._travel_distance(blocked, start, patrol_cell, resolution)
            score = distance
            cells = 1
            candidates = [(score, world, cells, distance)]

        if goal_type == 'STRUCTURE_REOBSERVE' and self.structure_feedback_enabled:
            self.structure_observation = selected_structure_task
        self._dispatch_goal(message, candidates, world, goal_type, view_yaws, blocked,
                            start, distance, score, cells)

    def _dispatch_goal(self, message, candidates, world, goal_type, view_yaws, blocked,
                       start, distance, score, cells):
        now = self.get_clock().now().nanoseconds / 1e9
        resolution = float(message.info.resolution)
        origin_x = message.info.origin.position.x
        origin_y = message.info.origin.position.y
        if self.last_goal is not None and np.linalg.norm(world - self.last_goal) < 0.30:
            if now - self.last_goal_time < self.goal_republish_period:
                return
        goal = PoseStamped()
        goal_id_ns = max(self.get_clock().now().nanoseconds, self.last_goal_id_ns + 1)
        goal.header.stamp.sec, goal.header.stamp.nanosec = divmod(goal_id_ns, 1000000000)
        self.last_goal_id_ns = goal_id_ns
        goal.header.frame_id = "world"
        goal.pose.position.x = float(world[0])
        goal.pose.position.y = float(world[1])
        goal.pose.orientation.w = 1.0
        goal_yaw = None
        if self.floor_viewpoint_enabled:
            goal_yaw = view_yaws.get(tuple(world))
            if goal_yaw is None:
                end = world_to_cell(world[0], world[1], origin_x, origin_y, resolution)
                path = astar(blocked, start, end)
                if len(path) >= 2:
                    goal_yaw = math.atan2(path[-1][0]-path[-2][0], path[-1][1]-path[-2][1])
                else:
                    goal_yaw = math.atan2(world[1]-self.robot_xy[1], world[0]-self.robot_xy[0])
            goal.pose.orientation.z = math.sin(goal_yaw/2)
            goal.pose.orientation.w = math.cos(goal_yaw/2)
        self.goal_publisher.publish(goal)
        self._record_scheduled_goal(goal_type)
        self.last_goal = world
        self.last_goal_time = now
        self.last_goal_monotonic = time.monotonic()
        self.active_timeout = (max(self.goal_timeout, distance / self.travel_speed + 30.0)
                               if self.path_cost_enabled else self.goal_timeout)
        if goal_type in ('FLOOR_REOBSERVE','STRUCTURE_REOBSERVE') and self.floor_detour_budget > 0:
            allocation = self._floor_allocation(distance, goal_type)
            self.floor_reserved_seconds += allocation
            wait = self.structure_score_wait if self.structure_feedback_enabled and goal_type == 'STRUCTURE_REOBSERVE' else 0.0
            self.active_timeout = min(self.active_timeout, allocation - self.settle_seconds - wait)
        self.goal_active = True
        self.last_goal_type = goal_type
        if self.structure_feedback_enabled and goal_type == 'STRUCTURE_REOBSERVE':
            self.structure_observation.begin(goal_id_ns, world)
            self._publish_structure_feedback()
        self._publish_markers(message, candidates[:20], world, goal_type)
        status = {
            "state": "GOAL_SELECTED",
            "goal_type": goal_type,
            "goal_id_ns": goal_id_ns,
            "x": float(world[0]),
            "y": float(world[1]),
            "cluster_cells": int(cells),
            "distance_m": distance,
            "distance_type": 'planned_path' if self.path_cost_enabled or goal_type in ('FLOOR_REOBSERVE', 'STRUCTURE_REOBSERVE') else 'straight_line',
            "goal_timeout_seconds": self.active_timeout,
            "observation_yaw": goal_yaw,
            "score": score,
            "candidate_count": len(candidates),
            "reachable_filter": True,
            "known_free_goal": True,
            "patrol_history_size": len(self.reached_patrol_world),
        }
        self.status_publisher.publish(String(data=json.dumps(status)))
        self.get_logger().info(json.dumps(status))

    def _new_structure_task(self, message, cluster):
        parts = [self.structure_gap_targets[tuple(cell)] for cell in cluster]
        targets = np.vstack([part[0] for part in parts])
        baseline = np.concatenate([part[1] for part in parts])
        # Match fixed 3-D probes, not a broad XY region: an exhausted top patch
        # must not hide a new low-wall or adjacent wall deficit. Resolved probes
        # can lose support later; retain their prior attempts/views on reopening.
        retained = np.ones(len(targets), dtype=bool)
        previous_matches = []
        for previous in self.finished_structure_regions:
            matches = cKDTree(previous.targets).query(targets)[0] <= .08 + 1e-6
            if previous.state == 'UNRESOLVED' or previous.attempt >= previous.max_attempts:
                retained &= ~matches
            else:
                previous_matches.append((previous, matches))
        if not np.any(retained):
            return None
        targets, baseline = targets[retained], baseline[retained]
        centre = targets[:, :2].mean(axis=0)
        anchor = targets[np.argmin(np.sum((targets[:, :2]-centre)**2, axis=1)), :2]
        task = StructureObservation(targets, baseline, self.floor_source_ns, message.header.frame_id,
                                    anchor, threshold=self.structure_support_tolerance)
        for previous, matches in previous_matches:
            if np.any(matches & retained):
                task.attempt = max(task.attempt, previous.attempt)
                for view in previous.views:
                    if not any(np.linalg.norm(view-other) < .01 for other in task.views):
                        task.views.append(view.copy())
        return task

    def _publish_structure_feedback(self):
        task = self.structure_observation
        if task is not None:
            payload = task.payload()
            payload['detour_reserved_seconds'] = self.floor_reserved_seconds
            payload['runtime_remaining_seconds'] = max(0.0, self.quality_hard_runtime -
                                                       (time.monotonic()-self.selection_started_at))
            encoded = json.dumps(payload, allow_nan=False)
            self.status_publisher.publish(String(data=encoded))
            self.get_logger().info(encoded)

    def _finish_structure_task(self, reason=None):
        task = self.structure_observation
        if task is None:
            return
        if reason is not None:
            task.state = 'UNRESOLVED'
            task.reason = reason
            self._publish_structure_feedback()
        self.finished_structure_regions.append(task)
        # Recheck remaining measured deficits after a successful repair before
        # spending another three frontier trips. Existing travel/time budgets
        # still apply to every additional observation.
        self.quality_review_due = task.state == 'RESOLVED'
        self.structure_observation = None

    def _publish_floor_review(self, state, source_ns):
        payload = dict(self.floor_review)
        payload.update(state='FLOOR_OBSERVATION', observation_state=state,
                       fresh_source_ns=source_ns, frame=self.floor_review['frame'])
        self.status_publisher.publish(String(data=json.dumps(payload, allow_nan=False)))
        self.get_logger().info(json.dumps(payload, allow_nan=False))

    def _wait_for_floor_review(self):
        if self.floor_review is None:
            return False
        if time.monotonic() <= self.floor_review['deadline']:
            return True
        self._publish_floor_review('FRESH_MAP_TIMEOUT', self.floor_source_ns)
        self.floor_review = None
        return False

    def _publish_markers(self, message, candidates, selected, goal_type) -> None:
        markers = MarkerArray()
        points = Marker()
        points.header = message.header
        points.header.frame_id = "world"
        points.ns = "frontiers" if goal_type == "FRONTIER" else "patrol_candidates"
        points.id = 0
        points.type = Marker.SPHERE_LIST
        points.action = Marker.ADD
        points.scale.x = points.scale.y = points.scale.z = 0.20
        points.color.r = 0.10
        points.color.g = 1.0 if goal_type == "FRONTIER" else 0.55
        points.color.b = 0.20 if goal_type == "FRONTIER" else 1.0
        points.color.a = 0.85
        for _, world, _, _ in candidates:
            points.points.append(Point(x=float(world[0]), y=float(world[1]), z=0.15))
        markers.markers.append(points)

        chosen = Marker()
        chosen.header = points.header
        chosen.ns = (
            "selected_frontier" if goal_type == "FRONTIER" else "selected_patrol"
        )
        chosen.id = 0
        chosen.type = Marker.SPHERE
        chosen.action = Marker.ADD
        chosen.pose.position.x = float(selected[0])
        chosen.pose.position.y = float(selected[1])
        chosen.pose.position.z = 0.20
        chosen.pose.orientation.w = 1.0
        chosen.scale.x = chosen.scale.y = chosen.scale.z = 0.45
        chosen.color.r = 1.0
        chosen.color.g = 0.35
        chosen.color.a = 1.0
        markers.markers.append(chosen)
        self.marker_publisher.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrontierSelector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass


if __name__ == "__main__":
    main()
