"""Midterm online exploration: no route file or scene geometry input."""
from pathlib import Path
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    stationary = os.environ.get('ROBOTAC_STATIONARY_MAPPING', '1') == '1'
    workspace = Path(__file__).resolve().parents[1]
    share = Path(get_package_share_directory('robotac_finals'))
    config = str(share / 'config' / 'task1.yaml')
    output = LaunchConfiguration('output_pcd')
    status = LaunchConfiguration('status_path')
    rviz = LaunchConfiguration('enable_rviz')
    speed = ParameterValue(LaunchConfiguration('speed'), value_type=float)
    hard = ParameterValue(LaunchConfiguration('hard_runtime'), value_type=float)
    actions = [
        DeclareLaunchArgument('output_pcd'),
        DeclareLaunchArgument('status_path'),
        DeclareLaunchArgument('enable_rviz', default_value='true'),
        DeclareLaunchArgument('speed', default_value='0.32'),
        DeclareLaunchArgument('hard_runtime', default_value='1800.0'),
    ]

    def node(executable, name, overrides=None, package='robotac_finals'):
        return Node(
            package=package, executable=executable, name=name,
            parameters=[config, overrides or {}], output='screen',
            on_exit=Shutdown(reason=f'required node exited: {name}'),
        )

    actions += [
        node('odom_voxel_mapper', 'odom_voxel_mapper', {
            'save_path': output,
            'lidar_pose_time_offset': -0.04,
            'adaptive_ground_timing': False,
            'stationary_mapping': stationary,
            'depth_ground_topic': '/robotac/depth_ground_candidates',
            'native_confidence_filter_enabled': True,
            'native_temporal_support_enabled': True,
            'native_temporal_patch_enabled': True,
            'native_ground_temporal_filter_enabled': True,
            'native_lidar_ground_patch_minimum_frames': 6,
            'native_lidar_ground_patch_minimum_span': 1.0,
            'scan_surface_filter_enabled': True,
            'filtered_lidar_topic': '/robotac/filtered_lidar',
            'native_temporal_minimum_frames': 3,
            'native_temporal_minimum_span': 0.30,
            'native_rejected_evidence_enabled': True,
            'native_minimum_distinct_scans': 2,
            'native_neighbor_radius_voxels': 1,
            'native_minimum_neighbor_voxels': 1,
            'native_preserve_ground': True,
            'native_ground_tolerance': 0.10,
            'publish_every_n_scans': 300,
            'checkpoint_every_n_scans': 900,
        }, package='robotac_mapping_baseline'),
        ExecuteProcess(
            cmd=['python3', str(workspace / 'tools' / 'depth_ground_bridge.py')],
            output='screen',
            on_exit=Shutdown(reason='required depth bridge exited'),
        ),
        node('ray_occupancy_grid', 'robotac_ray_occupancy_grid'),
        node('frontier_selector', 'robotac_frontier_selector', {
            'path_cost_enabled': True,
            'expected_travel_speed': 0.20,
            'goal_settle_seconds': 5.0,
            'floor_cloud_topic': '/robotac/finals/score_cloud',
            # Observe substantial measured floor deficits after initial exploration.
            # Reference scene geometry never enters this live selector.
            'floor_minimum_points_per_cell': 2,
            'frontiers_before_floor': 3,
            'floor_priority_after_seconds': 600.0,
            'floor_minimum_cluster_area': 0.02,
            'structure_reobserve_enabled': True,
            'structure_feedback_enabled': True,
            'floor_feedback_enabled': True,
            'structure_support_tolerance': 0.18,
            'structure_score_wait_seconds': 45.0,
            'quality_hard_runtime': hard,
            'floor_detour_budget_seconds': 1800.0,
            'floor_maximum_path_m': 60.0,
            # The independent SDK 0.65 m front stop remains enforced.
            'obstacle_clearance': 0.50,
            'floor_viewpoint_enabled': True,
        }),
        node('exploration_controller', 'robotac_exploration_controller', {
            'maximum_linear_speed': speed,
            'scan_pause_period': 8.0 if stationary else 0.0,
            'scan_pause_duration': 4.0,
            'maximum_yaw_rate': 0.24,
            'heading_kp': 0.8,
            'inflation_margin': 0.18,  # 0.32 + 0.18 = 0.50 m
            'maximum_drive_heading_error': 0.65,
            'align_goal_orientation': True,
            'align_only_observation_goals': True,
            'initial_panorama': stationary,
        }),
        node('sdk_twist_bridge', 'robotac_sdk_twist_bridge', {'dry_run': False}),
        node('task1_mission_manager', 'robotac_task1_mission_manager', {
            'output_pcd': output, 'status_path': status,
            'minimum_runtime': 0.0, 'hard_runtime': hard,
            'runtime_health_required': True,
            # No-frontier / low growth alone cannot prove complete ground coverage.
            'enable_coverage_converged_stop': False,
        }),
        Node(package='robotac_finals', executable='robot_visualizer',
             name='robotac_robot_visualizer', parameters=[config], output='log',
             ),
        Node(package='rviz2', executable='rviz2', name='robotac_midterm_rviz',
             arguments=['-d', str(share / 'rviz' / 'task1.rviz')], output='log',
             condition=IfCondition(rviz)),
    ]
    return LaunchDescription(actions)
