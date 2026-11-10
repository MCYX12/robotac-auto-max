from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("robotac_finals"))
    config = str(share / "config" / "task1.yaml")
    rviz_config = str(share / "rviz" / "task1.rviz")
    output_pcd = LaunchConfiguration("output_pcd")
    status_path = LaunchConfiguration("status_path")
    enable_control = LaunchConfiguration("enable_control")
    enable_rviz = LaunchConfiguration("enable_rviz")
    lidar_topic = LaunchConfiguration("lidar_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    minimum_runtime = LaunchConfiguration("minimum_runtime")
    hard_runtime = LaunchConfiguration("hard_runtime")
    enable_coverage_stop = LaunchConfiguration("enable_coverage_converged_stop")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "output_pcd",
                default_value="/workspace/finals_ws/output/task1_native.pcd",
            ),
            DeclareLaunchArgument("enable_control", default_value="false"),
            DeclareLaunchArgument("enable_rviz", default_value="true"),
            DeclareLaunchArgument("lidar_topic", default_value="/front_lidar"),
            DeclareLaunchArgument("odom_topic", default_value="/odom/mujoco_odom"),
            DeclareLaunchArgument("minimum_runtime", default_value="600.0"),
            DeclareLaunchArgument("hard_runtime", default_value="1500.0"),
            DeclareLaunchArgument(
                "enable_coverage_converged_stop", default_value="false"
            ),
            DeclareLaunchArgument(
                "status_path",
                default_value="/workspace/finals_ws/output/task1_status.json",
            ),
            Node(
                package="robotac_mapping_baseline",
                executable="odom_voxel_mapper",
                name="odom_voxel_mapper",
                output="screen",
                parameters=[
                    config,
                    {
                        "save_path": output_pcd,
                        "lidar_topic": lidar_topic,
                        "odom_topic": odom_topic,
                    },
                ],
                on_exit=Shutdown(reason="score-cloud mapper exited"),
            ),
            Node(
                package="robotac_finals",
                executable="ray_occupancy_grid",
                name="robotac_ray_occupancy_grid",
                output="screen",
                parameters=[
                    config,
                    {"lidar_topic": lidar_topic, "odom_topic": odom_topic},
                ],
                on_exit=Shutdown(reason="navigation ray grid exited"),
            ),
            Node(
                package="robotac_finals",
                executable="frontier_selector",
                name="robotac_frontier_selector",
                output="screen",
                parameters=[config, {"odom_topic": odom_topic}],
                on_exit=Shutdown(reason="frontier selector exited"),
            ),
            Node(
                package="robotac_finals",
                executable="exploration_controller",
                name="robotac_exploration_controller",
                output="screen",
                parameters=[config, {"odom_topic": odom_topic}],
                condition=IfCondition(enable_control),
                on_exit=Shutdown(reason="exploration controller exited"),
            ),
            Node(
                package="robotac_finals",
                executable="task1_mission_manager",
                name="robotac_task1_mission_manager",
                output="screen",
                parameters=[
                    config,
                    {
                        "output_pcd": output_pcd,
                        "status_path": status_path,
                        "minimum_runtime": minimum_runtime,
                        "hard_runtime": hard_runtime,
                        "enable_coverage_converged_stop": ParameterValue(
                            enable_coverage_stop, value_type=bool
                        ),
                    },
                ],
                condition=IfCondition(enable_control),
                on_exit=Shutdown(reason="task-one mission manager exited"),
            ),
            Node(
                package="robotac_finals",
                executable="sdk_twist_bridge",
                name="robotac_sdk_twist_bridge",
                output="screen",
                parameters=[
                    config,
                    {
                        "dry_run": False,
                        "odom_topic": odom_topic,
                        "lidar_topic": lidar_topic,
                    },
                ],
                condition=IfCondition(enable_control),
                on_exit=Shutdown(reason="SDK safety bridge exited"),
            ),
            Node(
                package="robotac_finals",
                executable="robot_visualizer",
                name="robotac_robot_visualizer",
                output="log",
                parameters=[config, {"odom_topic": odom_topic}],
                condition=IfCondition(enable_rviz),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="robotac_finals_rviz",
                arguments=["-d", rviz_config],
                output="log",
                condition=IfCondition(enable_rviz),
            ),
        ]
    )
