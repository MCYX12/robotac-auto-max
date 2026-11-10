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
    base_config = str(share / "config" / "task2.yaml")
    default_targets = str(share / "config" / "task2_targets_unverified.yaml")
    task1_pcd_path = LaunchConfiguration("task1_pcd_path")
    target_config = LaunchConfiguration("target_config")
    status_path = LaunchConfiguration("status_path")
    lidar_topic = LaunchConfiguration("lidar_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    enable_control = LaunchConfiguration("enable_control")
    enable_sdk_bridge = LaunchConfiguration("enable_sdk_bridge")
    auto_start = LaunchConfiguration("auto_start")
    hard_runtime = LaunchConfiguration("hard_runtime")
    simulation_mode = LaunchConfiguration("simulation_mode")
    sim_event_path = LaunchConfiguration("sim_event_path")
    enable_rviz = LaunchConfiguration("enable_rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("task1_pcd_path"),
            DeclareLaunchArgument("target_config", default_value=default_targets),
            DeclareLaunchArgument(
                "status_path",
                default_value="/workspace/finals_ws/output/task2_status.json",
            ),
            DeclareLaunchArgument("lidar_topic", default_value="/front_lidar"),
            DeclareLaunchArgument("odom_topic", default_value="/odom/mujoco_odom"),
            DeclareLaunchArgument("enable_control", default_value="false"),
            DeclareLaunchArgument("enable_sdk_bridge", default_value="false"),
            DeclareLaunchArgument("auto_start", default_value="false"),
            DeclareLaunchArgument("hard_runtime", default_value="1800.0"),
            DeclareLaunchArgument("simulation_mode", default_value="false"),
            DeclareLaunchArgument(
                "sim_event_path",
                default_value="/workspace/finals_ws/output/task2_sim_events.json",
            ),
            DeclareLaunchArgument("enable_rviz", default_value="false"),
            Node(
                package="robotac_finals",
                executable="pcd_map_server",
                name="robotac_task2_pcd_map_server",
                output="screen",
                parameters=[base_config, {"pcd_path": task1_pcd_path}],
                on_exit=Shutdown(reason="task-one PCD server exited"),
            ),
            Node(
                package="robotac_finals",
                executable="cloud_to_grid",
                name="robotac_task2_cloud_to_grid",
                output="screen",
                parameters=[base_config],
                on_exit=Shutdown(reason="task-one map grid projector exited"),
            ),
            Node(
                package="robotac_finals",
                executable="exploration_controller",
                name="robotac_task2_controller",
                output="screen",
                parameters=[base_config, {"odom_topic": odom_topic}],
                condition=IfCondition(enable_control),
                on_exit=Shutdown(reason="task-two controller exited"),
            ),
            Node(
                package="robotac_finals",
                executable="task2_mission_manager",
                name="robotac_task2_mission_manager",
                output="screen",
                parameters=[
                    base_config,
                    target_config,
                    {
                        "odom_topic": odom_topic,
                        "status_path": status_path,
                        "auto_start": ParameterValue(auto_start, value_type=bool),
                        "hard_runtime": ParameterValue(hard_runtime, value_type=float),
                    },
                ],
                condition=IfCondition(enable_control),
                on_exit=Shutdown(reason="task-two mission manager exited"),
            ),
            Node(
                package="robotac_finals",
                executable="sdk_twist_bridge",
                name="robotac_task2_sdk_twist_bridge",
                output="screen",
                parameters=[
                    base_config,
                    {
                        "dry_run": False,
                        "odom_topic": odom_topic,
                        "lidar_topic": lidar_topic,
                    },
                ],
                condition=IfCondition(enable_sdk_bridge),
                on_exit=Shutdown(reason="task-two SDK safety bridge exited"),
            ),
            Node(
                package="robotac_finals",
                executable="task2_box_simulator",
                name="robotac_task2_box_simulator",
                output="screen",
                parameters=[
                    base_config,
                    target_config,
                    {
                        "odom_topic": odom_topic,
                        "event_path": sim_event_path,
                    },
                ],
                condition=IfCondition(simulation_mode),
                on_exit=Shutdown(reason="local task-two box simulator exited"),
            ),
            Node(
                package="robotac_finals",
                executable="robot_visualizer",
                name="robotac_task2_robot_visualizer",
                output="log",
                parameters=[base_config, {"odom_topic": odom_topic}],
                condition=IfCondition(enable_rviz),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="robotac_task2_sim_rviz",
                arguments=["-d", str(share / "rviz" / "task2_sim.rviz")],
                output="screen",
                condition=IfCondition(enable_rviz),
            ),
        ]
    )
