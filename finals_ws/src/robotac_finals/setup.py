from glob import glob
from setuptools import find_packages, setup


package_name = "robotac_finals"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/meshes", glob("meshes/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ROBOTAC Team",
    maintainer_email="robotac@example.com",
    description="Autonomous mapping and target-routing stack for ROBOTAC finals.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sdk_twist_bridge = robotac_finals.sdk_twist_bridge:main",
            "cloud_to_grid = robotac_finals.cloud_to_grid:main",
            "ray_occupancy_grid = robotac_finals.ray_occupancy_grid:main",
            "frontier_selector = robotac_finals.frontier_selector:main",
            "exploration_controller = robotac_finals.exploration_controller:main",
            "task1_mission_manager = robotac_finals.task1_mission_manager:main",
            "pcd_map_server = robotac_finals.pcd_map_server:main",
            "task2_mission_manager = robotac_finals.task2_mission_manager:main",
            "generate_task2_sim_scenario = robotac_finals.generate_task2_sim_scenario:main",
            "task2_box_simulator = robotac_finals.task2_box_simulator:main",
            "robot_visualizer = robotac_finals.robot_visualizer:main",
        ],
    },
)
