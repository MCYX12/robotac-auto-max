#!/usr/bin/env bash
# Run only on Ubuntu 22.04 with the official ROS Humble APT source configured.
set -euo pipefail
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 ]] || { echo 'Requires Ubuntu 22.04 / ROS Humble'; exit 2; }
sudo apt-get update
sudo apt-get install -y build-essential cmake libpython3.10 python3-colcon-common-extensions python3-numpy python3-scipy python3-matplotlib python3-opencv libpcl-dev ros-humble-ros-base ros-humble-rviz2 ros-humble-pcl-conversions ros-humble-sensor-msgs-py ros-humble-launch-ros ros-humble-nav-msgs ros-humble-visualization-msgs
