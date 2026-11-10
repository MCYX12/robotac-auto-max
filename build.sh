#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"
source /opt/ros/humble/setup.bash
cd "$ROOT/finals_ws"
colcon build --build-base build_humble --install-base install_humble --packages-select robotac_finals --symlink-install
source install_humble/setup.bash
cd "$ROOT/dev_ws"
colcon build --build-base build_humble --install-base install_humble --packages-select robotac_mapping_baseline --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
