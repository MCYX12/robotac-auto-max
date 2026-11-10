#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspace/finals_ws/install_humble/setup.bash
source /workspace/dev_ws/install_humble/setup.bash
export ROBOTAC_SDK_ROOT=/workspace/deps/zsibot_sdk
export LD_LIBRARY_PATH="$ROBOTAC_SDK_ROOT/lib/zsl-1/x86_64:/workspace/runtime/src/robot_mc/build/export/mc/bin:${LD_LIBRARY_PATH:-}"
exec "$@"
