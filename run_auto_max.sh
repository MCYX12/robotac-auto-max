#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /opt/ros/humble/setup.bash
source finals_ws/install_humble/setup.bash
source dev_ws/install_humble/setup.bash
export ROBOTAC_SDK_ROOT="$PWD/deps/zsibot_sdk"
export LD_LIBRARY_PATH="$ROBOTAC_SDK_ROOT/lib/zsl-1/x86_64:${LD_LIBRARY_PATH:-}"
exec python3 "$PWD/deployment/run_session.py" "${1:-auto_max_$(date +%Y%m%d_%H%M%S)}" --speed .45 --duration "${ROBOTAC_RUNTIME_SECONDS:-1800}"
