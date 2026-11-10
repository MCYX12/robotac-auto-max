#!/usr/bin/env bash
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
source "$ROOT_DIR/finals_ws/install_humble/setup.bash"
source "$ROOT_DIR/dev_ws/install_humble/setup.bash"
export ROBOTAC_SDK_ROOT="$ROOT_DIR/deps/zsibot_sdk"
export LD_LIBRARY_PATH="$ROBOTAC_SDK_ROOT/lib/zsl-1/x86_64:${LD_LIBRARY_PATH:-}"
exec python3 "$ROOT_DIR/deployment/run_session.py" "${1:?provide a new run id}" --speed "${2:-0.45}" --duration "${MIDTERM_AUTO_RUNTIME:-1800}"
