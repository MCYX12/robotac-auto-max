#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"
NAME=robotac-auto-max
IMAGE=robotac-humble:local
validate_container() {
  docker inspect "$NAME" | python3 -c 'import json,sys; c=json.load(sys.stdin)[0]; assert any(m.get("Source")==sys.argv[1] and m.get("Destination")=="/workspace" for m in c.get("Mounts",[])), "同名容器挂载不匹配，已拒绝操作"' "$ROOT"
}
ensure_container() {
  if ! docker container inspect "$NAME" >/dev/null 2>&1; then
    args=(run -d --name "$NAME" --init --network bridge --shm-size 1g --gpus all
      --user "$(id -u):$(id -g)" --workdir /workspace --entrypoint bash
      --env ROS_DOMAIN_ID=24 --env ROS_LOCALHOST_ONLY=1 --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp
      --env TZ=Asia/Shanghai --env PYTHONUNBUFFERED=1 --env __NV_PRIME_RENDER_OFFLOAD=1 --env __GLX_VENDOR_LIBRARY_NAME=nvidia --env SDK_CLIENT_IP=127.0.0.1 --env QT_X11_NO_MITSHM=1
      --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json --env NVIDIA_DRIVER_CAPABILITIES=all --env XDG_RUNTIME_DIR=/tmp/xdg-runtime
      --mount "type=bind,src=$ROOT,dst=/workspace"
      --mount type=bind,src=/usr/share/vulkan,dst=/usr/share/vulkan,readonly
      --mount type=bind,src=/usr/share/glvnd,dst=/usr/share/glvnd,readonly)
    # Explicit device entries make GPU access part of the OCI container configuration.
    for gpu_device in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-modeset /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
      [[ ! -c "$gpu_device" ]] || args+=(--device "$gpu_device:$gpu_device")
    done
    if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
      args+=(--env "DISPLAY=$DISPLAY" --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix)
      auth="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
      if [[ -f "$auth" ]]; then args+=(--env XAUTHORITY=/tmp/.Xauthority --mount "type=bind,src=$auth,dst=/tmp/.Xauthority,readonly"); fi
    fi
    docker "${args[@]}" "$IMAGE" -lc 'mkdir -p /tmp/xdg-runtime; chmod 700 /tmp/xdg-runtime; exec sleep infinity' >/dev/null
  else
    validate_container
    if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != true ]]; then
    docker start "$NAME" >/dev/null
    fi
  fi
  if ! docker exec "$NAME" nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    echo '容器 GPU 不可用。先运行 bash docker.sh repair 重建本项目容器；源码和结果保留。' >&2
    return 2
  fi
  docker exec "$NAME" bash /workspace/deployment/env.sh python3 /workspace/deployment/control.py recover
}
remote() {
  observer_token=''
  if [[ "$action" == check || "$action" == audit ]]; then
    observer_token="$$-$RANDOM-$(date +%s%N)"
    set -- python3 /workspace/deployment/observer.py run "$observer_token" "$@"
  fi
  docker exec --env TZ=Asia/Shanghai --env PYTHONUNBUFFERED=1 \
    --env __NV_PRIME_RENDER_OFFLOAD=1 --env __GLX_VENDOR_LIBRARY_NAME=nvidia \
    --env "ROBOTAC_HEADLESS=${ROBOTAC_HEADLESS:-1}" --env "MIDTERM_RVIZ=${MIDTERM_RVIZ:-1}" \
    --env "ROBOTAC_RUNTIME_SECONDS=${ROBOTAC_RUNTIME_SECONDS:-1800}" \
    --env "ROBOTAC_STATIONARY_MAPPING=${ROBOTAC_STATIONARY_MAPPING:-1}" \
    "$NAME" bash /workspace/deployment/env.sh "$@" &
  remote_pid=$!
  trap 'if [[ -n "$observer_token" ]]; then docker exec "$NAME" python3 /workspace/deployment/observer.py cancel "$observer_token" || true; else docker exec "$NAME" bash /workspace/deployment/env.sh python3 /workspace/deployment/control.py stop || true; fi; wait "$remote_pid" || true; exit 130' INT TERM
  if wait "$remote_pid"; then result=0; else result=$?; fi
  trap - INT TERM
  return "$result"
}
action="${1:-status}"; [[ $# == 0 ]] || shift
case "$action" in
  up) ensure_container; echo "$NAME ready" ;;
  build) ensure_container; docker exec "$NAME" bash /workspace/build.sh ;;
  repair)
    if docker inspect "$NAME" >/dev/null 2>&1; then
      validate_container
      if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" == true ]]; then
        docker exec "$NAME" bash /workspace/deployment/env.sh python3 /workspace/deployment/control.py stop || true
        docker stop --timeout 15 "$NAME" >/dev/null
      fi
      docker rm "$NAME" >/dev/null
    fi
    ensure_container; echo '项目容器已重建，源码和历史结果保留。' ;;
  start) ensure_container; remote python3 /workspace/deployment/control.py start "$@" ;;
  sim) ensure_container; remote python3 /workspace/deployment/simulation.py ;;
  check) ensure_container; remote python3 /workspace/deployment/check.py "$@" ;;
  audit) ensure_container; remote python3 /workspace/deployment/audit_run.py "$@" ;;
  run) ensure_container; remote bash /workspace/run_auto_max.sh "$@" ;;
  shell) ensure_container; docker exec -it "$NAME" bash ;;
  stop)
    if docker inspect "$NAME" >/dev/null 2>&1; then
      validate_container
      if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" == true ]]; then
        docker exec "$NAME" bash /workspace/deployment/env.sh python3 /workspace/deployment/control.py stop || true
        docker stop --timeout 15 "$NAME"
      else echo '环境已停止。'; fi
    else echo '环境未创建。'; fi ;;
  status)
    if docker inspect "$NAME" >/dev/null 2>&1; then
      validate_container
      docker ps -a --filter "name=^/${NAME}$" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
      if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" == true ]]; then
        docker exec "$NAME" bash /workspace/deployment/env.sh python3 /workspace/deployment/control.py status
      fi
    else echo '环境未创建。'; fi ;;
  *) echo 'Usage: bash docker.sh {start [new_id]|sim|check|run [new_id]|audit id|stop|status|repair|up|build|shell}' >&2; exit 2 ;;
esac
