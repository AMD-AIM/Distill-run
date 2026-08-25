#!/usr/bin/env bash
# Start / stop / enter the distill2 container.
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh

start() {
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    return 0
  fi
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    say "restarting container $CONTAINER_NAME"
    docker start "$CONTAINER_NAME" >/dev/null
    return 0
  fi

  mkdir -p "$WORK_ROOT" "$OUTPUT_ROOT"
  say "creating container $CONTAINER_NAME (GPUs $TRAIN_GPUS)"
  docker run -d --name "$CONTAINER_NAME" \
    --entrypoint sleep \
    --network host \
    --ipc host \
    --shm-size "$DOCKER_SHM_SIZE" \
    --device /dev/kfd --device /dev/dri \
    --group-add "$KFD_GID" --group-add "$RENDER_GID" \
    --security-opt seccomp=unconfined \
    --cap-add SYS_PTRACE \
    -e ROCR_VISIBLE_DEVICES="$TRAIN_GPUS" \
    -e HF_ENDPOINT="$HF_ENDPOINT" \
    -e HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
    -e WORK_DIR="$RUNTIME_WORK_DIR" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    -e DISTILL_RUN_ENVS_ROOT="$DISTILL_RUN_ENVS_ROOT" \
    -e HOME="$RUNTIME_WORK_DIR/home" \
    -v "$REPO_DIR:$CONTAINER_WORKDIR" \
    -v "$MODEL_ROOT:$MODELS_DIR:ro" \
    -v "$WORK_ROOT:$WORK_DIR" \
    -v "$OUTPUT_ROOT:$OUTPUT_DIR" \
    -w "$CONTAINER_WORKDIR" \
    "$DOCKER_IMAGE" infinity >/dev/null
  ok "container $CONTAINER_NAME is up"
}

case "${1:-}" in
  start) start ;;
  stop)
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 && ok "stopped $CONTAINER_NAME" || warn "not running"
    ;;
  rm)
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 && ok "removed $CONTAINER_NAME" || warn "not present"
    ;;
  shell)
    start
    exec docker exec -it -w "$CONTAINER_WORKDIR" "$CONTAINER_NAME" bash
    ;;
  status)
    docker ps -a --filter "name=^${CONTAINER_NAME}$" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
    ;;
  *)
    echo "usage: $0 {start|stop|rm|shell|status}" >&2
    exit 2
    ;;
esac
