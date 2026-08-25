#!/usr/bin/env bash
# Build isolated core/engine environments inside the running container.
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh

./docker_ctl.sh start

profile="${PROFILE:-rocm714}"
engines="${ENGINES:-swift,trl,easydistill}"
say "creating isolated environments (profile=$profile, engines=$engines)"
docker exec -i -w "$CONTAINER_WORKDIR" \
  -u "$HOST_UID:$HOST_GID" \
  -e HOME="$RUNTIME_WORK_DIR/home" \
  -e PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}" \
  "$CONTAINER_NAME" \
  python3 scripts/setup_envs.py \
    --profile "$profile" --engines "$engines" --core-dir .venvs/core

ok "install complete. enter with: ./docker_ctl.sh shell"
