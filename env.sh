#!/usr/bin/env bash
# Shared configuration for the distill-run workflow.
# Override anything by exporting it before calling ./run.sh.

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# --- container ---------------------------------------------------------------
DOCKER_IMAGE="${DOCKER_IMAGE:-crpi-vedgrdboj0qrpia0.cn-shanghai.personal.cr.aliyuncs.com/infer-images/vllm-rocm-w7900-modelscope:1.36.3}"
CONTAINER_NAME="${CONTAINER_NAME:-distill2}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace/distill}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-$CONTAINER_ROOT/$(basename "$REPO_DIR")}"
DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-64g}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
KFD_GID="${KFD_GID:-$(stat -c %g /dev/kfd 2>/dev/null || echo "$HOST_GID")}"
RENDER_GID="${RENDER_GID:-$(stat -c %g /dev/dri/renderD128 2>/dev/null || echo "$KFD_GID")}"
DISTILL_RUN_ENVS_ROOT="${DISTILL_RUN_ENVS_ROOT:-$CONTAINER_WORKDIR/.venvs}"

# Framework source trees are pinned Git submodules inside this repository.
EASYDISTILL_SRC="${EASYDISTILL_SRC:-$CONTAINER_WORKDIR/third_party/easydistill}"
TRL_SRC="${TRL_SRC:-$CONTAINER_WORKDIR/third_party/trl}"

# GPUs for training. Teacher vLLM holds 0,1, so stay clear of those.
TRAIN_GPUS="${TRAIN_GPUS:-2,3,4,5,6,7}"

# --- host paths mounted into the container -----------------------------------
MODEL_ROOT="${MODEL_ROOT:-/disk/ssd1/models}"
WORK_ROOT="${WORK_ROOT:-$REPO_DIR/work}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/output}"

# --- paths inside the container -----------------------------------------------
MODELS_DIR="${MODELS_DIR:-/models}"
WORK_DIR="${WORK_DIR:-/work}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
RUNTIME_WORK_DIR="${RUNTIME_WORK_DIR:-$WORK_DIR/users/$HOST_UID}"

# --- models ------------------------------------------------------------------
STUDENT_MODEL="${STUDENT_MODEL:-$MODELS_DIR/Qwen3-0.6B}"
TEACHER_MODEL="${TEACHER_MODEL:-$MODELS_DIR/Qwen3.8-27B}"

# --- teacher service (black box) ---------------------------------------------
# Provided by the separate start-model container; host networking means the
# container reaches it on 127.0.0.1.
TEACHER_PORT="${TEACHER_PORT:-8000}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://127.0.0.1:${TEACHER_PORT}/v1}"
TEACHER_MODEL_ID="${TEACHER_MODEL_ID:-$TEACHER_MODEL}"
TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"

# --- mirrors -----------------------------------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

say()  { printf '\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
