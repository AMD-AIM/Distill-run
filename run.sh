#!/usr/bin/env bash
# Single entrypoint for the distill-run workflow on this machine.
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh

CMD="${1:-help}"
[[ $# -gt 0 ]] && shift || true

# Run distill-run inside the container with the shared env pre-exported, so a
# command line only has to name what differs from env.sh.
in_container() {
  ./docker_ctl.sh start
  local -a extra_env=()
  local runner="distill-run"
  if [[ -x "$REPO_DIR/.venvs/core/bin/distill-run" ]]; then
    runner="$DISTILL_RUN_ENVS_ROOT/core/bin/distill-run"
  fi
  if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    extra_env+=(-e "NPROC_PER_NODE=$NPROC_PER_NODE")
  fi
  if [[ -n "${DISTILL_GPU:-}" ]]; then
    # Logical index within TRAIN_GPUS. A single-process TRL run must see one
    # device; exposing all six makes Trainer and the embedding weights disagree.
    extra_env+=(-e "HIP_VISIBLE_DEVICES=$DISTILL_GPU")
  fi
  docker exec -i -w "$CONTAINER_WORKDIR" \
    -e HOME="$RUNTIME_WORK_DIR/home" \
    -e DISTILL_RUN_ENVS_ROOT="$DISTILL_RUN_ENVS_ROOT" \
    -e TEACHER_BASE_URL="$TEACHER_BASE_URL" \
    -e TEACHER_MODEL_ID="$TEACHER_MODEL_ID" \
    -e TEACHER_API_KEY="$TEACHER_API_KEY" \
    -e STUDENT_MODEL="$STUDENT_MODEL" \
    -e TEACHER_MODEL="$TEACHER_MODEL" \
    -e WORK_DIR="$RUNTIME_WORK_DIR" \
    "${extra_env[@]}" \
    "$CONTAINER_NAME" setpriv \
      --reuid "$HOST_UID" --regid "$HOST_GID" \
      --groups "$KFD_GID,$RENDER_GID" -- \
      "$runner" "$@"
}

in_container_distributed() {
  ./docker_ctl.sh start
  local launch_config="$1"
  shift
  local processes="${DISTRIBUTED_PROCESSES:-${FSDP_PROCESSES:-6}}"
  local accelerate="accelerate"
  if [[ -x "$REPO_DIR/.venvs/trl/bin/accelerate" ]]; then
    accelerate="$DISTILL_RUN_ENVS_ROOT/trl/bin/accelerate"
  fi
  docker exec -i -w "$CONTAINER_WORKDIR" \
    -e HOME="$RUNTIME_WORK_DIR/home" \
    -e DISTILL_RUN_ENVS_ROOT="$DISTILL_RUN_ENVS_ROOT" \
    -e STUDENT_MODEL="$STUDENT_MODEL" \
    -e TEACHER_MODEL="$TEACHER_MODEL" \
    -e WORK_DIR="$RUNTIME_WORK_DIR" \
    "$CONTAINER_NAME" setpriv \
      --reuid "$HOST_UID" --regid "$HOST_GID" \
      --groups "$KFD_GID,$RENDER_GID" -- \
      "$accelerate" launch \
      --config_file "$launch_config" \
      --num_processes "$processes" \
      --main_process_port "${MASTER_PORT:-29510}" \
      --module distill_run "$@"
}

require_teacher() {
  curl -sf --max-time 5 "$TEACHER_BASE_URL/models" >/dev/null 2>&1 \
    || die "teacher not reachable at $TEACHER_BASE_URL (start it with: repo/start-model/run.sh serve)"
  ok "teacher is up: $TEACHER_BASE_URL"
}

run_id() { echo "${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"; }

case "$CMD" in
  up)      ./docker_ctl.sh start ;;
  down)    ./docker_ctl.sh stop ;;
  shell)   ./docker_ctl.sh shell ;;
  status)  ./docker_ctl.sh status ;;
  install) ./install.sh ;;

  teacher-check) require_teacher ;;

  generate)
    require_teacher
    id="$(run_id)"
    in_container --engine easydistill \
      --config configs/easydistill/instruct_distill.yaml \
      --input "${INPUT:-examples/seed_instructions.jsonl}" \
      --output "$OUTPUT_DIR/generate-$id/sft.jsonl" \
      --run-id "$id" "$@"
    ;;

  sft)
    id="$(run_id)"
    in_container --engine swift \
      --config "${CONFIG:-configs/swift/sft_student.yaml}" \
      --dataset "${DATASET:?set DATASET=/output/.../sft.jsonl}" \
      --output "$OUTPUT_DIR/sft-$id" \
      --run-id "$id" "$@"
    ;;

  trl)
    id="$(run_id)"
    export DISTILL_GPU="${DISTILL_GPU:-0}"
    in_container --engine trl \
      --config "${CONFIG:-configs/trl/trl_distill.yaml}" \
      --data "${DATASET:-examples/seed_instructions.jsonl}" \
      --output "$OUTPUT_DIR/trl-$id" \
      --run-id "$id" "$@"
    ;;

  trl-fsdp)
    id="$(run_id)"
    in_container_distributed configs/accelerate/fsdp2.yaml --engine trl \
      --distributed-strategy fsdp2 \
      --config "${CONFIG:-configs/trl/trl_distill.yaml}" \
      --data "${DATASET:-examples/seed_instructions.jsonl}" \
      --output "$OUTPUT_DIR/trl-fsdp-$id" \
      --run-id "$id" "$@"
    ;;

  trl-zero3)
    id="$(run_id)"
    in_container_distributed configs/accelerate/deepspeed_zero3.yaml --engine trl \
      --distributed-strategy zero3 \
      --config "${CONFIG:-configs/trl/trl_distill.yaml}" \
      --data "${DATASET:-examples/seed_instructions.jsonl}" \
      --output "$OUTPUT_DIR/trl-zero3-$id" \
      --run-id "$id" "$@"
    ;;

  easydistill-swift)
    require_teacher
    id="$(run_id)"
    in_container --engine easydistill-swift \
      --config "${CONFIG:-configs/easydistill-swift/easydistill-swift.yaml}" \
      --input "${INPUT:-examples/seed_instructions.jsonl}" \
      --output "$OUTPUT_DIR/easydistill-swift-$id" \
      --run-id "$id" "$@"
    ;;

  cli) in_container "$@" ;;

  test)
    say "unit tests (host, no GPU)"
    make test
    say "CLI smoke (host)"
    ./scripts/smoke-cli.sh
    ;;

  smoke)
    say "in-container smoke: easydistill-swift + trl with tiny step counts"
    require_teacher
    id="smoke-$(date +%H%M%S)"
    in_container --engine easydistill-swift \
      --config configs/easydistill-swift/smoke.yaml \
      --input examples/seed_instructions.jsonl \
      --output "$OUTPUT_DIR/$id-easydistill-swift" --run-id "$id-es"
    DISTILL_GPU="${DISTILL_GPU:-0}" in_container --engine trl \
      --config configs/smoke/trl_distill.yaml \
      --teacher "${TRL_SMOKE_TEACHER:-$STUDENT_MODEL}" \
      --data examples/seed_instructions.jsonl \
      --output "$OUTPUT_DIR/$id-trl" --run-id "$id-trl"
    ok "in-container smoke passed"
    ;;

  help|*)
    cat <<EOF
usage: $0 <command> [extra distill-run args...]

  up | down | shell | status     container lifecycle ($CONTAINER_NAME)
  install                        install frameworks + distill-run in the container
  teacher-check                  verify the teacher endpoint answers

  generate                       seed -> teacher API -> SFT JSONL
  sft                            train the student on an SFT JSONL (needs DATASET)
  trl                            white-box distillation from both weight sets
  trl-fsdp                       sharded white-box distillation across train GPUs
  trl-zero3                      ZeRO-3 white-box distillation for oversized teachers
  easydistill-swift             EasyDistill generation + Swift training
  cli ...                        raw distill-run invocation in the container

  test                           host-side unit tests + CLI smoke
  smoke                          in-container end-to-end smoke (needs GPUs)

settings go in front of the command, e.g.
  DATASET=$OUTPUT_DIR/generate-x/sft.jsonl $0 sft
  NPROC_PER_NODE=4 $0 easydistill-swift
  INPUT=my_seed.jsonl CONFIG=configs/easydistill-swift/smoke.yaml $0 easydistill-swift

recognised: INPUT, DATASET, CONFIG, RUN_ID, NPROC_PER_NODE, DISTILL_GPU,
            DISTRIBUTED_PROCESSES, FSDP_PROCESSES, MASTER_PORT

teacher : $TEACHER_BASE_URL ($TEACHER_MODEL_ID)
student : $STUDENT_MODEL
output  : $OUTPUT_ROOT -> $OUTPUT_DIR
EOF
    ;;
esac
