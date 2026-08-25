#!/usr/bin/env bash
# Exercise the CLI contract with no GPU, no frameworks and no network:
# exit codes, engine-specific parameter sets, and the noop engine's happy path.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

DISTILL_RUN="${DISTILL_RUN:-.venv/bin/python -m distill_run}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAILED=0

expect() {
  local want="$1" name="$2"
  shift 2
  local log="$TMP/$name.log"
  $DISTILL_RUN "$@" >"$log" 2>&1
  local got=$?
  if [[ "$got" == "$want" ]]; then
    printf '\033[1;32m[ OK ]\033[0m %-42s exit=%s\n' "$name" "$got"
  else
    printf '\033[1;31m[FAIL]\033[0m %-42s exit=%s (wanted %s)\n' "$name" "$got" "$want"
    sed 's/^/       /' "$log" | tail -5
    FAILED=1
  fi
}

expect 0 "noop happy path" \
  --engine noop --output "$TMP/out" --work-dir "$TMP/work" --run-id smoke-1

expect 2 "missing --output" \
  --engine noop --work-dir "$TMP/work"

expect 2 "unknown engine" \
  --engine nope --output "$TMP/out" --work-dir "$TMP/work"

expect 2 "teacher URL rejected for noop" \
  --engine noop --output "$TMP/out" --work-dir "$TMP/work" \
  --teacher-base-url http://127.0.0.1:9/v1

expect 2 "swift without a student" \
  --engine swift --output "$TMP/out" --work-dir "$TMP/work" --dataset "$TMP/x.jsonl"

expect 3 "missing config file" \
  --engine noop --output "$TMP/out" --work-dir "$TMP/work" --config "$TMP/absent.yaml"

expect 5 "dataset path does not exist" \
  --engine swift --output "$TMP/out" --work-dir "$TMP/work" \
  --model "$TMP/work" --dataset "$TMP/absent.jsonl"

if [[ -f "$TMP/out/noop.json" ]]; then
  printf '\033[1;32m[ OK ]\033[0m %-42s artifact written\n' "noop artifact"
else
  printf '\033[1;31m[FAIL]\033[0m %-42s no artifact\n' "noop artifact"
  FAILED=1
fi

[[ $FAILED == 0 ]] && echo "CLI smoke passed" || echo "CLI smoke FAILED"
exit $FAILED
