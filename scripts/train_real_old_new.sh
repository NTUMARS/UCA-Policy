#!/usr/bin/env bash
# Train real-robot policies on old-frame vs new-frame UR5 datasets.
#
# Datasets (under UCA/data/):
#   old:  ur5_pick_old_dataset.zarr  / ur5_stack_old_dataset.zarr
#   new:  ur5_pick_rebuttal_dataset.zarr / ur5_stack_qube_dataset.zarr
#
# Task yamls:
#   old:  real_pick_old, real_stack_old  (aliases: real_pick, real_stack)
#   new:  real_pick_rebuttal, real_stack_qube
#
# Usage:
#   bash scripts/train_real_old_new.sh <GPU_ID> <CONFIG> <VERSION> [TASKS] [SEED]
#
#   VERSION: old | new | both
#   CONFIG : policy config stem, e.g. uca / dp3 / act / dit / mp1 / pi0 ...
#   TASKS  : optional comma list: pick,stack  (default: pick,stack)
#
# Examples:
#   bash scripts/train_real_old_new.sh 0 uca old
#   bash scripts/train_real_old_new.sh 1 dp3 new pick
#   bash scripts/train_real_old_new.sh 2 act both pick,stack 0

set -euo pipefail

if (( $# < 3 )); then
  cat <<'EOF'
use:
  bash scripts/train_real_old_new.sh <GPU_ID> <CONFIG> <VERSION> [TASKS] [SEED]

VERSION: old | new | both
TASKS  : pick,stack  (default pick,stack)
EOF
  exit 1
fi

GPU_ID=$1
CONFIG=$2
VERSION=$3
TASKS_ARG=${4:-pick,stack}
SEED=${5:-0}

IFS=',' read -ra TASK_KINDS <<< "$TASKS_ARG"

resolve_task() {
  local kind=$1
  local ver=$2
  case "${kind}_${ver}" in
    pick_old) echo real_pick_old ;;
    pick_new) echo real_pick_rebuttal ;;
    stack_old) echo real_stack_old ;;
    stack_new) echo real_stack_qube ;;
    *)
      echo "unknown task/version: kind=${kind} ver=${ver}" >&2
      exit 1
      ;;
  esac
}

VERSIONS=()
case "$VERSION" in
  old) VERSIONS=(old) ;;
  new) VERSIONS=(new) ;;
  both) VERSIONS=(old new) ;;
  *)
    echo "VERSION must be old|new|both, got: $VERSION" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for ver in "${VERSIONS[@]}"; do
  for kind in "${TASK_KINDS[@]}"; do
    kind=$(echo "$kind" | xargs)
    task=$(resolve_task "$kind" "$ver")
    tag="real${ver}"
    echo "▶ GPU=$GPU_ID | config=$CONFIG | task=$task | seed=$SEED | tag=$tag"
    bash scripts/train_policy.sh "$CONFIG" "$task" "$tag" "$SEED" "$GPU_ID"
  done
done
