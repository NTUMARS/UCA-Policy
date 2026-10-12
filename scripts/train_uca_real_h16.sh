#!/usr/bin/env bash
# Train uca_real (horizon=16, n_action_steps=15, 301 epochs) on new real tasks.
# Phase 1: pick rebuttal raw + aligned
# Phase 2: stack qube raw + aligned
#
# Usage:
#   bash scripts/train_uca_real_h16.sh [SEED] [GPU_A] [GPU_B]
# Example (idle GPUs):
#   bash scripts/train_uca_real_h16.sh 0 0 1

set -euo pipefail

SEED=${1:-0}
GPU_A=${2:-0}
GPU_B=${3:-1}
TAG="h16a15_301"
CONFIG=uca_real

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$HOME/miniforge3/envs/ICT/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate ICT
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1
mkdir -p logs

run_pair() {
  local t0=$1
  local t1=$2
  local g0=$3
  local g1=$4
  local log0="logs/${CONFIG}_${t0}_${TAG}_seed${SEED}.log"
  local log1="logs/${CONFIG}_${t1}_${TAG}_seed${SEED}.log"
  echo "▶ phase: ${t0} (gpu ${g0}) + ${t1} (gpu ${g1})"
  bash scripts/train_policy.sh "$CONFIG" "$t0" "$TAG" "$SEED" "$g0" >"$log0" 2>&1 &
  local p0=$!
  bash scripts/train_policy.sh "$CONFIG" "$t1" "$TAG" "$SEED" "$g1" >"$log1" 2>&1 &
  local p1=$!
  echo "PIDs: ${p0} ${p1}  logs: ${log0} , ${log1}"
  local ec=0
  wait "$p0" || ec=1
  wait "$p1" || ec=1
  return "$ec"
}

pack_tasks() {
  local -a tasks=("$@")
  local pack_dir="UCA/data/outputs/${CONFIG}_${TAG}_seed${SEED}_ckpts"
  local pack_zip="UCA/data/outputs/${CONFIG}_${TAG}_seed${SEED}_ckpts.tar.gz"
  rm -rf "$pack_dir"
  mkdir -p "$pack_dir"
  for task in "${tasks[@]}"; do
    local src="UCA/data/outputs/${task}-${CONFIG}-${TAG}_seed${SEED}/checkpoints"
    local dst="${pack_dir}/${task}"
    mkdir -p "$dst"
    if [[ -f "${src}/epoch-0300.ckpt" ]]; then
      cp -f "${src}/epoch-0300.ckpt" "${dst}/epoch-0300.ckpt"
    elif [[ -f "${src}/latest.ckpt" ]]; then
      cp -f "${src}/latest.ckpt" "${dst}/epoch-0300.ckpt"
    fi
    local best
    best=$(find "$src" -maxdepth 1 -type f -name 'epoch=*-test_mean_score=*.ckpt' 2>/dev/null | sort | tail -n 1 || true)
    if [[ -n "${best}" ]]; then
      cp -f "$best" "${dst}/$(basename "$best")"
      cp -f "$best" "${dst}/best_loss.ckpt"
    fi
  done
  {
    echo "config=${CONFIG} tag=${TAG} seed=${SEED}"
    echo "horizon=16 n_action_steps=15 num_epochs=301"
    echo "tasks: ${tasks[*]}"
  } > "${pack_dir}/README.txt"
  tar -czf "$pack_zip" -C "$(dirname "$pack_dir")" "$(basename "$pack_dir")"
  echo "✅ packed -> ${pack_zip}"
}

# Phase 1: pick
run_pair real_pick_rebuttal real_pick_rebuttal_aligned "$GPU_A" "$GPU_B"

# Phase 2: stack
run_pair real_stack_qube real_stack_qube_aligned "$GPU_A" "$GPU_B"

pack_tasks \
  real_pick_rebuttal \
  real_pick_rebuttal_aligned \
  real_stack_qube \
  real_stack_qube_aligned
