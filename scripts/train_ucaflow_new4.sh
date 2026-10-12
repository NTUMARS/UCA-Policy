#!/usr/bin/env bash
# Train ucaflow on 4 new-data tasks:
#   1) real_pick_rebuttal            (raw new frame)
#   2) real_pick_rebuttal_aligned    (aligned to old pick frame)
#   3) real_stack_qube               (raw new frame)
#   4) real_stack_qube_aligned       (aligned to old pick frame)
#
# Usage:
#   bash scripts/train_ucaflow_new4.sh [SEED] [GPU0] [GPU1] [GPU2] [GPU3]
# Example:
#   bash scripts/train_ucaflow_new4.sh 0 0 1 2 3

set -euo pipefail

SEED=${1:-0}
GPU0=${2:-0}
GPU1=${3:-1}
GPU2=${4:-2}
GPU3=${5:-3}
TAG="new4_301"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$HOME/miniforge3/envs/ICT/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate ICT
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1

TASKS=(
  real_pick_rebuttal
  real_pick_rebuttal_aligned
  real_stack_qube
  real_stack_qube_aligned
)
GPUS=("$GPU0" "$GPU1" "$GPU2" "$GPU3")

mkdir -p logs
PIDS=()
for i in "${!TASKS[@]}"; do
  task=${TASKS[$i]}
  gpu=${GPUS[$i]}
  log="logs/ucaflow_${task}_${TAG}_seed${SEED}.log"
  echo "▶ gpu=${gpu} task=${task} -> ${log}"
  bash scripts/train_policy.sh ucaflow "$task" "$TAG" "$SEED" "$gpu" >"$log" 2>&1 &
  PIDS+=($!)
done

echo "PIDs: ${PIDS[*]}"
ec=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || ec=1
done
if (( ec != 0 )); then
  echo "❌ one or more trainings failed"
  exit "$ec"
fi

PACK_DIR="UCA/data/outputs/ucaflow_${TAG}_seed${SEED}_ckpts"
PACK_ZIP="UCA/data/outputs/ucaflow_${TAG}_seed${SEED}_ckpts.tar.gz"
rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR"

for task in "${TASKS[@]}"; do
  src="UCA/data/outputs/${task}-ucaflow-${TAG}_seed${SEED}/checkpoints"
  dst="${PACK_DIR}/${task}"
  mkdir -p "$dst"
  if [[ -f "${src}/epoch-0300.ckpt" ]]; then
    cp -f "${src}/epoch-0300.ckpt" "${dst}/epoch-0300.ckpt"
  elif [[ -f "${src}/latest.ckpt" ]]; then
    cp -f "${src}/latest.ckpt" "${dst}/epoch-0300.ckpt"
  fi
  best=$(find "$src" -maxdepth 1 -type f -name 'epoch=*-test_mean_score=*.ckpt' 2>/dev/null | sort | tail -n 1 || true)
  if [[ -n "${best}" ]]; then
    cp -f "$best" "${dst}/$(basename "$best")"
    cp -f "$best" "${dst}/best_loss.ckpt"
  fi
  ls -lh "$dst" || true
done

{
  echo "ucaflow 4-task new-data package"
  echo "tag=${TAG} seed=${SEED}"
  echo "tasks: ${TASKS[*]}"
  echo "best_loss.ckpt = lowest train loss (test_mean_score=-train_loss)"
  echo "epoch-0300.ckpt = weights at epoch index 300"
  echo "aligned_* uses R_aligned = R_new @ R_delta vs old pick frame"
} > "$PACK_DIR/README.txt"

tar -czf "$PACK_ZIP" -C "$(dirname "$PACK_DIR")" "$(basename "$PACK_DIR")"
echo "✅ packed -> ${PACK_ZIP}"
ls -lh "$PACK_ZIP"
