#!/usr/bin/env bash
# Train ucaflow on old vs new real pick datasets (301 epochs), then pack weights.
#
# Keeps:
#   - lowest-loss topk ckpt  (test_mean_score = -train_loss on real tasks)
#   - epoch-300 weights      (saved as latest.ckpt, copied to epoch-0300.ckpt)
# Logs to wandb project "ucaflow" (online).
#
# Usage:
#   bash scripts/train_ucaflow_real_old_new.sh [GPU_OLD] [GPU_NEW] [SEED]
# Examples:
#   bash scripts/train_ucaflow_real_old_new.sh 1 2 0

set -euo pipefail

GPU_OLD=${1:-1}
GPU_NEW=${2:-2}
SEED=${3:-0}
TAG="rebuttal301"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prefer ICT env if present
if [[ -x "$HOME/miniforge3/envs/ICT/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate ICT
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1

OLD_TASK=real_pick_old
NEW_TASK=real_pick_rebuttal
OLD_DIR="UCA/data/outputs/${OLD_TASK}-ucaflow-${TAG}_seed${SEED}"
NEW_DIR="UCA/data/outputs/${NEW_TASK}-ucaflow-${TAG}_seed${SEED}"
PACK_DIR="UCA/data/outputs/ucaflow_real_old_new_${TAG}_seed${SEED}_ckpts"
PACK_ZIP="UCA/data/outputs/ucaflow_real_old_new_${TAG}_seed${SEED}_ckpts.tar.gz"

echo "▶ old: task=${OLD_TASK} gpu=${GPU_OLD} -> ${OLD_DIR}"
echo "▶ new: task=${NEW_TASK} gpu=${GPU_NEW} -> ${NEW_DIR}"

bash scripts/train_policy.sh ucaflow "$OLD_TASK" "$TAG" "$SEED" "$GPU_OLD" \
  > "logs_ucaflow_${OLD_TASK}_${TAG}.log" 2>&1 &
PID_OLD=$!

bash scripts/train_policy.sh ucaflow "$NEW_TASK" "$TAG" "$SEED" "$GPU_NEW" \
  > "logs_ucaflow_${NEW_TASK}_${TAG}.log" 2>&1 &
PID_NEW=$!

echo "PIDs: old=${PID_OLD} new=${PID_NEW}"
echo "logs: logs_ucaflow_${OLD_TASK}_${TAG}.log , logs_ucaflow_${NEW_TASK}_${TAG}.log"

ec=0
wait "$PID_OLD" || ec=1
wait "$PID_NEW" || ec=1
if (( ec != 0 )); then
  echo "❌ one or both trainings failed (exit aggregate=${ec})"
  exit "$ec"
fi

echo "▶ packaging checkpoints"
rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR/old_pick" "$PACK_DIR/new_pick"

pack_one() {
  local src=$1
  local dst=$2
  local label=$3
  local ckpt_dir="${src}/checkpoints"
  if [[ ! -d "$ckpt_dir" ]]; then
    echo "missing checkpoints: $ckpt_dir" >&2
    return 1
  fi
  # epoch-300 (named file from save_epoch_ckpts, else latest)
  if [[ -f "${ckpt_dir}/epoch-0300.ckpt" ]]; then
    cp -f "${ckpt_dir}/epoch-0300.ckpt" "${dst}/epoch-0300.ckpt"
  elif [[ -f "${ckpt_dir}/latest.ckpt" ]]; then
    cp -f "${ckpt_dir}/latest.ckpt" "${dst}/epoch-0300.ckpt"
  fi
  # lowest-loss topk (exclude latest / named epoch snapshots)
  local best
  best=$(find "$ckpt_dir" -maxdepth 1 -type f -name 'epoch=*-test_mean_score=*.ckpt' | sort | tail -n 1 || true)
  if [[ -n "${best}" ]]; then
    cp -f "$best" "${dst}/$(basename "$best")"
    cp -f "$best" "${dst}/best_loss.ckpt"
  fi
  ls -lh "$dst"
  echo "packed ${label}: ${dst}"
}

pack_one "$OLD_DIR" "$PACK_DIR/old_pick" "old_pick"
pack_one "$NEW_DIR" "$PACK_DIR/new_pick" "new_pick"

# brief manifest
{
  echo "ucaflow real old/new pick checkpoints"
  echo "tag=${TAG} seed=${SEED}"
  echo "old_run=${OLD_DIR}"
  echo "new_run=${NEW_DIR}"
  echo "note: best_loss.ckpt = lowest train loss (via max test_mean_score=-train_loss)"
  echo "note: epoch-0300.ckpt = weights at epoch index 300"
  echo "note: do NOT mix old/new frames at deploy time"
} > "$PACK_DIR/README.txt"

tar -czf "$PACK_ZIP" -C "$(dirname "$PACK_DIR")" "$(basename "$PACK_DIR")"
echo "✅ packed -> ${PACK_ZIP}"
ls -lh "$PACK_ZIP"
