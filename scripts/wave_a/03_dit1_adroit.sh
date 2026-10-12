#!/usr/bin/env bash
# Wave A part 3: dit1 on Adroit pen → door, seed0, 3005 epochs, eval every 200.
# Exits when done so the tmux session is destroyed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

GPU="${GPU:-2}"
SEED="${SEED:-0}"
TASKS="${TASKS:-adroit_pen adroit_door}"
LOGFILE="${LOG_DIR}/dit1_adroit_gpu${GPU}.log"
: > "$LOGFILE"

echo "[$(date)] dit1-Adroit start GPU=${GPU} seed=${SEED}" | tee -a "$LOGFILE"

for task in $TASKS; do
  echo "[$(date)] ▶ dit1 ${task} seed=${SEED} epochs=3005" | tee -a "$LOGFILE"
  cd "${ROOT_DIR}/UCA"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  python train.py --config-name=dit1.yaml \
    task="${task}" \
    hydra.run.dir="data/outputs/${task}-dit1-${TAG}_seed${SEED}" \
    training.debug=False \
    training.seed="${SEED}" \
    training.device=cuda:0 \
    training.num_epochs=3005 \
    training.rollout_every=200 \
    training.checkpoint_every=200 \
    exp_name="${task}-dit1-${TAG}" \
    logging.mode=offline \
    checkpoint.save_ckpt=True \
    2>&1 | tee -a "$LOGFILE"
  cd "$ROOT_DIR"
  echo "[$(date)] ✓ dit1 ${task} seed=${SEED}" | tee -a "$LOGFILE"
done

echo "[$(date)] dit1-Adroit ALL DONE" | tee -a "$LOGFILE"
