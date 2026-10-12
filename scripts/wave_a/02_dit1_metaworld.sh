#!/usr/bin/env bash
# Wave A part 2: dit1 on 6 MetaWorld tasks, seed0, 1005 epochs, eval every 200.
# Exits when done so the tmux session is destroyed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

GPU="${GPU:-1}"
SEED="${SEED:-0}"
# Optional space-separated subset, e.g. TASKS="metaworld_dial-turn metaworld_lever-pull"
TASKS="${TASKS:-metaworld_dial-turn metaworld_peg-insert-side metaworld_lever-pull metaworld_pick-place-wall metaworld_stick-pull metaworld_shelf-place}"
LOGFILE="${LOG_DIR}/dit1_mw_gpu${GPU}.log"
: > "$LOGFILE"

echo "[$(date)] dit1-MW start GPU=${GPU} seed=${SEED}" | tee -a "$LOGFILE"

for task in $TASKS; do
  echo "[$(date)] ▶ dit1 ${task} seed=${SEED}" | tee -a "$LOGFILE"
  cd "${ROOT_DIR}/UCA"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  python train.py --config-name=dit1.yaml \
    task="${task}" \
    hydra.run.dir="data/outputs/${task}-dit1-${TAG}_seed${SEED}" \
    training.debug=False \
    training.seed="${SEED}" \
    training.device=cuda:0 \
    training.num_epochs=1005 \
    training.rollout_every=200 \
    training.checkpoint_every=200 \
    exp_name="${task}-dit1-${TAG}" \
    logging.mode=offline \
    checkpoint.save_ckpt=True \
    2>&1 | tee -a "$LOGFILE"
  cd "$ROOT_DIR"
  echo "[$(date)] ✓ dit1 ${task} seed=${SEED}" | tee -a "$LOGFILE"
done

echo "[$(date)] dit1-MW ALL DONE" | tee -a "$LOGFILE"
