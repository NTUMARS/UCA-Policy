#!/usr/bin/env bash
# Wave A part 4: UCA hyperparam ablation Stage B1 (seed0).
# Configs:
#   n_obs_steps in {1,2,3} with t=1,h=1
#   (t,h) in {(2,1),(1,2),(2,2)} with n_obs_steps=2
# Tasks: stick-pull, shelf-place (1005), adroit_pen (3005)
# Exits when done so the tmux session is destroyed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

GPU="${GPU:-3}"
SEED="${SEED:-0}"
LOGFILE="${LOG_DIR}/uca_ablation_gpu${GPU}.log"
: > "$LOGFILE"

CONFIGS=(
  "obs1_t1_h1|1|1|1"
  "obs2_t1_h1|2|1|1"
  "obs3_t1_h1|3|1|1"
  "obs2_t2_h1|2|2|1"
  "obs2_t1_h2|2|1|2"
  "obs2_t2_h2|2|2|2"
)

MW_TASKS=(metaworld_stick-pull metaworld_shelf-place)
ADROIT_TASKS=(adroit_pen)

run_one() {
  local task="$1"
  local cfg_name="$2"
  local n_obs="$3"
  local nt="$4"
  local nh="$5"
  local epochs="$6"
  local run_tag="${TAG}_${cfg_name}"

  echo "[$(date)] ▶ uca ${task} ${cfg_name} seed=${SEED} ep=${epochs}" | tee -a "$LOGFILE"
  cd "${ROOT_DIR}/UCA"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  python train.py --config-name=uca.yaml \
    task="${task}" \
    hydra.run.dir="data/outputs/${task}-uca-${run_tag}_seed${SEED}" \
    training.debug=False \
    training.seed="${SEED}" \
    training.device=cuda:0 \
    training.num_epochs="${epochs}" \
    training.rollout_every=200 \
    training.checkpoint_every=200 \
    n_obs_steps="${n_obs}" \
    policy.n_obs_steps="${n_obs}" \
    policy.num_t_tokens="${nt}" \
    policy.num_h_tokens="${nh}" \
    exp_name="${task}-uca-${run_tag}" \
    logging.mode=offline \
    checkpoint.save_ckpt=True \
    2>&1 | tee -a "$LOGFILE"
  cd "$ROOT_DIR"
  echo "[$(date)] ✓ uca ${task} ${cfg_name}" | tee -a "$LOGFILE"
}

echo "[$(date)] UCA ablation start GPU=${GPU}" | tee -a "$LOGFILE"

for task in "${MW_TASKS[@]}"; do
  for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name n_obs nt nh <<<"$entry"
    run_one "$task" "$name" "$n_obs" "$nt" "$nh" 1005
  done
done

for task in "${ADROIT_TASKS[@]}"; do
  for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name n_obs nt nh <<<"$entry"
    run_one "$task" "$name" "$n_obs" "$nt" "$nh" 3005
  done
done

echo "[$(date)] UCA ablation ALL DONE" | tee -a "$LOGFILE"
