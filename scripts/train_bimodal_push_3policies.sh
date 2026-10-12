#!/usr/bin/env bash
# Train ACT / DP3 / UCA on metaworld_push_bimodal, seeds 0,1,2.
# Each policy runs on a dedicated GPU; seeds run sequentially on that GPU.
# Uses wandb offline to avoid proxy hangs.
#
# Usage:
#   bash scripts/train_bimodal_push_3policies.sh
#   ACT_GPU=1 DP3_GPU=2 UCA_GPU=3 bash scripts/train_bimodal_push_3policies.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ACT_GPU="${ACT_GPU:-1}"
DP3_GPU="${DP3_GPU:-2}"
UCA_GPU="${UCA_GPU:-3}"
TASK="metaworld_push_bimodal"
TAG="${TAG:-bimodal0807}"
LOG_DIR="${ROOT_DIR}/logs/bimodal_push_${TAG}"
mkdir -p "$LOG_DIR"

write_runner() {
  local alg="$1"
  local gpu="$2"
  local logfile="${LOG_DIR}/${alg}_gpu${gpu}.log"
  local runner="${LOG_DIR}/run_${alg}.sh"

  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "\$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ICT
export MUJOCO_GL=egl
export MUJOCO_PATH="\$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}:\$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DISABLE_CODE=true
cd "$ROOT_DIR"
echo "[\$(date)] start ${alg} on GPU ${gpu} (wandb offline)" | tee -a "$logfile"
for seed in 0 1 2; do
  echo "[\$(date)] ▶ ${alg} seed=\${seed}" | tee -a "$logfile"
  cd "$ROOT_DIR/UCA"
  export CUDA_VISIBLE_DEVICES=${gpu}
  python train.py --config-name=${alg}.yaml \\
    task=${TASK} \\
    hydra.run.dir=data/outputs/${TASK}-${alg}-${TAG}_seed\${seed} \\
    training.debug=False \\
    training.seed=\${seed} \\
    training.device=cuda:0 \\
    exp_name=${TASK}-${alg}-${TAG} \\
    logging.mode=offline \\
    checkpoint.save_ckpt=True \\
    2>&1 | tee -a "$logfile"
  cd "$ROOT_DIR"
  echo "[\$(date)] ✓ ${alg} seed=\${seed} done" | tee -a "$logfile"
done
echo "[\$(date)] all seeds finished for ${alg}" | tee -a "$logfile"
EOF
  chmod +x "$runner"
  : > "$logfile"
}

# Prefer the GPU-capable tmux server if present (Cursor sandbox often lacks /dev/nvidia*).
TMUX_SOCK="${TMUX_SOCK:-/tmp/tmux-1011/default}"
USE_TMUX=0
if [[ -S "$TMUX_SOCK" ]]; then
  USE_TMUX=1
fi

launch_one() {
  local alg="$1"
  local gpu="$2"
  write_runner "$alg" "$gpu"
  local runner="${LOG_DIR}/run_${alg}.sh"
  local session="train_${alg}_bimodal"
  if [[ "$USE_TMUX" == "1" ]]; then
    if tmux -S "$TMUX_SOCK" has-session -t "$session" 2>/dev/null; then
      tmux -S "$TMUX_SOCK" kill-session -t "$session"
    fi
    tmux -S "$TMUX_SOCK" new-session -d -s "$session" "$runner"
    echo "Started tmux -S $TMUX_SOCK [$session] GPU=${gpu}"
  else
    nohup "$runner" > "${LOG_DIR}/${alg}_gpu${gpu}.log.nohup" 2>&1 &
    echo "Started nohup ${alg} pid=$! GPU=${gpu}"
  fi
}

launch_one act "$ACT_GPU"
launch_one dp3 "$DP3_GPU"
launch_one uca "$UCA_GPU"

echo
echo "Monitor:"
echo "  tmux -S ${TMUX_SOCK} attach -t train_act_bimodal"
echo "  tmux -S ${TMUX_SOCK} attach -t train_dp3_bimodal"
echo "  tmux -S ${TMUX_SOCK} attach -t train_uca_bimodal"
echo "  tail -f ${LOG_DIR}/*.log"
echo "  nvidia-smi"
echo "Outputs under: ${ROOT_DIR}/UCA/data/outputs/${TASK}-*-${TAG}_seed*"