#!/usr/bin/env bash
# Wave A part 1: fixed-init mode coverage on push_bimodal (UCA / DP3 / ACT h32).
# Exits when done so the tmux session is destroyed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

GPU="${GPU:-1}"
N_INITS="${N_INITS:-10}"
K_SAMPLES="${K_SAMPLES:-20}"
SEEDS="${SEEDS:-0 1 2}"
LOGFILE="${LOG_DIR}/mode_coverage_gpu${GPU}.log"
: > "$LOGFILE"
export LOG_DIR

echo "[$(date)] mode-coverage start GPU=${GPU}" | tee -a "$LOGFILE"
cd "$ROOT_DIR"

for alg in uca dp3 act; do
  for seed in $SEEDS; do
    echo "[$(date)] ▶ ${alg} seed=${seed}" | tee -a "$LOGFILE"
    CUDA_VISIBLE_DEVICES="${GPU}" python scripts/eval_bimodal_mode_coverage.py \
      --alg "${alg}" \
      --seed "${seed}" \
      --tag bimodal0807 \
      --n_inits "${N_INITS}" \
      --k_samples "${K_SAMPLES}" \
      --out_dir "${LOG_DIR}/mode_eval/${alg}_seed${seed}" \
      2>&1 | tee -a "$LOGFILE"
    echo "[$(date)] ✓ ${alg} seed=${seed}" | tee -a "$LOGFILE"
  done
done

python - <<'PY' 2>&1 | tee -a "$LOGFILE"
import json, pathlib, os
root = pathlib.Path(os.environ["LOG_DIR"]) / "mode_eval"
print("alg seed  SR   visitL_s visitR_s both_init_frac  ckpt")
for p in sorted(root.glob("*/summary.json")):
    s = json.loads(p.read_text())
    print(f"{s['alg']:4s} {s['train_seed']}  "
          f"{s['success_rate']*100:5.1f} "
          f"{s['visit_left_success']*100:6.1f} "
          f"{s['visit_right_success']*100:6.1f} "
          f"{s['frac_inits_both_modes']*100:8.1f}  "
          f"{pathlib.Path(s['ckpt']).name}")
PY

echo "[$(date)] mode-coverage ALL DONE" | tee -a "$LOGFILE"
