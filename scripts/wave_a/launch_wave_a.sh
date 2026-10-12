#!/usr/bin/env bash
# Launch Wave A in tmux sessions that exit (and are destroyed) when jobs finish.
# GPU plan (GPU0 occupied by other users):
#   GPU1: mode-eval → dit1 × 6 MW seed0
#   GPU2: dit1 Adroit pen→door 3005
#   GPU3: UCA ablation B1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Prefer a dedicated socket; recreate if stale.
TMUX_SOCK="${TMUX_SOCK:-/tmp/tmux-waveA-s125mdg55_01}"
TAG="${TAG:-waveA0809}"
LOG_DIR="${ROOT_DIR}/logs/wave_a_${TAG}"
mkdir -p "$LOG_DIR"
chmod +x "${SCRIPT_DIR}"/*.sh

# Drop stale socket
if [[ -S "$TMUX_SOCK" ]] && ! tmux -S "$TMUX_SOCK" list-sessions >/dev/null 2>&1; then
  rm -f "$TMUX_SOCK"
fi

launch() {
  local session="$1"
  local script="$2"
  local gpu="$3"
  if tmux -S "$TMUX_SOCK" has-session -t "$session" 2>/dev/null; then
    tmux -S "$TMUX_SOCK" kill-session -t "$session"
  fi
  # Explicit /bin/bash -c avoids "fork failed: No such file or directory"
  # Session dies when the script exits (no lingering shell).
  tmux -S "$TMUX_SOCK" new-session -d -s "$session" /bin/bash -c \
    "export GPU=${gpu} TAG=${TAG} TMUX_SOCK=${TMUX_SOCK}; bash ${script}; echo DONE_\$(date); exit 0"
  echo "Started tmux -S ${TMUX_SOCK} [${session}] GPU=${gpu} → ${script}"
}

launch waveA_gpu1 "${SCRIPT_DIR}/00_gpu1_mode_then_dit1_mw.sh" 1
launch waveA_gpu2 "${SCRIPT_DIR}/03_dit1_adroit.sh" 2
launch waveA_gpu3 "${SCRIPT_DIR}/04_uca_ablation.sh" 3

chmod 777 "$TMUX_SOCK" 2>/dev/null || true
sleep 3
echo
echo "=== sessions ==="
tmux -S "$TMUX_SOCK" list-sessions
echo
echo "Monitor:"
echo "  tmux -S ${TMUX_SOCK} attach -t waveA_gpu1"
echo "  tmux -S ${TMUX_SOCK} attach -t waveA_gpu2"
echo "  tmux -S ${TMUX_SOCK} attach -t waveA_gpu3"
echo "  tail -f ${LOG_DIR}/*.log"
echo "  nvidia-smi"
echo
echo "Outputs: ${ROOT_DIR}/UCA/data/outputs/*-${TAG}*"
echo "Mode eval: ${LOG_DIR}/mode_eval/"
