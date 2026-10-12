#!/usr/bin/env bash
# Shared env for Wave A jobs (sourced by other scripts).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate ICT

export MUJOCO_GL=egl
export MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DISABLE_CODE=true

TMUX_SOCK="${TMUX_SOCK:-/tmp/tmux-1011/default}"
TAG="${TAG:-waveA0809}"
LOG_DIR="${ROOT_DIR}/logs/wave_a_${TAG}"
mkdir -p "$LOG_DIR"
