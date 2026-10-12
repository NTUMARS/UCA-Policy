#!/usr/bin/env bash
# GPU1 queue: mode-coverage eval, then dit1 MetaWorld (6 tasks).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPU="${GPU:-1}"
bash "${SCRIPT_DIR}/01_mode_coverage_eval.sh"
bash "${SCRIPT_DIR}/02_dit1_metaworld.sh"
