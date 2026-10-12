#!/bin/bash
# Generate MetaWorld Push left/right bimodal expert demos.
# Usage:
#   bash scripts/gen_bimodal_push.sh
#   bash scripts/gen_bimodal_push.sh 40          # episodes per mode
#   bash scripts/gen_bimodal_push.sh 40 0.10     # + lateral offset (m)
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_PATH="${MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/third_party/Metaworld"

NUM_PER_MODE="${1:-40}"
LATERAL="${2:-0.10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python gen_bimodal_push_expert.py \
  --num_episodes_per_mode "${NUM_PER_MODE}" \
  --lateral "${LATERAL}" \
  --root_dir "${ROOT_DIR}/UCA/data/" \
  --device "cuda:0" \
  --overwrite
#!/bin/bash
# Generate MetaWorld Push left/right bimodal expert demos.
# Usage:
#   bash scripts/gen_bimodal_push.sh
#   bash scripts/gen_bimodal_push.sh 40          # episodes per mode
#   bash scripts/gen_bimodal_push.sh 40 0.10     # + lateral offset (m)
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_PATH="${MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/third_party/Metaworld"

NUM_PER_MODE="${1:-40}"
LATERAL="${2:-0.10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python gen_bimodal_push_expert.py \
  --num_episodes_per_mode "${NUM_PER_MODE}" \
  --lateral "${LATERAL}" \
  --root_dir "${ROOT_DIR}/UCA/data/" \
  --device "cuda:0" \
  --overwrite
