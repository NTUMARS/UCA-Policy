#!/usr/bin/env bash
# -------------------------------------------------------------------------
# run_train.sh  GPU_ID CONFIG  [TASKS]  [SEEDS]
#
# - GPU_ID  : necessary
# - CONFIG  : necessary
# - TASKS   : option
# - SEEDS   : option 0,1,2
# -------------------------------------------------------------------------
# 1. mujoco environment variables
export MUJOCO_GL=egl
export MUJOCO_PATH=$HOME/.mujoco/mujoco210

# 2. library path
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$CONDA_PREFIX/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH

set -euo pipefail      #

show_usage() {
  cat <<EOF
use:
  $0 <GPU_ID> <CONFIG> [TASKS] [SEEDS]

example:
  $0 0 baseline metaworld_door-close 42

  # explain:
  #   - run a task on GPU 0 
  #   - config file is baseline
  #   - random seed is 42
EOF
}

# ----------- parameters checking-----------------------------------------------------
if (( $# < 2 )); then
  echo "❌ parameters are not enough!"
  show_usage
  exit 1
fi

GPU_ID=$1
CONFIG=$2
TASKS_ARG=${3:-}
SEEDS_ARG=${4:-}

# ----------- default tasks and seeds ---------------------------------------------
TASKS_DEFAULT=(
  # adroit_pen
  # adroit_door
  # real_close
  # real_stack
  metaworld_dial-turn
  metaworld_peg-insert-side
  metaworld_lever-pull
  metaworld_pick-place-wall
  metaworld_stick-pull
  metaworld_shelf-place
  # metaworld_push
  # metaworld_pick-place
  # metaworld_soccer
)
SEEDS_DEFAULT=(0)

# ---------- tasks ------------------------------------------------------
if [[ -n $TASKS_ARG ]]; then
  IFS=',' read -ra TASKS <<< "$TASKS_ARG"   #
else
  TASKS=("${TASKS_DEFAULT[@]}")             #
fi

# ---------- seeds ------------------------------------------------------
if [[ -n $SEEDS_ARG ]]; then
  IFS=',' read -ra SEEDS <<< "$SEEDS_ARG"
else
  SEEDS=("${SEEDS_DEFAULT[@]}")             #
fi

# ----------- train loop -----------------------------------------------------
for seed in "${SEEDS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "▶ GPU=$GPU_ID | config=$CONFIG | task=$task | seed=$seed"
    bash scripts/train_policy.sh "$CONFIG" "$task" 0001 "$seed" "$GPU_ID"
  done
done
#!/usr/bin/env bash
# -------------------------------------------------------------------------
# run_train.sh  GPU_ID CONFIG  [TASKS]  [SEEDS]
#
# - GPU_ID  : necessary
# - CONFIG  : necessary
# - TASKS   : option
# - SEEDS   : option 0,1,2
# -------------------------------------------------------------------------
# 1. mujoco environment variables
export MUJOCO_GL=egl
export MUJOCO_PATH=$HOME/.mujoco/mujoco210

# 2. library path
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$CONDA_PREFIX/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH

set -euo pipefail      #

show_usage() {
  cat <<EOF
use:
  $0 <GPU_ID> <CONFIG> [TASKS] [SEEDS]

example:
  $0 0 baseline metaworld_door-close 42

  # explain:
  #   - run a task on GPU 0 
  #   - config file is baseline
  #   - random seed is 42
EOF
}

# ----------- parameters checking-----------------------------------------------------
if (( $# < 2 )); then
  echo "❌ parameters are not enough!"
  show_usage
  exit 1
fi

GPU_ID=$1
CONFIG=$2
TASKS_ARG=${3:-}
SEEDS_ARG=${4:-}

# ----------- default tasks and seeds ---------------------------------------------
TASKS_DEFAULT=(
  # adroit_pen
  # adroit_door
  # real_close
  # real_stack
  metaworld_dial-turn
  metaworld_peg-insert-side
  metaworld_lever-pull
  metaworld_pick-place-wall
  metaworld_stick-pull
  metaworld_shelf-place
  # metaworld_push
  # metaworld_pick-place
  # metaworld_soccer
)
SEEDS_DEFAULT=(0)

# ---------- tasks ------------------------------------------------------
if [[ -n $TASKS_ARG ]]; then
  IFS=',' read -ra TASKS <<< "$TASKS_ARG"   #
else
  TASKS=("${TASKS_DEFAULT[@]}")             #
fi

# ---------- seeds ------------------------------------------------------
if [[ -n $SEEDS_ARG ]]; then
  IFS=',' read -ra SEEDS <<< "$SEEDS_ARG"
else
  SEEDS=("${SEEDS_DEFAULT[@]}")             #
fi

# ----------- train loop -----------------------------------------------------
for seed in "${SEEDS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "▶ GPU=$GPU_ID | config=$CONFIG | task=$task | seed=$seed"
    bash scripts/train_policy.sh "$CONFIG" "$task" 0001 "$seed" "$GPU_ID"
  done
done
