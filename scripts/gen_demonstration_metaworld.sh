# bash scripts/gen_demonstration_metaworld.sh assembly
# bash scripts/gen_demonstration_metaworld.sh disassemble
# bash scripts/gen_demonstration_metaworld.sh shelf-place   
# bash scripts/gen_demonstration_metaworld.sh bin-picking
# bash scripts/gen_demonstration_metaworld.sh basketball
export MUJOCO_GL=egl
export MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$CONDA_PREFIX/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH



cd third_party/Metaworld

task_name=${1}

export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 20 \
            --root_dir "../../UCA/data/" \
