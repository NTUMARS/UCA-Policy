# bash scripts/gen_demonstration_adroit.sh door
# bash scripts/gen_demonstration_adroit.sh hammer
# bash scripts/gen_demonstration_adroit.sh pen
export MUJOCO_GL=egl
export MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$CONDA_PREFIX/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH

cd third_party/VRL3/src

task=${1}

CUDA_VISIBLE_DEVICES=0 python gen_demonstration_expert.py --env_name $task \
                        --num_episodes 10 \
                        --root_dir "../../../UCA/data/" \
                        --expert_ckpt_path "../ckpts/vrl3_${task}.pt" \
                        --img_size 84 \
                        --not_use_multi_view \
                        --use_point_crop
