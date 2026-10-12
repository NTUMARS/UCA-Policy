#!/bin/bash
# Examples:
#   bash scripts/deploy_policy.sh idp3 pick_and_place 0520

# 1. 确保传入了足够的参数
if [ "$#" -lt 3 ]; then
    echo -e "\033[31mError: Need 3 arguments. Usage: bash deploy_policy.sh <alg_name> <task_name> <addition_info>\033[0m"
    exit 1
fi

alg_name=${1}
task_name=${2}
addition_info=${3}

# 配置参数
config_name=${alg_name}
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

# [非常重要] 确保这个路径下有你训练好的模型权重！
echo -e "\033[32mTarget Run Dir (for loading weights): ${run_dir}\033[0m"

dataset_path=/home/zikun/czk/UCF_flow_real/UCF_flow/data/ur5_pick1_dataset.zarr

DEBUG=False
save_ckpt=False  # 部署阶段不需要再保存 ckpt
wandb_mode="disabled" # [修复] 禁用 wandb，防止未定义变量导致 Hydra 报错

gpu_id=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# 进入工作目录 (请确保 deploy.py, deployment.py, realsense.py 都在此目录或能被找到)
cd UCA || exit 1

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

# 运行部署脚本
python3 deployur5.py --config-name=${config_name}.yaml \
                 task=${task_name} \
                 hydra.run.dir=${run_dir} \
                 training.debug=$DEBUG \
                 training.seed=${seed} \
                 training.device="cuda:0" \
                 exp_name=${exp_name} \
                 logging.mode=${wandb_mode} \
                 checkpoint.save_ckpt=${save_ckpt} \
                 task.dataset.zarr_path=$dataset_path