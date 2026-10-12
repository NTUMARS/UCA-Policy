# deployur5：UR5 真机部署说明

本文说明 `deployur5.py` 的配置选择、整体工作流、代码调用关系与使用方法。

---

## 1. 需要使用哪些 YAML

部署时通过 Hydra 加载 **policy 配置** + **task 配置**。二者必须与训练时一致（尤其是 `horizon` / `n_action_steps` / 坐标系约定）。

### 1.1 Policy 配置（算法侧）

| 配置文件 | 用途 | 关键超参 |
|---|---|---|
| `uca/config/uca_real.yaml` | **推荐**：真机 UCA（长 horizon） | `horizon=16`, `n_obs_steps=2`, `n_action_steps=15`, `num_epochs=301` |
| `uca/config/uca.yaml` | 仿真默认 UCA（短 horizon） | `horizon=4`, `n_action_steps=3`（**不要**直接拿去部署 h16 真机权重） |
| `uca/config/ucaflow.yaml` | 另一套真机/实验配置 | 需与对应训练 run 一致 |

部署入口脚本 `scripts/deploy_policy.sh` 的第一个参数即 config 名（不含 `.yaml`），例如 `uca_real`。

### 1.2 Task 配置（任务 / 数据侧）

| Task YAML | 数据 | 坐标系 | 适用场景 |
|---|---|---|---|
| `uca/config/task/real_pick_rebuttal.yaml` | `data/ur5_pick_rebuttal_dataset.zarr` | 新采集 frame（与当前机器人 TCP 一致） | **当前 `deployur5.py` 直接读 TCP，优先用此权重** |
| `uca/config/task/real_pick_rebuttal_aligned.yaml` | `data/ur5_pick_rebuttal_aligned_dataset.zarr` | 已对齐到旧 pick 的 6D 旋转 | 仅评估/对比；部署需额外做姿态变换（当前脚本未做） |
| `uca/config/task/real_stack_qube.yaml` | `data/ur5_stack_qube_dataset.zarr` | 新采集 frame | 堆叠任务部署（同 pick，优先 raw） |
| `uca/config/task/real_stack_qube_aligned.yaml` | `data/ur5_stack_qube_aligned_dataset.zarr` | 对齐到旧 pick | 同上，部署需变换 |

Task YAML 中的关键字段：

- `shape_meta`：`point_cloud=[1024,3]`，`agent_pos/action=[10]`（xyz + 6D rot + gripper）
- `env_runner: null`（真机无仿真 runner）
- `dataset.zarr_path`：训练用；部署时脚本会覆盖/解析，但 **权重对应的 normalizer 来自 ckpt**，task 名仍要与训练一致

### 1.3 配置与权重如何对应

训练产物目录（由 `scripts/train_policy.sh` 决定）：

```text
UCA/data/outputs/<task>-<config>-<tag>_seed<seed>/
  checkpoints/
    latest.ckpt          # deploy 默认加载
    epoch-0300.ckpt      # 第 300 个 epoch
    epoch=XXXX-test_mean_score=YYYY.ckpt  # loss 最低（topk）
  wandb/
```

示例（当前真机 h16 训练）：

```text
real_pick_rebuttal-uca_real-h16a15_301_seed0/checkpoints/latest.ckpt
real_pick_rebuttal_aligned-uca_real-h16a15_301_seed0/checkpoints/...
```

**部署时 `hydra.run.dir` 必须指向上述 run 目录**，因为 `deployur5.py` 固定读取：

```text
${hydra.run.dir}/checkpoints/latest.ckpt
```

若要用 `best_loss` / `epoch-0300`，请先复制或软链为 `checkpoints/latest.ckpt`，或改 `deployur5.py` 中的路径。

---

## 2. 整体工作流程

```text
┌─────────────┐    zarr + yaml     ┌──────────────────┐
│ 真机采集数据 │ ───────────────► │ train.py (Hydra)  │
│ (zarr)      │                   │ + uca_real.yaml   │
└─────────────┘                   │ + task yaml       │
                                  └────────┬─────────┘
                                           │ 保存
                                           ▼
                                  checkpoints/*.ckpt
                                           │
                     scripts/deploy_policy.sh
                                           │
                                           ▼
┌─────────────┐   obs(10D+PC)     ┌──────────────────┐
│ UR5 + L515  │ ◄───────────────► │ deployur5.py     │
│ + Gripper   │   action(10D)×H   │ UR5EnvInference  │
└─────────────┘                   └──────────────────┘
```

分步说明：

1. **训练**：`scripts/train_policy.sh uca_real <task> <tag> <seed> <gpu>`  
   或 `scripts/train_uca_real_h16.sh`（pick 两版 → stack 两版）。
2. **选权重**：`latest.ckpt` / `epoch-0300.ckpt` / topk best-loss。
3. **部署**：`scripts/deploy_policy.sh uca_real <task> <tag>`  
   将 `run_dir` 指到对应训练输出，加载 ckpt，连接相机/机械臂，闭环推理执行。
4. **（可选）记录**：运行结束写入 `deploy_logs/demo_*.h5`（点云 / agent_pos / action）。

控制时序（`uca_real`）：

- 观测窗口：`n_obs_steps = 2`
- 预测长度：`horizon = 16`
- 每轮执行步数：代码中  
  `action_horizon = policy.horizon - policy.n_obs_steps + 1`  
  对 `uca_real` 即为 **15**，与 `n_action_steps` 一致
- 点云：每帧采样 **1024** 点（与训练一致）

---

## 3. 代码调用关系

### 3.1 入口与加载

```text
scripts/deploy_policy.sh
  └─ python deployur5.py --config-name=<alg>.yaml task=<task> hydra.run.dir=<run_dir> ...
        └─ hydra.main → main(cfg)
              ├─ TrainUCAWorkspace.create_from_checkpoint(run_dir/checkpoints/latest.ckpt)
              │     └─ 恢复 model / ema_model / normalizer / 训练 cfg
              ├─ policy = ema_model if use_ema else model
              └─ UR5EnvInference(...).reset() → 闭环 while
```

### 3.2 硬件与观测

```text
UR5EnvInference
  ├─ L515PointCloudCamera          # RealSense L515 → 过滤/裁剪 → (1024,3)
  ├─ URRobot
  │    ├─ rtde_control / rtde_receive   # 关节 / TCP / moveL
  │    └─ MyGripper (Modbus TCP)        # 夹爪 0~1 ↔ 0~255
  ├─ _get_robot_proprio()          # TCP xyz + 6D rot + gripper → 10D
  ├─ _get_obs_dict()               # 堆叠最近 n_obs_steps 帧
  ├─ execute_action(10D)           # 6D→Rodrigues → moveL + gripper.move
  └─ step(action_list)             # 执行 action_horizon 步，刷新 obs
```

### 3.3 单步推理循环

```text
obs_dict = env.reset()
while step_count < roll_out_length:
    result = policy.predict_action(obs_dict)
        # 内部: encode obs → 预测 horizon 步 → unnormalize
        # 返回 action (n_action_steps) 与 action_pred (horizon)
    action_list = result['action_pred'][0]   # 当前实现用 full pred
    obs_dict = env.step(action_list[:action_horizon])
    step_count += action_horizon
```

### 3.4 与训练代码的关系

| 模块 | 训练 | 部署 |
|---|---|---|
| `train.py` / `TrainUCAWorkspace` | 训练循环、存 ckpt | `create_from_checkpoint` 加载 |
| `uca/policy/ict1policy.py` | `compute_loss` | `predict_action` |
| `uca/config/*.yaml` | Hydra 构图 | 同名配置解析结构；权重内含 normalizer |
| `uca/dataset/real_ur5.py` | 读 zarr | 部署不直接读数据 |
| `deployur5.py` | — | 真机 I/O + 闭环 |

---

## 4. 使用方法

### 4.1 环境与硬件前提

- 在 **真机工控机** 上运行（需 USB 相机、机器人网段）。
- 依赖（示例）：`torch`, `hydra-core`, `omegaconf`, `pyrealsense2`, `pymodbus`, `ur-rtde`, `opencv-python`, `h5py`。
- 默认硬件地址（见 `deployur5.py`，可按现场修改）：
  - UR5：`192.168.1.2`
  - 夹爪 Modbus：`192.168.1.11:502`
  - L515 serial：`f1420840`
- 确认急停、工作空间安全；脚本 `reset()` 会提示确认后回 home。

### 4.2 推荐命令

在仓库根目录 `/path/to/UCA`：

```bash
# 用法: bash scripts/deploy_policy.sh <config> <task> <addition_info>
# run_dir = data/outputs/<task>-<config>-<addition_info>_seed0

# 例：部署 uca_real 在 pick rebuttal（新坐标系）上训好的权重
bash scripts/deploy_policy.sh uca_real real_pick_rebuttal h16a15_301
```

脚本内部等价于（核心逻辑）：

```bash
cd UCA
export CUDA_VISIBLE_DEVICES=0
python3 deployur5.py --config-name=uca_real.yaml \
  task=real_pick_rebuttal \
  hydra.run.dir=data/outputs/real_pick_rebuttal-uca_real-h16a15_301_seed0 \
  training.seed=0 \
  training.device=cuda:0 \
  logging.mode=disabled \
  checkpoint.save_ckpt=False
```

堆叠任务同理：

```bash
bash scripts/deploy_policy.sh uca_real real_stack_qube h16a15_301
```

### 4.3 指定 best / epoch-300 权重

```bash
RUN=data/outputs/real_pick_rebuttal-uca_real-h16a15_301_seed0
# 使用最低 loss
cp -f "$RUN/checkpoints"/epoch=*-test_mean_score=*.ckpt "$RUN/checkpoints/latest.ckpt"
# 或使用第 300 epoch
cp -f "$RUN/checkpoints/epoch-0300.ckpt" "$RUN/checkpoints/latest.ckpt"

bash scripts/deploy_policy.sh uca_real real_pick_rebuttal h16a15_301
```

### 4.4 运行时交互

1. 启动后加载 ckpt，初始化 L515 / UR5 / 夹爪。
2. 提示 `Move robot to home position? Press Enter...` → 确认后回 home。
3. 进入闭环：预测 → 执行 15 步 → 更新观测，直到约 900 步（可在 `deployur5.py` 改 `roll_out_length`）。
4. 若 `record_data=True`，结果写入 `UCA/deploy_logs/demo_<timestamp>.h5`。

### 4.5 修改 `deploy_policy.sh` 时注意

当前脚本里仍有历史路径：

```bash
dataset_path=/home/zikun/czk/UCF_flow_real/...
```

请改成你机器上真实 zarr 路径（与 task 一致即可），例如：

```bash
dataset_path=data/ur5_pick_rebuttal_dataset.zarr
```

否则 Hydra 解析 task.dataset 时可能报错（即使推理本身不读数据集）。

---

## 5. 坐标系与选型建议（重要）

| 训练数据 | 与当前机器人 TCP | 能否直接用现有 `deployur5.py` |
|---|---|---|
| `real_pick_rebuttal` / `real_stack_qube`（raw） | 一致 | **可以** |
| `*_aligned`（旋转对齐到旧 pick） | 6D 约定不同 | **不建议**直接部署；需在观测/动作上加 `R_delta` 变换 |
| 旧 `ur5_pick_dataset` 权重 | 与旧采集一致 | 仅当现场 TCP/工具坐标仍是旧约定时可用 |

对齐变换定义见：`scripts/align_new_real_to_old_pick.py`（`R_aligned = R_new @ R_delta`）。

---

## 6. 常见问题

1. **找不到 `checkpoints/latest.ckpt`**  
   检查 `addition_info` / `seed` 是否与训练输出目录名一致；确认权重文件存在。

2. **动作幅度异常 / 姿态翻转**  
   多半是用了 aligned 权重却未做姿态变换，或用了短 horizon 的 `uca.yaml` 去加载 `uca_real` 权重。

3. **点云为空或全零**  
   检查 L515 serial、深度流、桌面平面过滤参数（`L515PointCloudCamera` 内 `A,B,C,D` / 阈值）。

4. **夹爪不动**  
   检查 Modbus IP、以及 gripper 值是否在 `[0,1]`。

5. **执行频率**  
   `UR5EnvInference(frequency=30.0)` 控制每步 sleep；可按示教采集频率调整。

---

## 7. 相关文件索引

| 路径 | 说明 |
|---|---|
| `UCA/deployur5.py` | 真机部署主程序 |
| `scripts/deploy_policy.sh` | 部署启动脚本 |
| `scripts/train_policy.sh` | 通用训练入口 |
| `scripts/train_uca_real_h16.sh` | 真机 h16 四任务训练编排 |
| `uca/config/uca_real.yaml` | 真机 UCA 配置（16/15，301 epoch） |
| `uca/config/task/real_*.yaml` | 真机 task |
| `scripts/align_new_real_to_old_pick.py` | 新数据 → 旧 pick 坐标系对齐 |
