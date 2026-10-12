import sys
import os
import time
import pathlib
import cv2
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from termcolor import cprint

# UCA 相关导入
# 假设 train.py 在根目录或者 uca 目录下，请根据实际路径调整导入
from train import TrainUCAWorkspace

# UR5 相关导入
import pyrealsense2 as rs
from pymodbus.client import ModbusTcpClient
import threading
from typing import Dict

# System configs
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
os.environ['WANDB_SILENT'] = "True"
OmegaConf.register_new_resolver("eval", eval, replace=True)

class Logger:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'

    @staticmethod
    def _timestamp():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def info(cls, msg):
        print(f"{cls.BLUE}[INFO]{cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()

    @classmethod
    def success(cls, msg):
        print(f"{cls.GREEN}[OK]  {cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()

    @classmethod
    def warn(cls, msg):
        print(f"{cls.YELLOW}[WARN]{cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()

    @classmethod
    def error(cls, msg):
        print(f"{cls.RED}[ERR] {cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()

log = Logger()

class L515PointCloudCamera:
    def __init__(self, serial="f1420840", width=640, height=480, fps=30, filter_magnitude=3, target_points=1024):
        self.serial = serial
        self.width = width
        self.height = height
        self.target_points = target_points
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, filter_magnitude)
        
        # 平面过滤参数
        self.A, self.B, self.C, self.D = 0.03, 0.97, 0.25, -0.50
        self.TABLE_THRESHOLD = 0.015
        
        self.pc = rs.pointcloud()
        self.last_pc_cache = None 

        try:
            self.profile = self.pipeline.start(config)
            log.success(f"L515 PointCloud Camera Started: {serial}")
            log.info("Warming up L515...")
            time.sleep(3.0)
            for _ in range(20):
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=8000)
                    if frames.get_depth_frame():
                        break
                except:
                    pass
            log.success("L515 ready for point cloud collection.")
        except Exception as e:
            log.error(f"Failed to start L515: {e}")
            self.pipeline = None

    def read(self):
        if not self.pipeline:
            return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                if self.last_pc_cache is not None:
                    return self.last_pc_cache.copy()
                return None

            depth_frame = self.decimation.process(depth_frame)
            points = self.pc.calculate(depth_frame)
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

            valid_mask = vtx[:, 2] != 0
            x_cond = (vtx[:, 0] < 1.0) & (vtx[:, 0] > -0.6)
            z_cond = (vtx[:, 2] < 1.3)
            pts_dist = (vtx[:, 0] * self.A + vtx[:, 1] * self.B + vtx[:, 2] * self.C + self.D)
            plane_cond = pts_dist < -self.TABLE_THRESHOLD
            
            final_mask = valid_mask & x_cond & z_cond & plane_cond
            cropped_pc = vtx[final_mask]

            num_cropped = len(cropped_pc)
            if num_cropped > 0:
                choice_idx = np.random.choice(num_cropped, self.target_points, replace=(num_cropped < self.target_points))
                final_pc = cropped_pc[choice_idx]
            else:
                final_pc = np.zeros((self.target_points, 3), dtype=np.float32)

            self.last_pc_cache = final_pc.copy()
            return final_pc

        except Exception as e:
            log.error(f"L515 read error: {e}")
            return self.last_pc_cache.copy() if self.last_pc_cache is not None else None

    def stop(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass

class MyGripper:
    def __init__(self, ip_address='192.168.1.11', port=502):
        self.ip_address = ip_address
        self.port = port
        self.client = None
        self.lock = threading.Lock()
        
        self.current_normalized_pos = 0.0

        self.connect()
        self.activate_gripper()
        
        log.info("Gripper control updated: Continuous 0-1 to 0-255 mapping enabled.")

    def connect(self):
        try:
            if self.client:
                self.client.close()

            self.client = ModbusTcpClient(self.ip_address, port=self.port)

            if self.client.connect():
                log.success(f"Gripper connected to {self.ip_address}")
            else:
                log.error(f"Gripper connection failed to {self.ip_address}")

        except Exception as e:
            log.error(f"Gripper connection error: {e}")

    def activate_gripper(self):
        try:
            self.client.write_registers(0, [0x0100, 0x0000, 0x6464, 0, 0, 0, 0, 0])
            time.sleep(0.1)
        except Exception as e:
            log.warn(f"Gripper activation failed: {e}")

    def move(self, value: float):
        # 确保网络输出值被限制在 0.0 到 1.0 之间
        val_clamped = max(0.0, min(1.0, value))
        
        # 仅在目标位置发生变化时发送指令，避免无意义的通信开销
        if val_clamped != self.current_normalized_pos:
            self.current_normalized_pos = val_clamped
            self._send_cmd(val_clamped)

    def _send_cmd(self, position):
        try:
            # 将 0.0-1.0 映射到 0-255 以实现硬件的连续位置控制，并转换为整数
            hw_position = int(round(position * 255.0))
            
            # 确保数值绝对安全地落在硬件支持的 0-255 范围内
            hw_position = max(0, min(255, hw_position))
            
            cmd = [0x0900, hw_position, 0x6464, 0, 0, 0, 0, 1]
            self.client.write_registers(0, cmd)
        except Exception as e:
            log.warn(f"Gripper command failed: {e}, attempting reconnect...")
            try:
                self.connect()
                # 重连后重试指令
                self.client.write_registers(0, cmd)
            except Exception as e2:
                log.error(f"Gripper command failed after reconnect: {e2}")

    def get_current_position(self) -> float:
        # 直接返回 [0, 1] 范围的值给网络
        return self.current_normalized_pos

class URRobot:
    def __init__(self, robot_ip: str = "192.168.1.2", no_gripper: bool = False, gripper_ip: str = "192.168.1.11"):
        import rtde_control
        import rtde_receive

        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
            log.success(f"Connected to UR Robot at {robot_ip}")
        except Exception as e:
            log.error(f"UR Connection Error: {e}")
            raise e

        try:
            self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        except Exception as e:
            log.error(f"UR Receive Interface Error: {e}")

        self._use_gripper = not no_gripper
        if self._use_gripper:
            try:
                # 只需传入 IP 地址，去除之前多余的 threshold 和 lock_duration 参数
                self.gripper = MyGripper(ip_address=gripper_ip)
                log.success("Gripper connected successfully (Continuous Control).")
            except Exception as e:
                log.error(f"Gripper connection failed: {e}")
                self._use_gripper = False

        self._free_drive = False
        self.robot.endFreedriveMode()

    def num_dofs(self) -> int:
        return 7 if self._use_gripper else 6

    def get_joint_state(self) -> np.ndarray:
        """Get current joint state including gripper (closed-loop feedback).
        
        Returns:
            Joint positions [6 or 7] array:
            - [6]: robot joints only (no gripper)
            - [7]: robot joints + gripper position [0, 1] (0=OPEN, 1=CLOSED)
        """
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            # Read actual gripper state (closed-loop feedback)
            # current_normalized_pos is in [0, 1] physical format
            gripper_pos = self.gripper.current_normalized_pos
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return np.array(pos)

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]]) if self._use_gripper else np.array([0.0])

        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }
    
class UR5EnvInference:
    """
    针对 UR5 + L515 点云的 DP3 推理环境封装
    """
    def __init__(self, obs_horizon=2, action_horizon=8, device="cuda", num_points=1024, frequency=10.0):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_points = num_points
        self.dt = 1.0 / frequency
        self.device = torch.device(device)
        
        # 1. 硬件初始化
        cprint("Initializing L515 PointCloud Camera...", "cyan")
        self.camera_pc = L515PointCloudCamera(serial="f1420840", target_points=num_points)
        
        cprint("Initializing UR5 Robot...", "cyan")
        self.robot = URRobot(robot_ip="192.168.1.2", gripper_ip="192.168.1.11")
        self.home_joints = np.array([-1.57, -1.57, -1.57, -1.57, 1.57, 3.14])

        # 缓存 Buffer
        self.cloud_array = []
        self.agent_pos_array = []
        self.action_array = []
    
    def _get_robot_proprio(self):
        """获取真实的 10D 状态 (xyz, 6D rot, gripper)"""
        try:
            tcp_pose = self.robot.r_inter.getActualTCPPose()
            ee_xyz = np.array(tcp_pose[:3])  # meters
            ee_rot_vec = np.array(tcp_pose[3:6])  # rot vector

            # 转换为 6D 旋转
            rot_mat, _ = cv2.Rodrigues(ee_rot_vec)
            ee_rot_6d = np.concatenate([rot_mat[:, 0], rot_mat[:, 1]])

        except Exception as e:
            cprint(f"Failed to get TCP pose: {e}", "red")
            ee_xyz = np.zeros(3)
            ee_rot_6d = np.array([1, 0, 0, 0, 1, 0])

        gripper_state = self.robot.gripper.current_normalized_pos if self.robot._use_gripper else 0.0
        proprio = np.concatenate([ee_xyz, ee_rot_6d, [gripper_state]])
        
        return proprio

    def execute_action(self, action_10d):
        """执行 10D 动作指令"""
        try:
            target_pos = action_10d[:3]
            target_rot_6d = action_10d[3:9]
            target_gripper = action_10d[9]

            # 6D 还原为旋转矩阵，再转为旋转向量
            x = target_rot_6d[:3] / np.linalg.norm(target_rot_6d[:3])
            y = target_rot_6d[3:6] - np.dot(x, target_rot_6d[3:6]) * x
            y = y / np.linalg.norm(y)
            z = np.cross(x, y)
            target_rot_mat = np.column_stack([x, y, z])
            
            target_rot_vec, _ = cv2.Rodrigues(target_rot_mat)
            target_pose = np.concatenate([target_pos, target_rot_vec.flatten()])

            # 执行机械臂与夹爪
            self.robot.robot.moveL(target_pose.tolist(), 0.1, 0.3, False)
            if self.robot._use_gripper:
                self.robot.gripper.move(target_gripper)

        except Exception as e:
            cprint(f"Target proprio execution failed: {e}", "red")

    def _get_obs_dict(self):
        """打包 Observation 给模型"""
        agent_pos = np.stack(self.agent_pos_array[-self.obs_horizon:], axis=0)
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)
            
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).float().unsqueeze(0).to(self.device),
            'point_cloud': torch.from_numpy(obs_cloud).float().unsqueeze(0).to(self.device)
        }
        return obs_dict

    def step(self, action_list):
        """执行动作序列并记录新状态"""
        for action_id in range(self.action_horizon):
            start_time = time.time()
            
            act = action_list[action_id]
            self.action_array.append(act)
            
            # 1. 下发物理指令
            self.execute_action(act)
            
            # 2. 读取新状态
            pc = self.camera_pc.read()
            if pc is None:
                pc = self.cloud_array[-1] # fallback
            
            proprio = self._get_robot_proprio()
            
            self.cloud_array.append(pc)
            self.agent_pos_array.append(proprio)
            
            # 控制频率
            elapsed = time.time() - start_time
            time.sleep(max(0, self.dt - elapsed))
            
        return self._get_obs_dict()
    
    def reset(self, confirm=True):
        """复位机器人并初始化 Buffer"""
        if confirm:
            input("\nMove robot to home position? Press Enter...")

        cprint("Resetting robot to home...", "yellow")
        current_joints = self.robot.get_joint_state()[:6]
        max_delta = np.abs(current_joints - self.home_joints).max()
        steps = min(int(max_delta / 0.01), 100)
        
        for joint_target in np.linspace(current_joints, self.home_joints, max(steps, 10)):
            self.robot.robot.moveJ(joint_target.tolist(), 0.3, 0.5, False)
            time.sleep(0.01)

        time.sleep(1.0) # 等待稳定
        cprint("Robot ready!", "green")
        
        # 清空 Buffer
        self.cloud_array, self.agent_pos_array, self.action_array = [], [], []

        # 获取初始观测，填充 obs_horizon
        pc = self.camera_pc.read()
        proprio = self._get_robot_proprio()

        for _ in range(self.obs_horizon):
            self.cloud_array.append(pc)
            self.agent_pos_array.append(proprio)
            
        return self._get_obs_dict()


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath('uca','config'))
)
def main(cfg: OmegaConf):
    # torch.manual_seed(42)
    OmegaConf.resolve(cfg)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    import os
    ckpt_path = os.path.join(os.getcwd(), "checkpoints", "latest.ckpt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"\n找不到权重文件，请检查路径是否正确: {ckpt_path}\n"
                                f"当前 Hydra 运行目录是: {os.getcwd()}")
        
    print(f"\n[INFO] 正在从 {ckpt_path} 加载预训练权重...\n")
    
    # 1. 直接使用 workspace 的类方法加载
    workspace = TrainUCAWorkspace.create_from_checkpoint(ckpt_path)
    
    # 2. 判断是否使用 EMA 模型 (与 train.py 中的 eval 逻辑对齐)
    if workspace.cfg.training.use_ema:
        policy = workspace.ema_model
    else:
        policy = workspace.model
        
    policy.eval().to(device)
    # =================================================
    
    # 获取预测长度（通常模型预测比如16步，我们执行前8步）
    action_horizon = policy.horizon - policy.n_obs_steps + 1

    # 2. 初始化环境
    env = UR5EnvInference(
        obs_horizon=policy.n_obs_steps, 
        action_horizon=action_horizon, 
        device=device,
        num_points=1024,
        frequency=30.0 # 根据你训练时的频率修改
    )

    obs_dict = env.reset(confirm=True)

    # 3. 部署参数
    roll_out_length = 900 
    step_count = 0
    record_data = True
    
    cprint("=== Starting DP3 Rollout ===", "green", attrs=["bold"])
    
    # 4. 主控循环
    while step_count < roll_out_length:
        with torch.no_grad():
            # 这里的输入 obs_dict (物理数值)，经过 policy 内部通常会自动 normalize
            # 预测出的 action 通常会自动 unnormalize 变回物理数值
            result = policy.predict_action(obs_dict)
            action = result['action_pred'][0]
            action_list = [act.cpu().numpy() for act in action]
        
        # 执行动作序列
        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"Executed Steps: {step_count}/{roll_out_length}")

    # 5. 保存遥操验证数据 (可选)
    if record_data:
        import h5py
        save_dir = "deploy_logs"
        os.makedirs(save_dir, exist_ok=True)
        
        record_file_name = f"{save_dir}/demo_{int(time.time())}.h5"
        with h5py.File(record_file_name, "w") as f:
            f.create_dataset("point_cloud", data=np.array(env.cloud_array))
            f.create_dataset("agent_pos", data=np.array(env.agent_pos_array))
            f.create_dataset("action", data=np.array(env.action_array))
        
        cprint(f"Data saved to {record_file_name}", "yellow")

if __name__ == "__main__":
    main()