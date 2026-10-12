import wandb
import time 
import numpy as np
import os
import torch
import collections
import tqdm
import imageio
from uca.env import MetaWorldEnv
from uca.gym_util.multistep_wrapper import MultiStepWrapper
from uca.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from uca.policy.base_policy import BasePolicy
from uca.common.pytorch_util import dict_apply
from uca.env_runner.base_runner import BaseRunner
import uca.common.logger_util as logger_util
from termcolor import cprint

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda",
                 use_point_crop=True,
                 num_points=512,
                 detail_timing=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.detail_timing = detail_timing

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
        self.output_dir = 'save_videos'

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        all_time = []
        all_encoder_time = []
        all_policy_time = []
        env = self.env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            actual_step_count = 0
            total_time = 0
            total_encoder_time = 0
            total_policy_time = 0
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    if self.detail_timing:
                        action_dict, times = self._run_policy_step(
                            policy, obs_dict_input, detail_timing=True)
                        total_time += times['total_s']
                        total_encoder_time += times['encoder_s']
                        total_policy_time += times['policy_s']
                    else:
                        start_time = time.time()
                        action_dict = policy.predict_action(obs_dict_input)
                        end_time = time.time()
                        total_time += end_time - start_time

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)


                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])
                actual_step_count += 1

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            all_time.append(total_time / actual_step_count)
            if self.detail_timing and actual_step_count > 0:
                all_encoder_time.append(total_encoder_time / actual_step_count)
                all_policy_time.append(total_policy_time / actual_step_count)
            

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['mean_time'] = np.mean(all_time)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)*100}", 'green')
        cprint(f"test_mean_time: {np.mean(all_time)*1000}", 'red')
        if self.detail_timing and len(all_encoder_time) > 0:
            log_data['mean_encoder_time'] = np.mean(all_encoder_time)
            log_data['mean_policy_time'] = np.mean(all_policy_time)
            cprint(f"test_mean_encoder_time: {log_data['mean_encoder_time']*1000}", 'red')
            cprint(f"test_mean_policy_time: {log_data['mean_policy_time']*1000}", 'red')
        
        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        print(videos.shape)
        if np.mean(all_success_rates)*100 >90 :
            video_path = os.path.join(
                self.output_dir,
                f"{self.task_name}_eval.mp4"
            )
            if videos is not None and videos.size > 0:
                # 目录不存在就先建
                os.makedirs(self.output_dir, exist_ok=True)
                self._save_video_mp4(videos, video_path, fps=self.fps)
                cprint(f"Saved evaluation video → {video_path}", "cyan")
            else:
                cprint("No video frames captured -- check wrapper settings.", "yellow")
                
            # exit()

        _ = env.reset()
        videos = None

        return log_data
    
    def _save_video_mp4(self, frames: np.ndarray, fname: str, fps: int = 10):
        """
        frames : ndarray
                 支持形状 (T, H, W, C)，(T, C, H, W)，(B, T, H, W, C) 或 (B, T, C, H, W)
        fname  : 目标文件名（以 .mp4 结尾）
        fps    : 帧率
        """
        # -------- 1. 处理批量维度 (B, ...) --------
        if frames.ndim == 5:            # (B, T, H, W, C) or (B, T, C, H, W)
            frames = frames[0]          # 只保存第 0 号 env 的视频

        # -------- 2. 把通道放到最后 ----------
        if frames.ndim == 4:
            # 形如 (T, C, H, W) → (T, H, W, C)
            if frames.shape[-1] not in (1, 2, 3, 4):
                frames = frames.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Unsupported frame ndim: {frames.ndim}")

        # -------- 3. dtype → uint8 ----------
        if frames.dtype != np.uint8:
            frames = np.clip(frames * 255, 0, 255).astype(np.uint8)

        # -------- 4. 写入 mp4 --------------
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        imageio.mimsave(
            fname,
            frames,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None   # 防止分辨率不是 16 的倍数时报错
        )
        cprint(f"Saved evaluation video → {fname}", "cyan")
import wandb
import time 
import numpy as np
import os
import torch
import collections
import tqdm
import imageio
from uca.env import MetaWorldEnv
from uca.gym_util.multistep_wrapper import MultiStepWrapper
from uca.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from uca.policy.base_policy import BasePolicy
from uca.common.pytorch_util import dict_apply
from uca.env_runner.base_runner import BaseRunner
import uca.common.logger_util as logger_util
from termcolor import cprint

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda",
                 use_point_crop=True,
                 num_points=512,
                 detail_timing=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.detail_timing = detail_timing

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
        self.output_dir = 'save_videos'

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        all_time = []
        all_encoder_time = []
        all_policy_time = []
        env = self.env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            actual_step_count = 0
            total_time = 0
            total_encoder_time = 0
            total_policy_time = 0
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    if self.detail_timing:
                        action_dict, times = self._run_policy_step(
                            policy, obs_dict_input, detail_timing=True)
                        total_time += times['total_s']
                        total_encoder_time += times['encoder_s']
                        total_policy_time += times['policy_s']
                    else:
                        start_time = time.time()
                        action_dict = policy.predict_action(obs_dict_input)
                        end_time = time.time()
                        total_time += end_time - start_time

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)


                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])
                actual_step_count += 1

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            all_time.append(total_time / actual_step_count)
            if self.detail_timing and actual_step_count > 0:
                all_encoder_time.append(total_encoder_time / actual_step_count)
                all_policy_time.append(total_policy_time / actual_step_count)
            

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['mean_time'] = np.mean(all_time)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)*100}", 'green')
        cprint(f"test_mean_time: {np.mean(all_time)*1000}", 'red')
        if self.detail_timing and len(all_encoder_time) > 0:
            log_data['mean_encoder_time'] = np.mean(all_encoder_time)
            log_data['mean_policy_time'] = np.mean(all_policy_time)
            cprint(f"test_mean_encoder_time: {log_data['mean_encoder_time']*1000}", 'red')
            cprint(f"test_mean_policy_time: {log_data['mean_policy_time']*1000}", 'red')
        
        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        print(videos.shape)
        if np.mean(all_success_rates)*100 >90 :
            video_path = os.path.join(
                self.output_dir,
                f"{self.task_name}_eval.mp4"
            )
            if videos is not None and videos.size > 0:
                # 目录不存在就先建
                os.makedirs(self.output_dir, exist_ok=True)
                self._save_video_mp4(videos, video_path, fps=self.fps)
                cprint(f"Saved evaluation video → {video_path}", "cyan")
            else:
                cprint("No video frames captured -- check wrapper settings.", "yellow")
                
            # exit()

        _ = env.reset()
        videos = None

        return log_data
    
    def _save_video_mp4(self, frames: np.ndarray, fname: str, fps: int = 10):
        """
        frames : ndarray
                 支持形状 (T, H, W, C)，(T, C, H, W)，(B, T, H, W, C) 或 (B, T, C, H, W)
        fname  : 目标文件名（以 .mp4 结尾）
        fps    : 帧率
        """
        # -------- 1. 处理批量维度 (B, ...) --------
        if frames.ndim == 5:            # (B, T, H, W, C) or (B, T, C, H, W)
            frames = frames[0]          # 只保存第 0 号 env 的视频

        # -------- 2. 把通道放到最后 ----------
        if frames.ndim == 4:
            # 形如 (T, C, H, W) → (T, H, W, C)
            if frames.shape[-1] not in (1, 2, 3, 4):
                frames = frames.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Unsupported frame ndim: {frames.ndim}")

        # -------- 3. dtype → uint8 ----------
        if frames.dtype != np.uint8:
            frames = np.clip(frames * 255, 0, 255).astype(np.uint8)

        # -------- 4. 写入 mp4 --------------
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        imageio.mimsave(
            fname,
            frames,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None   # 防止分辨率不是 16 的倍数时报错
        )
        cprint(f"Saved evaluation video → {fname}", "cyan")
