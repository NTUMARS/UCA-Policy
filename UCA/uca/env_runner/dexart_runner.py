import wandb
import time  # <-- 新增导入
import numpy as np
import torch
import collections
import tqdm
from termcolor import cprint
from uca.env import DexArtEnv
from uca.gym_util.multistep_wrapper import MultiStepWrapper
from uca.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from uca.policy.base_policy import BasePolicy
from uca.common.pytorch_util import dict_apply
from uca.env_runner.base_runner import BaseRunner
import uca.common.logger_util as logger_util


class DexArtRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 n_train=10,
                 max_steps=250,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_test_env=False,
                 agent_pos_length=None, # 32, to align for multitasks
                 detail_timing=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn(is_test=True):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(DexArtEnv(
                    task_name=task_name,
                    use_test_set=is_test,
                )),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.env_train = env_fn(is_test=False)
        self.episode_train = n_train

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.detail_timing = detail_timing

        # 对齐 Adroit 的记录器命名
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        
    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env_train = self.env_train

        all_returns = []
        all_success_rates = []
        all_time = []
        all_encoder_time = []
        all_policy_time = []

        ##############################
        # train env loop
        for episode_id in tqdm.tqdm(range(self.episode_train), desc=f"DexArt {self.task_name} Train Env",leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env_train.reset()

            policy.reset()

            done = False
            reward_sum = 0.
            total_time = 0
            total_encoder_time = 0
            total_policy_time = 0
            actual_step_count = 0

            for step_id in range(self.max_steps):
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    # add batch dim to match. (1,2,3,84,84)
                    # and multiply by 255, align with all envs
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['imagin_robot'] = obs_dict['imagin_robot'].unsqueeze(0)
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
                    actual_step_count += 1

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)

                # step env
                obs, reward, done, info = env_train.step(action)
                reward_sum += reward
                done = np.all(done)

                if done:
                    break

            all_returns.append(reward_sum)
            all_success_rates.append(env_train.is_success())
            # 计算每步平均耗时并存入列表
            all_time.append(total_time / max(1, actual_step_count))
            if self.detail_timing and actual_step_count > 0:
                all_encoder_time.append(total_encoder_time / actual_step_count)
                all_policy_time.append(total_policy_time / actual_step_count)


        SR_mean = np.mean(all_success_rates)
        returns_mean = np.mean(all_returns)
        time_mean = np.mean(all_time)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        
        # 对齐 Adroit 的 WandB log key
        log_data['mean_success_rates'] = SR_mean
        log_data['mean_returns'] = returns_mean
        log_data['mean_time'] = time_mean

        log_data['test_mean_score'] = SR_mean

        self.logger_util_test.record(SR_mean)
        self.logger_util_test10.record(SR_mean)

        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        
        # 对齐 Adroit 的终端打印格式和颜色
        cprint(f"test_mean_score: {SR_mean*100}", 'green')
        cprint(f"test_mean_time: {time_mean*1000}", 'red')
        if self.detail_timing and len(all_encoder_time) > 0:
            log_data['mean_encoder_time'] = np.mean(all_encoder_time)
            log_data['mean_policy_time'] = np.mean(all_policy_time)
            cprint(f"test_mean_encoder_time: {log_data['mean_encoder_time']*1000}", 'red')
            cprint(f"test_mean_policy_time: {log_data['mean_policy_time']*1000}", 'red')

        # visualize sim
        videos_train = env_train.env.get_video()

        if len(videos_train.shape) == 5:
            videos_train = videos_train[:, 0]
        sim_video_train = wandb.Video(videos_train, fps=self.fps, format="mp4")
        # 对齐视频的 key
        log_data[f'sim_video_eval'] = sim_video_train

        # clear out video buffer
        _ = env_train.reset()
        videos_train = None
        del env_train

        return log_dataimport wandb
import time  # <-- 新增导入
import numpy as np
import torch
import collections
import tqdm
from termcolor import cprint
from uca.env import DexArtEnv
from uca.gym_util.multistep_wrapper import MultiStepWrapper
from uca.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from uca.policy.base_policy import BasePolicy
from uca.common.pytorch_util import dict_apply
from uca.env_runner.base_runner import BaseRunner
import uca.common.logger_util as logger_util


class DexArtRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 n_train=10,
                 max_steps=250,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_test_env=False,
                 agent_pos_length=None, # 32, to align for multitasks
                 detail_timing=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn(is_test=True):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(DexArtEnv(
                    task_name=task_name,
                    use_test_set=is_test,
                )),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.env_train = env_fn(is_test=False)
        self.episode_train = n_train

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.detail_timing = detail_timing

        # 对齐 Adroit 的记录器命名
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        
    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env_train = self.env_train

        all_returns = []
        all_success_rates = []
        all_time = []
        all_encoder_time = []
        all_policy_time = []

        ##############################
        # train env loop
        for episode_id in tqdm.tqdm(range(self.episode_train), desc=f"DexArt {self.task_name} Train Env",leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env_train.reset()

            policy.reset()

            done = False
            reward_sum = 0.
            total_time = 0
            total_encoder_time = 0
            total_policy_time = 0
            actual_step_count = 0

            for step_id in range(self.max_steps):
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    # add batch dim to match. (1,2,3,84,84)
                    # and multiply by 255, align with all envs
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['imagin_robot'] = obs_dict['imagin_robot'].unsqueeze(0)
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
                    actual_step_count += 1

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)

                # step env
                obs, reward, done, info = env_train.step(action)
                reward_sum += reward
                done = np.all(done)

                if done:
                    break

            all_returns.append(reward_sum)
            all_success_rates.append(env_train.is_success())
            # 计算每步平均耗时并存入列表
            all_time.append(total_time / max(1, actual_step_count))
            if self.detail_timing and actual_step_count > 0:
                all_encoder_time.append(total_encoder_time / actual_step_count)
                all_policy_time.append(total_policy_time / actual_step_count)


        SR_mean = np.mean(all_success_rates)
        returns_mean = np.mean(all_returns)
        time_mean = np.mean(all_time)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        
        # 对齐 Adroit 的 WandB log key
        log_data['mean_success_rates'] = SR_mean
        log_data['mean_returns'] = returns_mean
        log_data['mean_time'] = time_mean

        log_data['test_mean_score'] = SR_mean

        self.logger_util_test.record(SR_mean)
        self.logger_util_test10.record(SR_mean)

        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        
        # 对齐 Adroit 的终端打印格式和颜色
        cprint(f"test_mean_score: {SR_mean*100}", 'green')
        cprint(f"test_mean_time: {time_mean*1000}", 'red')
        if self.detail_timing and len(all_encoder_time) > 0:
            log_data['mean_encoder_time'] = np.mean(all_encoder_time)
            log_data['mean_policy_time'] = np.mean(all_policy_time)
            cprint(f"test_mean_encoder_time: {log_data['mean_encoder_time']*1000}", 'red')
            cprint(f"test_mean_policy_time: {log_data['mean_policy_time']*1000}", 'red')

        # visualize sim
        videos_train = env_train.env.get_video()

        if len(videos_train.shape) == 5:
            videos_train = videos_train[:, 0]
        sim_video_train = wandb.Video(videos_train, fps=self.fps, format="mp4")
        # 对齐视频的 key
        log_data[f'sim_video_eval'] = sim_video_train

        # clear out video buffer
        _ = env_train.reset()
        videos_train = None
        del env_train

        return log_data