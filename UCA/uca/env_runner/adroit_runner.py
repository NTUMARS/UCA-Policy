
import wandb
import time
import numpy as np
import torch
import tqdm
from uca.env import AdroitEnv
from uca.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit
from uca.gym_util.multistep_wrapper import MultiStepWrapper
from uca.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from uca.policy.base_policy import BasePolicy
from uca.common.pytorch_util import dict_apply
from uca.env_runner.base_runner import BaseRunner
import uca.common.logger_util as logger_util
from termcolor import cprint


class AdroitRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=20,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 detail_timing=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(20 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True),
                                                  env_name='adroit_'+task_name, use_point_crop=use_point_crop)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.detail_timing = detail_timing

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        all_time = []
        all_encoder_time = []
        all_policy_time = []
        
         
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            total_time = 0
            total_encoder_time = 0
            total_policy_time = 0
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
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
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                # all_goal_achieved.append(info['goal_achieved']
                num_goal_achieved += np.sum(info['goal_achieved'])
                done = np.all(done)
                actual_step_count += 1

             # print(total_time / actual_step_count)
            all_success_rates.append(info['goal_achieved'])
            all_goal_achieved.append(num_goal_achieved)
            all_time.append(total_time / actual_step_count)
            if self.detail_timing and actual_step_count > 0:
                all_encoder_time.append(total_encoder_time / actual_step_count)
                all_policy_time.append(total_policy_time / actual_step_count)


        # log
        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
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
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env

        return log_data

import wandb
import time
import numpy as np
import torch
import tqdm
from uca.env import AdroitEnv
from uca.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit
from uca.gym_util.multistep_wrapper import MultiStepWrapper
from uca.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from uca.policy.base_policy import BasePolicy
from uca.common.pytorch_util import dict_apply
from uca.env_runner.base_runner import BaseRunner
import uca.common.logger_util as logger_util
from termcolor import cprint


class AdroitRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=20,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 detail_timing=False,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(20 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True),
                                                  env_name='adroit_'+task_name, use_point_crop=use_point_crop)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.detail_timing = detail_timing

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        all_time = []
        all_encoder_time = []
        all_policy_time = []
        
         
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            total_time = 0
            total_encoder_time = 0
            total_policy_time = 0
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
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
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                # all_goal_achieved.append(info['goal_achieved']
                num_goal_achieved += np.sum(info['goal_achieved'])
                done = np.all(done)
                actual_step_count += 1

             # print(total_time / actual_step_count)
            all_success_rates.append(info['goal_achieved'])
            all_goal_achieved.append(num_goal_achieved)
            all_time.append(total_time / actual_step_count)
            if self.detail_timing and actual_step_count > 0:
                all_encoder_time.append(total_encoder_time / actual_step_count)
                all_policy_time.append(total_policy_time / actual_step_count)


        # log
        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
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
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env

        return log_data
