"""
Generate MetaWorld Push demos with two geometrically distinct approach modes.

Mode A (left,  +1): approach puck from +X via two airborne waypoints, then push.
Mode B (right, -1): approach puck from -X via two airborne waypoints, then push.

After the approach phases, control is identical to SawyerPushV2Policy.
Output zarr matches the existing expert format, plus meta/mode_labels.
"""
import argparse
import copy
import os

import numpy as np
import zarr
from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.policies import SawyerPushV2Policy
from termcolor import cprint

from uca.env import MetaWorldEnv


class SawyerPushBimodalV2Policy(Policy):
    """
    Two airborne lateral waypoints, then exact original Push policy.
    Recreate (or call reset()) every episode so phase state is fresh.
    """

    def __init__(self, mode: str = "left", lateral: float = 0.10):
        assert mode in ("left", "right")
        self.mode = mode
        self.side = 1.0 if mode == "left" else -1.0
        self.lateral = float(lateral)
        self.phase = 0  # 0: far wp, 1: near wp, 2: original push

    def reset(self):
        self.phase = 0

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            "hand_pos": obs[:3],
            "unused_1": obs[3],
            "puck_pos": obs[4:7],
            "unused_2": obs[7:-3],
            "goal_pos": obs[-3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        action = Action({
            "delta_pos": np.arange(3),
            "grab_effort": 3,
        })
        action["delta_pos"] = move(o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=10.0)
        if self.phase < 2:
            action["grab_effort"] = 0.0
        else:
            action["grab_effort"] = SawyerPushV2Policy._grab_effort(o_d)
        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["puck_pos"] + np.array([-0.005, 0.0, 0.0])
        pos_goal = o_d["goal_pos"]

        wp1 = pos_puck + np.array([self.side * self.lateral, -0.12, 0.28])
        wp2 = pos_puck + np.array([self.side * self.lateral * 0.5, -0.02, 0.22])

        if self.phase == 0:
            if np.linalg.norm(pos_curr - wp1) > 0.035:
                return wp1
            self.phase = 1

        if self.phase == 1:
            if np.linalg.norm(pos_curr - wp2) > 0.03:
                return wp2
            self.phase = 2

        # Exact SawyerPushV2Policy logic
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02:
            return pos_puck + np.array([0.0, 0.0, 0.2])
        if abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return pos_puck + np.array([0.0, 0.0, 0.03])
        return pos_goal


def _infer_mode_label(full_states):
    """Label by mean (hand_x - puck_x) while airborne (z > 0.18)."""
    full = np.asarray(full_states)
    hand = full[:, :3]
    puck = full[:, 4:7]
    mask = hand[:, 2] > 0.18
    if mask.sum() < 5:
        mask = np.zeros(len(full), dtype=bool)
        mask[: max(20, len(full) // 5)] = True
    delta = float(np.mean(hand[mask, 0] - puck[mask, 0]))
    return 1 if delta > 0 else -1


def main(args):
    save_dir = os.path.join(args.root_dir, "metaworld_push_bimodal_expert.zarr")
    if os.path.exists(save_dir):
        cprint(f"Data already exists at {save_dir}", "red")
        if args.overwrite:
            cprint(f"Overwriting {save_dir}", "red")
            os.system(f"rm -rf {save_dir}")
        else:
            cprint("Pass --overwrite to replace it. Exiting.", "red")
            return
    os.makedirs(save_dir, exist_ok=True)

    env = MetaWorldEnv("push", device=args.device, use_point_crop=True)
    modes = [("left", 1), ("right", -1)]
    per_mode = args.num_episodes_per_mode
    cprint(f"Collecting {per_mode} episodes x {len(modes)} modes → {save_dir}", "yellow")

    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    full_state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    mode_label_arrays = []

    for mode_name, side_sign in modes:
        episode_idx = 0
        attempts = 0
        max_attempts = per_mode * args.max_attempts_factor
        cprint(f"=== Mode {mode_name} (label={side_sign}), lateral={args.lateral} ===", "cyan")

        while episode_idx < per_mode and attempts < max_attempts:
            attempts += 1
            policy = SawyerPushBimodalV2Policy(mode=mode_name, lateral=args.lateral)

            reset_obs = env.reset()
            raw_state = reset_obs["full_state"]
            obs_dict = env.get_visual_obs()

            done = False
            ep_reward = 0.0
            ep_success = False
            ep_success_times = 0

            img_sub, pc_sub, depth_sub = [], [], []
            state_sub, full_sub, action_sub = [], [], []
            total_count_sub = 0

            while not done:
                total_count_sub += 1
                img_sub.append(obs_dict["image"])
                pc_sub.append(obs_dict["point_cloud"])
                depth_sub.append(obs_dict["depth"])
                state_sub.append(obs_dict["agent_pos"])
                full_sub.append(raw_state)

                action = policy.get_action(raw_state)
                action_sub.append(action)

                obs_dict, reward, done, info = env.step(action)
                raw_state = obs_dict["full_state"]
                ep_reward += reward
                ep_success = ep_success or bool(info["success"])
                ep_success_times += int(info["success"])
                if done:
                    break

            if (not ep_success) or ep_success_times < args.min_success_times:
                cprint(
                    f"[{mode_name}] attempt {attempts} fail "
                    f"reward={ep_reward:.1f} success_times={ep_success_times}",
                    "red",
                )
                continue

            label = _infer_mode_label(full_sub)
            if label != side_sign:
                cprint(
                    f"[{mode_name}] attempt {attempts} rejected: "
                    f"inferred label={label} != {side_sign}",
                    "yellow",
                )
                continue

            total_count += total_count_sub
            episode_ends_arrays.append(copy.deepcopy(total_count))
            mode_label_arrays.append(side_sign)
            img_arrays.extend(copy.deepcopy(img_sub))
            point_cloud_arrays.extend(copy.deepcopy(pc_sub))
            depth_arrays.extend(copy.deepcopy(depth_sub))
            state_arrays.extend(copy.deepcopy(state_sub))
            full_state_arrays.extend(copy.deepcopy(full_sub))
            action_arrays.extend(copy.deepcopy(action_sub))
            cprint(
                f"[{mode_name}] ep {episode_idx + 1}/{per_mode} "
                f"reward={ep_reward:.1f} success_times={ep_success_times} label={label}",
                "green",
            )
            episode_idx += 1

        if episode_idx < per_mode:
            raise RuntimeError(
                f"Only collected {episode_idx}/{per_mode} episodes for mode={mode_name} "
                f"after {attempts} attempts. Try adjusting --lateral."
            )

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3:
        img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
    state_arrays = np.stack(state_arrays, axis=0).astype(np.float32)
    full_state_arrays = np.stack(full_state_arrays, axis=0).astype(np.float32)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0).astype(np.float32)
    depth_arrays = np.stack(depth_arrays, axis=0).astype(np.float32)
    action_arrays = np.stack(action_arrays, axis=0).astype(np.float32)
    episode_ends_arrays = np.asarray(episode_ends_arrays, dtype=np.int64)
    mode_label_arrays = np.asarray(mode_label_arrays, dtype=np.int64)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset(
        "img", data=img_arrays,
        chunks=(100, *img_arrays.shape[1:]), dtype="uint8",
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "state", data=state_arrays,
        chunks=(100, state_arrays.shape[1]), dtype="float32",
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "full_state", data=full_state_arrays,
        chunks=(100, full_state_arrays.shape[1]), dtype="float32",
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "point_cloud", data=point_cloud_arrays,
        chunks=(100, *point_cloud_arrays.shape[1:]), dtype="float32",
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "depth", data=depth_arrays,
        chunks=(100, *depth_arrays.shape[1:]), dtype="float32",
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "action", data=action_arrays,
        chunks=(100, action_arrays.shape[1]), dtype="float32",
        overwrite=True, compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends", data=episode_ends_arrays,
        dtype="int64", overwrite=True, compressor=compressor,
    )
    zarr_meta.create_dataset(
        "mode_labels", data=mode_label_arrays,
        dtype="int64", overwrite=True, compressor=compressor,
    )

    n_left = int(np.sum(mode_label_arrays == 1))
    n_right = int(np.sum(mode_label_arrays == -1))
    cprint("-" * 50, "cyan")
    cprint(f"episodes: {len(episode_ends_arrays)} (left=+1:{n_left}, right=-1:{n_right})", "green")
    cprint(f"img: {img_arrays.shape}", "green")
    cprint(f"point_cloud: {point_cloud_arrays.shape}", "green")
    cprint(f"state: {state_arrays.shape}", "green")
    cprint(f"action: {action_arrays.shape}", "green")
    cprint(f"Saved → {save_dir}", "green")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes_per_mode", type=int, default=40)
    parser.add_argument("--root_dir", type=str, default="../../UCA/data/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lateral", type=float, default=0.10,
                        help="Lateral approach offset in meters (+X left / -X right).")
    parser.add_argument("--min_success_times", type=int, default=5)
    parser.add_argument("--max_attempts_factor", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
"""
Generate MetaWorld Push demos with two geometrically distinct approach modes.

Mode A (left,  +1): approach puck from +X via two airborne waypoints, then push.
Mode B (right, -1): approach puck from -X via two airborne waypoints, then push.

After the approach phases, control is identical to SawyerPushV2Policy.
Output zarr matches the existing expert format, plus meta/mode_labels.
"""
import argparse
import copy
import os

import numpy as np
import zarr
from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.policies import SawyerPushV2Policy
from termcolor import cprint

from uca.env import MetaWorldEnv


class SawyerPushBimodalV2Policy(Policy):
    """
    Two airborne lateral waypoints, then exact original Push policy.
    Recreate (or call reset()) every episode so phase state is fresh.
    """

    def __init__(self, mode: str = "left", lateral: float = 0.10):
        assert mode in ("left", "right")
        self.mode = mode
        self.side = 1.0 if mode == "left" else -1.0
        self.lateral = float(lateral)
        self.phase = 0  # 0: far wp, 1: near wp, 2: original push

    def reset(self):
        self.phase = 0

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            "hand_pos": obs[:3],
            "unused_1": obs[3],
            "puck_pos": obs[4:7],
            "unused_2": obs[7:-3],
            "goal_pos": obs[-3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        action = Action({
            "delta_pos": np.arange(3),
            "grab_effort": 3,
        })
        action["delta_pos"] = move(o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=10.0)
        if self.phase < 2:
            action["grab_effort"] = 0.0
        else:
            action["grab_effort"] = SawyerPushV2Policy._grab_effort(o_d)
        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["puck_pos"] + np.array([-0.005, 0.0, 0.0])
        pos_goal = o_d["goal_pos"]

        wp1 = pos_puck + np.array([self.side * self.lateral, -0.12, 0.28])
        wp2 = pos_puck + np.array([self.side * self.lateral * 0.5, -0.02, 0.22])

        if self.phase == 0:
            if np.linalg.norm(pos_curr - wp1) > 0.035:
                return wp1
            self.phase = 1

        if self.phase == 1:
            if np.linalg.norm(pos_curr - wp2) > 0.03:
                return wp2
            self.phase = 2

        # Exact SawyerPushV2Policy logic
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02:
            return pos_puck + np.array([0.0, 0.0, 0.2])
        if abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return pos_puck + np.array([0.0, 0.0, 0.03])
        return pos_goal


def _infer_mode_label(full_states):
    """Label by mean (hand_x - puck_x) while airborne (z > 0.18)."""
    full = np.asarray(full_states)
    hand = full[:, :3]
    puck = full[:, 4:7]
    mask = hand[:, 2] > 0.18
    if mask.sum() < 5:
        mask = np.zeros(len(full), dtype=bool)
        mask[: max(20, len(full) // 5)] = True
    delta = float(np.mean(hand[mask, 0] - puck[mask, 0]))
    return 1 if delta > 0 else -1


def main(args):
    save_dir = os.path.join(args.root_dir, "metaworld_push_bimodal_expert.zarr")
    if os.path.exists(save_dir):
        cprint(f"Data already exists at {save_dir}", "red")
        if args.overwrite:
            cprint(f"Overwriting {save_dir}", "red")
            os.system(f"rm -rf {save_dir}")
        else:
            cprint("Pass --overwrite to replace it. Exiting.", "red")
            return
    os.makedirs(save_dir, exist_ok=True)

    env = MetaWorldEnv("push", device=args.device, use_point_crop=True)
    modes = [("left", 1), ("right", -1)]
    per_mode = args.num_episodes_per_mode
    cprint(f"Collecting {per_mode} episodes x {len(modes)} modes → {save_dir}", "yellow")

    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    full_state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    mode_label_arrays = []

    for mode_name, side_sign in modes:
        episode_idx = 0
        attempts = 0
        max_attempts = per_mode * args.max_attempts_factor
        cprint(f"=== Mode {mode_name} (label={side_sign}), lateral={args.lateral} ===", "cyan")

        while episode_idx < per_mode and attempts < max_attempts:
            attempts += 1
            policy = SawyerPushBimodalV2Policy(mode=mode_name, lateral=args.lateral)

            reset_obs = env.reset()
            raw_state = reset_obs["full_state"]
            obs_dict = env.get_visual_obs()

            done = False
            ep_reward = 0.0
            ep_success = False
            ep_success_times = 0

            img_sub, pc_sub, depth_sub = [], [], []
            state_sub, full_sub, action_sub = [], [], []
            total_count_sub = 0

            while not done:
                total_count_sub += 1
                img_sub.append(obs_dict["image"])
                pc_sub.append(obs_dict["point_cloud"])
                depth_sub.append(obs_dict["depth"])
                state_sub.append(obs_dict["agent_pos"])
                full_sub.append(raw_state)

                action = policy.get_action(raw_state)
                action_sub.append(action)

                obs_dict, reward, done, info = env.step(action)
                raw_state = obs_dict["full_state"]
                ep_reward += reward
                ep_success = ep_success or bool(info["success"])
                ep_success_times += int(info["success"])
                if done:
                    break

            if (not ep_success) or ep_success_times < args.min_success_times:
                cprint(
                    f"[{mode_name}] attempt {attempts} fail "
                    f"reward={ep_reward:.1f} success_times={ep_success_times}",
                    "red",
                )
                continue

            label = _infer_mode_label(full_sub)
            if label != side_sign:
                cprint(
                    f"[{mode_name}] attempt {attempts} rejected: "
                    f"inferred label={label} != {side_sign}",
                    "yellow",
                )
                continue

            total_count += total_count_sub
            episode_ends_arrays.append(copy.deepcopy(total_count))
            mode_label_arrays.append(side_sign)
            img_arrays.extend(copy.deepcopy(img_sub))
            point_cloud_arrays.extend(copy.deepcopy(pc_sub))
            depth_arrays.extend(copy.deepcopy(depth_sub))
            state_arrays.extend(copy.deepcopy(state_sub))
            full_state_arrays.extend(copy.deepcopy(full_sub))
            action_arrays.extend(copy.deepcopy(action_sub))
            cprint(
                f"[{mode_name}] ep {episode_idx + 1}/{per_mode} "
                f"reward={ep_reward:.1f} success_times={ep_success_times} label={label}",
                "green",
            )
            episode_idx += 1

        if episode_idx < per_mode:
            raise RuntimeError(
                f"Only collected {episode_idx}/{per_mode} episodes for mode={mode_name} "
                f"after {attempts} attempts. Try adjusting --lateral."
            )

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3:
        img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
    state_arrays = np.stack(state_arrays, axis=0).astype(np.float32)
    full_state_arrays = np.stack(full_state_arrays, axis=0).astype(np.float32)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0).astype(np.float32)
    depth_arrays = np.stack(depth_arrays, axis=0).astype(np.float32)
    action_arrays = np.stack(action_arrays, axis=0).astype(np.float32)
    episode_ends_arrays = np.asarray(episode_ends_arrays, dtype=np.int64)
    mode_label_arrays = np.asarray(mode_label_arrays, dtype=np.int64)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset(
        "img", data=img_arrays,
        chunks=(100, *img_arrays.shape[1:]), dtype="uint8",
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "state", data=state_arrays,
        chunks=(100, state_arrays.shape[1]), dtype="float32",
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "full_state", data=full_state_arrays,
        chunks=(100, full_state_arrays.shape[1]), dtype="float32",
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "point_cloud", data=point_cloud_arrays,
        chunks=(100, *point_cloud_arrays.shape[1:]), dtype="float32",
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "depth", data=depth_arrays,
        chunks=(100, *depth_arrays.shape[1:]), dtype="float32",
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "action", data=action_arrays,
        chunks=(100, action_arrays.shape[1]), dtype="float32",
        overwrite=True, compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends", data=episode_ends_arrays,
        dtype="int64", overwrite=True, compressor=compressor,
    )
    zarr_meta.create_dataset(
        "mode_labels", data=mode_label_arrays,
        dtype="int64", overwrite=True, compressor=compressor,
    )

    n_left = int(np.sum(mode_label_arrays == 1))
    n_right = int(np.sum(mode_label_arrays == -1))
    cprint("-" * 50, "cyan")
    cprint(f"episodes: {len(episode_ends_arrays)} (left=+1:{n_left}, right=-1:{n_right})", "green")
    cprint(f"img: {img_arrays.shape}", "green")
    cprint(f"point_cloud: {point_cloud_arrays.shape}", "green")
    cprint(f"state: {state_arrays.shape}", "green")
    cprint(f"action: {action_arrays.shape}", "green")
    cprint(f"Saved → {save_dir}", "green")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes_per_mode", type=int, default=40)
    parser.add_argument("--root_dir", type=str, default="../../UCA/data/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lateral", type=float, default=0.10,
                        help="Lateral approach offset in meters (+X left / -X right).")
    parser.add_argument("--min_success_times", type=int, default=5)
    parser.add_argument("--max_attempts_factor", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
