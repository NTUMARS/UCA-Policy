#!/usr/bin/env python3
"""Fixed-init multimodal coverage eval for metaworld_push_bimodal.

Protocol:
  - N fixed environment seeds (shared across methods)
  - K stochastic rollouts per init (policy noise free)
  - Label mode by airborne mean (hand_x - puck_x): +1 left / -1 right
  - Report success rate and mode visit rates (all / success-only)

Example:
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_bimodal_mode_coverage.py \\
    --ckpt UCA/data/outputs/.../checkpoints/latest.ckpt \\
    --alg uca --seed 0 --n_inits 10 --k_samples 20 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch
from termcolor import cprint

ROOT = pathlib.Path(__file__).resolve().parents[1]
UCA_DIR = ROOT / "UCA"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UCA_DIR))
os.chdir(UCA_DIR)


def infer_mode_label(full_states: np.ndarray) -> int:
    """Same rule as gen_bimodal_push_expert._infer_mode_label."""
    full = np.asarray(full_states)
    hand = full[:, :3]
    puck = full[:, 4:7]
    mask = hand[:, 2] > 0.18
    if mask.sum() < 5:
        mask = np.zeros(len(full), dtype=bool)
        mask[: max(20, len(full) // 5)] = True
    delta = float(np.mean(hand[mask, 0] - puck[mask, 0]))
    return 1 if delta > 0 else -1


def find_best_ckpt(run_dir: pathlib.Path) -> pathlib.Path:
    ckpt_dir = run_dir / "checkpoints"
    latest = ckpt_dir / "latest.ckpt"
    if latest.is_file():
        return latest
    best, best_score = None, -1e9
    for p in ckpt_dir.glob("epoch=*-test_mean_score=*.ckpt"):
        try:
            score = float(p.name.split("test_mean_score=")[1].split(".ckpt")[0])
        except Exception:
            continue
        if score > best_score:
            best, best_score = p, score
    if best is None:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    return best


def load_policy(ckpt_path: pathlib.Path, device: str):
    from train import TrainUCAWorkspace

    workspace = TrainUCAWorkspace.create_from_checkpoint(str(ckpt_path))
    cfg = workspace.cfg
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    policy.to(device)
    return workspace, policy, cfg


def make_env(cfg):
    from uca.env import MetaWorldEnv
    from uca.gym_util.multistep_wrapper import MultiStepWrapper

    # No video wrapper — faster for coverage stats.
    base = MetaWorldEnv(
        task_name="push",
        device=cfg.training.device,
        use_point_crop=cfg.policy.use_point_crop,
        num_points=512,
    )
    env = MultiStepWrapper(
        base,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=200,
        reward_agg_method="sum",
    )
    return env


def rollout_once(env, policy, device, init_seed: int):
    from uca.common.pytorch_util import dict_apply

    obs = env.reset(seed=init_seed)
    policy.reset()
    done = False
    success = False
    states = []

    if "full_state" in obs:
        fs = np.asarray(obs["full_state"])
        states.append(fs[-1] if fs.ndim == 2 else fs)

    while not done:
        obs_dict = dict_apply(dict(obs), lambda x: torch.from_numpy(x).to(device))
        with torch.no_grad():
            inp = {
                "point_cloud": obs_dict["point_cloud"].unsqueeze(0),
                "agent_pos": obs_dict["agent_pos"].unsqueeze(0),
            }
            action_dict = policy.predict_action(inp)
        action = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())["action"].squeeze(0)
        obs, reward, done, info = env.step(action)
        done = bool(np.all(done))
        success = success or bool(np.max(info["success"]))
        if "full_state" in obs:
            fs = np.asarray(obs["full_state"])
            states.append(fs[-1] if fs.ndim == 2 else fs)

    states = np.asarray(states)
    mode = infer_mode_label(states) if len(states) else 0
    return success, mode, states


def eval_ckpt(ckpt_path: pathlib.Path, device: str, init_seeds, k_samples: int):
    workspace, policy, cfg = load_policy(ckpt_path, device)
    env = make_env(cfg)

    rows = []
    for init_seed in init_seeds:
        for k in range(k_samples):
            torch.manual_seed(10_000 * int(init_seed) + k)
            np.random.seed(10_000 * int(init_seed) + k)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(10_000 * int(init_seed) + k)
            success, mode, _ = rollout_once(env, policy, device, int(init_seed))
            rows.append(
                {
                    "init_seed": int(init_seed),
                    "sample": k,
                    "success": bool(success),
                    "mode": int(mode),
                }
            )
            tag = "L" if mode > 0 else "R"
            cprint(
                f"  init={init_seed} k={k:02d} success={int(success)} mode={tag}",
                "cyan",
            )
    env.close()
    return rows, cfg


def summarize(rows):
    n = len(rows)
    n_succ = sum(1 for r in rows if r["success"])
    n_left = sum(1 for r in rows if r["mode"] > 0)
    n_right = sum(1 for r in rows if r["mode"] < 0)
    n_left_s = sum(1 for r in rows if r["success"] and r["mode"] > 0)
    n_right_s = sum(1 for r in rows if r["success"] and r["mode"] < 0)

    by_init = defaultdict(list)
    for r in rows:
        by_init[r["init_seed"]].append(r)

    both_modes = 0
    for _, rs in by_init.items():
        succ = [r for r in rs if r["success"]]
        modes = {r["mode"] for r in succ}
        if 1 in modes and -1 in modes:
            both_modes += 1

    return {
        "n_rollouts": n,
        "success_rate": n_succ / max(n, 1),
        "visit_left_all": n_left / max(n, 1),
        "visit_right_all": n_right / max(n, 1),
        "visit_left_success": n_left_s / max(n_succ, 1) if n_succ else 0.0,
        "visit_right_success": n_right_s / max(n_succ, 1) if n_succ else 0.0,
        "n_success": n_succ,
        "frac_inits_both_modes": both_modes / max(len(by_init), 1),
        "n_inits_both_modes": both_modes,
        "n_inits": len(by_init),
    }


def default_ckpt(alg: str, seed: int, tag: str) -> pathlib.Path:
    if alg == "act":
        run = UCA_DIR / "data" / "outputs" / f"metaworld_push_bimodal-act-{tag}h32_seed{seed}"
        if (run / "checkpoints").is_dir():
            return find_best_ckpt(run)
        run = UCA_DIR / "data" / "outputs" / f"metaworld_push_bimodal-act-{tag}_seed{seed}"
        return find_best_ckpt(run)
    run = UCA_DIR / "data" / "outputs" / f"metaworld_push_bimodal-{alg}-{tag}_seed{seed}"
    return find_best_ckpt(run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--alg", type=str, required=True, choices=["uca", "dp3", "act"])
    parser.add_argument("--seed", type=int, default=0, help="training seed of the run")
    parser.add_argument("--tag", type=str, default="bimodal0807")
    parser.add_argument("--n_inits", type=int, default=10)
    parser.add_argument("--k_samples", type=int, default=20)
    parser.add_argument("--init_seed_base", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    # Respect caller CUDA_VISIBLE_DEVICES (Wave A sets the physical GPU outside).
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else default_ckpt(args.alg, args.seed, args.tag)
    cprint(f"Loading {ckpt}", "yellow")

    init_seeds = [args.init_seed_base + i for i in range(args.n_inits)]
    rows, _cfg = eval_ckpt(ckpt, device, init_seeds, args.k_samples)
    summary = summarize(rows)
    summary.update(
        {
            "alg": args.alg,
            "train_seed": args.seed,
            "ckpt": str(ckpt),
            "n_inits": args.n_inits,
            "k_samples": args.k_samples,
            "init_seeds": init_seeds,
        }
    )

    out_dir = pathlib.Path(
        args.out_dir
        or (ROOT / "logs" / "bimodal_mode_eval" / f"{args.alg}_seed{args.seed}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    cprint("======== MODE COVERAGE SUMMARY ========", "green")
    for k, v in summary.items():
        if k == "init_seeds":
            continue
        cprint(f"{k}: {v}", "green")
    cprint(f"Wrote {out_dir}", "yellow")


if __name__ == "__main__":
    main()
