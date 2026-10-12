#!/usr/bin/env python3
"""
Offline FLOPs / parameter profiling for a trained UCA policy checkpoint.

Does NOT run during training. Example:

  cd UCA
  python ../scripts/compute_flops.py --ckpt data/outputs/.../checkpoints/latest.ckpt

  # or instantiate from a Hydra config without weights:
  python ../scripts/compute_flops.py --config-name dp3 task=adroit_hammer
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn
from termcolor import cprint


ROOT = pathlib.Path(__file__).resolve().parents[1]
UCA_DIR = ROOT / "UCA"
sys.path.insert(0, str(UCA_DIR))


def _fmt_flops(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.3f} TFLOPs"
    if n >= 1e9:
        return f"{n / 1e9:.3f} GFLOPs"
    if n >= 1e6:
        return f"{n / 1e6:.3f} MFLOPs"
    return f"{n:.0f} FLOPs"


def _fmt_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.3f} M"
    if n >= 1e3:
        return f"{n / 1e3:.3f} K"
    return str(n)


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class _EncoderWrapper(nn.Module):
    """fvcore-friendly wrapper: positional tensors -> obs_encoder(dict)."""

    def __init__(self, encoder: nn.Module, keys):
        super().__init__()
        self.encoder = encoder
        self.keys = list(keys)

    def forward(self, *tensors):
        return self.encoder({k: t for k, t in zip(self.keys, tensors)})


class _ModelWrapper(nn.Module):
    """fvcore-friendly wrapper around policy.model forward."""

    def __init__(self, model: nn.Module, use_r: bool = False):
        super().__init__()
        self.model = model
        self.use_r = use_r

    def forward(self, sample, timestep, global_cond, r=None):
        kwargs = dict(
            sample=sample,
            timestep=timestep,
            local_cond=None,
            global_cond=global_cond,
        )
        if self.use_r:
            kwargs["r"] = r if r is not None else torch.zeros_like(timestep)
        return self.model(**kwargs)


def _infer_global_cond_shape(policy, batch_size: int, obs_feat_dim: int):
    """Match how each policy builds global_cond in predict_action."""
    n_obs = int(policy.n_obs_steps)
    cond_type = getattr(policy, "condition_type", "")
    # DiT / ICT / Pi0 keep (B, To, Do); FiLM U-Nets flatten to (B, To*Do)
    if cond_type and "cross_attention" in cond_type:
        return (batch_size, n_obs, obs_feat_dim), True
    # MeanFlow DiT policies (ict/ict1/pi0/dit/ditx) reshape to sequence
    policy_cls = type(policy).__name__
    module_name = type(policy).__module__
    if any(x in module_name for x in ("ict", "dit", "pi0")):
        return (batch_size, n_obs, obs_feat_dim), True
    if getattr(policy, "obs_as_global_cond", True):
        return (batch_size, n_obs * obs_feat_dim), False
    return (batch_size, n_obs, obs_feat_dim), True


def _needs_r(policy) -> bool:
    module_name = type(policy).__module__
    return any(x in module_name for x in ("ict", "dit", "pi0", "meanpolicy"))


def _num_denoise_steps(policy) -> int:
    if hasattr(policy, "num_inference_steps") and policy.num_inference_steps:
        return int(policy.num_inference_steps)
    if hasattr(policy, "num_inference_step") and policy.num_inference_step:
        return int(policy.num_inference_step)
    # MeanFlow / one-step policies
    return 1


def _build_dummy_obs(policy, batch_size: int, device: torch.device):
    shape_meta = None
    # shape_meta is usually stored only in cfg; fall back to encoder attrs
    obs_keys = []
    tensors = []
    encoder = policy.obs_encoder
    n_obs = int(policy.n_obs_steps)
    # flattened encoder batch = B * n_obs_steps
    be = batch_size * n_obs

    pc_key = getattr(encoder, "point_cloud_key", "point_cloud")
    state_key = getattr(encoder, "state_key", "agent_pos")
    use_pc_color = getattr(policy, "use_pc_color", False)
    # point cloud channels: 3 or 6
    in_ch = 6 if use_pc_color else 3
    # try to read from encoder cfg
    pc_cfg = getattr(encoder, "pointcloud_encoder_cfg", None)
    if isinstance(pc_cfg, dict) and "in_channels" in pc_cfg:
        in_ch = int(pc_cfg["in_channels"])

    num_points = 512
    # DexArt may concat imagined robot points inside encoder; input still uses env points
    pc = torch.randn(be, num_points, in_ch, device=device)
    obs_keys.append(pc_key)
    tensors.append(pc)

    state_shape = getattr(encoder, "state_shape", None)
    if state_shape is None:
        state_dim = 24
    else:
        state_dim = int(state_shape[0])
    state = torch.randn(be, state_dim, device=device)
    obs_keys.append(state_key)
    tensors.append(state)

    if getattr(encoder, "use_imagined_robot", False):
        imag_key = getattr(encoder, "imagination_key", "imagin_robot")
        # imagined points typically (B, M, C); C aligned to pc last dim inside encoder
        imag = torch.randn(be, 32, in_ch, device=device)
        obs_keys.append(imag_key)
        tensors.append(imag)

    return obs_keys, tensors


def compute_flops(policy, batch_size: int = 1, device: str = "cuda"):
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    device_t = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    policy = policy.to(device_t)
    policy.eval()

    obs_feat_dim = int(policy.obs_encoder.output_shape())
    horizon = int(policy.horizon)
    action_dim = int(policy.action_dim)

    enc_keys, enc_tensors = _build_dummy_obs(policy, batch_size, device_t)
    enc_wrap = _EncoderWrapper(policy.obs_encoder, enc_keys).to(device_t).eval()

    with torch.no_grad():
        enc_flops = FlopCountAnalysis(enc_wrap, tuple(enc_tensors))
        enc_total = float(enc_flops.total())

    gshape, is_seq = _infer_global_cond_shape(policy, batch_size, obs_feat_dim)
    sample = torch.randn(batch_size, horizon, action_dim, device=device_t)
    timestep = torch.rand(batch_size, device=device_t)
    global_cond = torch.randn(*gshape, device=device_t)
    use_r = _needs_r(policy)
    r = torch.zeros(batch_size, device=device_t) if use_r else None

    model_wrap = _ModelWrapper(policy.model, use_r=use_r).to(device_t).eval()
    model_inputs = (sample, timestep, global_cond, r) if use_r else (sample, timestep, global_cond)

    with torch.no_grad():
        model_flops = FlopCountAnalysis(model_wrap, model_inputs)
        model_step = float(model_flops.total())

    n_steps = _num_denoise_steps(policy)
    model_infer = model_step * n_steps
    total_infer = enc_total + model_infer

    result = {
        "encoder_flops": enc_total,
        "policy_step_flops": model_step,
        "num_denoise_steps": n_steps,
        "policy_infer_flops": model_infer,
        "total_infer_flops": total_infer,
        "encoder_params": _count_params(policy.obs_encoder),
        "policy_params": _count_params(policy.model),
        "total_params": _count_params(policy),
        "global_cond_shape": list(gshape),
        "batch_size": batch_size,
    }

    cprint("========== FLOPs / Params ==========", "cyan")
    cprint(f"batch_size           : {batch_size}", "cyan")
    cprint(f"global_cond shape    : {gshape}", "cyan")
    cprint(f"denoise / sample steps: {n_steps}", "cyan")
    cprint(f"3D encoder FLOPs     : {_fmt_flops(enc_total)}", "yellow")
    cprint(f"policy 1-step FLOPs  : {_fmt_flops(model_step)}", "yellow")
    cprint(f"policy infer FLOPs   : {_fmt_flops(model_infer)}  (= 1-step x {n_steps})", "yellow")
    cprint(f"total infer FLOPs    : {_fmt_flops(total_infer)}  (encoder + policy infer)", "green")
    cprint(f"encoder params       : {_fmt_params(result['encoder_params'])}", "yellow")
    cprint(f"policy backbone params: {_fmt_params(result['policy_params'])}", "yellow")
    cprint(f"total policy params  : {_fmt_params(result['total_params'])}", "green")
    print()
    print(parameter_count_table(policy))
    return result


def load_policy_from_ckpt(ckpt: str, use_ema: bool = True):
    from train import TrainUCAWorkspace
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    workspace = TrainUCAWorkspace.create_from_checkpoint(ckpt)
    policy = workspace.ema_model if (use_ema and workspace.ema_model is not None) else workspace.model
    return policy, workspace.cfg


def load_policy_from_config(config_name: str, overrides):
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from train import TrainUCAWorkspace

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = str(UCA_DIR / "uca" / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    workspace = TrainUCAWorkspace(cfg)
    return workspace.model, cfg


def main():
    parser = argparse.ArgumentParser(description="Offline FLOPs for UCA policies")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to .ckpt")
    parser.add_argument("--config-name", type=str, default=None, help="Hydra config name, e.g. dp3")
    parser.add_argument("--override", nargs="*", default=[], help="Hydra overrides, e.g. task=adroit_hammer")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-ema", action="store_true", help="Use raw model instead of EMA")
    args = parser.parse_args()

    if args.ckpt:
        policy, cfg = load_policy_from_ckpt(args.ckpt, use_ema=not args.no_ema)
        cprint(f"Loaded checkpoint: {args.ckpt}", "magenta")
    elif args.config_name:
        policy, cfg = load_policy_from_config(args.config_name, args.override)
        cprint(f"Instantiated from config: {args.config_name}", "magenta")
    else:
        parser.error("Provide --ckpt or --config-name")

    compute_flops(policy, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Offline FLOPs / parameter profiling for a trained UCA policy checkpoint.

Does NOT run during training. Example:

  cd UCA
  python ../scripts/compute_flops.py --ckpt data/outputs/.../checkpoints/latest.ckpt

  # or instantiate from a Hydra config without weights:
  python ../scripts/compute_flops.py --config-name dp3 task=adroit_hammer
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn
from termcolor import cprint


ROOT = pathlib.Path(__file__).resolve().parents[1]
UCA_DIR = ROOT / "UCA"
sys.path.insert(0, str(UCA_DIR))


def _fmt_flops(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.3f} TFLOPs"
    if n >= 1e9:
        return f"{n / 1e9:.3f} GFLOPs"
    if n >= 1e6:
        return f"{n / 1e6:.3f} MFLOPs"
    return f"{n:.0f} FLOPs"


def _fmt_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.3f} M"
    if n >= 1e3:
        return f"{n / 1e3:.3f} K"
    return str(n)


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class _EncoderWrapper(nn.Module):
    """fvcore-friendly wrapper: positional tensors -> obs_encoder(dict)."""

    def __init__(self, encoder: nn.Module, keys):
        super().__init__()
        self.encoder = encoder
        self.keys = list(keys)

    def forward(self, *tensors):
        return self.encoder({k: t for k, t in zip(self.keys, tensors)})


class _ModelWrapper(nn.Module):
    """fvcore-friendly wrapper around policy.model forward."""

    def __init__(self, model: nn.Module, use_r: bool = False):
        super().__init__()
        self.model = model
        self.use_r = use_r

    def forward(self, sample, timestep, global_cond, r=None):
        kwargs = dict(
            sample=sample,
            timestep=timestep,
            local_cond=None,
            global_cond=global_cond,
        )
        if self.use_r:
            kwargs["r"] = r if r is not None else torch.zeros_like(timestep)
        return self.model(**kwargs)


def _infer_global_cond_shape(policy, batch_size: int, obs_feat_dim: int):
    """Match how each policy builds global_cond in predict_action."""
    n_obs = int(policy.n_obs_steps)
    cond_type = getattr(policy, "condition_type", "")
    # DiT / ICT / Pi0 keep (B, To, Do); FiLM U-Nets flatten to (B, To*Do)
    if cond_type and "cross_attention" in cond_type:
        return (batch_size, n_obs, obs_feat_dim), True
    # MeanFlow DiT policies (ict/ict1/pi0/dit/ditx) reshape to sequence
    policy_cls = type(policy).__name__
    module_name = type(policy).__module__
    if any(x in module_name for x in ("ict", "dit", "pi0")):
        return (batch_size, n_obs, obs_feat_dim), True
    if getattr(policy, "obs_as_global_cond", True):
        return (batch_size, n_obs * obs_feat_dim), False
    return (batch_size, n_obs, obs_feat_dim), True


def _needs_r(policy) -> bool:
    module_name = type(policy).__module__
    return any(x in module_name for x in ("ict", "dit", "pi0", "meanpolicy"))


def _num_denoise_steps(policy) -> int:
    if hasattr(policy, "num_inference_steps") and policy.num_inference_steps:
        return int(policy.num_inference_steps)
    if hasattr(policy, "num_inference_step") and policy.num_inference_step:
        return int(policy.num_inference_step)
    # MeanFlow / one-step policies
    return 1


def _build_dummy_obs(policy, batch_size: int, device: torch.device):
    shape_meta = None
    # shape_meta is usually stored only in cfg; fall back to encoder attrs
    obs_keys = []
    tensors = []
    encoder = policy.obs_encoder
    n_obs = int(policy.n_obs_steps)
    # flattened encoder batch = B * n_obs_steps
    be = batch_size * n_obs

    pc_key = getattr(encoder, "point_cloud_key", "point_cloud")
    state_key = getattr(encoder, "state_key", "agent_pos")
    use_pc_color = getattr(policy, "use_pc_color", False)
    # point cloud channels: 3 or 6
    in_ch = 6 if use_pc_color else 3
    # try to read from encoder cfg
    pc_cfg = getattr(encoder, "pointcloud_encoder_cfg", None)
    if isinstance(pc_cfg, dict) and "in_channels" in pc_cfg:
        in_ch = int(pc_cfg["in_channels"])

    num_points = 512
    # DexArt may concat imagined robot points inside encoder; input still uses env points
    pc = torch.randn(be, num_points, in_ch, device=device)
    obs_keys.append(pc_key)
    tensors.append(pc)

    state_shape = getattr(encoder, "state_shape", None)
    if state_shape is None:
        state_dim = 24
    else:
        state_dim = int(state_shape[0])
    state = torch.randn(be, state_dim, device=device)
    obs_keys.append(state_key)
    tensors.append(state)

    if getattr(encoder, "use_imagined_robot", False):
        imag_key = getattr(encoder, "imagination_key", "imagin_robot")
        # imagined points typically (B, M, C); C aligned to pc last dim inside encoder
        imag = torch.randn(be, 32, in_ch, device=device)
        obs_keys.append(imag_key)
        tensors.append(imag)

    return obs_keys, tensors


def compute_flops(policy, batch_size: int = 1, device: str = "cuda"):
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    device_t = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    policy = policy.to(device_t)
    policy.eval()

    obs_feat_dim = int(policy.obs_encoder.output_shape())
    horizon = int(policy.horizon)
    action_dim = int(policy.action_dim)

    enc_keys, enc_tensors = _build_dummy_obs(policy, batch_size, device_t)
    enc_wrap = _EncoderWrapper(policy.obs_encoder, enc_keys).to(device_t).eval()

    with torch.no_grad():
        enc_flops = FlopCountAnalysis(enc_wrap, tuple(enc_tensors))
        enc_total = float(enc_flops.total())

    gshape, is_seq = _infer_global_cond_shape(policy, batch_size, obs_feat_dim)
    sample = torch.randn(batch_size, horizon, action_dim, device=device_t)
    timestep = torch.rand(batch_size, device=device_t)
    global_cond = torch.randn(*gshape, device=device_t)
    use_r = _needs_r(policy)
    r = torch.zeros(batch_size, device=device_t) if use_r else None

    model_wrap = _ModelWrapper(policy.model, use_r=use_r).to(device_t).eval()
    model_inputs = (sample, timestep, global_cond, r) if use_r else (sample, timestep, global_cond)

    with torch.no_grad():
        model_flops = FlopCountAnalysis(model_wrap, model_inputs)
        model_step = float(model_flops.total())

    n_steps = _num_denoise_steps(policy)
    model_infer = model_step * n_steps
    total_infer = enc_total + model_infer

    result = {
        "encoder_flops": enc_total,
        "policy_step_flops": model_step,
        "num_denoise_steps": n_steps,
        "policy_infer_flops": model_infer,
        "total_infer_flops": total_infer,
        "encoder_params": _count_params(policy.obs_encoder),
        "policy_params": _count_params(policy.model),
        "total_params": _count_params(policy),
        "global_cond_shape": list(gshape),
        "batch_size": batch_size,
    }

    cprint("========== FLOPs / Params ==========", "cyan")
    cprint(f"batch_size           : {batch_size}", "cyan")
    cprint(f"global_cond shape    : {gshape}", "cyan")
    cprint(f"denoise / sample steps: {n_steps}", "cyan")
    cprint(f"3D encoder FLOPs     : {_fmt_flops(enc_total)}", "yellow")
    cprint(f"policy 1-step FLOPs  : {_fmt_flops(model_step)}", "yellow")
    cprint(f"policy infer FLOPs   : {_fmt_flops(model_infer)}  (= 1-step x {n_steps})", "yellow")
    cprint(f"total infer FLOPs    : {_fmt_flops(total_infer)}  (encoder + policy infer)", "green")
    cprint(f"encoder params       : {_fmt_params(result['encoder_params'])}", "yellow")
    cprint(f"policy backbone params: {_fmt_params(result['policy_params'])}", "yellow")
    cprint(f"total policy params  : {_fmt_params(result['total_params'])}", "green")
    print()
    print(parameter_count_table(policy))
    return result


def load_policy_from_ckpt(ckpt: str, use_ema: bool = True):
    from train import TrainUCAWorkspace
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    workspace = TrainUCAWorkspace.create_from_checkpoint(ckpt)
    policy = workspace.ema_model if (use_ema and workspace.ema_model is not None) else workspace.model
    return policy, workspace.cfg


def load_policy_from_config(config_name: str, overrides):
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from train import TrainUCAWorkspace

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = str(UCA_DIR / "uca" / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    workspace = TrainUCAWorkspace(cfg)
    return workspace.model, cfg


def main():
    parser = argparse.ArgumentParser(description="Offline FLOPs for UCA policies")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to .ckpt")
    parser.add_argument("--config-name", type=str, default=None, help="Hydra config name, e.g. dp3")
    parser.add_argument("--override", nargs="*", default=[], help="Hydra overrides, e.g. task=adroit_hammer")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-ema", action="store_true", help="Use raw model instead of EMA")
    args = parser.parse_args()

    if args.ckpt:
        policy, cfg = load_policy_from_ckpt(args.ckpt, use_ema=not args.no_ema)
        cprint(f"Loaded checkpoint: {args.ckpt}", "magenta")
    elif args.config_name:
        policy, cfg = load_policy_from_config(args.config_name, args.override)
        cprint(f"Instantiated from config: {args.config_name}", "magenta")
    else:
        parser.error("Provide --ckpt or --config-name")

    compute_flops(policy, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
