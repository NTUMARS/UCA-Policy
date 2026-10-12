import sys
sys.path.append('UCA/uca')
from typing import Dict
import torch
from termcolor import cprint
from uca.model.common.normalizer import LinearNormalizer
from uca.policy.base_policy import BasePolicy
from uca.model.mean.ditx import DiTModel
from uca.common.pytorch_util import dict_apply
from uca.common.model_util import print_params
from uca.model.vision.pointnet_extractor import UCAEncoder 
from functools import partial
import warnings
from einops import rearrange, reduce

warnings.filterwarnings("ignore")

class DiTMeanpolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_as_global_cond=True,
            n_layer=8,
            n_head=8,
            n_emb=256,
            n_cond_layers=2,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            num_t_tokens: int = 1,
            num_h_tokens: int = 1,
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # ==================== 使用 UCAEncoder ====================
        obs_encoder = UCAEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            state_mlp_size=(64, 64),
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        self.obs_encoder = obs_encoder
        obs_feature_dim = obs_encoder.output_shape()   # 单一总维度 (例如 320)
        self.obs_feature_dim = obs_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiTMeanpolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiTMeanpolicy] pointnet_type: {self.pointnet_type}", "yellow")

        # ==================== DiTModel 初始化 ====================
        model = DiTModel(
            input_dim=action_dim,
            horizon=horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            n_cond_layers=n_cond_layers,
            cond_dim=obs_feature_dim,          # ← 推荐使用 cond_dim（需同步修改 DiTModel）
            num_t_tokens=num_t_tokens,
            num_h_tokens=num_h_tokens,
        )

        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self.flow_ratio = 0.50
        self.time_dist = ['lognorm', -0.4, 1.0]
        print_params(self)

    # ==================== predict_action ====================
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)                    # (B*To, total_dim)

        # 关键：reshape 成 (B, n_obs_steps, emb) 序列形式
        global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # 采样
        model = self.model
        model.eval()

        z = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)
        t = torch.ones((cond_data.shape[0],), device=cond_data.device)
        r = torch.zeros((cond_data.shape[0],), device=cond_data.device)

        z = z - model(sample=z, timestep=t, local_cond=None, global_cond=global_cond, r=r)

        naction_pred = z[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred,
        }

    # ==================== compute_loss ====================
    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        x = nactions
        device = x.device

        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)                    # (B*To, total_dim)

        # 关键：reshape 成 (B, n_obs_steps, emb)
        global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)

        t, r = self.sample_t_r(batch_size, device)
        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")
        e = torch.randn_like(x)

        z = (1 - t_) * x + t_ * e
        v = e - x

        u = self.model(sample=z, timestep=t, global_cond=global_cond, r=r)

        model_partial = partial(self.model, global_cond=global_cond)
        def u_fn(z_in, t_in, r_in):
            return model_partial(sample=z_in, timestep=t_in, r=r_in)

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            _, dudt = torch.autograd.functional.jvp(
                u_fn,
                (z, t, r),
                (v, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=False
            )

        u_tgt = v - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)

        mse_val = (stopgrad(error) ** 2).mean()

        return loss, {
            'bc_loss': loss.item(),
            'mse_val': mse_val.item()
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = torch.rand((batch_size, 2), device=device, dtype=torch.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = torch.randn((batch_size, 2), device=device, dtype=torch.float32) * sigma + mu
            samples = 1.0 / (1.0 + torch.exp(-normal_samples))

        t_tensor = torch.maximum(samples[:, 0], samples[:, 1])
        r_tensor = torch.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = torch.randperm(batch_size, device=device)[:num_selected]
        r_tensor[indices] = t_tensor[indices]

        return t_tensor, r_tensor


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()