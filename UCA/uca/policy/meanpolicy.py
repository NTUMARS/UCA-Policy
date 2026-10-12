import sys
sys.path.append('UCA/uca')
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
import copy
import time
import numpy as np
from uca.sde_lib import ConsistencyFM
from uca.model.common.normalizer import LinearNormalizer
from uca.policy.base_policy import BasePolicy
from uca.model.mean.conditional_unet1d_meanflow import ConditionalUnet1D
from uca.model.mean.mask_generator import LowdimMaskGenerator
from uca.common.pytorch_util import dict_apply
from uca.common.model_util import print_params
from uca.model.vision.pointnet_extractor import UCAEncoder
from functools import partial
import warnings
from einops import rearrange, reduce

warnings.filterwarnings("ignore")

class Meanpolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = UCAEncoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")



        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps

        self.flow_ratio=0.50
        self.time_dist=['lognorm', -0.4, 1.0]
        self.cfg_ratio=0.10
        cfg_scale=2.0
        # experimental
        self.cfg_uncond='u'
        self.w = cfg_scale
        print_params(self)
                
        self.t_start = 0.0
        self.t_end = 1.0
        self.kappa = None


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        model = self.model
        model.eval()
        
        z = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device)

        t = torch.ones((cond_data.shape[0],), device=cond_data.device)
        r = torch.zeros((cond_data.shape[0],), device=cond_data.device)

        z = z - model(sample=z,
                    timestep=t, 
                    local_cond=local_cond, 
                    global_cond=global_cond, r=r)
           
        
        # unnormalize prediction
        naction_pred = z[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        x = trajectory
        
        device = cond_data.device
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda n: n[:,:self.n_obs_steps,...].reshape(-1,*n.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        
        t, r = self.sample_t_r(batch_size, device)
        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")
        e = torch.randn_like(x)
        # x = normalize_to_neg1_1(x)
        z = (1 - t_) * x + t_ * e
        v = e - x
        with torch.no_grad():
            u_t = self.model(
                            sample=z, 
                            timestep=t, 
                            global_cond=global_cond, 
                            r=t)
        v_hat = 2 * v + (1 - 2) * u_t

        # model_partial = partial(self.model, global_cond=global_cond)
        # u, dudt = torch.autograd.functional.jvp(
        #     lambda z, t, r: model_partial(sample=z, timestep=t, r=r),
        #     # model,
        #     (z, t, r),
        #     (v, torch.ones_like(t), torch.zeros_like(r)),
        #     create_graph=True
        # )

        # u_tgt = v - (t_ - r_) * dudt
        
        # 1. 正常单次前向传播，保留完整的计算图（用于后续 loss.backward 更新权重）
        u = self.model(sample=z, timestep=t, global_cond=global_cond, r=r)

        # 2. 构建纯函数
        model_partial = partial(self.model, global_cond=global_cond)
        def u_fn(z_in, t_in, r_in):
            return model_partial(sample=z_in, timestep=t_in, r=r_in)

        # 3. 在 no_grad 上下文中计算 JVP，彻底关闭计算图，获得极致加速
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            _, dudt = torch.autograd.functional.jvp(
                u_fn,
                (z, t, r),
                (v_hat, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=False  # 关闭图
            )

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
      

        mse_val = (stopgrad(error) ** 2).mean()

        loss_dict = {
                'bc_loss': loss.item(),
                'mse_val': mse_val.item()
            }
        
        return loss, loss_dict

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

def normalize_to_neg1_1(x):
    return x * 2 - 1


def unnormalize_to_0_1(x):
    return (x + 1) * 0.5

def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))    
    # delta_sq = torch.sum(error ** 2, dim=tuple(range(1, error.ndim)))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()
