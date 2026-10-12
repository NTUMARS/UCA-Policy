import sys
sys.path.append('UCA/uca')
from typing import Dict
import torch
from termcolor import cprint
from uca.model.common.normalizer import LinearNormalizer
from uca.policy.base_policy import BasePolicy
from uca.model.mean.ict import DiTModel
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
            # --- 以下替换为 Transformer/DiTModel 的参数 ---
            n_layer=8,
            n_head=8,
            n_emb=256,
            num_t_tokens: int = 1,  # 可调参数
            num_h_tokens: int = 1,  # 可调参数

            encoder_output_dim=384,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # parameters passed to step
            **kwargs):
        super().__init__()


        # parse shape_meta [保持不变]
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
            cond_dim=obs_feature_dim,          # ← 推荐使用 cond_dim（需同步修改 DiTModel）
            num_t_tokens=num_t_tokens,
            num_h_tokens=num_h_tokens,
        )

        self.model = model
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs
        self.time_dist = ['lognorm', -0.4, 1.0]
        self.flow_ratio = 0
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

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

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
        v_g=stopgrad(v)

        # ====== 1. 强制 r=t，获取瞬时速度 v (保持梯度！) ======
        v_c = self.model(
            sample=z, 
            timestep=t, 
            global_cond=global_cond, 
            r=t  
        )

        # 使用 detach() 截断梯度，仅仅作为 JVP 的数学方向
        v_surrogate = stopgrad(v_c)

        # ====== 2. 正常的前向传播，计算带有梯度的 u_c ======
        u_c = self.model(
            sample=z, 
            timestep=t, 
            global_cond=global_cond, 
            r=r
        )

        # ====== 3. 构建供 JVP 调用的纯函数 u_fn ======
        def u_fn(z_in, t_in, r_in):
            return self.model(sample=z_in, timestep=t_in, global_cond=global_cond, r=r_in)

        # ====== 4. 计算 JVP ======
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            _, dudt = torch.autograd.functional.jvp(
                u_fn,
                (z, t, r),
                (v_surrogate, torch.ones_like(t), torch.zeros_like(r)), 
                create_graph=False 
            )

        # ====== 5. 计算复合函数 V ======
        # 复合速度 V，注意 dudt 通常需要 detach 避免二阶导数的不稳定
        V = u_c + (t_ - r_) * stopgrad(dudt) 

        # ====== 6. 双重loss ======
        error_u = V - v_g
        error_v = v_c - v_g  # 辅助 v-head 去拟合真实轨迹
        
        loss_u = adaptive_l2_loss(error_u)
        loss_v = adaptive_l2_loss(error_v)

        # 总 Loss (可以根据实际规模给 loss_v 加个权重，但通常 1:1 就很好)
        loss = loss_u + loss_v

        mse_val = (stopgrad(error_u) ** 2).mean()

        loss_dict = {
                'bc_loss': loss.item(),
                'loss_u': loss_u.item(),
                'loss_v': loss_v.item(),
                'mse_val': mse_val.item()
            }
        return loss, loss_dict

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = torch.rand((batch_size, 2), device=device, dtype=torch.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = torch.randn((batch_size, 2), device=device, dtype=torch.float32) * sigma + mu
            samples = 1.0 / (1.0 + torch.exp(-normal_samples))  # PyTorch 的 sigmoid 操作

        t_tensor = torch.maximum(samples[:, 0], samples[:, 1])
        r_tensor = torch.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = torch.randperm(batch_size, device=device)[:num_selected]
        
        r_tensor[indices] = t_tensor[indices]

        return t_tensor, r_tensor

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
    