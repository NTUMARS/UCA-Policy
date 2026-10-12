import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def modulate_rmsnorm(x, rms_weight, shift, scale, eps=1e-6):
    """AdaRMSNorm 调制"""
    rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if rms_weight is not None:
        rms = rms * rms_weight
    return rms * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ==========================================
# 1. 基础组件
# ==========================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_dim=256):
        super().__init__()
        self.freq_embed = SinusoidalPosEmb(frequency_embedding_dim)
        # 输出维度为 hidden_size (即 n_emb)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_size),
            nn.SiLU(), 
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t):
        return self.mlp(self.freq_embed(t))

# ==========================================
# 2. 全局观测条件编码器
# ==========================================
class GlobalCondEncoder(nn.Module):
    """将序列形态的观测条件展平并编码为全局特征向量，供 AdaLN 注入"""
    def __init__(self, cond_dim, n_emb, cond_seq_len):
        super().__init__()
        flat_dim = cond_dim * cond_seq_len if cond_seq_len > 0 else cond_dim
        # 输出维度为 n_emb
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, n_emb),
            nn.SiLU(),
            nn.Linear(n_emb, n_emb)
        )

    def forward(self, cond):
        B = cond.shape[0]
        cond_flat = cond.view(B, -1)
        return self.mlp(cond_flat)

# ==========================================
# 3. AdaRMSNorm Transformer 块
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, n_emb, n_head, p_drop_attn=0.0, p_drop=0.0):
        super().__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        self.head_dim = n_emb // n_head

        self.norm1_weight = nn.Parameter(torch.ones(n_emb))
        self.qkv = nn.Linear(n_emb, 3 * n_emb, bias=False)
        self.proj_msa = nn.Linear(n_emb, n_emb, bias=False)
        self.dropout_attn = p_drop_attn

        self.norm2_weight = nn.Parameter(torch.ones(n_emb))
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_emb, n_emb, bias=False),
            nn.Dropout(p_drop),
        )

        self.ada_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_emb, 6 * n_emb, bias=True)
        )

    def forward(self, x, time_cond, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_modulation(time_cond).chunk(6, dim=1)

        B, T, D = x.shape
        residual = x
        
        x_mod = modulate_rmsnorm(x, self.norm1_weight, shift_msa, scale_msa)
        qkv = self.qkv(x_mod).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_attn if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, D)
        
        x = residual + gate_msa.unsqueeze(1) * self.proj_msa(attn_output)

        residual = x
        x_mod = modulate_rmsnorm(x, self.norm2_weight, shift_mlp, scale_mlp)
        ffn_output = self.mlp(x_mod)
        
        x = residual + gate_mlp.unsqueeze(1) * ffn_output
        return x

# ==========================================
# 4. 主干模型
# ==========================================
class DiTModel(nn.Module):
    def __init__(self,
            input_dim: int,
            horizon: int,
            cond_dim: int,
            n_layer: int = 8,
            n_emb: int = 256,
            cond_seq_len: int = 2,
            **kwargs
        ):
        super().__init__()
        self.n_emb = n_emb
        self.horizon = horizon
        self.cond_seq_len = cond_seq_len

        self.action_emb = nn.Linear(input_dim, n_emb)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))

        self.cond_encoder = GlobalCondEncoder(
            cond_dim=cond_dim,
            n_emb=n_emb,
            cond_seq_len=cond_seq_len
        )

        self.time_emb_net = TimestepEmbedder(hidden_size=n_emb)
        self.h_emb_net = TimestepEmbedder(hidden_size=n_emb)


        self.blocks = nn.ModuleList([
            TransformerBlock(n_emb, kwargs.get('n_head', 8)) for _ in range(n_layer)
        ])
        
        self.final_norm_weight = nn.Parameter(torch.ones(n_emb))
        self.final_ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_emb, 2 * n_emb, bias=True)
        )
        self.final_linear = nn.Linear(n_emb, input_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        nn.init.normal_(self.action_pos_emb, std=0.02)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.ada_modulation[-1].weight, 0)
            nn.init.constant_(block.ada_modulation[-1].bias, 0)
        nn.init.constant_(self.final_ada_mod[-1].weight, 0)
        nn.init.constant_(self.final_ada_mod[-1].bias, 0)

    def forward(self, sample, timestep, r=None, cond=None, global_cond=None, **kwargs):
        if cond is None and global_cond is not None:
            cond = global_cond
            
        if r is None:
            r = torch.zeros_like(timestep)

        # 1. 提取各个独立特征，它们各自的网络输出均已经为 n_emb 维度
        t_emb = self.time_emb_net(timestep)            # [B, n_emb]
        h_emb = self.h_emb_net(timestep - r)           # [B, n_emb]
        cond_emb = self.cond_encoder(cond)             # [B, n_emb]

        # 2. 直接在特征维度上进行逐元素相加融合
        c = t_emb + h_emb + cond_emb

        # 3. 动作序列编码
        seq = self.action_emb(sample) + self.action_pos_emb

        # 4. Transformer 推理 (通过 AdaLN 注入相加后的特征)
        for block in self.blocks:
            seq = block(seq, c, attn_mask=None)

        # 5. 提取输出
        shift, scale = self.final_ada_mod(c).chunk(2, dim=1)
        action_out = modulate_rmsnorm(seq, self.final_norm_weight, shift, scale)
        pred = self.final_linear(action_out)
        
        return pred