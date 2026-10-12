import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ==========================================
# 1. 基础组件 (PosEmb & Timestep & Final)
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
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t):
        return self.mlp(self.freq_embed(t))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x)


# ==========================================
# 2. 简化版前缀编码器（适配 UCAEncoder）
# ==========================================
class PrefixConditionEncoder(nn.Module):
    def __init__(self, cond_dim, n_emb, 
                 cond_seq_len=2, 
                 num_t_tokens=1, 
                 num_h_tokens=1):
        super().__init__()
        self.n_emb = n_emb
        self.cond_seq_len = cond_seq_len
        self.num_t_tokens = num_t_tokens
        self.num_h_tokens = num_h_tokens

        # 核心改动：只保留一个简单的 MLP 映射
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, n_emb),
            nn.GELU(),
            nn.Linear(n_emb, n_emb)
        )
        self.cond_pos_emb = nn.Parameter(torch.randn(1, cond_seq_len, n_emb) * 0.02)

        # 时间特征提取器
        self.time_emb_net = TimestepEmbedder(hidden_size=n_emb)
        self.h_emb_net = TimestepEmbedder(hidden_size=n_emb)

        # 多 Token 身份偏置
        self.t_token_bias = nn.Parameter(torch.randn(num_t_tokens, n_emb) * 0.02)
        self.h_token_bias = nn.Parameter(torch.randn(num_h_tokens, n_emb) * 0.02)

    def forward(self, cond, timestep, r):
        """
        cond: (B, cond_seq_len, cond_dim)   ← 现在是单一 tensor
        """
        B = cond.shape[0]

        # === 1. 观测条件 Token（简单 MLP 映射）===
        cond_tokens = self.cond_proj(cond) + self.cond_pos_emb   # (B, cond_seq_len, n_emb)

        # === 2. 时间 Token ===
        timestep = timestep.expand(B) if not torch.is_tensor(timestep) else timestep
        r = r.expand(B) if not torch.is_tensor(r) else r

        t_base = self.time_emb_net(timestep)
        h_base = self.h_emb_net(timestep - r)

        t_tokens = t_base.unsqueeze(1) + self.t_token_bias
        h_tokens = h_base.unsqueeze(1) + self.h_token_bias

        # === 3. 拼接前缀 ===
        prefix_sequence = torch.cat([t_tokens, h_tokens, cond_tokens], dim=1)
        return prefix_sequence


# ==========================================
# 3. 纯粹的自注意力块
# ==========================================
class PureTransformerBlock(nn.Module):
    def __init__(self, n_emb, n_head, p_drop_attn=0.0, p_drop=0.0):
        super().__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        self.head_dim = n_emb // n_head

        self.norm1 = nn.LayerNorm(n_emb, eps=1e-6)
        self.qkv = nn.Linear(n_emb, 3 * n_emb, bias=True)
        self.proj_msa = nn.Linear(n_emb, n_emb, bias=True)
        self.dropout_attn = p_drop_attn

        self.norm2 = nn.LayerNorm(n_emb, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(p_drop),
        )

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_attn if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, D)
        x = residual + self.proj_msa(attn_output)

        residual = x
        ffn_output = self.mlp(self.norm2(x))
        x = residual + ffn_output
        return x


# ==========================================
# 4. 主干模型 (已适配 cond_dim)
# ==========================================
class DiTModel(nn.Module):
    def __init__(self,
            input_dim: int,
            horizon: int,
            cond_dim: int,                    # ← 新参数：替换原来的 pc_dim + state_dim
            n_layer: int = 8,
            n_emb: int = 256,
            num_t_tokens: int = 1,
            num_h_tokens: int = 1,
            cond_seq_len: int = 2,
            **kwargs
        ):
        super().__init__()
        self.n_emb = n_emb
        self.horizon = horizon
        self.prefix_len = num_t_tokens + num_h_tokens + cond_seq_len
        self.cond_seq_len = cond_seq_len

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))

        # 使用新的 PrefixConditionEncoder
        self.cond_encoder = PrefixConditionEncoder(
            cond_dim=cond_dim,
            n_emb=n_emb,
            cond_seq_len=cond_seq_len,
            num_t_tokens=num_t_tokens,
            num_h_tokens=num_h_tokens
        )

        self.blocks = nn.ModuleList([
            PureTransformerBlock(n_emb, kwargs.get('n_head', 8)) for _ in range(n_layer)
        ])
        self.final_layer = FinalLayer(n_emb, input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        nn.init.normal_(self.input_emb.weight, std=0.02)
        if self.input_emb.bias is not None:
            nn.init.zeros_(self.input_emb.bias)
        nn.init.normal_(self.pos_emb, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        def _zero_out_residual_projs(module):
            if isinstance(module, PureTransformerBlock):
                nn.init.constant_(module.proj_msa.weight, 0)
                nn.init.constant_(module.proj_msa.bias, 0)
                nn.init.constant_(module.mlp[2].weight, 0)
                nn.init.constant_(module.mlp[2].bias, 0)
        self.apply(_zero_out_residual_projs)

    # def _create_custom_mask(self, seq_len, device):
    #     prefix_len = self.prefix_len
    #     horizon = seq_len - prefix_len
        
    #     # 初始化全 False 矩阵
    #     mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        
    #     # === 1. Cond 内部：不互相通信，只关注自己 ===
    #     # 生成一个对角线为 True 的矩阵 (Identity Matrix)
    #     mask[:prefix_len, :prefix_len] = torch.eye(prefix_len, dtype=torch.bool, device=device)
        
    #     # === 2. Cond 看 Action：禁止 ===
    #     # mask[:prefix_len, prefix_len:] = False (默认已经是 False)
        
    #     # === 3. Action 看 Cond：完全允许 ===
    #     # 所有的 Action 都可以完整地看到 t, r 和 cond
    #     mask[prefix_len:, :prefix_len] = True
        
    #     # === 4. Action 内部：因果掩码 (Causal Mask) ===
    #     # 生成下三角矩阵，防止死记硬背未来的轨迹
    #     causal_mask = torch.tril(torch.ones((horizon, horizon), dtype=torch.bool, device=device))
    #     mask[prefix_len:, prefix_len:] = causal_mask
        
    #     # 增加 Batch 和 Head 维度以匹配 F.scaled_dot_product_attention (1, 1, seq_len, seq_len)
    #     return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, sample, timestep, r=None, cond=None, global_cond=None, **kwargs):
        if cond is None and global_cond is not None:
            cond = global_cond
        if r is None:
            r = torch.zeros_like(timestep)

        action_tokens = self.input_emb(sample) + self.pos_emb
        prefix_tokens = self.cond_encoder(cond, timestep, r)
        seq = torch.cat([prefix_tokens, action_tokens], dim=1)

        # attn_mask = self._create_custom_mask(seq_len=seq.shape[1], device=seq.device)

        for block in self.blocks:
            seq = block(seq, attn_mask=None)

        action_output = seq[:, self.prefix_len:, :]
        x = self.final_layer(action_output)
        return x