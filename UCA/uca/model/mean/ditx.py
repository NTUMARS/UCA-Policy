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
# 2. 条件编码器（现在只输出 cond_tokens）
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

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, n_emb),
            nn.GELU(),
            nn.Linear(n_emb, n_emb)
        )
        self.cond_pos_emb = nn.Parameter(torch.randn(1, cond_seq_len, n_emb) * 0.02)

        self.time_emb_net = TimestepEmbedder(hidden_size=n_emb)
        self.h_emb_net = TimestepEmbedder(hidden_size=n_emb)

        self.t_token_bias = nn.Parameter(torch.randn(num_t_tokens, n_emb) * 0.02)
        self.h_token_bias = nn.Parameter(torch.randn(num_h_tokens, n_emb) * 0.02)

    def forward(self, cond, timestep, r):
        B = cond.shape[0]

        cond_tokens = self.cond_proj(cond) + self.cond_pos_emb

        timestep = timestep.expand(B) if not torch.is_tensor(timestep) else timestep
        r = r.expand(B) if not torch.is_tensor(r) else r

        t_base = self.time_emb_net(timestep)
        h_base = self.h_emb_net(timestep - r)

        t_tokens = t_base.unsqueeze(1) + self.t_token_bias
        h_tokens = h_base.unsqueeze(1) + self.h_token_bias

        # 返回所有条件 token（共 4 个）
        prefix_sequence = torch.cat([t_tokens, h_tokens, cond_tokens], dim=1)
        return prefix_sequence


# ==========================================
# 3. 支持 Cross-Attention 的 Transformer Block
# ==========================================
class PureTransformerBlock(nn.Module):
    def __init__(self, n_emb, n_head, p_drop_attn=0.0, p_drop=0.0):
        super().__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        self.head_dim = n_emb // n_head

        # === Self-Attention ===
        self.norm1 = nn.LayerNorm(n_emb, eps=1e-6)
        self.qkv = nn.Linear(n_emb, 3 * n_emb, bias=True)
        self.proj_msa = nn.Linear(n_emb, n_emb, bias=True)
        self.dropout_attn = p_drop_attn

        # === Cross-Attention (新增) ===
        self.norm_cross = nn.LayerNorm(n_emb, eps=1e-6)
        self.q_proj_cross = nn.Linear(n_emb, n_emb, bias=True)
        self.kv_proj_cross = nn.Linear(n_emb, 2 * n_emb, bias=True)
        self.proj_cross = nn.Linear(n_emb, n_emb, bias=True)

        # === FFN ===
        self.norm2 = nn.LayerNorm(n_emb, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(p_drop),
        )

    def forward(self, x, cond_tokens=None, attn_mask=None):
        B, T, D = x.shape

        # === 1. Self-Attention ===
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

        # === 2. Cross-Attention (条件注入) ===
        if cond_tokens is not None:
            residual = x
            x_norm = self.norm_cross(x)

            # Query 来自主序列
            q_cross = self.q_proj_cross(x_norm)
            q_cross = q_cross.reshape(B, T, self.n_head, self.head_dim).permute(0, 2, 1, 3)

            # Key / Value 来自 cond_tokens
            kv_cross = self.kv_proj_cross(cond_tokens)
            k_cross, v_cross = kv_cross.chunk(2, dim=-1)

            T_cond = cond_tokens.shape[1]
            k_cross = k_cross.reshape(B, T_cond, self.n_head, self.head_dim).permute(0, 2, 1, 3)
            v_cross = v_cross.reshape(B, T_cond, self.n_head, self.head_dim).permute(0, 2, 1, 3)

            cross_attn_output = F.scaled_dot_product_attention(
                q_cross, k_cross, v_cross,
                dropout_p=0.0
            )
            cross_attn_output = cross_attn_output.transpose(1, 2).contiguous().reshape(B, T, D)
            x = residual + self.proj_cross(cross_attn_output)

        # === 3. FFN ===
        residual = x
        ffn_output = self.mlp(self.norm2(x))
        x = residual + ffn_output
        return x


# ==========================================
# 4. 主干模型 (Cross-Attention 版本)
# ==========================================
class DiTModel(nn.Module):
    def __init__(self,
            input_dim: int,
            horizon: int,
            cond_dim: int,
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
        self.cond_seq_len = cond_seq_len

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))

        # 条件编码器（输出 4 个 cond token）
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
                nn.init.constant_(module.proj_cross.weight, 0)   # 新增
                nn.init.constant_(module.proj_cross.bias, 0)     # 新增
                nn.init.constant_(module.mlp[2].weight, 0)
                nn.init.constant_(module.mlp[2].bias, 0)
        self.apply(_zero_out_residual_projs)

    def forward(self, sample, timestep, r=None, cond=None, global_cond=None, **kwargs):
        if cond is None and global_cond is not None:
            cond = global_cond
        if r is None:
            r = torch.zeros_like(timestep)

        # 主序列（不再拼接 prefix）
        action_tokens = self.input_emb(sample) + self.pos_emb   # (B, horizon, n_emb)

        # 生成条件 token（t + h + cond，共 4 个）
        cond_tokens = self.cond_encoder(cond, timestep, r)     # (B, 4, n_emb)

        # 通过所有 Transformer Block（使用 cross-attn 注入条件）
        seq = action_tokens
        for block in self.blocks:
            seq = block(seq, cond_tokens=cond_tokens)

        x = self.final_layer(seq)
        return x