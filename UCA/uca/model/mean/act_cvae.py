"""
ACT-style Transformer-CVAE action chunk decoder.

Observation tokens come from an external encoder (e.g. UCAEncoder).
Training uses a CVAE style latent; inference samples z ~ N(0, I).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ACTCVAEModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        cond_dim: int,
        n_obs_steps: int = 2,
        n_emb: int = 384,
        n_layer: int = 8,
        n_head: int = 8,
        n_enc_layers: int = 4,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.cond_dim = cond_dim
        self.n_obs_steps = n_obs_steps
        self.n_emb = n_emb
        self.latent_dim = latent_dim

        # ---- observation tokens ----
        self.obs_proj = nn.Sequential(
            nn.Linear(cond_dim, n_emb),
            nn.GELU(),
            nn.Linear(n_emb, n_emb),
        )
        self.obs_pos = nn.Parameter(torch.zeros(1, n_obs_steps, n_emb))

        # lightweight obs encoder (self-attn over obs tokens)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.obs_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # ---- CVAE style encoder q(z | o, a) ----
        style_in = n_emb + horizon * action_dim
        self.style_encoder = nn.Sequential(
            nn.Linear(style_in, n_emb),
            nn.GELU(),
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
        )
        self.style_mu = nn.Linear(n_emb, latent_dim)
        self.style_logvar = nn.Linear(n_emb, latent_dim)

        # ---- latent -> decoder conditioning ----
        self.latent_proj = nn.Linear(latent_dim, n_emb)

        # ---- action query decoder ----
        self.query_embed = nn.Parameter(torch.zeros(1, horizon, n_emb))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layer)
        self.action_head = nn.Sequential(
            nn.LayerNorm(n_emb),
            nn.Linear(n_emb, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.obs_pos, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_obs(self, obs_feat: torch.Tensor) -> torch.Tensor:
        """
        obs_feat: (B, To, cond_dim) from UCAEncoder (per-timestep)
        returns memory: (B, To, n_emb)
        """
        x = self.obs_proj(obs_feat) + self.obs_pos[:, : obs_feat.shape[1], :]
        return self.obs_encoder(x)

    def encode_style(
        self,
        memory: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        q(z | o, a). memory: (B, To, D), actions: (B, T, Da)
        """
        obs_pool = memory.mean(dim=1)
        a_flat = actions.reshape(actions.shape[0], -1)
        h = self.style_encoder(torch.cat([obs_pool, a_flat], dim=-1))
        mu = self.style_mu(h)
        logvar = self.style_logvar(h).clamp(-10.0, 10.0)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, memory: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        memory: (B, To, D), z: (B, latent_dim) -> actions (B, T, Da)
        """
        B = memory.shape[0]
        latent = self.latent_proj(z).unsqueeze(1)  # (B, 1, D)
        # prepend latent token to memory so decoder can cross-attend to style
        memory_z = torch.cat([latent, memory], dim=1)
        queries = self.query_embed.expand(B, -1, -1)
        hs = self.action_decoder(tgt=queries, memory=memory_z)
        return self.action_head(hs)

    def forward(
        self,
        obs_feat: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ):
        """
        Train: pass actions -> returns action_hat, mu, logvar
        Infer: leave actions=None, optional z; returns action_hat, None, None
        """
        memory = self.encode_obs(obs_feat)
        if actions is not None:
            z, mu, logvar = self.encode_style(memory, actions)
            action_hat = self.decode(memory, z)
            return action_hat, mu, logvar

        if z is None:
            z = torch.randn(
                obs_feat.shape[0], self.latent_dim,
                device=obs_feat.device, dtype=obs_feat.dtype,
            )
        action_hat = self.decode(memory, z)
        return action_hat, None, None


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|o,a) || N(0, I)), mean over batch."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
