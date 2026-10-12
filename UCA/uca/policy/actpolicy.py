"""
ACT-style Transformer-CVAE policy.

Uses the existing UCAEncoder (point cloud + state) for observations.
Action chunk is decoded via CVAE Transformer (not MeanFlow / diffusion).
"""
from typing import Dict

import torch
import torch.nn.functional as F
from termcolor import cprint

from uca.common.model_util import print_params
from uca.common.pytorch_util import dict_apply
from uca.model.common.normalizer import LinearNormalizer
from uca.model.mean.act_cvae import ACTCVAEModel, kl_divergence
from uca.model.vision.pointnet_extractor import UCAEncoder
from uca.policy.base_policy import BasePolicy


class ACTPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        horizon,
        n_action_steps,
        n_obs_steps,
        n_layer=8,
        n_head=8,
        n_emb=384,
        n_enc_layers=4,
        latent_dim=32,
        kl_weight=10.0,
        dropout=0.1,
        encoder_output_dim=64,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="mlp",
        pointcloud_encoder_cfg=None,
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])

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
        obs_feature_dim = obs_encoder.output_shape()
        self.obs_feature_dim = obs_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[ACTPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[ACTPolicy] pointnet_type: {self.pointnet_type}", "yellow")
        cprint(f"[ACTPolicy] latent_dim={latent_dim}, kl_weight={kl_weight}", "yellow")

        self.model = ACTCVAEModel(
            action_dim=action_dim,
            horizon=horizon,
            cond_dim=obs_feature_dim,
            n_obs_steps=n_obs_steps,
            n_emb=n_emb,
            n_layer=n_layer,
            n_head=n_head,
            n_enc_layers=n_enc_layers,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kl_weight = float(kl_weight)
        self.kwargs = kwargs
        print_params(self)

    def _encode_obs(self, nobs: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        To = self.n_obs_steps
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)  # (B*To, Do)
        return nobs_features.reshape(batch_size, To, -1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps

        obs_feat = self._encode_obs(nobs, B)

        self.model.eval()
        # inference: z ~ N(0, I), single forward (no diffusion / flow steps)
        naction_pred, _, _ = self.model(obs_feat=obs_feat, actions=None, z=None)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        batch_size = nactions.shape[0]
        obs_feat = self._encode_obs(nobs, batch_size)

        action_hat, mu, logvar = self.model(obs_feat=obs_feat, actions=nactions)

        # ACT uses L1 reconstruction + weighted KL to N(0, I)
        recon_loss = F.l1_loss(action_hat, nactions)
        kl_loss = kl_divergence(mu, logvar)
        loss = recon_loss + self.kl_weight * kl_loss

        mse_val = F.mse_loss(action_hat.detach(), nactions)

        return loss, {
            "bc_loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "mse_val": mse_val.item(),
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
