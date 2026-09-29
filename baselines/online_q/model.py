"""
baselines/online_q/model.py
--------------------------------
Discrete-action Q-network operating on the SAME observation the on-policy
baselines already use: obs = [z_t (raw normalised field) || layer_token ||
cool_time_token], exactly what online_RL_ucpg_v2.env.TwoStageLatentLPBFEnv's
step()/reset() return (see that module's `_make_obs`). No separate raw-field
normaliser (unlike baselines/offline_q, which never touches the surrogate
and fits its own standardisation from the offline dataset) — this baseline
trains BY INTERACTING with the surrogate environment, same as naive_pg /
UCPG v2, so it reuses the surrogate's own normalisation for free via obs.

Architecture mirrors online_RL_ucpg_v2.model.ContinuousLatentPolicyNet's
trunk (input projection + ResidualBlocks + learned per-layer embedding),
swapping the continuous Gaussian head for a discrete Q-head, so this is as
close to "the same network, but value-based instead of policy-gradient" as
offline_q/model.py's RawQNet is to online_RL_ucpg's (single-stage) policy.

Action grid: reuses baselines.offline_q.model.ACTION_GRID (100:10:400W, 31
choices, matching the training data's own discrete laser-power grid) rather
than redefining it, so both Q-learning baselines are directly comparable and
any future change to the grid only needs to happen in one place.
"""

import torch
import torch.nn as nn

from online_RL_ucpg_v2.model import ResidualBlock
from baselines.offline_q.model import ACTION_GRID   # re-exported for convenience

__all__ = ["ACTION_GRID", "LatentQNet", "LatentQController", "load_online_q_controller"]


class LatentQNet(nn.Module):
    """
    Q(obs, ·) over the discrete laser-power grid, built directly from the
    environment's own observation vector plus a learned per-layer embedding
    (obs already carries a continuous layer token, but embedding the exact
    integer layer index gives the net the same inductive bias
    ContinuousLatentPolicyNet has — see module docstring).

    Parameters
    ----------
    obs_dim         : environment observation dimension (latent_dim + 2)
    latent_dim      : raw (normalised) temperature-field dimension (env's z_t slice of obs)
    n_actions       : size of the discrete action grid (len(ACTION_GRID))
    hidden          : trunk width
    depth           : number of ResidualBlocks
    n_layers        : number of LPBF build layers (embedding table size)
    layer_embed_dim : learned per-layer embedding dimension
    """

    def __init__(
        self,
        obs_dim:         int,
        latent_dim:      int,
        n_actions:       int,
        hidden:          int = 128,
        depth:           int = 3,
        n_layers:        int = 12,
        layer_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.obs_dim    = obs_dim
        self.latent_dim = latent_dim
        self.n_actions  = n_actions
        self.n_layers   = n_layers

        self.layer_embed = nn.Embedding(n_layers, layer_embed_dim)
        # input = z (latent_dim) ‖ layer_embed ‖ cool_time_token (1) — the layer_token
        # slice of obs is dropped (redundant with the learned embedding, same as
        # offline_q/model.py's RawQNet never feeding the raw layer index in twice).
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + layer_embed_dim + 1, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(hidden) for _ in range(depth)])
        self.head  = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor, layer_idx: torch.Tensor) -> torch.Tensor:
        """
        obs       : (B, obs_dim) = [z_t ‖ layer_token ‖ cool_time_token], as produced by
                    TwoStageLatentLPBFEnv.reset()/step() (or StepContext.obs at eval time)
        layer_idx : (B,) int64 — the exact 0-indexed layer (ctx.layer / the training
                    loop's own `t`, not decoded from obs's continuous layer_token)
        -> (B, n_actions) Q-values
        """
        z          = obs[:, : self.latent_dim]
        cool_token = obs[:, -1:]
        e = self.layer_embed(layer_idx.clamp(0, self.n_layers - 1))
        x = torch.cat([z, e, cool_token], dim=-1)
        return self.head(self.trunk(self.input_proj(x)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"LatentQNet(obs_dim={self.obs_dim}, latent_dim={self.latent_dim}, "
                f"n_actions={self.n_actions}, n_layers={self.n_layers}, "
                f"params={self.count_parameters():,})")


class LatentQController:
    """Wraps a trained LatentQNet to the Harness's act(ctx) interface: greedy
    argmax over the discrete action grid, from ctx.obs directly (exactly
    what the environment produced during training — no separate normaliser,
    unlike offline_q's RawQController) and ctx.layer for the embedding."""

    def __init__(self, qnet: LatentQNet, action_grid, device: str = "cpu"):
        self.qnet        = qnet.to(device).eval()
        self.action_grid = torch.tensor(action_grid, dtype=torch.float32, device=device)
        self.device      = device

    @torch.no_grad()
    def act(self, ctx) -> float:
        obs   = torch.tensor(ctx.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        layer = torch.tensor([ctx.layer], dtype=torch.long, device=self.device)
        q = self.qnet(obs, layer).squeeze(0)
        return float(self.action_grid[q.argmax()].item())


def load_online_q_controller(checkpoint_path: str, device: str = "cpu") -> LatentQController:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    qnet = LatentQNet(**ckpt["model_config"])
    qnet.load_state_dict(ckpt["qnet_state_dict"])
    return LatentQController(qnet, ckpt["action_grid"], device=device)
