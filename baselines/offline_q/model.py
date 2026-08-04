"""
baselines/offline_q/model.py
--------------------------------
Discrete-action Q-network operating on the RAW temperature field — no
surrogate, no latent space. Reuses ResidualBlock from online_RL_ucpg_v2.model
(read-only import) to avoid duplicating that one building block; everything
else here is specific to this baseline.

Action grid
-----------
Laser power was originally sampled from a discrete grid during data
generation (multilayer_random_v2.m: LP_values = 100:10:400 -> 31 choices).
Offline Q-learning uses that SAME grid — no discretisation choice to make,
since it already matches the actions actually present in the dataset.

cool_time as an explicit input
----------------------------------
Reward never depends on cool_time (it's computed from the end-of-heating
field, before cooling runs), but cool_time DOES determine s_{t+1} given
(s_t, a_t) — so from this agent's perspective, not observing it makes the
transition look stochastic (same (s,a) can land on different s' depending
on the episode's cool_time), and the Bellman target it bootstraps from
(max_a' Q(s',a')) is only ever learned as an average over whatever
cool_time happened to occur. Adding cool_time as an input (concatenated
directly, like the RL policy's cool_time token — see
online_RL_ucpg_v2/model.py) lets Q depend on the ACTUAL realized cool_time
per trajectory rather than marginalising over it, closing that gap.
"""

import numpy as np
import torch
import torch.nn as nn

from online_RL_ucpg_v2.model import ResidualBlock

ACTION_GRID = np.arange(100.0, 401.0, 10.0, dtype=np.float32)   # 31 discrete laser powers [W]


class RawQNet(nn.Module):
    """
    Q(s_t, layer_t, ·) over the discrete laser-power grid, from the RAW
    (standardised) temperature field — no encoder, no latent space.

    Parameters
    ----------
    state_dim       : raw temperature field dimension (1053)
    n_actions       : size of the discrete action grid (len(ACTION_GRID))
    hidden          : trunk width
    depth           : number of ResidualBlocks
    n_layers        : number of LPBF build layers (embedding table size)
    layer_embed_dim : learned per-layer embedding dimension
    """

    def __init__(
        self,
        state_dim:       int,
        n_actions:       int,
        hidden:          int = 256,
        depth:           int = 3,
        n_layers:        int = 12,
        layer_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_layers  = n_layers

        self.layer_embed = nn.Embedding(n_layers, layer_embed_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim + layer_embed_dim + 1, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(hidden) for _ in range(depth)])
        self.head  = nn.Linear(hidden, n_actions)

    def forward(self, s: torch.Tensor, layer_idx: torch.Tensor, cool_time: torch.Tensor) -> torch.Tensor:
        """
        s         : (B, state_dim) standardised raw field
        layer_idx : (B,) int64
        cool_time : (B,) standardised cool_time (this trajectory's fixed value)
        -> (B, n_actions) Q-values
        """
        e = self.layer_embed(layer_idx.clamp(0, self.n_layers - 1))
        x = torch.cat([s, e, cool_time.unsqueeze(-1)], dim=-1)
        return self.head(self.trunk(self.input_proj(x)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"RawQNet(state_dim={self.state_dim}, n_actions={self.n_actions}, "
                f"n_layers={self.n_layers}, params={self.count_parameters():,})")


class RawQController:
    """Wraps a trained RawQNet to the Harness's act(ctx) interface: greedy
    argmax over the discrete action grid, from ctx.raw_state (decoded,
    denormalised — this baseline standardises it with its OWN mean/std,
    fit on the offline dataset, never the surrogate's normalisation) AND
    ctx.cool_time (this episode's actual realised value, standardised the
    same way)."""

    def __init__(self, qnet: RawQNet, state_mean: np.ndarray, state_std: np.ndarray,
                cool_mean: float, cool_std: float,
                action_grid: np.ndarray, device: str = "cpu"):
        self.qnet        = qnet.to(device).eval()
        self.state_mean  = torch.tensor(state_mean, dtype=torch.float32, device=device)
        self.state_std   = torch.tensor(state_std,  dtype=torch.float32, device=device)
        self.cool_mean   = cool_mean
        self.cool_std    = cool_std
        self.action_grid = torch.tensor(action_grid, dtype=torch.float32, device=device)
        self.device      = device

    @torch.no_grad()
    def act(self, ctx) -> float:
        s = torch.tensor(ctx.raw_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        s_n = (s - self.state_mean) / self.state_std
        layer = torch.tensor([ctx.layer], dtype=torch.long, device=self.device)
        c_n = torch.tensor([(ctx.cool_time - self.cool_mean) / self.cool_std],
                           dtype=torch.float32, device=self.device)
        q = self.qnet(s_n, layer, c_n).squeeze(0)
        return float(self.action_grid[q.argmax()].item())


def load_offline_q_controller(checkpoint_path: str, device: str = "cpu") -> RawQController:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]
    qnet = RawQNet(**cfg)
    qnet.load_state_dict(ckpt["qnet_state_dict"])
    return RawQController(
        qnet, ckpt["state_mean"], ckpt["state_std"],
        ckpt["cool_mean"], ckpt["cool_std"], ckpt["action_grid"], device=device,
    )
