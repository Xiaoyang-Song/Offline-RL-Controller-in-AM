"""
online_RL_ucpg/model.py
--------------------------
Latent-space stochastic policy network for Uncertainty-Constrained Policy
Gradient: pi_theta: Z -> A.

Architecture
------------
  Input  : obs (latent_dim+1,) = latent state z_t (latent_dim) + float layer
           token in [0,1] (matches online_RL/model.py's QNet convention)
  Embed  : float layer token -> integer index -> nn.Embedding(n_layers, layer_embed_dim)
  Stem   : Linear(latent_dim + layer_embed_dim -> H) -> LayerNorm(H) -> SiLU
  Trunk  : depth x ResidualBlock(H)
  Head   : Linear(H -> A)  — categorical logits over the discrete action grid

The policy is intentionally categorical (discrete action grid), matching the
laser-power discretisation used throughout this repo (online_RL/env.py's
ACTION_LIST) — this makes "fraction of chosen actions above the surrogate's
training range" a simple, direct diagnostic (see evaluate.py), and makes
nabla_theta log pi_theta(a_t|s_t) exact (Categorical.log_prob), with no need
for a reparameterisation trick.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block  (identical to online_RL/model.py's ResidualBlock)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """x → Linear → LayerNorm → SiLU → Dropout → Linear → LayerNorm → (+ skip) → SiLU"""
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class LatentPolicyNet(nn.Module):
    """
    Categorical policy over a discrete laser-power grid, conditioned on the
    surrogate's latent state and a learned per-layer embedding.

    Parameters
    ----------
    latent_dim      : surrogate latent space dimension
    n_actions       : number of discrete laser-power choices
    hidden          : width of every hidden layer
    depth           : number of ResidualBlocks in the trunk
    dropout         : dropout probability inside each block (0 = off)
    n_layers        : number of LPBF build layers (sets embedding table size)
    layer_embed_dim : dimension of the learned layer embedding
    """

    def __init__(
        self,
        latent_dim:      int,
        n_actions:       int,
        hidden:          int   = 128,
        depth:           int   = 3,
        dropout:         float = 0.0,
        n_layers:        int   = 12,
        layer_embed_dim: int   = 8,
    ) -> None:
        super().__init__()
        self.latent_dim      = latent_dim
        self.n_actions       = n_actions
        self.hidden          = hidden
        self.depth           = depth
        self.n_layers        = n_layers
        self.layer_embed_dim = layer_embed_dim

        self.layer_embed = nn.Embedding(n_layers, layer_embed_dim)

        stem_in = latent_dim + layer_embed_dim
        self.stem = nn.Sequential(
            nn.Linear(stem_in, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden, dropout=dropout) for _ in range(depth)]
        )
        self.head = nn.Linear(hidden, n_actions)
        # Near-zero init → near-uniform initial categorical distribution,
        # i.e. maximum-entropy exploration at the start of training.
        nn.init.uniform_(self.head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs     : (B, latent_dim+1) — last column = normalised float layer token in [0,1]
        returns : (B, n_actions) categorical logits
        """
        z           = obs[:, :-1]                                            # (B, latent_dim)
        layer_float = obs[:, -1]                                             # (B,)
        layer_idx   = (layer_float * (self.n_layers - 1)).round().long()
        layer_idx   = layer_idx.clamp(0, self.n_layers - 1)
        e = self.layer_embed(layer_idx)                                      # (B, embed_dim)
        x = torch.cat([z, e], dim=-1)
        x = self.stem(x)
        x = self.trunk(x)
        return self.head(x)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"LatentPolicyNet(latent_dim={self.latent_dim}, n_actions={self.n_actions}, "
            f"hidden={self.hidden}, depth={self.depth}, "
            f"n_layers={self.n_layers}, layer_embed_dim={self.layer_embed_dim}, "
            f"params={self.count_parameters():,})"
        )
