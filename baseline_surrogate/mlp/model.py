"""
baseline_surrogate/mlp/model.py
-----------------------------------
Baseline 1: plain feedforward MLP. No two-stage decomposition, no latent
bottleneck, no ensemble — the floor baseline every other method (including
the main surrogate) should beat.

    s_{t+1} = s_t + MLP([s_t, a_t, cool_t, layer_embed])

Predicting the residual Delta-s (rather than s_{t+1} directly) is the one
concession to training stability shared with every other model in this
repo (main surrogate predicts Delta-z the same way) — everything else
(no residual blocks, no LayerNorm-heavy trunk, plain dropout-MLP) is kept
deliberately simple.
"""

import torch
import torch.nn as nn


class PlainMLPSurrogate(nn.Module):
    def __init__(
        self,
        state_dim:       int   = 1053,
        hidden:          int   = 512,
        depth:           int   = 4,
        n_layers:        int   = 12,
        layer_embed_dim: int   = 8,
        dropout:         float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.layer_embed = nn.Embedding(n_layers, layer_embed_dim)

        in_dim = state_dim + 1 + 1 + layer_embed_dim
        blocks = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            blocks += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*blocks)
        self.head  = nn.Linear(hidden, state_dim)

        # near-zero init so the model starts near the identity (Delta-s = 0),
        # matching the main surrogate's mu_head init rationale
        nn.init.uniform_(self.head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        s:         torch.Tensor,  # (B, state_dim) normalised
        a:         torch.Tensor,  # (B, 1)         normalised
        c:         torch.Tensor,  # (B, 1)         normalised
        layer_idx: torch.Tensor,  # (B,) int64
    ) -> torch.Tensor:
        e = self.layer_embed(layer_idx)
        x = torch.cat([s, a, c, e], dim=-1)
        delta = self.head(self.trunk(x))
        return s + delta

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
