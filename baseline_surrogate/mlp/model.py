"""
baseline_surrogate/mlp/model.py
-----------------------------------
Baseline 1: plain feedforward MLP. No two-stage decomposition, no latent
bottleneck, no ensemble, and — deliberately — none of the techniques
surrogate_model_v3 (the proposed method) itself relies on: no layer
embedding (conditioning is (s_t, a_t, cool_t) only) and no residual/delta
prediction (the network regresses s_{t+1} directly, not s_t + Delta). The
floor baseline every other method should beat, kept as plain as possible so
a win by the proposed method isn't attributable to borrowed tricks.

    s_{t+1} = MLP([s_t, a_t, cool_t])

The LayerNorm+SiLU+Dropout stacked trunk below is just generic network
plumbing (how to build a deep net), not a technique the proposed method
specifically contributes — kept for capacity, not "fanciness".
"""

import torch
import torch.nn as nn


class PlainMLPSurrogate(nn.Module):
    def __init__(
        self,
        state_dim: int   = 1053,
        hidden:    int   = 512,
        depth:     int   = 4,
        dropout:   float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim

        in_dim = state_dim + 1 + 1
        blocks = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            blocks += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*blocks)
        self.head  = nn.Linear(hidden, state_dim)

    def forward(
        self,
        s: torch.Tensor,  # (B, state_dim) normalised
        a: torch.Tensor,  # (B, 1)         normalised
        c: torch.Tensor,  # (B, 1)         normalised
    ) -> torch.Tensor:
        x = torch.cat([s, a, c], dim=-1)
        return self.head(self.trunk(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
