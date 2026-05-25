"""
online_RL/model.py
------------------
Residual Q-network for discrete-action online RL on the LPBF environment.

Architecture
------------
  Input  : normalised temperature field  (D = 1053)
  Stem   : Linear(D → H) → LayerNorm(H) → SiLU
  Trunk  : depth × ResidualBlock(H)
  Head   : Linear(H → A)   — Q-values per action

Using residual blocks (same style as the surrogate model) provides stable
gradient flow through the high-dimensional (1053-D) state.  LayerNorm
prevents exploding activations without the dead-neuron risk of BatchNorm.

The output head is initialised near zero so early Q-value estimates start
close to zero rather than arbitrary large values.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block  (identical to surrogate_model/model.py for consistency)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Two-layer residual block with skip connection.
    Layout:  x → Linear → LN → SiLU → Dropout → Linear → LN → (+x) → SiLU
    """
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
# Q-network
# ---------------------------------------------------------------------------

class QNet(nn.Module):
    """
    Residual Q-network:  state → Q-values for each discrete action.

    Parameters
    ----------
    state_dim   : dimension of the (normalised) temperature field  (default 1053)
    num_actions : number of discrete laser-power actions           (default 26)
    hidden      : width of every hidden layer
    depth       : number of ResidualBlocks in the trunk
    dropout     : dropout probability inside each block (0 = off)
    """

    def __init__(
        self,
        state_dim:   int   = 1053,
        num_actions: int   = 26,
        hidden:      int   = 256,
        depth:       int   = 4,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim   = state_dim
        self.num_actions = num_actions
        self.hidden      = hidden
        self.depth       = depth

        # ── stem: project state dim → hidden ──────────────────────────────
        self.stem = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        # ── trunk: stack of residual blocks ───────────────────────────────
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden, dropout=dropout) for _ in range(depth)]
        )

        # ── output head: Q(s, a) for every a ──────────────────────────────
        self.head = nn.Linear(hidden, num_actions)
        # Small init so early Q-values are near zero (avoids huge initial targets)
        nn.init.uniform_(self.head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state  : (B, state_dim)  normalised temperature field
        returns: (B, num_actions) Q-values
        """
        x = self.stem(state)
        x = self.trunk(x)
        return self.head(x)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"QNet(state_dim={self.state_dim}, num_actions={self.num_actions}, "
            f"hidden={self.hidden}, depth={self.depth}, "
            f"params={self.count_parameters():,})"
        )
