"""
surrogate_model/model.py
------------------------
Residual MLP dynamics model for the LPBF digital twin.

Architecture
------------
  Input  : [s_t (1053,) || a_t (1,)]  — both normalised before entering
  Hidden : depth × (Linear → LayerNorm → SiLU) blocks
  Output : Δ (1053,)  →  ŝ_{t+1} = s_t + Δ

Using a *delta / residual* output is critical for LPBF:
  • Most nodes change very little between layers (slow conduction).
  • The network only needs to learn the localised hot-spot delta,
    which is a much easier regression target than the full field.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Single hidden block with an optional skip connection when dimensions match.
    Layout:  x → Linear → LayerNorm → SiLU → Linear → LayerNorm → (+ skip) → SiLU
    """
    def __init__(self, dim: int, dropout: float = 0.0):
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
        return self.act(x + self.block(x))   # skip connection


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ResidualDynamicsModel(nn.Module):
    """
    Parameters
    ----------
    state_dim   : dimensionality of the flattened temperature field (1053)
    action_dim  : number of action dimensions (1 — laser power)
    hidden      : width of every hidden layer
    depth       : number of ResidualBlocks in the trunk
    dropout     : dropout probability inside each block (0 = off)

    Forward
    -------
    state  : (B, state_dim)  — normalised temperature field
    action : (B, action_dim) — normalised laser power
    returns: (B, state_dim)  — predicted *next* normalised temperature field
    """

    def __init__(
        self,
        state_dim:  int   = 1053,
        action_dim: int   = 1,
        hidden:     int   = 512,
        depth:      int   = 4,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim

        # ── input projection ──────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        # ── trunk: stack of residual blocks ───────────────────────────────
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden, dropout=dropout) for _ in range(depth)]
        )

        # ── output head: predict Δ (same size as state) ───────────────────
        self.output_head = nn.Linear(hidden, state_dim)

        # Initialise the output head near zero so early predictions are
        # close to pure pass-through (s_{t+1} ≈ s_t at random init).
        nn.init.zeros_(self.output_head.bias)
        nn.init.uniform_(self.output_head.weight, -1e-3, 1e-3)

    # ------------------------------------------------------------------
    def forward(
        self,
        state:  torch.Tensor,   # (B, 1053) normalised
        action: torch.Tensor,   # (B, 1)    normalised
    ) -> torch.Tensor:
        """Returns predicted next normalised temperature field (B, 1053)."""
        x     = torch.cat([state, action], dim=-1)   # (B, 1054)
        x     = self.input_proj(x)                   # (B, hidden)
        x     = self.trunk(x)                        # (B, hidden)
        delta = self.output_head(x)                  # (B, 1053)
        return state + delta                         # residual prediction

    # ------------------------------------------------------------------
    def predict_unnorm(
        self,
        state_raw:   torch.Tensor,   # (B, 1053) raw temperature [K]
        action_raw:  torch.Tensor,   # (B, 1)    raw laser power [W]
        state_mean:  torch.Tensor,   # (1053,)
        state_std:   torch.Tensor,   # (1053,)
        action_mean: float,
        action_std:  float,
    ) -> torch.Tensor:
        """
        Convenience wrapper: accepts raw (unnormalised) tensors,
        normalises internally, runs forward, and returns the prediction
        in the *original temperature scale* [K].
        """
        s_norm = (state_raw  - state_mean)  / state_std
        a_norm = (action_raw - action_mean) / action_std
        with torch.no_grad():
            out_norm = self.forward(s_norm, a_norm)
        return out_norm * state_std + state_mean

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ResidualDynamicsModel("
            f"state={self.state_dim}, action={self.action_dim}, "
            f"hidden={self.input_proj[0].out_features}, "
            f"depth={len(self.trunk)}, "
            f"params={self.count_parameters():,})"
        )
