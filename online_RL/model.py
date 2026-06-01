"""
online_RL/model.py
------------------
Residual Q-network for discrete-action online RL on the LPBF environment.

Architecture
------------
  Input  : obs (D+1,) = normalised temperature field (D=1053) + float layer token [0,1]
  Embed  : float layer token → integer index → nn.Embedding(n_layers, layer_embed_dim)
  Stem   : Linear(D + layer_embed_dim → H) → LayerNorm(H) → SiLU
  Trunk  : depth × ResidualBlock(H)
  Head   : Linear(H → A)   — Q-values per action

Layer embedding
---------------
  The observation carries a normalised float layer token (layer_idx / (n_layers-1)).
  The Q-net converts it to an integer index and looks up a learned embedding vector,
  replacing the raw scalar with a richer layer_embed_dim-dimensional representation.
  This mirrors the layer conditioning in the surrogate transition MLP and lets the
  Q-net learn layer-specific Q-value adjustments without increasing hidden width.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
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
    Residual Q-network with layer-index embedding.

    Parameters
    ----------
    state_dim       : dimension of the observation (D + 1, where the last
                      element is the normalised float layer token)
    num_actions     : number of discrete laser-power actions (default 26)
    hidden          : width of every hidden layer
    depth           : number of ResidualBlocks in the trunk
    dropout         : dropout probability inside each block (0 = off)
    n_layers        : number of LPBF build layers (sets embedding table size)
    layer_embed_dim : dimension of the learned layer embedding
    """

    def __init__(
        self,
        state_dim:       int   = 1054,   # 1053 field + 1 layer float token
        num_actions:     int   = 26,
        hidden:          int   = 256,
        depth:           int   = 4,
        dropout:         float = 0.0,
        n_layers:        int   = 12,
        layer_embed_dim: int   = 8,
    ) -> None:
        super().__init__()
        self.state_dim       = state_dim
        self.num_actions     = num_actions
        self.hidden          = hidden
        self.depth           = depth
        self.n_layers        = n_layers
        self.layer_embed_dim = layer_embed_dim

        self.field_dim = state_dim - 1          # D = 1053

        # Learned layer embedding — replaces the raw scalar with a richer vector
        self.layer_embed = nn.Embedding(n_layers, layer_embed_dim)

        stem_in = self.field_dim + layer_embed_dim   # D + embed_dim

        self.stem = nn.Sequential(
            nn.Linear(stem_in, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden, dropout=dropout) for _ in range(depth)]
        )
        self.head = nn.Linear(hidden, num_actions)
        nn.init.uniform_(self.head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state  : (B, state_dim)  = (B, D+1)
                 last column = normalised float layer token in [0, 1]
        returns: (B, num_actions) Q-values
        """
        field       = state[:, :-1]                                           # (B, D)
        layer_float = state[:, -1]                                            # (B,)
        # Convert normalised float → integer layer index, clamp for safety
        layer_idx = (layer_float * (self.n_layers - 1)).round().long()
        layer_idx = layer_idx.clamp(0, self.n_layers - 1)
        e = self.layer_embed(layer_idx)                                       # (B, embed_dim)
        x = torch.cat([field, e], dim=-1)                                     # (B, D+embed_dim)
        x = self.stem(x)
        x = self.trunk(x)
        return self.head(x)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"QNet(state_dim={self.state_dim}, num_actions={self.num_actions}, "
            f"hidden={self.hidden}, depth={self.depth}, "
            f"n_layers={self.n_layers}, layer_embed_dim={self.layer_embed_dim}, "
            f"params={self.count_parameters():,})"
        )
