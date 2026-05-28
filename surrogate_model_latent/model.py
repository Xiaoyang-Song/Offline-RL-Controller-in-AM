"""
surrogate_model_latent/model.py
--------------------------------
Latent-space dynamics model for the LPBF digital twin.

Architecture
------------
  Encoder      : s_t (1053,) → z_t (latent_dim,)   [deterministic]
  Transition   : (z_t, a_t)  → μ_Δz, log σ_Δz (latent_dim,)   [Gaussian MLP]
  Decoder      : z_t (latent_dim,) → ŝ_t (1053,)

Transition
----------
  z_{t+1} = z_t + μ_Δz + ε · exp(log σ_Δz),   ε ~ N(0, I)

  Reconstruction uses the deterministic (mean) path.
  Stochastic sampling is used during rollout / uncertainty estimation.

Loss terms (combined in train.py)
----------------------------------
  L_recon_st  : weighted MSE(decoder(encoder(s_t)), s_t)
  L_recon_st1 : weighted MSE(decoder(z_t + μ_Δz), s_{t+1})
  L_nll       : Gaussian NLL of (encoder(s_{t+1}) − z_t) under N(μ_Δz, exp(2·log σ_Δz))

Physics-aware per-node weights are applied in train.py via a precomputed ROI
weight table indexed by build-layer, based on the LPBF heat-source position.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    x → Linear → LayerNorm → SiLU → Dropout → Linear → LayerNorm → (+ skip) → SiLU
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
        return self.act(x + self.block(x))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Deterministic encoder: s_t → z_t.

    Parameters
    ----------
    state_dim   : input state dimension (1053)
    latent_dim  : latent space dimension
    hidden      : hidden layer width
    depth       : number of ResidualBlocks in trunk
    dropout     : dropout probability (0 = off)
    """
    def __init__(
        self,
        state_dim:  int,
        latent_dim: int,
        hidden:     int   = 256,
        depth:      int   = 3,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.latent_dim = latent_dim

        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(depth)]
        )
        self.head = nn.Linear(hidden, latent_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, state_dim) → z: (B, latent_dim)"""
        return self.head(self.trunk(self.input_proj(s)))


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Decoder: z_t → ŝ_t.

    Parameters
    ----------
    latent_dim  : latent space dimension
    state_dim   : output state dimension (1053)
    hidden      : hidden layer width
    depth       : number of ResidualBlocks in trunk
    dropout     : dropout probability (0 = off)
    """
    def __init__(
        self,
        latent_dim: int,
        state_dim:  int,
        hidden:     int   = 256,
        depth:      int   = 3,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim  = state_dim

        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(depth)]
        )
        self.head = nn.Linear(hidden, state_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → ŝ: (B, state_dim)"""
        return self.head(self.trunk(self.input_proj(z)))


# ---------------------------------------------------------------------------
# Gaussian Transition MLP
# ---------------------------------------------------------------------------

class GaussianTransitionMLP(nn.Module):
    """
    Gaussian transition: (z_t, a_t) → (μ_Δz, log σ_Δz).

    The next latent is:
        z_{t+1} = z_t + μ_Δz + ε · exp(log σ_Δz),   ε ~ N(0, I)

    Parameters
    ----------
    latent_dim      : latent space dimension
    action_dim      : action dimension (1 — laser power)
    hidden          : hidden layer width
    depth           : number of ResidualBlocks in trunk
    dropout         : dropout probability (0 = off)
    log_sigma_min   : lower clamp for log σ (prevents sigma → 0 / NLL explosion)
    log_sigma_max   : upper clamp for log σ (prevents over-diffuse predictions)
    """
    def __init__(
        self,
        latent_dim:     int,
        action_dim:     int   = 1,
        hidden:         int   = 128,
        depth:          int   = 3,
        dropout:        float = 0.0,
        log_sigma_min:  float = -5.0,
        log_sigma_max:  float = 2.0,
    ):
        super().__init__()
        self.latent_dim    = latent_dim
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(depth)]
        )
        self.mu_head        = nn.Linear(hidden, latent_dim)
        self.log_sigma_head = nn.Linear(hidden, latent_dim)

        # Near-zero init so early predictions start near Δz = 0
        nn.init.uniform_(self.mu_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mu_head.bias)
        # Log-sigma starts at -1 → σ ≈ 0.37 (moderate initial uncertainty)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, -1.0)

    def forward(
        self, z: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: (B, latent_dim), a: (B, action_dim)
        Returns (mu_delta, log_sigma_delta), each (B, latent_dim).
        """
        x = torch.cat([z, a], dim=-1)
        h = self.trunk(self.input_proj(x))
        mu        = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma


# ---------------------------------------------------------------------------
# Full latent dynamics model
# ---------------------------------------------------------------------------

class LatentDynamicsModel(nn.Module):
    """
    Latent-space LPBF surrogate: Encoder → GaussianTransition → Decoder.

    Parameters
    ----------
    state_dim       : temperature field dimension (1053)
    action_dim      : laser power dimension (1)
    latent_dim      : latent space dimension
    enc_hidden      : encoder hidden width
    enc_depth       : encoder ResidualBlock depth
    trans_hidden    : transition MLP hidden width
    trans_depth     : transition MLP depth
    dec_hidden      : decoder hidden width
    dec_depth       : decoder ResidualBlock depth
    dropout         : shared dropout for all sub-networks
    log_sigma_min/max : bounds for transition log-sigma clamping
    """

    def __init__(
        self,
        state_dim:     int   = 1053,
        action_dim:    int   = 1,
        latent_dim:    int   = 64,
        enc_hidden:    int   = 256,
        enc_depth:     int   = 3,
        trans_hidden:  int   = 128,
        trans_depth:   int   = 3,
        dec_hidden:    int   = 256,
        dec_depth:     int   = 3,
        dropout:       float = 0.0,
        log_sigma_min: float = -5.0,
        log_sigma_max: float = 2.0,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            state_dim, latent_dim, enc_hidden, enc_depth, dropout
        )
        self.transition = GaussianTransitionMLP(
            latent_dim, action_dim, trans_hidden, trans_depth, dropout,
            log_sigma_min, log_sigma_max,
        )
        self.decoder = Decoder(
            latent_dim, state_dim, dec_hidden, dec_depth, dropout
        )

    # ------------------------------------------------------------------
    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def predict_delta(
        self, z: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu_delta, log_sigma_delta)."""
        return self.transition(z, a)

    # ------------------------------------------------------------------
    def forward(
        self,
        s_t:  torch.Tensor,                      # (B, state_dim) normalised
        a_t:  torch.Tensor,                      # (B, action_dim) normalised
        s_t1: Optional[torch.Tensor] = None,     # (B, state_dim) normalised
    ) -> Tuple:
        """
        Full forward pass.

        Returns
        -------
        s_t_recon       : (B, state_dim)    decoder(encoder(s_t))
        s_t1_pred       : (B, state_dim)    decoder(z_t + μ_Δz)  [deterministic]
        mu_delta        : (B, latent_dim)   predicted mean Δz
        log_sigma_delta : (B, latent_dim)   predicted log σ_Δz
        z_t             : (B, latent_dim)   encoded s_t
        z_t1_enc        : (B, latent_dim) or None  — encoder(s_t1), if provided
        """
        z_t                       = self.encoder(s_t)
        s_t_recon                 = self.decoder(z_t)
        mu_delta, log_sigma_delta = self.transition(z_t, a_t)
        z_t1_pred                 = z_t + mu_delta
        s_t1_pred                 = self.decoder(z_t1_pred)
        z_t1_enc                  = self.encoder(s_t1) if s_t1 is not None else None

        return s_t_recon, s_t1_pred, mu_delta, log_sigma_delta, z_t, z_t1_enc

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        s_0:          torch.Tensor,   # (B, state_dim) normalised
        actions:      torch.Tensor,   # (B, T, action_dim) normalised
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Auto-regressive rollout in latent space.

        Returns
        -------
        pred_states : (B, T, state_dim)   decoded prediction at each step
        pred_sigmas : (B, T, latent_dim)  transition σ at each step
        """
        B, T, _ = actions.shape
        z_t      = self.encoder(s_0)
        pred_states, pred_sigmas = [], []

        for t in range(T):
            a_t                       = actions[:, t, :]
            mu_delta, log_sigma_delta = self.transition(z_t, a_t)
            if deterministic:
                z_t = z_t + mu_delta
            else:
                eps = torch.randn_like(mu_delta)
                z_t = z_t + mu_delta + eps * torch.exp(log_sigma_delta)
            pred_states.append(self.decoder(z_t))
            pred_sigmas.append(torch.exp(log_sigma_delta))

        return torch.stack(pred_states, dim=1), torch.stack(pred_sigmas, dim=1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_unnorm(
        self,
        state_raw:   torch.Tensor,
        action_raw:  torch.Tensor,
        state_mean:  torch.Tensor,
        state_std:   torch.Tensor,
        action_mean: float,
        action_std:  float,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Accept raw [K] / [W] tensors, return predicted next state in raw [K]."""
        s_norm = (state_raw  - state_mean)  / state_std
        a_norm = (action_raw - action_mean) / action_std
        z_t    = self.encoder(s_norm)
        mu_delta, log_sigma_delta = self.transition(z_t, a_norm)
        if deterministic:
            z_t1 = z_t + mu_delta
        else:
            eps  = torch.randn_like(mu_delta)
            z_t1 = z_t + mu_delta + eps * torch.exp(log_sigma_delta)
        return self.decoder(z_t1) * state_std + state_mean

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        enc_p   = sum(p.numel() for p in self.encoder.parameters())
        trans_p = sum(p.numel() for p in self.transition.parameters())
        dec_p   = sum(p.numel() for p in self.decoder.parameters())
        return (
            f"LatentDynamicsModel("
            f"state={self.state_dim}, latent={self.latent_dim}, "
            f"action={self.action_dim} | "
            f"enc={enc_p:,}  trans={trans_p:,}  dec={dec_p:,} | "
            f"total={self.count_parameters():,})"
        )
