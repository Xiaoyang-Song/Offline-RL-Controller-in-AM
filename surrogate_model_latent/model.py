"""
surrogate_model_latent/model.py
--------------------------------
Latent-space ensemble dynamics model for the LPBF digital twin.

Architecture
------------
  Encoder        : s_t (1053,) → z_t (latent_dim,)          [shared, deterministic]
  Transition_k   : (z_t, a_t)  → μ_Δz_k (latent_dim,)      [K independent MLPs]
  Decoder        : z_t (latent_dim,) → ŝ_t (1053,)          [shared, deterministic]

Ensemble Transition
-------------------
  K independent DeterministicTransitionMLP members share one Encoder and one Decoder.

  At inference:
    μ_mean  = mean_k(μ_Δz_k)   ← best next-latent prediction
    σ_epist = std_k( μ_Δz_k)   ← epistemic uncertainty (large = OOD)

  The ensemble std is a meaningful signal: all K members agree on in-distribution
  (z_t, a_t) and disagree on out-of-distribution inputs, giving the online RL
  controller a reliable "trust the model" flag.

Loss terms (combined in train.py)
----------------------------------
  L_recon_st   : MSE( decoder(encoder(s_t)), s_t )                  [autoencoder]
  L_recon_st1  : mean_k MSE( decoder(z_t + μ_Δz_k), s_{t+1} )      [per-member]
  L_rollout    : k-step autoregressive MSE using ensemble mean       [optional]

Physics-aware per-node weights are applied via a precomputed ROI weight table.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """x → Linear → LayerNorm → SiLU → Dropout → Linear → LayerNorm → (+ skip) → SiLU"""
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
    """Deterministic encoder: s_t → z_t."""
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
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.head  = nn.Linear(hidden, latent_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, state_dim) → z: (B, latent_dim)"""
        return self.head(self.trunk(self.input_proj(s)))


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Decoder: z_t → ŝ_t."""
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
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.head  = nn.Linear(hidden, state_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → ŝ: (B, state_dim)"""
        return self.head(self.trunk(self.input_proj(z)))


# ---------------------------------------------------------------------------
# Deterministic Transition MLP  (one ensemble member)
# ---------------------------------------------------------------------------

class DeterministicTransitionMLP(nn.Module):
    """
    Deterministic transition: (z_t, a_t) → μ_Δz.

    No σ head — epistemic uncertainty is captured at the ensemble level
    via disagreement across K members (std of K predictions).

    Parameters
    ----------
    latent_dim  : latent space dimension
    action_dim  : action dimension (1 — laser power)
    hidden      : hidden layer width
    depth       : number of ResidualBlocks
    dropout     : dropout probability (0 = off)
    """
    def __init__(
        self,
        latent_dim: int,
        action_dim: int   = 1,
        hidden:     int   = 128,
        depth:      int   = 3,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.trunk   = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.mu_head = nn.Linear(hidden, latent_dim)

        # Near-zero init so early predictions start near Δz = 0
        nn.init.uniform_(self.mu_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mu_head.bias)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim), a: (B, action_dim) → μ_Δz: (B, latent_dim)"""
        x = torch.cat([z, a], dim=-1)
        return self.mu_head(self.trunk(self.input_proj(x)))


# ---------------------------------------------------------------------------
# Ensemble latent dynamics model
# ---------------------------------------------------------------------------

class EnsembleLatentDynamicsModel(nn.Module):
    """
    Ensemble latent-space LPBF surrogate.

    Shared Encoder and Decoder with K independent DeterministicTransitionMLP
    members.  Epistemic uncertainty = std of K predictions at any (z_t, a_t).

    Parameters
    ----------
    state_dim    : temperature field dimension (1053)
    action_dim   : laser power dimension (1)
    latent_dim   : latent space dimension
    n_ensemble   : number of ensemble members K (default 5)
    enc_hidden / enc_depth     : encoder MLP width / depth
    trans_hidden / trans_depth : transition MLP width / depth per member
    dec_hidden / dec_depth     : decoder MLP width / depth
    dropout      : shared dropout for all sub-networks
    """

    def __init__(
        self,
        state_dim:    int   = 1053,
        action_dim:   int   = 1,
        latent_dim:   int   = 64,
        n_ensemble:   int   = 5,
        enc_hidden:   int   = 256,
        enc_depth:    int   = 3,
        trans_hidden: int   = 128,
        trans_depth:  int   = 3,
        dec_hidden:   int   = 256,
        dec_depth:    int   = 3,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.n_ensemble = n_ensemble

        self.encoder = Encoder(state_dim, latent_dim, enc_hidden, enc_depth, dropout)
        self.transitions = nn.ModuleList([
            DeterministicTransitionMLP(latent_dim, action_dim, trans_hidden, trans_depth, dropout)
            for _ in range(n_ensemble)
        ])
        self.decoder = Decoder(latent_dim, state_dim, dec_hidden, dec_depth, dropout)

    # ------------------------------------------------------------------
    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def predict_ensemble(
        self, z: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run all K transitions; return (mean_delta, std_delta).

        Returns
        -------
        mu_mean : (B, latent_dim)  ensemble mean Δz
        mu_std  : (B, latent_dim)  ensemble std  Δz  ← epistemic uncertainty
        """
        deltas = torch.stack([t(z, a) for t in self.transitions], dim=0)  # (K, B, D_z)
        return deltas.mean(0), deltas.std(0)

    # ------------------------------------------------------------------
    def forward(
        self,
        s_t:  torch.Tensor,                    # (B, state_dim)  normalised
        a_t:  torch.Tensor,                    # (B, action_dim) normalised
        s_t1: Optional[torch.Tensor] = None,   # unused, kept for API consistency
    ) -> Tuple:
        """
        Full forward pass.

        Returns
        -------
        s_t_recon  : (B, state_dim)        decoder(encoder(s_t))
        s_t1_pred  : (B, state_dim)        decoder(z_t + mean(μ_Δz_k))
        mu_deltas  : (K, B, latent_dim)    all K member delta predictions
        z_t        : (B, latent_dim)       encoded s_t
        """
        z_t       = self.encoder(s_t)
        s_t_recon = self.decoder(z_t)

        mu_deltas = torch.stack(
            [t(z_t, a_t) for t in self.transitions], dim=0
        )  # (K, B, latent_dim)

        z_t1_pred = z_t + mu_deltas.mean(0)
        s_t1_pred = self.decoder(z_t1_pred)

        return s_t_recon, s_t1_pred, mu_deltas, z_t

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        s_0:     torch.Tensor,   # (B, state_dim)       normalised
        actions: torch.Tensor,   # (B, T, action_dim)   normalised
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Auto-regressive rollout using ensemble mean for stepping.

        Returns
        -------
        pred_states    : (B, T, state_dim)  decoded predictions at each step
        epistemic_stds : (B, T)             mean latent-space ensemble std per step
                                            (large = ensemble disagrees = OOD region)
        """
        B, T, _ = actions.shape
        z_t = self.encoder(s_0)
        pred_states, epistemic_stds = [], []

        for t in range(T):
            a_t              = actions[:, t, :]
            mu_mean, mu_std  = self.predict_ensemble(z_t, a_t)
            z_t              = z_t + mu_mean
            pred_states.append(self.decoder(z_t))
            epistemic_stds.append(mu_std.mean(dim=-1))   # (B,) averaged over latent dims

        return torch.stack(pred_states, dim=1), torch.stack(epistemic_stds, dim=1)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accept raw [K] / [W] tensors; return (mean prediction, epistemic std) in raw [K].

        epistemic_std is in normalised latent space — use as a relative signal,
        not an absolute Kelvin value.
        """
        s_norm = (state_raw  - state_mean)  / state_std
        a_norm = (action_raw - action_mean) / action_std
        z_t    = self.encoder(s_norm)
        mu_mean, mu_std = self.predict_ensemble(z_t, a_norm)
        pred_raw = self.decoder(z_t + mu_mean) * state_std + state_mean
        return pred_raw, mu_std

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        enc_p   = sum(p.numel() for p in self.encoder.parameters())
        trans_p = sum(p.numel() for p in self.transitions[0].parameters())
        dec_p   = sum(p.numel() for p in self.decoder.parameters())
        total   = self.count_parameters()
        return (
            f"EnsembleLatentDynamicsModel("
            f"state={self.state_dim}, latent={self.latent_dim}, "
            f"action={self.action_dim}, K={self.n_ensemble} | "
            f"enc={enc_p:,}  trans×{self.n_ensemble}={trans_p * self.n_ensemble:,}  "
            f"dec={dec_p:,} | total={total:,})"
        )
