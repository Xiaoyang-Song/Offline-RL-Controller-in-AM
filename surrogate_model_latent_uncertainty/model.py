"""
surrogate_model_latent_uncertainty/model.py
--------------------------------------------
Latent-space ensemble dynamics model for the LPBF digital twin, with a
Gaussian transition head per ensemble member so the model captures both
aleatoric and epistemic uncertainty (extends surrogate_model_latent/model.py).

Architecture
------------
  Encoder        : s_t (1053,) → z_t (latent_dim,)               [shared, deterministic]
  Transition_k   : (z_t, a_t, embed(layer_idx)) → (μ_Δz_k, σ_Δz_k)  [K Gaussian MLPs]
  Decoder        : z_t (latent_dim,) → ŝ_t (1053,)               [shared, deterministic]

Layer-Index Embedding
---------------------
  Unchanged from surrogate_model_latent: each transition member has a learned
  nn.Embedding(n_layers, layer_embed_dim), concatenated to (z_t, a_t).

Gaussian Transition Head (aleatoric uncertainty)
-------------------------------------------------
  Each member now predicts a diagonal Gaussian over the latent delta instead
  of a point estimate:
      Δz_k ~ N(μ_k(z_t, a_t), diag(σ_k(z_t, a_t)^2))
  σ_k is bounded with the PETS-style learnable soft clamp (Chua et al., 2018)
  to keep the Gaussian NLL numerically stable without a hard, zero-gradient
  clamp.

Bootstrap Ensemble (epistemic uncertainty)
-------------------------------------------
  Each member is trained on an independent bootstrap resample of the training
  set (see dataset.py: make_bootstrap_masks). Members therefore disagree more
  in regions with limited data coverage — their mean-prediction spread is the
  epistemic signal, exactly as in the deterministic ensemble, but now combined
  with each member's own aleatoric variance via mixture-of-Gaussians moment
  matching (Lakshminarayanan et al., 2017):

      μ_mean         = mean_k(μ_k)                          ← next-latent step
      σ_epist^2      = var_k(μ_k)                            ← epistemic (disagreement)
      σ_aleat^2      = mean_k(σ_k^2)                          ← aleatoric (avg. member noise)
      σ_total^2      = σ_epist^2 + σ_aleat^2                  ← total predictive variance

Loss terms (combined in train.py)
----------------------------------
  L_recon_st   : MSE( decoder(encoder(s_t)), s_t )                    [autoencoder]
  L_recon_st1  : mean_k MSE( decoder(z_t + μ_Δz_k), s_{t+1} )        [per-member, bootstrap-weighted]
  L_nll        : mean_k GaussianNLL( μ_Δz_k, σ_Δz_k ; Δz_target )    [per-member, bootstrap-weighted]
  L_rollout    : k-step autoregressive MSE using ensemble mean         [optional]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Gaussian Transition MLP  (one ensemble member)
# ---------------------------------------------------------------------------

class GaussianTransitionMLP(nn.Module):
    """
    Gaussian transition: (z_t, a_t, embed(layer_idx)) → (μ_Δz, σ_Δz).

    Predicts a diagonal Gaussian over the latent delta, capturing aleatoric
    (irreducible, per-input) uncertainty. Combined with bootstrap-resampled
    training across the K ensemble members (see dataset.py), disagreement
    between members' means captures epistemic uncertainty.

    log σ is bounded with a learnable soft clamp (Chua et al., "Deep RL in a
    Handful of Trials using Probabilistic Dynamics Models", PETS, 2018) so the
    Gaussian NLL loss stays finite and well-conditioned throughout training —
    a hard clamp has zero gradient outside the bounds and can get the head
    permanently stuck saturated.

    Parameters
    ----------
    latent_dim      : latent space dimension
    action_dim      : action dimension (1 — laser power)
    hidden          : hidden layer width
    depth           : number of ResidualBlocks
    dropout         : dropout probability (0 = off)
    n_layers        : total number of build layers (sets embedding table size)
    layer_embed_dim : dimension of the learned layer embedding
    """
    def __init__(
        self,
        latent_dim:      int,
        action_dim:      int   = 1,
        hidden:          int   = 128,
        depth:           int   = 3,
        dropout:         float = 0.0,
        n_layers:        int   = 12,
        layer_embed_dim: int   = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.layer_embed = nn.Embedding(n_layers, layer_embed_dim)

        in_dim = latent_dim + action_dim + layer_embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.trunk           = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.mu_head         = nn.Linear(hidden, latent_dim)
        self.log_sigma_head  = nn.Linear(hidden, latent_dim)

        # Near-zero init so early predictions start near Δz = 0, σ ≈ near-bound
        nn.init.uniform_(self.mu_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.uniform_(self.log_sigma_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.log_sigma_head.bias)

        # PETS-style learnable soft bounds on log σ (per latent dimension)
        self.max_log_sigma = nn.Parameter(torch.full((latent_dim,),  0.5))
        self.min_log_sigma = nn.Parameter(torch.full((latent_dim,), -5.0))

    def forward(
        self,
        z:         torch.Tensor,   # (B, latent_dim)
        a:         torch.Tensor,   # (B, action_dim)
        layer_idx: torch.Tensor,   # (B,) int64, 0-indexed
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """→ (μ_Δz, log σ_Δz), each (B, latent_dim)"""
        e = self.layer_embed(layer_idx)          # (B, layer_embed_dim)
        x = torch.cat([z, a, e], dim=-1)
        h = self.trunk(self.input_proj(x))

        mu             = self.mu_head(h)
        raw_log_sigma  = self.log_sigma_head(h)

        # Soft clamp: log_sigma ∈ (min_log_sigma, max_log_sigma), differentiable everywhere
        log_sigma = self.max_log_sigma - F.softplus(self.max_log_sigma - raw_log_sigma)
        log_sigma = self.min_log_sigma + F.softplus(log_sigma - self.min_log_sigma)

        return mu, log_sigma


# ---------------------------------------------------------------------------
# Ensemble latent dynamics model (Gaussian, bootstrap-trained)
# ---------------------------------------------------------------------------

class EnsembleGaussianLatentDynamicsModel(nn.Module):
    """
    Ensemble latent-space LPBF surrogate with layer-index conditioning,
    Gaussian transition heads, and bootstrap-resampled training.

    Shared Encoder and Decoder with K independent GaussianTransitionMLP
    members. Each member receives a learned layer embedding so it can express
    layer-specific dynamics, and predicts a full diagonal Gaussian so it can
    express per-input (aleatoric) noise in addition to the ensemble's
    between-member (epistemic) disagreement.

    Parameters
    ----------
    state_dim       : temperature field dimension (1053)
    action_dim      : laser power dimension (1)
    latent_dim      : latent space dimension
    n_ensemble      : number of ensemble members K (default 5)
    n_layers        : number of build layers — sets embedding table size (default 12)
    layer_embed_dim : dimension of learned layer embedding per member (default 8)
    enc_hidden / enc_depth     : encoder MLP width / depth
    trans_hidden / trans_depth : transition MLP width / depth per member
    dec_hidden / dec_depth     : decoder MLP width / depth
    dropout         : shared dropout for all sub-networks
    """

    def __init__(
        self,
        state_dim:       int   = 1053,
        action_dim:      int   = 1,
        latent_dim:      int   = 64,
        n_ensemble:      int   = 5,
        n_layers:        int   = 12,
        layer_embed_dim: int   = 8,
        enc_hidden:      int   = 256,
        enc_depth:       int   = 3,
        trans_hidden:    int   = 128,
        trans_depth:     int   = 3,
        dec_hidden:      int   = 256,
        dec_depth:       int   = 3,
        dropout:         float = 0.0,
    ):
        super().__init__()
        self.state_dim       = state_dim
        self.action_dim      = action_dim
        self.latent_dim      = latent_dim
        self.n_ensemble      = n_ensemble
        self.n_layers        = n_layers
        self.layer_embed_dim = layer_embed_dim

        self.encoder = Encoder(state_dim, latent_dim, enc_hidden, enc_depth, dropout)
        self.transitions = nn.ModuleList([
            GaussianTransitionMLP(
                latent_dim, action_dim, trans_hidden, trans_depth, dropout,
                n_layers, layer_embed_dim,
            )
            for _ in range(n_ensemble)
        ])
        self.decoder = Decoder(latent_dim, state_dim, dec_hidden, dec_depth, dropout)

    # ------------------------------------------------------------------
    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def _run_transitions(
        self,
        z:         torch.Tensor,
        a:         torch.Tensor,
        layer_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run all K Gaussian transitions. → (mu_deltas, log_sigma_deltas), each (K, B, latent_dim)."""
        mus, log_sigmas = [], []
        for t in self.transitions:
            mu, log_sigma = t(z, a, layer_idx)
            mus.append(mu)
            log_sigmas.append(log_sigma)
        return torch.stack(mus, dim=0), torch.stack(log_sigmas, dim=0)

    def predict_ensemble(
        self,
        z:         torch.Tensor,   # (B, latent_dim)
        a:         torch.Tensor,   # (B, action_dim)
        layer_idx: torch.Tensor,   # (B,) int64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run all K Gaussian transitions and combine via mixture-of-Gaussians
        moment matching (Lakshminarayanan et al., 2017).

        Returns
        -------
        mu_mean       : (B, latent_dim)  ensemble mean Δz          ← used to step z_t
        epistemic_std : (B, latent_dim)  std across member means   ← epistemic uncertainty (large = OOD)
        aleatoric_std : (B, latent_dim)  sqrt(mean member variance) ← aleatoric uncertainty (inherent noise)
        total_std     : (B, latent_dim)  sqrt(epistemic² + aleatoric²)
        """
        mu_deltas, log_sigma_deltas = self._run_transitions(z, a, layer_idx)
        var_deltas = (2.0 * log_sigma_deltas).exp()          # (K, B, latent_dim)

        mu_mean       = mu_deltas.mean(0)
        epistemic_var = mu_deltas.var(0, unbiased=False)
        aleatoric_var = var_deltas.mean(0)
        total_var     = epistemic_var + aleatoric_var

        return (
            mu_mean,
            epistemic_var.clamp_min(0.0).sqrt(),
            aleatoric_var.clamp_min(0.0).sqrt(),
            total_var.clamp_min(0.0).sqrt(),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        s_t:       torch.Tensor,                    # (B, state_dim)
        a_t:       torch.Tensor,                    # (B, action_dim)
        layer_idx: torch.Tensor,                    # (B,) int64
        s_t1:      Optional[torch.Tensor] = None,   # unused, kept for API consistency
    ) -> Tuple:
        """
        Full forward pass.

        Returns
        -------
        s_t_recon        : (B, state_dim)        decoder(encoder(s_t))
        s_t1_pred        : (B, state_dim)        decoder(z_t + mean(μ_Δz_k))
        mu_deltas        : (K, B, latent_dim)    all K member delta means
        log_sigma_deltas : (K, B, latent_dim)    all K member delta log-std
        z_t              : (B, latent_dim)       encoded s_t
        """
        z_t       = self.encoder(s_t)
        s_t_recon = self.decoder(z_t)

        mu_deltas, log_sigma_deltas = self._run_transitions(z_t, a_t, layer_idx)

        z_t1_pred = z_t + mu_deltas.mean(0)
        s_t1_pred = self.decoder(z_t1_pred)

        return s_t_recon, s_t1_pred, mu_deltas, log_sigma_deltas, z_t

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        s_0:     torch.Tensor,   # (B, state_dim)       normalised
        actions: torch.Tensor,   # (B, T, action_dim)   normalised
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Auto-regressive rollout using ensemble mean for stepping.
        Layer indices are generated automatically: step t → layer index t.

        Returns
        -------
        pred_states    : (B, T, state_dim)
        epistemic_stds : (B, T)   mean latent-space epistemic std per step
        aleatoric_stds : (B, T)   mean latent-space aleatoric std per step
        """
        B, T, _ = actions.shape
        device   = s_0.device
        z_t      = self.encoder(s_0)
        pred_states, epistemic_stds, aleatoric_stds = [], [], []

        for t in range(T):
            a_t        = actions[:, t, :]
            layer_idx  = torch.full((B,), t, dtype=torch.long, device=device)
            mu_mean, epi_std, ale_std, _ = self.predict_ensemble(z_t, a_t, layer_idx)
            z_t        = z_t + mu_mean
            pred_states.append(self.decoder(z_t))
            epistemic_stds.append(epi_std.mean(dim=-1))   # (B,)
            aleatoric_stds.append(ale_std.mean(dim=-1))    # (B,)

        return (
            torch.stack(pred_states, dim=1),
            torch.stack(epistemic_stds, dim=1),
            torch.stack(aleatoric_stds, dim=1),
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_unnorm(
        self,
        state_raw:   torch.Tensor,
        action_raw:  torch.Tensor,
        layer_idx:   torch.Tensor,   # (B,) int64
        state_mean:  torch.Tensor,
        state_std:   torch.Tensor,
        action_mean: float,
        action_std:  float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Accept raw [K]/[W] tensors; return (mean prediction [K], epistemic std,
        aleatoric std, total std). Uncertainties are reported in normalised
        latent space (consistent with surrogate_model_latent's predict_unnorm).
        """
        s_norm = (state_raw  - state_mean)  / state_std
        a_norm = (action_raw - action_mean) / action_std
        z_t    = self.encoder(s_norm)
        mu_mean, epi_std, ale_std, tot_std = self.predict_ensemble(z_t, a_norm, layer_idx)
        pred_raw = self.decoder(z_t + mu_mean) * state_std + state_mean
        return pred_raw, epi_std, ale_std, tot_std

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        enc_p   = sum(p.numel() for p in self.encoder.parameters())
        trans_p = sum(p.numel() for p in self.transitions[0].parameters())
        dec_p   = sum(p.numel() for p in self.decoder.parameters())
        return (
            f"EnsembleGaussianLatentDynamicsModel("
            f"state={self.state_dim}, latent={self.latent_dim}, "
            f"action={self.action_dim}, K={self.n_ensemble}, "
            f"n_layers={self.n_layers}, embed={self.layer_embed_dim} | "
            f"enc={enc_p:,}  trans×{self.n_ensemble}={trans_p * self.n_ensemble:,}  "
            f"dec={dec_p:,} | total={self.count_parameters():,})"
        )
