"""
surrogate_model_latent_uncertainty_v2/model.py
--------------------------------------------------
Two-stage latent-space ensemble dynamics model for the LPBF digital twin.
Extends surrogate_model_latent_uncertainty/model.py's Gaussian-ensemble /
bootstrap design to a physically motivated two-stage transition:

  Stage 1 (heating): s_t --[a_t = laser power]--> u_heat_t
                      action-DEPENDENT — this is where the controller acts.
  Stage 2 (cooling):  u_heat_t --[cool_time_t]--> s_{t+1}
                      action-INDEPENDENT — cooling physics is the same
                      regardless of what laser power produced u_heat_t, so
                      this stage is conditioned on cool_time instead of a_t.

Architecture
------------
  Encoder            : any field (s_t / u_heat_t / s_{t+1}) → z   [shared, deterministic]
  HeatingTransition_k : (z_t, a_t, embed(layer_idx))   → (μ_Δz, σ_Δz)   [K Gaussian MLPs]
  CoolingTransition_k : (z_heat, cool_t, embed(layer_idx)) → (μ_Δz, σ_Δz) [K Gaussian MLPs]
  Decoder            : z → field                                  [shared, deterministic]

The encoder/decoder are shared across all three field "views" because they
are literally the same kind of object — a temperature field over the same
mesh — just captured at three different points in one layer's heat/cool
cycle. HeatingTransition and CoolingTransition are separate ensembles (not
shared weights) since they model different physics, but both reuse the exact
same GaussianTransitionMLP architecture (PETS-style soft-clamped log σ,
bootstrap-trained members, moment-matched epistemic/aleatoric decomposition)
as surrogate_model_latent_uncertainty — see that package for the derivation.

Reward is NOT modelled here: it is a deterministic function of u_heat_t
(mean deviation of the decoded heating-end field from the target temperature
range), computed downstream (e.g. in the RL controller), exactly as in v1
where the transition model only ever predicted states, never rewards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


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
# Encoder / Decoder  (shared across s_t, u_heat_t, s_{t+1})
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Deterministic encoder: field → z."""
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


class Decoder(nn.Module):
    """Decoder: z → field."""
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
        """z: (B, latent_dim) → field: (B, state_dim)"""
        return self.head(self.trunk(self.input_proj(z)))


# ---------------------------------------------------------------------------
# Gaussian Transition MLP  (one ensemble member — used for BOTH stages)
# ---------------------------------------------------------------------------

class GaussianTransitionMLP(nn.Module):
    """
    Gaussian transition: (z, c, embed(layer_idx)) → (μ_Δz, σ_Δz).

    `c` is the stage's scalar conditioning input — laser power for the
    heating stage, cool_time for the cooling stage. Architecturally
    identical either way (one scalar concatenated to z + a learned layer
    embedding); only the physical meaning of `c` differs between the two
    ensembles instantiated in TwoStageEnsembleGaussianLatentDynamicsModel.

    log σ is bounded with a learnable soft clamp (Chua et al., "Deep RL in a
    Handful of Trials using Probabilistic Dynamics Models", PETS, 2018).
    """
    def __init__(
        self,
        latent_dim:      int,
        cond_dim:        int   = 1,
        hidden:          int   = 128,
        depth:           int   = 3,
        dropout:         float = 0.0,
        n_layers:        int   = 12,
        layer_embed_dim: int   = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.layer_embed = nn.Embedding(n_layers, layer_embed_dim)

        in_dim = latent_dim + cond_dim + layer_embed_dim
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
        c:         torch.Tensor,   # (B, cond_dim)
        layer_idx: torch.Tensor,   # (B,) int64, 0-indexed
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """→ (μ_Δz, log σ_Δz), each (B, latent_dim)"""
        e = self.layer_embed(layer_idx)          # (B, layer_embed_dim)
        x = torch.cat([z, c, e], dim=-1)
        h = self.trunk(self.input_proj(x))

        mu             = self.mu_head(h)
        raw_log_sigma  = self.log_sigma_head(h)

        log_sigma = self.max_log_sigma - F.softplus(self.max_log_sigma - raw_log_sigma)
        log_sigma = self.min_log_sigma + F.softplus(log_sigma - self.min_log_sigma)

        return mu, log_sigma


def _moment_match(
    mu_deltas:        torch.Tensor,   # (K, B, latent_dim)
    log_sigma_deltas: torch.Tensor,   # (K, B, latent_dim)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine K Gaussian ensemble members via mixture-of-Gaussians moment
    matching (Lakshminarayanan et al., 2017). Shared by both the heating and
    the cooling ensemble.

    Returns
    -------
    mu_mean       : (B, latent_dim)  ensemble mean Δz
    epistemic_std : (B, latent_dim)  std across member means   (large = OOD)
    aleatoric_std : (B, latent_dim)  sqrt(mean member variance) (inherent noise)
    total_std     : (B, latent_dim)  sqrt(epistemic² + aleatoric²)
    """
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


def combine_stage_uncertainties(
    heat_epistemic_std: torch.Tensor,
    heat_aleatoric_std: torch.Tensor,
    cool_epistemic_std: torch.Tensor,
    cool_aleatoric_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine the heating-stage and cooling-stage uncertainties into ONE
    full-step (z_t -> z_{t+1}) epistemic / aleatoric / total uncertainty.
    This is the number future RL code should read for "how uncertain is the
    surrogate about this transition" (e.g. an uncertainty-penalised reward
    or an uncertainty budget in a constrained MDP).

    Why variances simply add
    -------------------------
    The full step is literally a SUM of two latent increments:
        z_{t+1} = z_t + Δz_heat + Δz_cool
    Both Δz_heat and Δz_cool are themselves moment-matched Gaussian mixtures
    (see _moment_match): each is N(mu, epistemic_var + aleatoric_var) to
    first-order. Treating Δz_heat and Δz_cool as independent — the same
    simplifying assumption already implicit in chaining ensemble means
    step-to-step during rollout, i.e. we do not analytically propagate
    z_heat's own uncertainty through the cooling ensemble's mean/σ
    functions, only through the additive noise terms — gives, for two
    independent Gaussians, Var(X + Y) = Var(X) + Var(Y). Applied separately
    to each uncertainty SOURCE (they don't mix with each other):

        epistemic_var_total = epistemic_var_heat + epistemic_var_cool
        aleatoric_var_total = aleatoric_var_heat + aleatoric_var_cool
        total_var            = epistemic_var_total + aleatoric_var_total

    Must be called on the FULL (not yet layer/batch-averaged) σ tensors —
    e.g. (B, latent_dim) — since Var(X+Y)=Var(X)+Var(Y) operates on
    variances, and averaging stds first (as the per-layer report scalars
    do) would not commute with this sum. Callers that need a single
    scalar for logging should call .mean(-1) (or further .mean(0)) on the
    RETURNED combined tensors, not on the inputs.

    Returns
    -------
    epistemic_std_total, aleatoric_std_total, total_std : same shape as inputs
    """
    epistemic_var_total = heat_epistemic_std.pow(2) + cool_epistemic_std.pow(2)
    aleatoric_var_total = heat_aleatoric_std.pow(2) + cool_aleatoric_std.pow(2)
    total_var            = epistemic_var_total + aleatoric_var_total
    return (
        epistemic_var_total.clamp_min(0.0).sqrt(),
        aleatoric_var_total.clamp_min(0.0).sqrt(),
        total_var.clamp_min(0.0).sqrt(),
    )


# ---------------------------------------------------------------------------
# Two-stage ensemble latent dynamics model
# ---------------------------------------------------------------------------

class TwoStageEnsembleGaussianLatentDynamicsModel(nn.Module):
    """
    Two-stage (heating → cooling) latent-space LPBF surrogate with
    layer-index conditioning, Gaussian transition heads, and
    bootstrap-resampled training.

    Parameters
    ----------
    state_dim       : temperature field dimension (1053)
    lp_dim          : laser power dimension (1) — heating stage conditioning
    cool_dim        : cool time dimension (1)   — cooling stage conditioning
    latent_dim      : latent space dimension
    n_ensemble      : number of ensemble members K (default 5), shared by both stages
    n_layers        : number of build layers — sets embedding table size (default 12)
    layer_embed_dim : dimension of learned per-layer embedding per member (default 8)
    enc_hidden / enc_depth     : encoder MLP width / depth
    trans_hidden / trans_depth : transition MLP width / depth per member (both stages)
    dec_hidden / dec_depth     : decoder MLP width / depth
    dropout         : shared dropout for all sub-networks
    """

    def __init__(
        self,
        state_dim:       int   = 1053,
        lp_dim:          int   = 1,
        cool_dim:        int   = 1,
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
        self.lp_dim          = lp_dim
        self.cool_dim        = cool_dim
        self.latent_dim      = latent_dim
        self.n_ensemble      = n_ensemble
        self.n_layers        = n_layers
        self.layer_embed_dim = layer_embed_dim

        self.encoder = Encoder(state_dim, latent_dim, enc_hidden, enc_depth, dropout)
        self.decoder = Decoder(latent_dim, state_dim, dec_hidden, dec_depth, dropout)

        self.heating_transitions = nn.ModuleList([
            GaussianTransitionMLP(
                latent_dim, lp_dim, trans_hidden, trans_depth, dropout,
                n_layers, layer_embed_dim,
            )
            for _ in range(n_ensemble)
        ])
        self.cooling_transitions = nn.ModuleList([
            GaussianTransitionMLP(
                latent_dim, cool_dim, trans_hidden, trans_depth, dropout,
                n_layers, layer_embed_dim,
            )
            for _ in range(n_ensemble)
        ])

    # ------------------------------------------------------------------
    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def _run(
        self,
        transitions: nn.ModuleList,
        z:           torch.Tensor,
        c:           torch.Tensor,
        layer_idx:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run all K members of one ensemble. → (mu_deltas, log_sigma_deltas), each (K, B, latent_dim)."""
        mus, log_sigmas = [], []
        for t in transitions:
            mu, log_sigma = t(z, c, layer_idx)
            mus.append(mu)
            log_sigmas.append(log_sigma)
        return torch.stack(mus, dim=0), torch.stack(log_sigmas, dim=0)

    def predict_heating_ensemble(
        self, z_t: torch.Tensor, a_t: torch.Tensor, layer_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(z_t, laser_power) → moment-matched (mu_mean, epistemic_std, aleatoric_std, total_std) for Δz_heat."""
        mu_deltas, log_sigma_deltas = self._run(self.heating_transitions, z_t, a_t, layer_idx)
        return _moment_match(mu_deltas, log_sigma_deltas)

    def predict_cooling_ensemble(
        self, z_heat: torch.Tensor, cool_t: torch.Tensor, layer_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(z_heat, cool_time) → moment-matched (mu_mean, epistemic_std, aleatoric_std, total_std) for Δz_cool."""
        mu_deltas, log_sigma_deltas = self._run(self.cooling_transitions, z_heat, cool_t, layer_idx)
        return _moment_match(mu_deltas, log_sigma_deltas)

    # ------------------------------------------------------------------
    def forward(
        self,
        s_t:       torch.Tensor,   # (B, state_dim)
        a_t:       torch.Tensor,   # (B, lp_dim)      normalised laser power
        cool_t:    torch.Tensor,   # (B, cool_dim)    normalised cool time
        u_heat_t:  torch.Tensor,   # (B, state_dim)   ground-truth end-of-heating field
        layer_idx: torch.Tensor,   # (B,) int64
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for single-step training. The cooling stage is
        teacher-forced on the ground-truth u_heat_t (not the heating stage's
        own prediction) so the two stages train independently, matching the
        physical fact that cooling doesn't care how u_heat_t was produced.

        Returns a dict with:
          z_t, s_t_recon                        — pre-heat state encode/AE-recon
          mu_heat, log_sigma_heat               — (K, B, Dz) heating ensemble outputs
          z_heat_enc, heat_recon                — encode/AE-recon of ground-truth u_heat_t
          mu_cool, log_sigma_cool               — (K, B, Dz) cooling ensemble outputs
        """
        z_t       = self.encoder(s_t)
        s_t_recon = self.decoder(z_t)
        mu_heat, log_sigma_heat = self._run(self.heating_transitions, z_t, a_t, layer_idx)

        z_heat_enc = self.encoder(u_heat_t)
        heat_recon = self.decoder(z_heat_enc)
        mu_cool, log_sigma_cool = self._run(self.cooling_transitions, z_heat_enc, cool_t, layer_idx)

        return dict(
            z_t=z_t, s_t_recon=s_t_recon,
            mu_heat=mu_heat, log_sigma_heat=log_sigma_heat,
            z_heat_enc=z_heat_enc, heat_recon=heat_recon,
            mu_cool=mu_cool, log_sigma_cool=log_sigma_cool,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        s_0:        torch.Tensor,   # (B, state_dim)     normalised
        actions:    torch.Tensor,   # (B, T, lp_dim)     normalised laser power
        cool_times: torch.Tensor,   # (B, T, cool_dim)   normalised cool time
    ) -> Dict[str, torch.Tensor]:
        """
        Auto-regressive rollout, fully chained in latent space: heating then
        cooling per layer, with the cooling stage consuming the heating
        stage's OWN predicted latent (no ground truth available at rollout
        time). Layer indices are generated automatically: step t → layer index t.

        Returns a dict with (all but the two decoded-field tensors are (B, T)):
          pred_heat_states, pred_next_states : (B, T, state_dim)  decoded fields
          heat_epistemic, heat_aleatoric     : mean latent σ, heating stage alone
          cool_epistemic, cool_aleatoric     : mean latent σ, cooling stage alone
          total_epistemic, total_aleatoric, total_std :
              combined full-step (z_t -> z_{t+1}) uncertainty — see
              combine_stage_uncertainties for the variance-addition derivation.
              THIS is what RL code should read for "how uncertain is the
              surrogate about this transition."
        """
        B, T, _ = actions.shape
        device  = s_0.device
        z_t     = self.encoder(s_0)

        pred_heat, pred_next = [], []
        heat_epi_l, heat_ale_l, cool_epi_l, cool_ale_l = [], [], [], []
        total_epi_l, total_ale_l, total_std_l = [], [], []

        for t in range(T):
            a_t       = actions[:, t, :]
            c_t       = cool_times[:, t, :]
            layer_idx = torch.full((B,), t, dtype=torch.long, device=device)

            mu_heat, heat_epi, heat_ale, _ = self.predict_heating_ensemble(z_t, a_t, layer_idx)
            z_heat = z_t + mu_heat
            pred_heat.append(self.decoder(z_heat))

            mu_cool, cool_epi, cool_ale, _ = self.predict_cooling_ensemble(z_heat, c_t, layer_idx)
            z_t = z_heat + mu_cool
            pred_next.append(self.decoder(z_t))

            # combine at full (B, latent_dim) resolution BEFORE averaging over
            # latent dims — Var(X+Y)=Var(X)+Var(Y) must operate on variances,
            # so this has to happen before any lossy std-averaging.
            total_epi, total_ale, total_std = combine_stage_uncertainties(
                heat_epi, heat_ale, cool_epi, cool_ale
            )

            heat_epi_l.append(heat_epi.mean(dim=-1))
            heat_ale_l.append(heat_ale.mean(dim=-1))
            cool_epi_l.append(cool_epi.mean(dim=-1))
            cool_ale_l.append(cool_ale.mean(dim=-1))
            total_epi_l.append(total_epi.mean(dim=-1))
            total_ale_l.append(total_ale.mean(dim=-1))
            total_std_l.append(total_std.mean(dim=-1))

        return dict(
            pred_heat_states=torch.stack(pred_heat, dim=1),
            pred_next_states=torch.stack(pred_next, dim=1),
            heat_epistemic=torch.stack(heat_epi_l, dim=1),
            heat_aleatoric=torch.stack(heat_ale_l, dim=1),
            cool_epistemic=torch.stack(cool_epi_l, dim=1),
            cool_aleatoric=torch.stack(cool_ale_l, dim=1),
            total_epistemic=torch.stack(total_epi_l, dim=1),
            total_aleatoric=torch.stack(total_ale_l, dim=1),
            total_std=torch.stack(total_std_l, dim=1),
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_unnorm(
        self,
        state_raw:   torch.Tensor,
        lp_raw:      torch.Tensor,
        cool_raw:    torch.Tensor,
        layer_idx:   torch.Tensor,   # (B,) int64
        state_mean:  torch.Tensor,
        state_std:   torch.Tensor,
        lp_mean:     float,
        lp_std:      float,
        cool_mean:   float,
        cool_std:    float,
    ) -> Dict[str, torch.Tensor]:
        """
        Accept raw [K]/[W]/[s] tensors; chain heating → cooling. This is the
        method future RL / MPC code should call for one step.

        Returns a dict:
          heat_pred_raw : (B, state_dim) predicted end-of-heating field, raw
                          Kelvin — needed for reward, since meanDeviation is
                          computed FROM this field, not the next state.
          next_pred_raw : (B, state_dim) predicted next state, raw Kelvin.
          heat_epistemic, heat_aleatoric, heat_total   : (B, latent_dim), heating stage alone
          cool_epistemic, cool_aleatoric, cool_total   : (B, latent_dim), cooling stage alone
          total_epistemic, total_aleatoric, total_std  : (B, latent_dim), combined
              full-step uncertainty (see combine_stage_uncertainties) — use
              THIS for an uncertainty-penalised reward / uncertainty budget.
        All uncertainties are reported in normalised latent space (consistent
        with v1's predict_unnorm).
        """
        s_norm  = (state_raw - state_mean) / state_std
        a_norm  = (lp_raw    - lp_mean)    / lp_std
        c_norm  = (cool_raw  - cool_mean)  / cool_std

        z_t = self.encoder(s_norm)
        mu_heat, heat_epi, heat_ale, heat_tot = self.predict_heating_ensemble(z_t, a_norm, layer_idx)
        z_heat = z_t + mu_heat
        heat_pred_raw = self.decoder(z_heat) * state_std + state_mean

        mu_cool, cool_epi, cool_ale, cool_tot = self.predict_cooling_ensemble(z_heat, c_norm, layer_idx)
        z_next = z_heat + mu_cool
        next_pred_raw = self.decoder(z_next) * state_std + state_mean

        total_epi, total_ale, total_std = combine_stage_uncertainties(
            heat_epi, heat_ale, cool_epi, cool_ale
        )

        return dict(
            heat_pred_raw=heat_pred_raw, next_pred_raw=next_pred_raw,
            heat_epistemic=heat_epi, heat_aleatoric=heat_ale, heat_total=heat_tot,
            cool_epistemic=cool_epi, cool_aleatoric=cool_ale, cool_total=cool_tot,
            total_epistemic=total_epi, total_aleatoric=total_ale, total_std=total_std,
        )

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        enc_p   = sum(p.numel() for p in self.encoder.parameters())
        dec_p   = sum(p.numel() for p in self.decoder.parameters())
        heat_p  = sum(p.numel() for p in self.heating_transitions[0].parameters())
        cool_p  = sum(p.numel() for p in self.cooling_transitions[0].parameters())
        return (
            f"TwoStageEnsembleGaussianLatentDynamicsModel("
            f"state={self.state_dim}, latent={self.latent_dim}, "
            f"lp_dim={self.lp_dim}, cool_dim={self.cool_dim}, K={self.n_ensemble}, "
            f"n_layers={self.n_layers}, embed={self.layer_embed_dim} | "
            f"enc={enc_p:,}  heat×{self.n_ensemble}={heat_p * self.n_ensemble:,}  "
            f"cool×{self.n_ensemble}={cool_p * self.n_ensemble:,}  "
            f"dec={dec_p:,} | total={self.count_parameters():,})"
        )
