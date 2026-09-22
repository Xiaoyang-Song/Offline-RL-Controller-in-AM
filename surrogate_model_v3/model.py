"""
surrogate_model_v3/model.py
------------------------------------
Two-stage ensemble dynamics model for the LPBF digital twin — NO learned
latent bottleneck. This is the primary surrogate design going forward (see
../README.md and ../../baseline_surrogate/README.md's "Result: the latent
encoder/decoder underperforms" section, and the ablation it grew out of at
../../baseline_surrogate/ablation_no_latent/).

  Stage 1 (heating): s_t --[a_t = laser power]--> u_heat_t
                      action-DEPENDENT — this is where the controller acts.
  Stage 2 (cooling):  u_heat_t --[cool_time_t]--> s_{t+1}
                      action-INDEPENDENT — cooling physics is the same
                      regardless of what laser power produced u_heat_t, so
                      this stage is conditioned on cool_time instead of a_t.

Architecture
------------
  HeatingTransition_k : (s_t, a_t, embed(layer_idx))        → (μ_Δ, σ_Δ)   [K Gaussian MLPs]
  CoolingTransition_k : (u_heat_t, cool_t, embed(layer_idx)) → (μ_Δ, σ_Δ)   [K Gaussian MLPs]

Both ensembles act DIRECTLY on the raw (normalised) 1053-dim state — no
encoder/decoder indirection, since a learned latent bottleneck was found to
badly underperform on this dataset (see the README). HeatingTransition and
CoolingTransition are separate ensembles (not shared weights) since they
model different physics, but both reuse the exact same GaussianTransitionMLP
architecture: PETS-style (Chua et al., 2018) soft-clamped log σ,
bootstrap-trained members, moment-matched (Lakshminarayanan et al., 2017)
epistemic/aleatoric decomposition.

Reward is NOT modelled here: it is a deterministic function of u_heat_t
(mean deviation of the end-of-heating field from the target temperature
range), computed downstream (e.g. in the RL controller).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


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
# Gaussian Transition MLP (one ensemble member — used for BOTH stages)
# ---------------------------------------------------------------------------

class GaussianTransitionMLP(nn.Module):
    """
    Gaussian transition: (x, c, embed(layer_idx)) → (μ_Δ, σ_Δ).

    `x` is the stage's input state (s_t for heating, u_heat_t for cooling);
    `c` is the stage's scalar conditioning input — laser power for the
    heating stage, cool_time for the cooling stage. Architecturally
    identical either way (one scalar concatenated to x + a learned layer
    embedding); only the physical meaning of `c` differs between the two
    ensembles instantiated in TwoStageSurrogate.

    log σ is bounded with a learnable soft clamp (Chua et al., "Deep RL in a
    Handful of Trials using Probabilistic Dynamics Models", PETS, 2018).
    """
    def __init__(
        self,
        state_dim:       int,
        cond_dim:        int   = 1,
        hidden:          int   = 128,
        depth:           int   = 3,
        dropout:         float = 0.0,
        n_layers:        int   = 12,
        layer_embed_dim: int   = 8,
        mu_init_scale:   float = 1e-3,
    ):
        super().__init__()
        self.state_dim = state_dim

        self.layer_embed = nn.Embedding(n_layers, layer_embed_dim)

        in_dim = state_dim + cond_dim + layer_embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.trunk           = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.mu_head         = nn.Linear(hidden, state_dim)
        self.log_sigma_head  = nn.Linear(hidden, state_dim)

        # mu_head init: near-zero (default 1e-3) so early predictions start
        # near Δ = 0 for training stability. NOTE: this range is identical
        # across ensemble members by default, so at init every member's
        # FUNCTION (not just its exact weights) is near-indistinguishable
        # from every other member's -- any inter-member disagreement then has
        # to be earned via gradient descent, which only happens where there's
        # training data. In a training-data gap, no member gets a gradient to
        # pull it away from this shared near-zero start, so they stay close
        # together there too (epistemic collapse in OOD/gap regions). Raising
        # --mu_init_scale (esp. combined with per-member seeding, see
        # TwoStageSurrogate's member_init_seed) gives members genuinely
        # different starting functions instead, at the cost of some of that
        # early-training stability.
        nn.init.uniform_(self.mu_head.weight, -mu_init_scale, mu_init_scale)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.uniform_(self.log_sigma_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.log_sigma_head.bias)

        # PETS-style learnable soft bounds on log σ (per state dimension)
        self.max_log_sigma = nn.Parameter(torch.full((state_dim,),  0.5))
        self.min_log_sigma = nn.Parameter(torch.full((state_dim,), -5.0))

    def forward(
        self,
        x:         torch.Tensor,   # (B, state_dim)
        c:         torch.Tensor,   # (B, cond_dim)
        layer_idx: torch.Tensor,   # (B,) int64, 0-indexed
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """→ (μ_Δ, log σ_Δ), each (B, state_dim)"""
        e = self.layer_embed(layer_idx)          # (B, layer_embed_dim)
        h_in = torch.cat([x, c, e], dim=-1)
        h = self.trunk(self.input_proj(h_in))

        mu             = self.mu_head(h)
        raw_log_sigma  = self.log_sigma_head(h)

        log_sigma = self.max_log_sigma - F.softplus(self.max_log_sigma - raw_log_sigma)
        log_sigma = self.min_log_sigma + F.softplus(log_sigma - self.min_log_sigma)

        return mu, log_sigma


def _moment_match(
    mu_deltas:        torch.Tensor,   # (K, B, state_dim)
    log_sigma_deltas: torch.Tensor,   # (K, B, state_dim)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine K Gaussian ensemble members via mixture-of-Gaussians moment
    matching (Lakshminarayanan et al., 2017). Shared by both the heating and
    the cooling ensemble.

    Returns
    -------
    mu_mean       : (B, state_dim)  ensemble mean Δ
    epistemic_std : (B, state_dim)  std across member means   (large = OOD)
    aleatoric_std : (B, state_dim)  sqrt(mean member variance) (inherent noise)
    total_std     : (B, state_dim)  sqrt(epistemic² + aleatoric²)
    """
    var_deltas = (2.0 * log_sigma_deltas).exp()          # (K, B, state_dim)

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
    full-step (s_t -> s_{t+1}) epistemic / aleatoric / total uncertainty.
    This is the number RL/MPC code should read for "how uncertain is the
    surrogate about this transition" (e.g. an uncertainty-penalised reward
    or an uncertainty budget in a constrained MDP).

    Why variances simply add
    -------------------------
    The full step is literally a SUM of two increments:
        s_{t+1} = s_t + Δ_heat + Δ_cool
    Both Δ_heat and Δ_cool are themselves moment-matched Gaussian mixtures
    (see _moment_match): each is N(mu, epistemic_var + aleatoric_var) to
    first-order. Treating Δ_heat and Δ_cool as independent — the same
    simplifying assumption already implicit in chaining ensemble means
    step-to-step during rollout — gives, for two independent Gaussians,
    Var(X + Y) = Var(X) + Var(Y). Applied separately to each uncertainty
    SOURCE (they don't mix with each other):

        epistemic_var_total = epistemic_var_heat + epistemic_var_cool
        aleatoric_var_total = aleatoric_var_heat + aleatoric_var_cool
        total_var            = epistemic_var_total + aleatoric_var_total

    Must be called on the FULL (not yet layer/batch-averaged) σ tensors —
    e.g. (B, state_dim) — since Var(X+Y)=Var(X)+Var(Y) operates on
    variances, and averaging stds first would not commute with this sum.
    Callers that need a single scalar for logging should call .mean(-1) (or
    further .mean(0)) on the RETURNED combined tensors, not on the inputs.

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
# Two-stage ensemble surrogate (no latent space)
# ---------------------------------------------------------------------------

class TwoStageSurrogate(nn.Module):
    """
    Two-stage (heating → cooling) LPBF surrogate with layer-index
    conditioning, Gaussian transition heads, bootstrap-resampled training,
    and NO learned latent bottleneck — transitions act directly on the raw
    (normalised) 1053-dim state.

    Parameters
    ----------
    state_dim       : temperature field dimension (1053)
    lp_dim          : laser power dimension (1) — heating stage conditioning
    cool_dim        : cool time dimension (1)   — cooling stage conditioning
    n_ensemble      : number of ensemble members K (default 5), shared by both stages
    n_layers        : number of build layers — sets embedding table size (default 12)
    layer_embed_dim : dimension of learned per-layer embedding per member (default 8)
    trans_hidden / trans_depth : transition MLP width / depth per member (both stages)
    dropout         : shared dropout for all sub-networks
    """

    def __init__(
        self,
        state_dim:        int   = 1053,
        lp_dim:            int  = 1,
        cool_dim:          int  = 1,
        n_ensemble:        int  = 5,
        n_layers:          int  = 12,
        layer_embed_dim:   int  = 8,
        trans_hidden:      int  = 128,
        trans_depth:       int  = 3,
        dropout:           float = 0.0,
        mu_init_scale:     float = 1e-3,
        member_init_seed:  Optional[int] = None,
    ):
        super().__init__()
        self.state_dim       = state_dim
        self.lp_dim          = lp_dim
        self.cool_dim        = cool_dim
        self.n_ensemble      = n_ensemble
        self.n_layers        = n_layers
        self.layer_embed_dim = layer_embed_dim

        # member_init_seed (optional): give each of the K members its OWN
        # seed (base + offset) for weight init, instead of letting every
        # member draw from the same global RNG stream in sequence. Saves/
        # restores the global RNG state around member construction so this
        # doesn't change any other randomness (data shuffling, bootstrap
        # masks, etc.) in the training run when set.
        rng_state = torch.get_rng_state() if member_init_seed is not None else None

        def _build(cond_dim: int, seed_offset: int) -> nn.ModuleList:
            members = []
            for k in range(n_ensemble):
                if member_init_seed is not None:
                    torch.manual_seed(member_init_seed + seed_offset + k)
                members.append(GaussianTransitionMLP(
                    state_dim, cond_dim, trans_hidden, trans_depth, dropout,
                    n_layers, layer_embed_dim, mu_init_scale=mu_init_scale,
                ))
            return nn.ModuleList(members)

        self.heating_transitions = _build(lp_dim, seed_offset=0)
        self.cooling_transitions = _build(cool_dim, seed_offset=n_ensemble)

        if rng_state is not None:
            torch.set_rng_state(rng_state)

    # ------------------------------------------------------------------
    def _run(
        self,
        transitions: nn.ModuleList,
        x:           torch.Tensor,
        c:           torch.Tensor,
        layer_idx:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run all K members of one ensemble. → (mu_deltas, log_sigma_deltas), each (K, B, state_dim)."""
        mus, log_sigmas = [], []
        for t in transitions:
            mu, log_sigma = t(x, c, layer_idx)
            mus.append(mu)
            log_sigmas.append(log_sigma)
        return torch.stack(mus, dim=0), torch.stack(log_sigmas, dim=0)

    def predict_heating_ensemble(
        self, s_t: torch.Tensor, a_t: torch.Tensor, layer_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(s_t, laser_power) → moment-matched (mu_mean, epistemic_std, aleatoric_std, total_std) for Δ_heat."""
        mu_deltas, log_sigma_deltas = self._run(self.heating_transitions, s_t, a_t, layer_idx)
        return _moment_match(mu_deltas, log_sigma_deltas)

    def predict_cooling_ensemble(
        self, u_heat: torch.Tensor, cool_t: torch.Tensor, layer_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(u_heat_t, cool_time) → moment-matched (mu_mean, epistemic_std, aleatoric_std, total_std) for Δ_cool."""
        mu_deltas, log_sigma_deltas = self._run(self.cooling_transitions, u_heat, cool_t, layer_idx)
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
          mu_heat, log_sigma_heat — (K, B, D) heating ensemble outputs
          mu_cool, log_sigma_cool — (K, B, D) cooling ensemble outputs
        """
        mu_heat, log_sigma_heat = self._run(self.heating_transitions, s_t, a_t, layer_idx)
        mu_cool, log_sigma_cool = self._run(self.cooling_transitions, u_heat_t, cool_t, layer_idx)
        return dict(
            mu_heat=mu_heat, log_sigma_heat=log_sigma_heat,
            mu_cool=mu_cool, log_sigma_cool=log_sigma_cool,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_mean(self, s: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
                      layer_idx: torch.Tensor) -> torch.Tensor:
        """Chains heating then cooling ensemble means; returns the final
        s_{t+1} point estimate (normalised space)."""
        mu_heat, _, _, _ = self.predict_heating_ensemble(s, a, layer_idx)
        s_heat = s + mu_heat
        mu_cool, _, _, _ = self.predict_cooling_ensemble(s_heat, c, layer_idx)
        return s_heat + mu_cool

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        s_0:        torch.Tensor,   # (B, state_dim)     normalised
        actions:    torch.Tensor,   # (B, T, lp_dim)     normalised laser power
        cool_times: torch.Tensor,   # (B, T, cool_dim)   normalised cool time
    ) -> Dict[str, torch.Tensor]:
        """
        Auto-regressive rollout: heating then cooling per layer, with the
        cooling stage consuming the heating stage's OWN predicted state (no
        ground truth available at rollout time). Layer indices are
        generated automatically: step t → layer index t.

        Returns a dict with (all but the two field tensors are (B, T)):
          pred_heat_states, pred_next_states : (B, T, state_dim)
          heat_epistemic, heat_aleatoric     : mean per-node σ, heating stage alone
          cool_epistemic, cool_aleatoric     : mean per-node σ, cooling stage alone
          total_epistemic, total_aleatoric, total_std :
              combined full-step (s_t -> s_{t+1}) uncertainty — see
              combine_stage_uncertainties for the variance-addition
              derivation. THIS is what RL/MPC code should read for "how
              uncertain is the surrogate about this transition."
        """
        B, T, _ = actions.shape
        device  = s_0.device
        s_t     = s_0

        pred_heat, pred_next = [], []
        heat_epi_l, heat_ale_l, cool_epi_l, cool_ale_l = [], [], [], []
        total_epi_l, total_ale_l, total_std_l = [], [], []

        for t in range(T):
            a_t       = actions[:, t, :]
            c_t       = cool_times[:, t, :]
            layer_idx = torch.full((B,), t, dtype=torch.long, device=device)

            mu_heat, heat_epi, heat_ale, _ = self.predict_heating_ensemble(s_t, a_t, layer_idx)
            s_heat = s_t + mu_heat
            pred_heat.append(s_heat)

            mu_cool, cool_epi, cool_ale, _ = self.predict_cooling_ensemble(s_heat, c_t, layer_idx)
            s_t = s_heat + mu_cool
            pred_next.append(s_t)

            # combine at full (B, state_dim) resolution BEFORE averaging over
            # nodes — Var(X+Y)=Var(X)+Var(Y) must operate on variances, so
            # this has to happen before any lossy std-averaging.
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
        method RL / MPC code should call for one step.

        Returns a dict:
          heat_pred_raw : (B, state_dim) predicted end-of-heating field, raw
                          Kelvin — needed for reward, since meanDeviation is
                          computed FROM this field, not the next state.
          next_pred_raw : (B, state_dim) predicted next state, raw Kelvin.
          heat_epistemic, heat_aleatoric, heat_total   : (B, state_dim), heating stage alone
          cool_epistemic, cool_aleatoric, cool_total   : (B, state_dim), cooling stage alone
          total_epistemic, total_aleatoric, total_std  : (B, state_dim), combined
              full-step uncertainty (see combine_stage_uncertainties) — use
              THIS for an uncertainty-penalised reward / uncertainty budget.
        """
        s_norm = (state_raw - state_mean) / state_std
        a_norm = (lp_raw    - lp_mean)    / lp_std
        c_norm = (cool_raw  - cool_mean)  / cool_std

        mu_heat, heat_epi, heat_ale, heat_tot = self.predict_heating_ensemble(s_norm, a_norm, layer_idx)
        s_heat = s_norm + mu_heat
        heat_pred_raw = s_heat * state_std + state_mean

        mu_cool, cool_epi, cool_ale, cool_tot = self.predict_cooling_ensemble(s_heat, c_norm, layer_idx)
        s_next = s_heat + mu_cool
        next_pred_raw = s_next * state_std + state_mean

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
        heat_p = sum(p.numel() for p in self.heating_transitions[0].parameters())
        cool_p = sum(p.numel() for p in self.cooling_transitions[0].parameters())
        return (
            f"TwoStageSurrogate("
            f"state={self.state_dim}, lp_dim={self.lp_dim}, cool_dim={self.cool_dim}, "
            f"K={self.n_ensemble}, n_layers={self.n_layers}, embed={self.layer_embed_dim} | "
            f"heat×{self.n_ensemble}={heat_p * self.n_ensemble:,}  "
            f"cool×{self.n_ensemble}={cool_p * self.n_ensemble:,} | "
            f"total={self.count_parameters():,})"
        )


def load_surrogate(checkpoint_path: str, device: str = "cpu"):
    """Load a saved TwoStageSurrogate checkpoint.

    Returns: model (eval mode), state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std.
    """
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TwoStageSurrogate(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return (
        model,
        ckpt["state_mean"].to(device), ckpt["state_std"].to(device),
        ckpt["lp_mean"], ckpt["lp_std"], ckpt["cool_mean"], ckpt["cool_std"],
    )
