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
  HeatingTransition_k : (s_t, a_t, embed(layer_idx))        → (μ_Δ, σ_Δ[, U_Δ])   [K Gaussian MLPs]
  CoolingTransition_k : (u_heat_t, cool_t, embed(layer_idx)) → (μ_Δ, σ_Δ[, U_Δ])   [K Gaussian MLPs]

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

Uncertainty representation: diagonal vs. low-rank ("offdiag"), and
combination: naive-sum vs. propagated
-----------------------------------------------------------------------
Every prediction path in this file still works exactly as before by
default — nothing below changes unless you opt in via a constructor/call
argument. Two independent, orthogonal knobs were added on top:

  `rank` (GaussianTransitionMLP / TwoStageSurrogate constructor, default 0):
      0    → pure diagonal covariance per member, Σ_k = diag(exp(2·logσ_k)).
             Identical architecture/behavior to the original design.
      r>0  → each member ALSO outputs a low-rank factor U_k ∈ R^{D×r}, so
             Σ_k = diag(exp(2·logσ_k)) + U_k U_kᵀ. Captures spatially-
             correlated error modes a diagonal can't (physically: this
             field is a heat-diffusion result, so neighbouring-node errors
             are correlated, not independent — a diagonal covariance
             understates the uncertainty of anything spatially AGGREGATED,
             e.g. the ROI-averaged reward). Trained via a Woodbury-identity
             Gaussian NLL (train.py's `low_rank_gaussian_nll`) so only an
             (r×r) matrix needs inverting per sample, not the full (D×D) Σ.

      Independently of `rank`, the K ensemble MEANS' own empirical
      deviation is an EXACT rank-(K-1) covariance factor for the epistemic
      term — (μ_k − μ̄)/√K stacked over k — available at ZERO extra cost
      regardless of `rank`. `_moment_match_full` always returns this
      `epistemic_factor` alongside the usual diagonal `epistemic_var`
      (same number as `_moment_match` would give); only the ALEATORIC
      side needs `rank>0` to get a matching `aleatoric_factor`.

  `propagate_uncertainty` (rollout / predict_unnorm call argument, default
  False):
      False → naive combination: Var_total = Var_heat + Var_cool (assumes
              the cooling stage passes heating-stage error through
              unchanged — see combine_stage_uncertainties). Original
              behavior when rank=0.
      True  → EKF-style moment propagation: heating's uncertainty is
              propagated through the cooling MEAN function's local
              Jacobian before being combined with cooling's own
              uncertainty, since μ_cool is a nonlinear function of its
              input s_heat and can amplify or damp upstream error. The
              structured (low-rank) directions — the epistemic factor
              always, the aleatoric factor when rank>0 — are propagated
              EXACTLY via Jacobian-vector products (batched into a single
              extra forward+backward pass per stage, not one per
              direction). Whatever's left as a residual pure diagonal is
              propagated via a cheap Hutchinson stochastic diagonal
              estimate of the same Jacobian. See
              `combine_stage_uncertainties_propagated`.

Both knobs are independent: a rank=0 checkpoint can still be evaluated with
propagate_uncertainty=True (propagates the free epistemic factor exactly,
Hutchinson-approximates the rest); a rank>0 checkpoint can be evaluated with
propagate_uncertainty=False (naive sum, but now correctly including the
low-rank aleatoric contribution — see combine_stage_uncertainties_naive_full).
See compare_uncertainty_methods.py for a script that evaluates all
available combinations side by side.
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
    Gaussian transition: (x, c, embed(layer_idx)) → (μ_Δ, σ_Δ[, U_Δ]).

    `x` is the stage's input state (s_t for heating, u_heat_t for cooling);
    `c` is the stage's scalar conditioning input — laser power for the
    heating stage, cool_time for the cooling stage. Architecturally
    identical either way (one scalar concatenated to x + a learned layer
    embedding); only the physical meaning of `c` differs between the two
    ensembles instantiated in TwoStageSurrogate.

    log σ is bounded with a learnable soft clamp (Chua et al., "Deep RL in a
    Handful of Trials using Probabilistic Dynamics Models", PETS, 2018).

    `rank` (default 0, i.e. pure diagonal — unchanged from the original
    design): when > 0, an additional linear head outputs a low-rank factor
    U ∈ R^{D×rank} per sample, so this member's own covariance is
    diag(exp(2·logσ)) + U Uᵀ instead of just diag(exp(2·logσ)) — see this
    module's docstring for why. U's per-node row norm (sqrt of its
    contribution to that node's variance) gets the SAME kind of PETS-style
    soft clamp as log σ, via `min_log_u_norm`/`max_log_u_norm` — unlike log
    σ, U has no other architectural bound (its magnitude is otherwise only
    discouraged by --weight_decay), and was observed in practice to inflate
    without limit over training: the Gaussian NLL objective can reduce loss
    by growing U to "explain away" residual mean-prediction error instead of
    fitting the mean better, with nothing to stop it — resulting in a
    combined aleatoric σ ~2 (in z-scored units, i.e. comparable to a node's
    ENTIRE natural range) despite actual point-prediction RMSE around 0.02-0.03,
    and — worse — an uncertainty-vs-laser-power curve that no longer
    distinguished a narrow-trained checkpoint's ID range from its OOD region
    at all (see surrogate_model_v3/README.md's "Uncertainty representation"
    section). The clamp caps U's max per-node variance CONTRIBUTION well
    below log σ's own max variance, since U is layered ON TOP of the
    diagonal, not instead of it.
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
        rank:            int   = 0,
        min_log_u_norm:  float = -6.0,
        max_log_u_norm:  float = -1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.rank       = rank

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

        if rank > 0:
            self.u_head = nn.Linear(hidden, state_dim * rank)
            nn.init.uniform_(self.u_head.weight, -1e-3, 1e-3)
            nn.init.zeros_(self.u_head.bias)
            # PETS-style learnable soft bounds on U's per-node ROW NORM (not
            # on U itself — U has `rank` free directions per node, only their
            # combined magnitude is capped) — see class docstring for why
            # this exists. Defaults much tighter than log σ's own [-5, 0.5]:
            # exp(max_log_u_norm)=exp(-1)≈0.37 caps this member's low-rank
            # variance contribution at ≈0.135 per node, well under log σ's
            # own ceiling of exp(2·0.5)≈2.72, since U adds to that, not
            # replaces it.
            self.max_log_u_norm = nn.Parameter(torch.full((state_dim,), max_log_u_norm))
            self.min_log_u_norm = nn.Parameter(torch.full((state_dim,), min_log_u_norm))
        else:
            self.u_head = None

    def forward(
        self,
        x:         torch.Tensor,   # (B, state_dim)
        c:         torch.Tensor,   # (B, cond_dim)
        layer_idx: torch.Tensor,   # (B,) int64, 0-indexed
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """→ (μ_Δ, log σ_Δ, U_Δ) — μ_Δ/log σ_Δ each (B, state_dim); U_Δ is
        (B, state_dim, rank) if rank>0, else None."""
        e = self.layer_embed(layer_idx)          # (B, layer_embed_dim)
        h_in = torch.cat([x, c, e], dim=-1)
        h = self.trunk(self.input_proj(h_in))

        mu             = self.mu_head(h)
        raw_log_sigma  = self.log_sigma_head(h)

        log_sigma = self.max_log_sigma - F.softplus(self.max_log_sigma - raw_log_sigma)
        log_sigma = self.min_log_sigma + F.softplus(log_sigma - self.min_log_sigma)

        u = None
        if self.u_head is not None:
            raw_u = self.u_head(h).view(x.shape[0], self.state_dim, self.rank)

            row_norm = raw_u.norm(dim=-1).clamp_min(1e-8)                    # (B, D)
            log_norm = row_norm.log()
            log_norm = self.max_log_u_norm - F.softplus(self.max_log_u_norm - log_norm)
            log_norm = self.min_log_u_norm + F.softplus(log_norm - self.min_log_u_norm)
            capped_norm = log_norm.exp()                                     # (B, D)

            u = raw_u * (capped_norm / row_norm).unsqueeze(-1)               # same direction, capped magnitude

        return mu, log_sigma, u


def _moment_match(
    mu_deltas:        torch.Tensor,   # (K, B, state_dim)
    log_sigma_deltas: torch.Tensor,   # (K, B, state_dim)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine K Gaussian ensemble members via mixture-of-Gaussians moment
    matching (Lakshminarayanan et al., 2017). Shared by both the heating and
    the cooling ensemble. UNCHANGED from the original diagonal-only design
    — see `_moment_match_full` for the version that also returns exact/
    low-rank factor views on top of these same numbers.

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


def _moment_match_full(
    mu_deltas:        torch.Tensor,             # (K, B, D)
    log_sigma_deltas: torch.Tensor,              # (K, B, D)
    u_deltas:         Optional[torch.Tensor],    # (K, B, D, rank) or None
) -> Dict[str, torch.Tensor]:
    """
    Like `_moment_match`, but ALSO returns exact low-rank factor views —
    used by the propagation-aware / low-rank-aware prediction path (see this
    module's docstring). The diagonal numbers returned here
    (`epistemic_var`/`aleatoric_var`/`total_var`) are numerically IDENTICAL
    to what `_moment_match` computes from the same `mu_deltas`/
    `log_sigma_deltas` — this function only ADDS the factor views, it does
    not change them.

    epistemic_factor : (B, D, K) — ALWAYS available (no `rank` needed): the
        K member means' own empirical deviation (μ_k − μ̄)/√K IS an exact
        rank-(K-1) covariance factor for the epistemic term at zero extra
        cost — `epistemic_factor @ epistemic_factor.mT` has diagonal EXACTLY
        equal to `epistemic_var`. Moment-matching this into a diagonal (as
        `_moment_match` does) throws away whether the K members disagree
        TOGETHER (e.g. the whole field shifts one direction) or
        independently node-by-node.
    aleatoric_factor : (B, D, K·rank) or None — only when `u_deltas` is
        given (rank>0): the members' own low-rank factors U_k/√K
        concatenated, so the AVERAGE within-member covariance's low-rank
        part is diag(mean_k d_k) [= `aleatoric_var` below] + this factor's
        outer product, i.e. `aleatoric_var` alone is NOT the full aleatoric
        variance when this is not None — see combine_stage_uncertainties_naive_full.
    """
    K, B, D = mu_deltas.shape
    var_deltas = (2.0 * log_sigma_deltas).exp()

    mu_mean       = mu_deltas.mean(0)
    epistemic_var = mu_deltas.var(0, unbiased=False).clamp_min(0.0)
    aleatoric_var = var_deltas.mean(0).clamp_min(0.0)
    total_var     = epistemic_var + aleatoric_var

    epistemic_factor = ((mu_deltas - mu_mean.unsqueeze(0)) / (K ** 0.5)).permute(1, 2, 0)  # (B, D, K)

    aleatoric_factor = None
    if u_deltas is not None:
        rank = u_deltas.shape[-1]
        # (K, B, D, rank) -> (B, D, K, rank) -> (B, D, K*rank)
        aleatoric_factor = (u_deltas / (K ** 0.5)).permute(1, 2, 0, 3).reshape(B, D, K * rank)

    return dict(
        mu_mean=mu_mean,
        epistemic_var=epistemic_var, aleatoric_var=aleatoric_var, total_var=total_var,
        epistemic_std=epistemic_var.sqrt(), aleatoric_std=aleatoric_var.sqrt(), total_std=total_var.sqrt(),
        epistemic_factor=epistemic_factor, aleatoric_factor=aleatoric_factor,
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

    UNCHANGED — original naive (independence-assuming) combination. See
    `combine_stage_uncertainties_propagated` for the EKF-style alternative,
    and `combine_stage_uncertainties_naive_full` for the naive combination
    generalised to also cover a low-rank (rank>0) model's aleatoric factor.

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


def _factor_sq_sum(factor: Optional[torch.Tensor]) -> float:
    """diag(factor @ factor.mT) = row-wise sum of squares — 0.0 if factor is None."""
    return 0.0 if factor is None else (factor ** 2).sum(-1)


def _concat_factors(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Concatenate two (B, D, m)-or-None factors along the last dim — this is
    how independent covariance CONTRIBUTIONS combine (Σ1+Σ2 -> factors
    concatenated), used to build a combined low-rank view of the final
    next-state uncertainty."""
    if a is None:
        return b
    if b is None:
        return a
    return torch.cat([a, b], dim=-1)


def combine_stage_uncertainties_naive_full(
    heat_full: Dict[str, torch.Tensor],
    cool_full: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Naive (independence-assuming, no Jacobian propagation) combination, but
    operating on `predict_*_ensemble_full` dicts so it correctly INCLUDES
    any low-rank aleatoric contribution (`aleatoric_var` alone is only the
    diagonal `d`-part when rank>0 — see `_moment_match_full`). Reduces
    EXACTLY to `combine_stage_uncertainties`'s numbers when both stages have
    rank=0 (aleatoric_factor is None on both sides).

    Returns a dict with epistemic_std/aleatoric_std/total_std (diagonal,
    (B, D)) plus epistemic_factor/aleatoric_factor (the combined low-rank
    view of the final next-state uncertainty, via factor concatenation).
    """
    heat_ale_var = heat_full["aleatoric_var"] + _factor_sq_sum(heat_full["aleatoric_factor"])
    cool_ale_var = cool_full["aleatoric_var"] + _factor_sq_sum(cool_full["aleatoric_factor"])

    epistemic_var_total = heat_full["epistemic_var"] + cool_full["epistemic_var"]
    aleatoric_var_total = heat_ale_var + cool_ale_var
    total_var            = epistemic_var_total + aleatoric_var_total

    return dict(
        epistemic_std=epistemic_var_total.clamp_min(0.0).sqrt(),
        aleatoric_std=aleatoric_var_total.clamp_min(0.0).sqrt(),
        total_std=total_var.clamp_min(0.0).sqrt(),
        epistemic_factor=_concat_factors(heat_full["epistemic_factor"], cool_full["epistemic_factor"]),
        aleatoric_factor=_concat_factors(heat_full["aleatoric_factor"], cool_full["aleatoric_factor"]),
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
    rank            : low-rank aleatoric factor dimension (default 0 = pure
                      diagonal, i.e. the original design — see this module's
                      docstring for the full "diagonal vs. low-rank" story).
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
        rank:              int   = 0,
        min_log_u_norm:    float = -6.0,
        max_log_u_norm:    float = -1.0,
    ):
        super().__init__()
        self.state_dim       = state_dim
        self.lp_dim          = lp_dim
        self.cool_dim        = cool_dim
        self.n_ensemble      = n_ensemble
        self.n_layers        = n_layers
        self.layer_embed_dim = layer_embed_dim
        self.rank            = rank

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
                    n_layers, layer_embed_dim, mu_init_scale=mu_init_scale, rank=rank,
                    min_log_u_norm=min_log_u_norm, max_log_u_norm=max_log_u_norm,
                ))
            return nn.ModuleList(members)

        self.heating_transitions = _build(lp_dim, seed_offset=0)
        self.cooling_transitions = _build(cool_dim, seed_offset=n_ensemble)

        if rng_state is not None:
            torch.set_rng_state(rng_state)

    # ------------------------------------------------------------------
    def _run_full(
        self,
        transitions: nn.ModuleList,
        x:           torch.Tensor,
        c:           torch.Tensor,
        layer_idx:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run all K members of one ensemble, keeping each member's low-rank
        factor (if any). → (mu_deltas, log_sigma_deltas, u_deltas_or_None),
        (K, B, D) / (K, B, D) / (K, B, D, rank)."""
        mus, log_sigmas, us = [], [], []
        any_u = False
        for t in transitions:
            mu, log_sigma, u = t(x, c, layer_idx)
            mus.append(mu)
            log_sigmas.append(log_sigma)
            us.append(u)
            any_u = any_u or (u is not None)
        u_deltas = torch.stack(us, dim=0) if any_u else None
        return torch.stack(mus, dim=0), torch.stack(log_sigmas, dim=0), u_deltas

    def _run(
        self,
        transitions: nn.ModuleList,
        x:           torch.Tensor,
        c:           torch.Tensor,
        layer_idx:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """UNCHANGED signature/behavior: (mu_deltas, log_sigma_deltas), each
        (K, B, state_dim) — drops the low-rank factor (see `_run_full`)."""
        mu_deltas, log_sigma_deltas, _ = self._run_full(transitions, x, c, layer_idx)
        return mu_deltas, log_sigma_deltas

    def predict_heating_ensemble(
        self, s_t: torch.Tensor, a_t: torch.Tensor, layer_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """UNCHANGED. (s_t, laser_power) → moment-matched (mu_mean,
        epistemic_std, aleatoric_std, total_std) for Δ_heat. NOTE: when
        rank>0, `aleatoric_std` here is only the diagonal part — use
        `predict_heating_ensemble_full` to also get the low-rank factor."""
        mu_deltas, log_sigma_deltas = self._run(self.heating_transitions, s_t, a_t, layer_idx)
        return _moment_match(mu_deltas, log_sigma_deltas)

    def predict_cooling_ensemble(
        self, u_heat: torch.Tensor, cool_t: torch.Tensor, layer_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """UNCHANGED. (u_heat_t, cool_time) → moment-matched (mu_mean,
        epistemic_std, aleatoric_std, total_std) for Δ_cool. See note on
        `predict_heating_ensemble` above."""
        mu_deltas, log_sigma_deltas = self._run(self.cooling_transitions, u_heat, cool_t, layer_idx)
        return _moment_match(mu_deltas, log_sigma_deltas)

    def predict_heating_ensemble_full(
        self, s_t: torch.Tensor, a_t: torch.Tensor, layer_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Like `predict_heating_ensemble`, but returns the full dict from
        `_moment_match_full` (diag numbers identical, plus factor views)."""
        mu_deltas, log_sigma_deltas, u_deltas = self._run_full(self.heating_transitions, s_t, a_t, layer_idx)
        return _moment_match_full(mu_deltas, log_sigma_deltas, u_deltas)

    def predict_cooling_ensemble_full(
        self, u_heat: torch.Tensor, cool_t: torch.Tensor, layer_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Like `predict_cooling_ensemble`, but returns the full dict from
        `_moment_match_full` (diag numbers identical, plus factor views)."""
        mu_deltas, log_sigma_deltas, u_deltas = self._run_full(self.cooling_transitions, u_heat, cool_t, layer_idx)
        return _moment_match_full(mu_deltas, log_sigma_deltas, u_deltas)

    # ------------------------------------------------------------------
    # EKF-style moment propagation through the cooling stage's Jacobian
    # ------------------------------------------------------------------
    def _cooling_mean_fn(self, x: torch.Tensor, cool_t: torch.Tensor, layer_idx: torch.Tensor) -> torch.Tensor:
        """The (differentiable) cooling ensemble MEAN function alone
        (mu_cool_mean(x) — not x + mu_cool_mean), i.e. what
        combine_stage_uncertainties_propagated linearises around s_heat."""
        mu_deltas, _, _ = self._run_full(self.cooling_transitions, x, cool_t, layer_idx)
        return mu_deltas.mean(0)

    def _batched_jvp_through_cooling(
        self,
        s_heat:    torch.Tensor,             # (B, D)
        cool_t:    torch.Tensor,             # (B, cool_dim)
        layer_idx: torch.Tensor,             # (B,)
        tangents:  Optional[torch.Tensor],   # (B, D, M) direction columns, or None
    ) -> Optional[torch.Tensor]:
        """Jacobian-vector product of the cooling mean function w.r.t. its
        input state, for EACH of the M direction columns in `tangents` —
        computed in ONE batched forward+backward pass (fold M into the
        batch dimension) rather than M separate passes. Returns (B, D, M),
        or None if `tangents` is None / has zero columns.

        Works regardless of the caller's autograd context (e.g. inside
        `with torch.no_grad():`) since JVP needs its own local grad
        tracking — this is scoped to just this computation via
        `torch.enable_grad()` and everything is `.detach()`ed going in and
        out, so it never leaks a graph into the surrounding rollout.
        """
        if tangents is None or tangents.shape[-1] == 0:
            return None
        B, D, M = tangents.shape
        s_rep     = s_heat.detach().unsqueeze(0).expand(M, B, D).reshape(M * B, D)
        cool_rep  = cool_t.detach().unsqueeze(0).expand(M, B, cool_t.shape[-1]).reshape(M * B, cool_t.shape[-1])
        layer_rep = layer_idx.unsqueeze(0).expand(M, B).reshape(M * B)
        v         = tangents.detach().permute(2, 0, 1).reshape(M * B, D)

        def mean_fn(x):
            return self._cooling_mean_fn(x, cool_rep, layer_rep)

        with torch.enable_grad():
            _, jv = torch.autograd.functional.jvp(mean_fn, s_rep, v, create_graph=False, strict=False)

        return jv.detach().reshape(M, B, D).permute(1, 2, 0)

    def _hutchinson_diag_through_cooling(
        self, s_heat: torch.Tensor, cool_t: torch.Tensor, layer_idx: torch.Tensor, num_probes: int = 4,
    ) -> torch.Tensor:
        """Hutchinson stochastic estimate of diag(∂μ_cool/∂s_heat): E[v ⊙ Jv]
        over Rademacher probe vectors v, reusing the same batched-JVP
        machinery (num_probes columns in one pass) — cheap (num_probes+1
        forward+backward passes total) vs. exact (D passes). Returns (B, D)."""
        B, D = s_heat.shape
        probes = torch.randint(0, 2, (B, D, num_probes), device=s_heat.device, dtype=s_heat.dtype) * 2 - 1
        jv = self._batched_jvp_through_cooling(s_heat, cool_t, layer_idx, probes)
        return (probes * jv).mean(dim=-1)

    def combine_stage_uncertainties_propagated(
        self,
        s_heat:     torch.Tensor,
        cool_t:     torch.Tensor,
        layer_idx:  torch.Tensor,
        heat_full:  Dict[str, torch.Tensor],
        cool_full:  Dict[str, torch.Tensor],
        num_probes: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        EKF-style moment propagation: heating's uncertainty is propagated
        through the cooling mean function's local Jacobian J = ∂μ_cool/∂s_heat
        before being combined with cooling's own (unpropagated — cooling is
        the last stage here) uncertainty, instead of just adding the two
        stages' variances as if heating's error passes through cooling
        unchanged (see this module's docstring, and
        `combine_stage_uncertainties`'s docstring for the naive version).

        The structured directions are propagated EXACTLY via
        Jacobian-vector products: `heat_full["epistemic_factor"]` (always
        present) and `heat_full["aleatoric_factor"]` (present iff rank>0).
        Each propagated column `v'` = `v + Jv` (the `+v` accounts for the
        `s_heat + μ_cool` identity term of the full transition, `Jv` is this
        method's JVP of μ_cool alone). What's left over — the residual pure
        diagonal aleatoric variance not captured by any explicit factor —
        is propagated via a cheap Hutchinson diagonal Jacobian estimate
        (`num_probes` extra JVP columns) rather than an exact but D×-more-
        expensive full diagonal.

        Returns a dict with epistemic_std/aleatoric_std/total_std (diagonal,
        (B, D)), epistemic_factor/aleatoric_factor (combined low-rank view
        of the FINAL next-state uncertainty — heating's propagated factors
        concatenated with cooling's own), and jacobian_diag (the Hutchinson
        estimate of 1+∂μ_cool/∂s_heat, a diagnostic "amplification map":
        >1 means cooling amplifies upstream heating error at that node, <1
        means it damps it).
        """
        prop_epi_factor = self._batched_jvp_through_cooling(
            s_heat, cool_t, layer_idx, heat_full["epistemic_factor"])
        if prop_epi_factor is not None:
            prop_epi_factor = heat_full["epistemic_factor"] + prop_epi_factor

        prop_ale_factor = None
        if heat_full["aleatoric_factor"] is not None:
            jv = self._batched_jvp_through_cooling(s_heat, cool_t, layer_idx, heat_full["aleatoric_factor"])
            prop_ale_factor = heat_full["aleatoric_factor"] + jv

        jacobian_diag = 1.0 + self._hutchinson_diag_through_cooling(s_heat, cool_t, layer_idx, num_probes)

        # Epistemic is fully captured by the (always-present) exact factor —
        # its propagated diagonal is just read off the propagated factor,
        # NOT also separately Jacobian-scaled (that would double count).
        prop_epi_var = _factor_sq_sum(prop_epi_factor)
        # Aleatoric's residual pure-diagonal part (the per-member `d`, on
        # top of any low-rank `u`) is what the Hutchinson estimate propagates.
        prop_ale_diag_var = jacobian_diag.pow(2) * heat_full["aleatoric_var"]
        prop_ale_var = prop_ale_diag_var + _factor_sq_sum(prop_ale_factor)

        total_epistemic_var = prop_epi_var + cool_full["epistemic_var"]
        cool_ale_var = cool_full["aleatoric_var"] + _factor_sq_sum(cool_full["aleatoric_factor"])
        total_aleatoric_var = prop_ale_var + cool_ale_var
        total_var = total_epistemic_var + total_aleatoric_var

        return dict(
            epistemic_std=total_epistemic_var.clamp_min(0.0).sqrt(),
            aleatoric_std=total_aleatoric_var.clamp_min(0.0).sqrt(),
            total_std=total_var.clamp_min(0.0).sqrt(),
            jacobian_diag=jacobian_diag,
            epistemic_factor=_concat_factors(prop_epi_factor, cool_full["epistemic_factor"]),
            aleatoric_factor=_concat_factors(prop_ale_factor, cool_full["aleatoric_factor"]),
        )

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
          u_heat, u_cool — (K, B, D, rank) low-rank factors, or None if rank=0
              (NEW keys — mu_heat/log_sigma_heat/mu_cool/log_sigma_cool are
              numerically IDENTICAL to the original diagonal-only forward,
              existing callers that only read those four keys are
              unaffected).
        """
        mu_heat, log_sigma_heat, u_heat = self._run_full(self.heating_transitions, s_t, a_t, layer_idx)
        mu_cool, log_sigma_cool, u_cool = self._run_full(self.cooling_transitions, u_heat_t, cool_t, layer_idx)
        return dict(
            mu_heat=mu_heat, log_sigma_heat=log_sigma_heat, u_heat=u_heat,
            mu_cool=mu_cool, log_sigma_cool=log_sigma_cool, u_cool=u_cool,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_mean(self, s: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
                      layer_idx: torch.Tensor) -> torch.Tensor:
        """UNCHANGED. Chains heating then cooling ensemble means; returns
        the final s_{t+1} point estimate (normalised space)."""
        mu_heat, _, _, _ = self.predict_heating_ensemble(s, a, layer_idx)
        s_heat = s + mu_heat
        mu_cool, _, _, _ = self.predict_cooling_ensemble(s_heat, c, layer_idx)
        return s_heat + mu_cool

    # ------------------------------------------------------------------
    def rollout(
        self,
        s_0:                 torch.Tensor,   # (B, state_dim)     normalised
        actions:             torch.Tensor,   # (B, T, lp_dim)     normalised laser power
        cool_times:          torch.Tensor,   # (B, T, cool_dim)   normalised cool time
        propagate_uncertainty: bool = False,
        num_probes:          int   = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Auto-regressive rollout: heating then cooling per layer, with the
        cooling stage consuming the heating stage's OWN predicted state (no
        ground truth available at rollout time). Layer indices are
        generated automatically: step t → layer index t.

        `propagate_uncertainty=False` (default): EXACT original behavior —
        uses `combine_stage_uncertainties` (naive sum). When this model
        also has `rank==0`, output is bit-identical to before this feature
        existed.

        `propagate_uncertainty=True`: uses the EKF-style
        `combine_stage_uncertainties_propagated` instead (see its
        docstring) — needs `_full` predictions, so this also activates for
        rank>0 models regardless of this flag (naive combination on a
        rank>0 model still needs the `_full` path to correctly include the
        low-rank aleatoric term — see `combine_stage_uncertainties_naive_full`).

        Returns a dict with (all but the two field tensors are (B, T)):
          pred_heat_states, pred_next_states : (B, T, state_dim)
          heat_epistemic, heat_aleatoric     : mean per-node σ, heating stage alone
          cool_epistemic, cool_aleatoric     : mean per-node σ, cooling stage alone
          total_epistemic, total_aleatoric, total_std :
              combined full-step (s_t -> s_{t+1}) uncertainty — THIS is what
              RL/MPC code should read for "how uncertain is the surrogate
              about this transition."
        """
        B, T, _ = actions.shape
        device  = s_0.device
        s_t     = s_0
        use_full = propagate_uncertainty or (self.rank > 0)

        pred_heat, pred_next = [], []
        heat_epi_l, heat_ale_l, cool_epi_l, cool_ale_l = [], [], [], []
        total_epi_l, total_ale_l, total_std_l = [], [], []

        with torch.no_grad():
            for t in range(T):
                a_t       = actions[:, t, :]
                c_t       = cool_times[:, t, :]
                layer_idx = torch.full((B,), t, dtype=torch.long, device=device)

                if use_full:
                    heat_full = self.predict_heating_ensemble_full(s_t, a_t, layer_idx)
                    mu_heat, heat_epi, heat_ale = heat_full["mu_mean"], heat_full["epistemic_std"], heat_full["aleatoric_std"]
                else:
                    mu_heat, heat_epi, heat_ale, _ = self.predict_heating_ensemble(s_t, a_t, layer_idx)
                s_heat = s_t + mu_heat
                pred_heat.append(s_heat)

                if use_full:
                    cool_full = self.predict_cooling_ensemble_full(s_heat, c_t, layer_idx)
                    mu_cool, cool_epi, cool_ale = cool_full["mu_mean"], cool_full["epistemic_std"], cool_full["aleatoric_std"]
                else:
                    mu_cool, cool_epi, cool_ale, _ = self.predict_cooling_ensemble(s_heat, c_t, layer_idx)
                s_t = s_heat + mu_cool
                pred_next.append(s_t)

                # combine at full (B, state_dim) resolution BEFORE averaging
                # over nodes — Var(X+Y)=Var(X)+Var(Y) must operate on
                # variances, so this has to happen before any lossy
                # std-averaging.
                if propagate_uncertainty:
                    combo = self.combine_stage_uncertainties_propagated(
                        s_heat, c_t, layer_idx, heat_full, cool_full, num_probes=num_probes)
                elif use_full:
                    combo = combine_stage_uncertainties_naive_full(heat_full, cool_full)
                else:
                    e, a_, t_ = combine_stage_uncertainties(heat_epi, heat_ale, cool_epi, cool_ale)
                    combo = dict(epistemic_std=e, aleatoric_std=a_, total_std=t_)

                heat_epi_l.append(heat_epi.mean(dim=-1))
                heat_ale_l.append(heat_ale.mean(dim=-1))
                cool_epi_l.append(cool_epi.mean(dim=-1))
                cool_ale_l.append(cool_ale.mean(dim=-1))
                total_epi_l.append(combo["epistemic_std"].mean(dim=-1))
                total_ale_l.append(combo["aleatoric_std"].mean(dim=-1))
                total_std_l.append(combo["total_std"].mean(dim=-1))

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
        propagate_uncertainty: bool = False,
        num_probes:  int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Accept raw [K]/[W]/[s] tensors; chain heating → cooling. This is the
        method RL / MPC code should call for one step. See `rollout`'s
        docstring for `propagate_uncertainty`'s semantics (identical here,
        single-step instead of a full trajectory).

        Returns a dict:
          heat_pred_raw : (B, state_dim) predicted end-of-heating field, raw
                          Kelvin — needed for reward, since meanDeviation is
                          computed FROM this field, not the next state.
          next_pred_raw : (B, state_dim) predicted next state, raw Kelvin.
          heat_epistemic, heat_aleatoric, heat_total   : (B, state_dim), heating stage alone
          cool_epistemic, cool_aleatoric, cool_total   : (B, state_dim), cooling stage alone
          total_epistemic, total_aleatoric, total_std  : (B, state_dim), combined
              full-step uncertainty — use THIS for an uncertainty-penalised
              reward / uncertainty budget.
        """
        use_full = propagate_uncertainty or (self.rank > 0)

        with torch.no_grad():
            s_norm = (state_raw - state_mean) / state_std
            a_norm = (lp_raw    - lp_mean)    / lp_std
            c_norm = (cool_raw  - cool_mean)  / cool_std

            if use_full:
                heat_full = self.predict_heating_ensemble_full(s_norm, a_norm, layer_idx)
                mu_heat, heat_epi, heat_ale, heat_tot = (
                    heat_full["mu_mean"], heat_full["epistemic_std"], heat_full["aleatoric_std"], heat_full["total_std"])
            else:
                mu_heat, heat_epi, heat_ale, heat_tot = self.predict_heating_ensemble(s_norm, a_norm, layer_idx)
            s_heat = s_norm + mu_heat
            heat_pred_raw = s_heat * state_std + state_mean

            if use_full:
                cool_full = self.predict_cooling_ensemble_full(s_heat, c_norm, layer_idx)
                mu_cool, cool_epi, cool_ale, cool_tot = (
                    cool_full["mu_mean"], cool_full["epistemic_std"], cool_full["aleatoric_std"], cool_full["total_std"])
            else:
                mu_cool, cool_epi, cool_ale, cool_tot = self.predict_cooling_ensemble(s_heat, c_norm, layer_idx)
            s_next = s_heat + mu_cool
            next_pred_raw = s_next * state_std + state_mean

            if propagate_uncertainty:
                combo = self.combine_stage_uncertainties_propagated(
                    s_heat, c_norm, layer_idx, heat_full, cool_full, num_probes=num_probes)
                total_epi, total_ale, total_std = combo["epistemic_std"], combo["aleatoric_std"], combo["total_std"]
            elif use_full:
                combo = combine_stage_uncertainties_naive_full(heat_full, cool_full)
                total_epi, total_ale, total_std = combo["epistemic_std"], combo["aleatoric_std"], combo["total_std"]
            else:
                total_epi, total_ale, total_std = combine_stage_uncertainties(heat_epi, heat_ale, cool_epi, cool_ale)

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
            f"K={self.n_ensemble}, n_layers={self.n_layers}, embed={self.layer_embed_dim}, "
            f"rank={self.rank} | "
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
