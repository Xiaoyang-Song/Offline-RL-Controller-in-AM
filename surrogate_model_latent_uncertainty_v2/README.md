# surrogate_model_latent_uncertainty_v2

Two-stage extension of [`surrogate_model_latent_uncertainty/`](../surrogate_model_latent_uncertainty/),
built on the v2 simulation dataset (`../LPBF-Simulation/simulation_v2/RL_Dataset_v2/`).

## What changed vs. v1

The v1 pipeline computed the reward (`meanDeviation`) from the temperature
field **after cooling**. The v2 simulation (`simulateHeatingCooling_v2.m`)
computes it from the field **at the end of heating** instead, and now returns
both fields separately per layer:

- `uHeatFinal` — temperature field at the end of heating (reward input)
- `uFinal` — temperature field at the end of cooling (next state)
- `coolTime_step` — cooling duration used for that layer (randomized once per
  trajectory)

The surrogate is accordingly split into two independent stages:

| Stage | Input | Output | Action-dependent? |
|---|---|---|---|
| 1. Heating | `s_t`, laser power `a_t` | `u_heat_t` | Yes — this is where the controller acts |
| 2. Cooling | `u_heat_t`, `cool_time_t` | `s_{t+1}` | No — cooling physics doesn't depend on how `u_heat_t` was reached |

## Architecture

Same modeling philosophy as v1 (shared latent encoder/decoder + a
bootstrap-trained Gaussian ensemble for epistemic/aleatoric uncertainty), but
with **two** ensembles instead of one:

```
Encoder (shared)         : {s_t, u_heat_t, s_{t+1}} → z      (all literally
                            temperature fields over the same mesh)
HeatingTransition_k (×K) : (z_t, a_t, layer_idx)      → Δz_heat
CoolingTransition_k (×K) : (z_heat, cool_time, layer_idx) → Δz_cool
Decoder (shared)         : z → field
```

`HeatingTransition` and `CoolingTransition` reuse the exact same
`GaussianTransitionMLP` class (PETS-style soft-clamped log σ, moment-matched
epistemic/aleatoric decomposition) — they're separate weight sets, not a
shared module, since they model different physics.

During single-step training, the cooling stage is teacher-forced on the
**ground-truth** `u_heat_t`, not the heating stage's own prediction — the two
stages train independently, matching the physical fact that cooling doesn't
care how `u_heat_t` was produced. The optional rollout loss (`--rollout_steps`)
and `model.rollout(...)` chain both stages' own predictions end-to-end,
matching actual inference-time behaviour.

Reward is **not** modeled here — same as v1, this package only predicts
states. `meanDeviation` is a deterministic function of the decoded `u_heat_t`
field and should be computed downstream (e.g. by the RL controller), using
`model.predict_unnorm(...)`, which returns the raw predicted heating field
alongside the raw predicted next state for exactly this purpose.

## Mathematical details

Notation: `s_t` = pre-heat state, `a_t` = laser power, `u_h` = end-of-heating
field, `c_t` = cool time, `s_{t+1}` = end-of-cooling field (next state). All
of `s_t`, `u_h`, `s_{t+1}` live in the same `state_dim = 1053` space (nodal
temperatures over the shared PDE mesh) and share one normalisation.

### 1. Normalisation

Computed once from the training split only (`build_normalizers`):

```
state_mean, state_std = mean/std over the POOLED set {s_t, u_h_t, s_{t+1,t}}
                         for every t in every training trajectory
                         (pooled because encoder/decoder is shared across
                         all three field "views")
lp_mean,   lp_std      = mean/std of laser power [W]
cool_mean, cool_std    = mean/std of cool time [s]
```

`state_std`/`lp_std`/`cool_std` are clamped to ≥ 1e-6 (or fall back to 1.0 if
the raw std is ~0). All model inputs/targets below are in normalised space
unless stated as "raw".

### 2. Encoder / Decoder (shared across s_t, u_h, s_{t+1})

A residual MLP. One `ResidualBlock` is:

```
block(x) = LayerNorm(Linear(Dropout(SiLU(LayerNorm(Linear(x))))))
ResidualBlock(x) = SiLU(x + block(x))
```

```
Encoder:  z = Linear_head( ResidualBlock^{×enc_depth}( SiLU(LayerNorm(Linear(s))) ) )      s ∈ R^1053 → z ∈ R^latent_dim
Decoder: ŝ = Linear_head( ResidualBlock^{×dec_depth}( SiLU(LayerNorm(Linear(z))) ) )      z ∈ R^latent_dim → ŝ ∈ R^1053
```

### 3. Gaussian transition head (one ensemble member, one stage)

Both the heating ensemble `{HeatingTransition_k}_{k=1..K}` and the cooling
ensemble `{CoolingTransition_k}_{k=1..K}` use the *same* architecture,
`GaussianTransitionMLP`; only what's concatenated as the scalar conditioning
input `c` differs (`c = a_t` for heating, `c = cool_time_t` for cooling):

```
e = LayerEmbedding(layer_idx)                              e ∈ R^layer_embed_dim
h = ResidualBlock^{×trans_depth}( SiLU(LayerNorm(Linear([z, c, e]))) )
μ         = Linear_mu(h)                                    μ ∈ R^latent_dim
raw_logσ  = Linear_logsigma(h)
```

**PETS-style soft σ clamp** (Chua et al., 2018) — bounds `log σ` into
`(min_logσ, max_logσ)` (both learned, per latent dim, init `-5.0`/`0.5`)
smoothly, so the Gaussian NLL below never blows up but still has gradient
everywhere (unlike a hard `clamp`):

```
logσ' = max_logσ − softplus(max_logσ − raw_logσ)     # smooth ceiling
logσ  = min_logσ + softplus(logσ' − min_logσ)        # smooth floor
σ     = exp(logσ)
```

Member k's predictive distribution over the latent step is
`Δz_k ~ N(μ_k, diag(σ_k²))`.

### 4. Per-stage moment matching (K members → one Gaussian)

The ensemble's K members are combined as a uniform mixture,
`p(Δz) = (1/K) Σ_k N(μ_k, diag(σ_k²))`, moment-matched to a single Gaussian
(Lakshminarayanan et al., 2017 — deep ensembles) via the standard
law-of-total-variance decomposition:

```
μ̄            = (1/K) Σ_k μ_k                              ← mean prediction, used to step z
epistemic_var = Var_k[μ_k]           (population variance, ÷K not ÷(K−1))
aleatoric_var = (1/K) Σ_k σ_k²
total_var     = epistemic_var + aleatoric_var
```

`epistemic_std = √epistemic_var` is the ensemble members' *disagreement* —
large where training data was sparse (out-of-distribution signal).
`aleatoric_std = √aleatoric_var` is each member's own average predicted
noise — irreducible, data-inherent uncertainty. This is computed identically
for the heating ensemble (→ `heat_epistemic`, `heat_aleatoric`) and the
cooling ensemble (→ `cool_epistemic`, `cool_aleatoric`); see
`predict_heating_ensemble` / `predict_cooling_ensemble` in `model.py`.

### 5. Combining the two stages' uncertainty (for RL)

This is the part that's new relative to v1's single-stage model, where
there was only one epistemic and one aleatoric number per step. Here there
are two of each — one from the heating ensemble, one from the cooling
ensemble — and a single downstream consumer (the RL controller) needs one
number describing "how uncertain is the surrogate about this whole
`s_t → s_{t+1}` step."

The key structural fact: the full step is literally a **sum** of two latent
increments,

```
z_{t+1} = z_t + Δz_heat + Δz_cool
```

Approximating `Δz_heat` and `Δz_cool` each as the single moment-matched
Gaussian from §4, and treating them as **independent**, variance of a sum
of independent Gaussians is the sum of their variances — applied separately
per uncertainty *source* (epistemic doesn't mix with aleatoric):

```
epistemic_var_total = epistemic_var_heat + epistemic_var_cool
aleatoric_var_total = aleatoric_var_heat + aleatoric_var_cool
total_var            = epistemic_var_total + aleatoric_var_total
total_std             = √total_var
```

implemented as `combine_stage_uncertainties(...)` in `model.py`, and exposed
per-step by both `model.rollout(...)` (keys `total_epistemic`,
`total_aleatoric`, `total_std`, alongside the individual `heat_*`/`cool_*`
values) and `model.predict_unnorm(...)` (same keys, for single-step /
online use). **`total_std` is what future RL code should read** as the
per-transition uncertainty signal — e.g. as an uncertainty-penalised reward
term or an uncertainty budget in a constrained formulation (this matches the
existing `online_RL_ucpg` package's pattern of tracking a *separate*
uncertainty return alongside the reward return, resolved via Lagrangian dual
ascent — that package's "uncertainty return" should sum `total_std` per
step, not `heat_epistemic` or `cool_epistemic` alone).

**Caveat — what this approximation does *not* capture:** treating
`Δz_heat` and `Δz_cool` as independent means we only account for their
noise terms *adding up*; we do **not** analytically propagate the heating
stage's own uncertainty about `z_heat` through the cooling ensemble's
`μ_cool(z_heat, c)` / `σ_cool(z_heat, c)` functions (that would require a
Jacobian/unscented-transform-style linearisation, since those functions are
nonlinear in their input). In other words, we capture "the cooling stage
adds its own noise on top" but not "the cooling stage's own noise estimate
would itself be higher/lower if it knew z_heat were uncertain." This is the
same simplification implicitly made when chaining ensemble means
step-to-step in ANY multi-step ensemble rollout (single-stage models don't
propagate input distributions through their networks either — they just
step the mean and separately accumulate marginal uncertainty terms). Given
K=5 (default) and both stages sharing the same encoder/decoder geometry,
this is a reasonable and cheap approximation; if it proves too loose in
practice, the fix is Monte-Carlo particle propagation (sample a handful of
`z_heat` particles from stage 1, run each through stage 2, and take the
empirical variance of the resulting `z_{t+1}` particles) rather than
anything analytic.

### 6. Loss functions (`train.py`)

**Weighted MSE** — `weighted_mse(pred, target, w, bw)`, used for every
reconstruction term below:

```
per_sample = (1/D) Σ_d w_d (pred_d − target_d)²          # w = ROI weight row for this sample's layer, or 1
L = Σ_i bw_i · per_sample_i / Σ_i bw_i                     # bw = bootstrap sample weight, or plain mean if bw is None
```

**Gaussian NLL** — `gaussian_nll(μ, logσ, target, bw)`, per-dimension
average negative log-likelihood of a diagonal Gaussian, bootstrap-weighted
the same way:

```
per_sample = (1/D) Σ_d [ 0.5·log(2π) + logσ_d + 0.5·(target_d − μ_d)² / σ_d² ]
L = Σ_i bw_i · per_sample_i / Σ_i bw_i
```

**Per-step terms** (`compute_single_step_losses`), given a batch
`(s_t, a_t, c_t, u_h, s_{t+1})` and `z_t = encode(s_t)`,
`z_h = encode(u_h)` (both computed by the shared encoder in one `forward`
call):

```
L_recon_s       = weighted_mse( decode(z_t), s_t, w )
L_recon_heat_ae = weighted_mse( decode(z_h), u_h, w )

# heating stage — decode all K members' predictions in one batched call
z_h_pred_k   = z_t + μ_heat_k                    for k = 1..K
L_recon_heat = weighted_mse( decode(z_h_pred_k), u_h, w, bw )   # mean over K, bootstrap-weighted
L_nll_heat   = gaussian_nll( μ_heat, σ_heat, target=stopgrad(encode(u_h)) − z_t, bw )

# cooling stage — teacher-forced on GROUND-TRUTH u_h (not the heating stage's own prediction)
z_next_pred_k = z_h + μ_cool_k                   for k = 1..K
L_recon_cool  = weighted_mse( decode(z_next_pred_k), s_{t+1}, w, bw )
L_nll_cool    = gaussian_nll( μ_cool, σ_cool, target=stopgrad(encode(s_{t+1})) − z_h, bw )
```

`stopgrad(·)` (`.detach()` in code) on the NLL targets prevents the encoder
from trivially minimising the NLL by collapsing the latent geometry around
its own predictions; it is **not** applied to `z_t` / `z_h` themselves, so
gradients from `L_nll_heat`/`L_nll_cool` still flow into the encoder through
the "current latent" side of the subtraction, and gradients from
`L_recon_heat`/`L_recon_cool` flow into the encoder normally.

`w = roi_table[layer_idx]` (§7) or `None` (uniform); `bw = bootstrap_masks.T`
(§8) or `None` (plain mean, used at eval time).

**Total single-step loss:**

```
L_total = recon_s_w · (L_recon_s + L_recon_heat_ae)
        + recon_heat_w · L_recon_heat + nll_heat_w · L_nll_heat
        + recon_cool_w · L_recon_cool + nll_cool_w · L_nll_cool
        [+ rollout_w · L_rollout]                      # only if --rollout_steps > 1
```

Defaults: `recon_s_w = recon_heat_w = recon_cool_w = 1.0`,
`nll_heat_w = nll_cool_w = 0.1` (NLL is a calibration signal for σ, kept
low-weight so it doesn't dominate the reconstruction objective — same ratio
v1 used for its single `nll_weight`).

**Rollout loss** (optional, `compute_rollout_loss`, k = `--rollout_steps`) —
chains *the model's own* ensemble-mean predictions (no bootstrap weighting,
no teacher forcing — this is what happens at real inference time):

```
z_0 = encode(s_0)
for t = 0 .. k−1:
    z_heat_t   = z_t + mean_k(μ_heat_k(z_t, a_t))
    z_{t+1}    = z_heat_t + mean_k(μ_cool_k(z_heat_t, c_t))
L_rollout = (1 / 2k) Σ_{t=0}^{k−1} [ weighted_mse(decode(z_heat_t), u_h_t, w_t)
                                    + weighted_mse(decode(z_{t+1}), s_{t+1,t}, w_t) ]
```

### 7. Physics-aware ROI weight table (`compute_roi_weights_table`)

The laser scans a square region that **grows** with layer index (matches
`squareSideFraction` in the MATLAB generator). For layer `l` (0-indexed) of
`n_layers`, with `fractions = linspace(initial_fraction, final_fraction, n_layers)`:

```
half_side = min(width, height) · fractions[l] / 2
box_dist(x, y) = max( |x − cx| − half_side, |y − cy| − half_side )     # signed distance to square edge, <0 inside
smooth_inside  = 1 / (1 + exp(box_dist / σ))                          # σ = edge_sigma_frac · min(width, height)
w(x, y)        = 1 + (roi_boost − 1) · smooth_inside
table[l, :]    = w / mean(w)                                          # renormalised → mean weight 1 per layer
```

Nodes inside the (softened) square get up to `roi_boost×` (default 5×) more
weight in every reconstruction loss term above, since that's the region the
`meanDeviation` reward actually cares about.

### 8. Bootstrap ensemble masks (`make_bootstrap_masks`)

For each of the K members and a dataset of N samples, draw N indices with
replacement from `{0, ..., N−1}` and count how often each original sample
was drawn:

```
counts_k = bincount( randint(0, N, size=N), minlength=N )
masks[:, k] = counts_k
```

`masks[i, k]` is sample i's multiplicity in member k's (implicit) bootstrap
resample — 0 for the ~36.8% (→ 1/e) of samples member k never draws, ≥1
otherwise. Used directly as `bw` in §6, so every member is trained (in
expectation) on its own resample while all K members still share one
batched forward/decoder pass per step. Flat (`TwoStageLatentSurrogateDataset`)
and trajectory (`TwoStageLatentTrajectoryDataset`) datasets draw independent
masks at the transition level and the whole-trajectory level respectively,
since transitions within one trajectory aren't independent population draws.

## Files

- `dataset_v2.py` — extraction of the raw `.mat` trajectories
  (`extract_single_trajectory_v2` / `gather_dataset_v2`) + the two consumer
  `Dataset` classes (`TwoStageLatentSurrogateDataset`,
  `TwoStageLatentTrajectoryDataset`), normalizers, ROI weight table, and
  bootstrap mask helpers.
- `extract_dataset_v2.py` — CLI that calls `gather_dataset_v2` and pickles
  the result (kept as a separate file from `dataset_v2.py` to sidestep a
  `python -m` + `NamedTuple` pickling gotcha — see its module docstring).
- `model.py` — `TwoStageEnsembleGaussianLatentDynamicsModel`.
- `train.py` — training loop; loss terms documented in its module docstring.
- `evaluate.py` — per-stage single-step (teacher-forced) and rollout
  (auto-regressive) metrics, uncertainty decomposition, and paired
  GT/Pred/Error field plots for both the heating and cooling fields.

## Usage

```bash
# 1. Extract + pickle the first 200 trajectories (run from repo root)
python -m surrogate_model_latent_uncertainty_v2.extract_dataset_v2 --n 200

# 2. Train
python -m surrogate_model_latent_uncertainty_v2.train \
    --data_path Data/DatasetV2_layer_12_samples_200.pkl

# 3. Evaluate
python -m surrogate_model_latent_uncertainty_v2.evaluate \
    --checkpoint surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt \
    --data_path  Data/DatasetV2_layer_12_samples_200.pkl
```

## Not ported (yet)

`evaluate_ood.py` from v1 (OOD-generalization diagnostics) was intentionally
left out of this pass — revisit once the core two-stage model is trained and
it's clear which OOD axes (laser power range? cool time range?) matter most
for this dataset.
