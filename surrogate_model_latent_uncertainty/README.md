# Latent-Space Gaussian Ensemble Surrogate for LPBF Digital Twin

A physics-aware surrogate model for predicting the temperature field evolution
in a Laser Powder Bed Fusion (LPBF) process, formulated as a **bootstrap
ensemble of Gaussian transitions in a learned latent space**, with separate
**aleatoric** and **epistemic** uncertainty estimates for online RL.

This is a variant of [`surrogate_model_latent/`](../surrogate_model_latent/)
that changes two things:

1. Each ensemble member is a **Gaussian transition MLP** (predicts a mean
   *and* a variance over the latent delta) instead of a deterministic MLP,
   so the model can express **aleatoric** (inherent, per-input) uncertainty.
2. Each of the K members is trained on its own **bootstrap resample** of the
   training set, so between-member disagreement reflects **epistemic**
   uncertainty (limited data coverage) rather than just random initialization.

Everything else — encoder/decoder architecture, layer-index embedding,
physics-aware ROI weights, evaluation plot style — is unchanged.

---

## Problem Statement

Given the current temperature field $s_t \in \mathbb{R}^{1053}$ (flattened FEM
nodes) and laser power action $a_t \in \mathbb{R}^1$, predict the next
temperature field $s_{t+1}$ after the laser pass, over 12 build layers —
along with calibrated aleatoric and epistemic uncertainty on that prediction.

---

## Architecture

One shared Encoder and Decoder, with **K independent Gaussian transition
MLPs**, each trained on its own bootstrap resample of the training set:

```
s_t  ──[Encoder]──► z_t ──┬──[GaussTrans_1]──► (μ_1, σ_1) ──┐
        (shared)           ├──[GaussTrans_2]──► (μ_2, σ_2)   │ mean(μ_k)    → z_{t+1}
                           │      ⋮                            │ std(μ_k)     → epistemic σ
                           └──[GaussTrans_K]──► (μ_K, σ_K)  ──┘ mean(σ_k²)^½ → aleatoric σ
                                                            │
s_{t+1}  ◄──[Decoder]──────────────────────────────────── z_{t+1}
              (shared)

each GaussTrans_k trained on an independent bootstrap resample of the
training set (dataset.py: make_bootstrap_masks)
```

### Why a Gaussian MLP *and* an ensemble?

A single Gaussian transition MLP only models **aleatoric** uncertainty: since
this LPBF simulation is deterministic, the optimal σ under an NLL-based loss
would collapse to near-zero as a single model improves, so on its own it
gives no signal about *data coverage*.

An ensemble whose members are only diversified by random initialization
gives **epistemic** uncertainty (disagreement on out-of-distribution inputs),
but discards any signal about aleatoric noise the model itself believes is
present.

Combining both — **K Gaussian members, each on a different bootstrap
resample** — is the standard "deep ensembles" recipe
(Lakshminarayanan et al., 2017) used for combined aleatoric + epistemic
uncertainty, and the probabilistic-ensemble recipe used in PETS-style
model-based RL (Chua et al., 2018). The two uncertainty sources are combined
via mixture-of-Gaussians moment matching:

$$\bar{\mu}_{\Delta z} = \frac{1}{K}\sum_k \mu_k, \qquad
\sigma_{\text{epist}}^2 = \text{var}_k(\mu_k), \qquad
\sigma_{\text{aleat}}^2 = \frac{1}{K}\sum_k \sigma_k^2, \qquad
\sigma_{\text{total}}^2 = \sigma_{\text{epist}}^2 + \sigma_{\text{aleat}}^2$$

### Encoder / Decoder

Unchanged from `surrogate_model_latent`: deterministic residual MLPs,
`state_dim=1053 → hidden=256 (×3 residual blocks) → latent_dim=64` and back.

### Gaussian Transition MLP (per ensemble member)

Each member independently predicts a **diagonal Gaussian** over the latent
delta, conditioned on the current latent state, action, and a learned
per-layer embedding:

$$\Delta z_k \sim \mathcal{N}\bigl(\mu_k(z_t, a_t),\, \text{diag}(\sigma_k(z_t, a_t)^2)\bigr), \quad k = 1, \ldots, K$$

$\log \sigma_k$ is bounded with a **learnable soft clamp**
(Chua et al., 2018 — "PETS"): `log_sigma = max_ls - softplus(max_ls - raw)`
then `log_sigma = min_ls + softplus(log_sigma - min_ls)`. This keeps the
Gaussian NLL loss numerically stable throughout training without the
zero-gradient dead zones a hard `clamp()` would introduce.

Default: `(latent_dim+1+layer_embed_dim) → hidden=128 (×3 residual blocks) → (μ, log σ) ∈ ℝ^{2·latent_dim}`  ×K members

### Bootstrap Resampling

Each of the K members is trained on an independent bootstrap resample of the
training set, implemented as a per-sample **multiplicity mask** rather than a
physically reordered dataset copy (`dataset.py: make_bootstrap_masks`):

- For member $k$, draw $N$ indices with replacement from the training set and
  count how many times each original sample was drawn.
- That count is used as a per-sample loss weight for member $k$: 0 excludes
  the sample, 2+ up-weights it — reproducing bootstrap statistics while
  letting all K members share one batched forward/decoder pass.
- `LatentSurrogateDataset` (flat, single-step training) bootstraps at the
  **transition level**.
- `LatentTrajectoryDataset` (rollout training) bootstraps at the
  **trajectory level**, since transitions within one trajectory are
  temporally correlated, not independent population draws.

---

## Loss Function

All states and actions are z-score normalised before entering the network.
Let $w_t \in \mathbb{R}^{1053}$ denote the per-node physics weight vector for
layer $t$ (ROI section below), and $b_k^{(i)}$ the bootstrap multiplicity of
sample $i$ for member $k$.

### 1. Autoencoder Reconstruction Loss (unchanged)

$$\mathcal{L}_{\text{recon-st}} = \frac{1}{N} \sum_i w_t^{(i)} \bigl(\hat{s}_t^{(i)} - s_t^{(i)}\bigr)^2, \quad \hat{s}_t = h_\psi(f_\phi(s_t))$$

### 2. Per-Member Next-State Prediction Loss (bootstrap-weighted)

$$\mathcal{L}_{\text{recon-st1}} = \frac{\sum_{k,i} b_k^{(i)}\, w_t^{(i)} \bigl(\hat{s}_{t+1,k}^{(i)} - s_{t+1}^{(i)}\bigr)^2}{\sum_{k,i} b_k^{(i)}}, \quad \hat{s}_{t+1,k} = h_\psi\!\left(z_t + \mu_k\right)$$

### 3. Latent Gaussian NLL Loss (bootstrap-weighted, new)

Teaches each member's σ head to calibrate against its own prediction error.
The target latent delta uses a **detached** encoder pass on $s_{t+1}$ so the
encoder cannot trivially minimise this loss by collapsing its own geometry:

$$\Delta z^{\text{target}} = \text{stopgrad}\bigl(f_\phi(s_{t+1})\bigr) - z_t$$

$$\mathcal{L}_{\text{nll}} = \frac{\sum_{k,i} b_k^{(i)} \left[\tfrac{1}{2}\log(2\pi) + \log\sigma_k^{(i)} + \tfrac{(\Delta z^{\text{target},(i)} - \mu_k^{(i)})^2}{2\sigma_k^{(i)2}}\right]}{\sum_{k,i} b_k^{(i)}}$$

### 4. Multi-Step Rollout Loss (optional, unchanged mechanics)

Auto-regressive reconstruction loss over $k$ steps using the ensemble mean
for stepping (no teacher forcing, no bootstrap weighting — mirrors
`surrogate_model_latent`'s rollout loss):

$$\mathcal{L}_{\text{rollout}} = \frac{1}{k} \sum_{t=1}^{k} \frac{1}{N} \sum_i w_{t-1}^{(i)} \bigl(h_\psi(z_t^{\text{pred}})^{(i)} - s_t^{(i)}\bigr)^2$$

### Total Loss

$$\boxed{\mathcal{L} = \lambda_{\text{rs}}\,\mathcal{L}_{\text{recon-st}} + \lambda_{\text{rs1}}\,\mathcal{L}_{\text{recon-st1}} + \lambda_{\text{nll}}\,\mathcal{L}_{\text{nll}} + \lambda_{\text{ro}}\,\mathcal{L}_{\text{rollout}}}$$

Default weights: $\lambda_{\text{rs}} = 1.0,\; \lambda_{\text{rs1}} = 1.0,\; \lambda_{\text{nll}} = 0.1,\; \lambda_{\text{ro}} = 0.1$

---

## Uncertainty Outputs

At inference (`predict_ensemble`, `rollout`, `predict_unnorm`), the model
returns three uncertainty signals per latent dimension:

| Signal | Formula | Meaning |
|---|---|---|
| Epistemic σ | $\text{std}_k(\mu_k)$ | Between-member disagreement — large when (z_t, a_t) is out of the training distribution |
| Aleatoric σ | $\sqrt{\text{mean}_k(\sigma_k^2)}$ | Average per-member predicted noise — inherent uncertainty even in-distribution |
| Total σ | $\sqrt{\sigma_{\text{epist}}^2 + \sigma_{\text{aleat}}^2}$ | Moment-matched mixture std |

For online RL, the mean over latent dimensions of each is returned per
rollout step and can be used to:
- Gate model-based rollouts (trust model only when epistemic σ is small)
- Provide an exploration bonus (visit states with high epistemic σ)
- Detect when real environment interaction is needed (MBPO-style branching)
- Distinguish "the model is uncertain because data is thin here" (epistemic,
  reducible with more data) from "the process itself is noisy here"
  (aleatoric, irreducible)

---

## Physics-Aware ROI Weights

Unchanged from `surrogate_model_latent` — see that README for the full
derivation. Summary: the laser scans a square region, centred at the domain
centre, that grows across layers (`fractions = linspace(0.4, 0.5, 12)`); nodes
inside the region get boosted loss weight via a soft-edged box signed
distance.

---

## Training

### Quick start

```bash
# Local
python -m surrogate_model_latent_uncertainty.train \
    --data_path "Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl"

# SLURM
sbatch jobs/train_latent_surrogate_uncertainty.sh
```

Override any argument at submission time:

```bash
sbatch jobs/train_latent_surrogate_uncertainty.sh --n_ensemble 7 --latent_dim 64 --nll_weight 0.2
```

### Key hyperparameters (new / changed vs. `surrogate_model_latent`)

| Argument | Default | Meaning |
|----------|---------|---------|
| `--nll_weight` | `0.1` | $\lambda_{\text{nll}}$ — weight of the latent Gaussian NLL loss |
| `--bootstrap_seed` | `-1` (→ same as `--seed`) | RNG seed for the per-member bootstrap resamples |

All other arguments (`--latent_dim`, `--n_ensemble`, `--enc_hidden`,
`--trans_hidden`, `--recon_st_weight`, `--recon_st1_weight`,
`--rollout_steps`, `--rollout_weight`, ROI args) are unchanged from
`surrogate_model_latent`.

### Outputs

```
surrogate_model_latent_uncertainty/runs/<timestamp>/
  latent_best.pt          ← best validation checkpoint
  latent_final.pt         ← final epoch checkpoint
  loss_curves.png         ← total train / val loss
  loss_components.png     ← per-component breakdown (recon_st, recon_st1, nll, rollout)
```

---

## Evaluation

```bash
python -m surrogate_model_latent_uncertainty.evaluate \
    --checkpoint surrogate_model_latent_uncertainty/runs/<ts>/latent_best.pt \
    --data_path  "Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl"
```

### Metrics reported

| Metric | Description |
|--------|-------------|
| Per-layer MAE / RMSE [K] | Single-step (teacher-forced) vs auto-regressive rollout |
| Per-layer epistemic σ | Ensemble disagreement — mean latent std across K members |
| Per-layer aleatoric σ | Average per-member predicted noise |
| Autoencoder RMSE [K] | Encoder → decoder round-trip error |
| Loss components | `recon_st`, `recon_st1`, `nll` on test set |

### Outputs

```
<run_dir>/
  per_layer_mae.png
  per_layer_rmse.png
  per_layer_uncertainty.png       ← epistemic | aleatoric | total, 3 panels
  per_layer_diagnostics.png
  traj_000/
    layer_01_LP<p>W.png           ← GT | Pred | Error  (jet/hot colormap, MATLAB style)
    action_sequence.png
    sigma_per_layer.png           ← epistemic & aleatoric curves for this trajectory
    actions.txt
```

---

## File Structure

```
surrogate_model_latent_uncertainty/
  model.py      ← Encoder, GaussianTransitionMLP, Decoder,
                   EnsembleGaussianLatentDynamicsModel
  dataset.py    ← LatentSurrogateDataset, LatentTrajectoryDataset,
                   build_normalizers, compute_roi_weights_table,
                   make_bootstrap_masks
  train.py      ← loss functions (incl. Gaussian NLL), training loop,
                   load_latent_surrogate()
  evaluate.py   ← per-layer metrics, aleatoric/epistemic plots,
                   2-D field plots (MATLAB pdeplot style)
  README.md     ← this file

jobs/
  train_latent_surrogate_uncertainty.sh   ← SLURM job script
```

---

## Comparison with `surrogate_model_latent/`

| | `surrogate_model_latent` | `surrogate_model_latent_uncertainty` |
|---|---|---|
| Transition head | Deterministic MLP | Gaussian MLP (μ, σ) with PETS soft-clamped log σ |
| Ensemble diversity | Random init only | Random init **+ bootstrap resample per member** |
| Uncertainty | Epistemic only | Epistemic **and** aleatoric, decomposed |
| Loss | Recon-st + Recon-st1 (+ rollout) | Recon-st + Recon-st1 + **Latent NLL** (+ rollout) |
| Physics weights | Square ROI, per-layer, growing | Same |
| Parameters (K=5) | ~1.9 M | ~1.9 M + K×(latent_dim σ-head + 2·latent_dim soft-clamp params) |
