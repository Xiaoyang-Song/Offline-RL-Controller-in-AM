# Latent-Space Ensemble Dynamics Surrogate for LPBF Digital Twin

A physics-aware surrogate model for predicting the temperature field evolution
in a Laser Powder Bed Fusion (LPBF) process, formulated as a **deterministic
ensemble transition in a learned latent space** with epistemic uncertainty
estimates for online RL.

---

## Problem Statement

Given the current temperature field $s_t \in \mathbb{R}^{1053}$ (flattened FEM
nodes) and laser power action $a_t \in \mathbb{R}^1$, predict the next
temperature field $s_{t+1}$ after the laser pass, over 12 build layers.

---

## Architecture

One shared Encoder and Decoder, with **K independent deterministic transition
MLPs** forming the ensemble:

```
s_t  ──[Encoder]──► z_t ──┬──[Trans_1]──► μ_Δz_1 ──┐
        (shared)           ├──[Trans_2]──► μ_Δz_2    │  mean → z_{t+1}
                           │      ⋮                   │  std  → epistemic σ
                           └──[Trans_K]──► μ_Δz_K ──┘
                                                   │
s_{t+1}  ◄──[Decoder]──────────────────────────── z_{t+1}
              (shared)
```

### Why ensembles instead of a Gaussian MLP?

A single Gaussian transition MLP only models **aleatoric** uncertainty (inherent
process noise). Since the LPBF simulation is deterministic, the optimal σ under
any NLL-based loss collapses to zero as the model improves — making the
uncertainty signal useless.

Ensemble disagreement models **epistemic** uncertainty — uncertainty from
limited data coverage. K members agree on in-distribution (state, action) pairs
and diverge on out-of-distribution inputs, giving the online RL controller a
reliable signal about when not to trust the model.

### Encoder
A deterministic residual MLP mapping the temperature field to a compact latent
vector:

$$z_t = f_\phi(s_t), \quad z_t \in \mathbb{R}^{d_z}$$

Default: `state_dim=1053 → hidden=256 (×3 residual blocks) → latent_dim=64`

### Deterministic Transition MLP (per ensemble member)
Each member independently predicts the latent delta conditioned on the current
latent state and action:

$$\mu_{\Delta z}^{(k)} = g_{\theta_k}(z_t,\, a_t), \quad k = 1, \ldots, K$$

The ensemble mean and std are:

$$\bar{\mu}_{\Delta z} = \frac{1}{K}\sum_k \mu_{\Delta z}^{(k)}, \qquad
\sigma_{\text{epist}} = \text{std}_k\!\left(\mu_{\Delta z}^{(k)}\right)$$

The next latent is stepped with the mean: $z_{t+1} = z_t + \bar{\mu}_{\Delta z}$.

Default: `(latent_dim+1) → hidden=128 (×3 residual blocks) → latent_dim`  ×K members

### Decoder
A deterministic residual MLP mapping back to state space:

$$\hat{s} = h_\psi(z), \quad \hat{s} \in \mathbb{R}^{1053}$$

Default: `latent_dim=64 → hidden=256 (×3 residual blocks) → state_dim=1053`

All sub-networks use `Linear → LayerNorm → SiLU` residual blocks.

---

## Loss Function

All states and actions are z-score normalised before entering the network.
Let $w_t \in \mathbb{R}^{1053}$ denote the per-node physics weight vector for
layer $t$ (see ROI section below).

### 1. Autoencoder Reconstruction Loss

Ensures the shared encoder–decoder pair forms a faithful autoencoder of the
current state:

$$\mathcal{L}_{\text{recon-st}} = \frac{1}{N} \sum_i w_t^{(i)} \bigl(\hat{s}_t^{(i)} - s_t^{(i)}\bigr)^2, \quad \hat{s}_t = h_\psi(f_\phi(s_t))$$

### 2. Per-Member Next-State Prediction Loss

Each ensemble member independently predicts $s_{t+1}$. All K losses are
averaged so every member learns the full dynamics:

$$\mathcal{L}_{\text{recon-st1}} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{N} \sum_i w_t^{(i)} \bigl(\hat{s}_{t+1,k}^{(i)} - s_{t+1}^{(i)}\bigr)^2, \quad \hat{s}_{t+1,k} = h_\psi\!\left(z_t + \mu_{\Delta z}^{(k)}\right)$$

The batched decoder call `decode(K×B predictions)` makes this efficient — a
single decoder forward pass covers all K members.

### 3. Multi-Step Rollout Loss (optional)

Auto-regressive reconstruction loss over $k$ steps using the ensemble mean for
stepping (no teacher forcing):

$$z_{t+1}^{\text{pred}} = z_t^{\text{pred}} + \bar{\mu}_{\Delta z}(z_t^{\text{pred}},\, a_t)$$

$$\mathcal{L}_{\text{rollout}} = \frac{1}{k} \sum_{t=1}^{k} \frac{1}{N} \sum_i w_{t-1}^{(i)} \bigl(h_\psi(z_t^{\text{pred}})^{(i)} - s_t^{(i)}\bigr)^2$$

This penalises **error accumulation** across layers.

### Total Loss

$$\boxed{\mathcal{L} = \lambda_{\text{rs}}\,\mathcal{L}_{\text{recon-st}} + \lambda_{\text{rs1}}\,\mathcal{L}_{\text{recon-st1}} + \lambda_{\text{ro}}\,\mathcal{L}_{\text{rollout}}}$$

Default weights: $\lambda_{\text{rs}} = 1.0,\; \lambda_{\text{rs1}} = 1.0,\; \lambda_{\text{ro}} = 0.1$

---

## Epistemic Uncertainty

At inference, the ensemble std over latent deltas is the epistemic uncertainty
signal:

$$\sigma_{\text{epist}}(z_t, a_t) = \text{std}_k\!\left(\mu_{\Delta z}^{(k)}\right) \in \mathbb{R}^{d_z}$$

This quantity is:
- **Small** when (z_t, a_t) is well covered by training data — all K members agree
- **Large** when (z_t, a_t) is out of distribution — members diverge

For online RL, the mean over latent dimensions $\bar{\sigma}_{\text{epist}} = \frac{1}{d_z}\sum_j \sigma_{\text{epist},j}$ is returned per rollout step and can be used to:
- Gate model-based rollouts (trust model only when $\bar{\sigma}_{\text{epist}}$ is small)
- Provide an exploration bonus (visit states with high $\bar{\sigma}_{\text{epist}}$)
- Detect when real environment interaction is needed (MBPO-style branching)

---

## Physics-Aware ROI Weights

### Motivation

The LPBF simulation applies the laser within a **square scan region** that
is centred at the domain centre and **grows** across layers. Nodes inside this
region undergo the most significant temperature change and should receive higher
loss weight.

### Square Scan Region

From the MATLAB simulation parameters:

```matlab
fractions = linspace(initialFraction, finalFraction, nSteps)
% = linspace(0.4, 0.5, 12)

squareSide = min(width, height) * fractions(l)
halfSide   = squareSide / 2
centre     = (width/2, height/2)
```

So at layer $l$ (0-indexed), the ROI square has:

$$\text{half-side}^{(l)} = \min(W, H) \times \underbrace{\left[0.4 + \frac{l}{11} \times 0.1\right]}_{\text{linspace}(0.4,\,0.5,\,12)[l]} \;\Big/ \;2$$

The square is **always centred** at $(W/2,\, H/2)$; only its size evolves.

### Weight Computation

The weight for node $i$ at layer $l$ uses the **box signed distance** (positive
outside, negative inside):

$$d^{(i,l)} = \max\!\Bigl(\bigl|x^{(i)} - c_x\bigr| - h^{(l)},\;\bigl|y^{(i)} - c_y\bigr| - h^{(l)}\Bigr)$$

A sigmoid soft-step converts distance to weight:

$$\tilde{w}^{(i,l)} = 1 + (\lambda_{\text{roi}} - 1) \cdot \underbrace{\sigma\!\left(-\frac{d^{(i,l)}}{\delta}\right)}_{\to\,1 \text{ inside},\; \to\,0 \text{ outside}}$$

Each row is normalised so the mean weight per layer equals 1.

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ROI_INITIAL_FRACTION` | `0.4` | `squareSideFraction` at layer 0 |
| `ROI_FINAL_FRACTION` | `0.5` | `squareSideFraction` at layer 11 |
| `ROI_BOOST` | `5.0` | Weight multiplier inside the square |
| `ROI_EDGE_SIGMA_FRAC` | `0.05` | Soft-edge width as fraction of min(width, height) |

---

## Training

### Quick start

```bash
# Local
python -m surrogate_model_latent.train \
    --data_path "Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl"

# SLURM
sbatch jobs/train_latent_surrogate.sh
```

Override any argument at submission time:

```bash
sbatch jobs/train_latent_surrogate.sh --n_ensemble 7 --latent_dim 64
```

### Key hyperparameters

| Argument | Default | Meaning |
|----------|---------|---------|
| `--latent_dim` | `64` | Latent space dimension $d_z$ |
| `--n_ensemble` | `5` | Number of ensemble members $K$ |
| `--enc_hidden / enc_depth` | `256 / 3` | Encoder width and depth |
| `--trans_hidden / trans_depth` | `128 / 3` | Transition MLP width and depth (per member) |
| `--dec_hidden / dec_depth` | `256 / 3` | Decoder width and depth |
| `--recon_st_weight` | `1.0` | $\lambda_{\text{rs}}$ |
| `--recon_st1_weight` | `1.0` | $\lambda_{\text{rs1}}$ |
| `--rollout_steps` | `12` | $k$ for rollout loss (0 = off) |
| `--rollout_weight` | `0.1` | $\lambda_{\text{ro}}$ |

### Parameter count (K=5, latent_dim=32)

| Component | Parameters |
|-----------|-----------|
| Encoder (shared) | 676,384 |
| Transitions (×5) | 546,720 |
| Decoder (shared) | 677,405 |
| **Total** | **1,900,509** |

### Outputs

```
surrogate_model_latent/runs/<timestamp>/
  latent_best.pt          ← best validation checkpoint
  latent_final.pt         ← final epoch checkpoint
  loss_curves.png         ← total train / val loss
  loss_components.png     ← per-component breakdown (recon_st, recon_st1, rollout)
```

---

## Evaluation

```bash
python -m surrogate_model_latent.evaluate \
    --checkpoint surrogate_model_latent/runs/<ts>/latent_best.pt \
    --data_path  "Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl"
```

### Metrics reported

| Metric | Description |
|--------|-------------|
| Per-layer MAE [K] | Teacher-forced single-step, raw Kelvin |
| Per-layer RMSE [K] | Single-step vs auto-regressive rollout |
| Per-layer epistemic std | Mean ensemble std over latent dims per rollout step |
| Autoencoder RMSE [K] | Encoder → decoder round-trip error |
| Loss components | `recon_st`, `recon_st1` on test set |

### Outputs

```
<run_dir>/
  per_layer_mae.png
  per_layer_rmse.png
  per_layer_uncertainty.png       ← epistemic std per layer (ensemble disagreement)
  traj_000/
    layer_01_LP<p>W.png           ← GT | Pred | Error  (jet/hot colormap, MATLAB style)
    action_sequence.png
    sigma_per_layer.png           ← epistemic std per layer for this trajectory
    actions.txt
```

---

## File Structure

```
surrogate_model_latent/
  model.py      ← Encoder, DeterministicTransitionMLP, Decoder,
                   EnsembleLatentDynamicsModel
  dataset.py    ← LatentSurrogateDataset, LatentTrajectoryDataset,
                   build_normalizers, compute_roi_weights_table
  train.py      ← loss functions, training loop, load_latent_surrogate()
  evaluate.py   ← per-layer metrics, 2-D field plots (MATLAB pdeplot style)
  README.md     ← this file

jobs/
  train_latent_surrogate.sh   ← SLURM job script
```

---

## Comparison with `surrogate_model/`

| | `surrogate_model` | `surrogate_model_latent` |
|---|---|---|
| Prediction space | State space directly | Latent space $\mathbb{R}^{d_z}$ |
| Uncertainty | None | Epistemic: ensemble std of K members |
| Uncertainty type | — | Epistemic (OOD detection, online RL) |
| Loss | MSE + rollout MSE | Recon-st + Recon-st1 (per-member) + rollout |
| Physics weights | None | Square ROI, per-layer, growing |
| Parameters (K=5) | ~1.5 M | ~1.9 M (K-1 extra transition MLPs) |
