# Latent-Space Dynamics Surrogate for LPBF Digital Twin

A physics-aware surrogate model for predicting the temperature field evolution
in a Laser Powder Bed Fusion (LPBF) process, formulated as a **stochastic
transition in a learned latent space** with uncertainty-aware predictions.

---

## Problem Statement

Given the current temperature field $s_t \in \mathbb{R}^{1053}$ (flattened FEM
nodes) and laser power action $a_t \in \mathbb{R}^1$, predict the next
temperature field $s_{t+1}$ after the laser pass, over 12 build layers.

---

## Architecture

The model has three learned components:

```
s_t  ──[Encoder]──► z_t ──[GaussianTransitionMLP]──► μ_Δz, log σ_Δz
                                                            │
                          z_{t+1} = z_t + μ_Δz + ε·σ_Δz   │  ε ~ N(0, I)
                                                            ▼
s_{t+1}  ◄──[Decoder]──  z_{t+1}
```

### Encoder
A deterministic residual MLP mapping the temperature field to a compact latent
vector:

$$z_t = f_\phi(s_t), \quad z_t \in \mathbb{R}^{d_z}$$

Default: `state_dim=1053 → hidden=256 (×3 residual blocks) → latent_dim=64`

### Gaussian Transition MLP
A stochastic MLP predicting the **mean and log-standard-deviation of the latent
delta**, conditioned on the current latent state and action:

$$(\mu_{\Delta z},\, \log \sigma_{\Delta z}) = g_\theta(z_t,\, a_t)$$

The next latent state is:

$$z_{t+1} = z_t + \mu_{\Delta z} + \varepsilon \cdot \exp(\log \sigma_{\Delta z}), \quad \varepsilon \sim \mathcal{N}(0, I)$$

During training the **deterministic path** (mean only, $\varepsilon = 0$) is
used for reconstruction losses. Stochastic sampling is used at rollout /
inference time to propagate uncertainty.

Default: `(latent_dim+1=65) → hidden=128 (×3 residual blocks) → 2×latent_dim`

Log-sigma is clamped to $[-5,\, 2]$ to prevent NLL explosion or collapse.

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

Ensures the encoder–decoder pair forms a faithful autoencoder of the current
state:

$$\mathcal{L}_{\text{recon-st}} = \frac{1}{N} \sum_i w_t^{(i)} \bigl(\hat{s}_t^{(i)} - s_t^{(i)}\bigr)^2, \quad \hat{s}_t = h_\psi(f_\phi(s_t))$$

### 2. Next-State Prediction Loss

Ensures the full encoder → transition (mean) → decoder chain correctly predicts
$s_{t+1}$:

$$\mathcal{L}_{\text{recon-st1}} = \frac{1}{N} \sum_i w_t^{(i)} \bigl(\hat{s}_{t+1}^{(i)} - s_{t+1}^{(i)}\bigr)^2, \quad \hat{s}_{t+1} = h_\psi\!\left(z_t + \mu_{\Delta z}\right)$$

### 3. Gaussian NLL Transition Loss

Trains the transition model to be **calibrated in latent space**. The ground
truth latent delta is obtained by encoding the next state:

$$\Delta z^* = f_\phi(s_{t+1}) - z_t$$

The Gaussian NLL of $\Delta z^*$ under the predicted distribution $\mathcal{N}(\mu_{\Delta z},\, \sigma_{\Delta z}^2)$ is:

$$\mathcal{L}_{\text{nll}} = \frac{1}{d_z} \sum_{j=1}^{d_z} \frac{1}{2} \left[ 2 \log \sigma_{\Delta z}^{(j)} + \frac{\left(\Delta z^{*(j)} - \text{sg}[\mu_{\Delta z}^{(j)}]\right)^2}{\sigma_{\Delta z}^{(j)\,2}} \right]$$

where $\text{sg}[\cdot]$ denotes **stop-gradient** (`.detach()` in PyTorch).

The constant $\frac{1}{2}\log(2\pi)$ is omitted as it does not affect gradients.

**Why stop-gradient on $\mu$?**
Without it, the NLL drives $\sigma \to 0$ as $\mu$ improves (variance collapse):
the equilibrium $\sigma^{(j)} = |\Delta z^{*(j)} - \mu^{(j)}|$ means a more
accurate $\mu$ directly shrinks $\sigma$, eventually saturating the lower clamp
at `log_sigma_min=-5` ($\sigma \approx 0.0067$) for all layers.

With stop-gradient, $\mu$ and $\sigma$ are trained by **separate signals**:
- $\mu_{\Delta z}$ ← reconstruction losses $\mathcal{L}_{\text{recon-st1}}$ and $\mathcal{L}_{\text{rollout}}$
- $\log\sigma_{\Delta z}$ ← NLL only, learning the actual residual magnitude $|\Delta z^* - \text{sg}[\mu]|$ per layer

### 4. Multi-Step Rollout Loss (optional)

Auto-regressive reconstruction loss over $k$ steps in latent space (no teacher
forcing, no sampling — uses mean path):

$$z_{t+1}^{\text{pred}} = z_t^{\text{pred}} + \mu_{\Delta z}(z_t^{\text{pred}},\, a_t)$$

$$\mathcal{L}_{\text{rollout}} = \frac{1}{k} \sum_{t=1}^{k} \frac{1}{N} \sum_i w_{t-1}^{(i)} \bigl(h_\psi(z_t^{\text{pred}})^{(i)} - s_t^{(i)}\bigr)^2$$

This penalises **error accumulation** across layers, analogous to the rollout
loss in `surrogate_model/`.

### Total Loss

$$\boxed{\mathcal{L} = \lambda_{\text{rs}}\,\mathcal{L}_{\text{recon-st}} + \lambda_{\text{rs1}}\,\mathcal{L}_{\text{recon-st1}} + \lambda_{\text{nll}}\,\mathcal{L}_{\text{nll}} + \lambda_{\text{ro}}\,\mathcal{L}_{\text{rollout}}}$$

Default weights: $\lambda_{\text{rs}} = 1.0,\; \lambda_{\text{rs1}} = 1.0,\; \lambda_{\text{nll}} = 0.1,\; \lambda_{\text{ro}} = 0.1$

The NLL weight is kept smaller because the NLL is on a different scale
(latent space, dimensionality $d_z$) compared to the MSE losses (state space,
dimensionality 1053).

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

where $\delta = \delta_{\text{frac}} \times \min(W, H)$ is the soft-edge width.

Each row is then normalised so the **mean weight per layer equals 1**:

$$w^{(i,l)} = \frac{\tilde{w}^{(i,l)}}{\frac{1}{N}\sum_j \tilde{w}^{(j,l)}}$$

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ROI_INITIAL_FRACTION` | `0.4` | `squareSideFraction` at layer 0 — from MATLAB `initialFraction` |
| `ROI_FINAL_FRACTION` | `0.5` | `squareSideFraction` at layer 11 — from MATLAB `finalFraction` |
| `ROI_BOOST` | `5.0` | $\lambda_{\text{roi}}$: weight multiplier inside the square |
| `ROI_EDGE_SIGMA_FRAC` | `0.05` | $\delta_{\text{frac}}$: soft-edge width (numerical only, no physical meaning) |

`ROI_INITIAL_FRACTION` and `ROI_FINAL_FRACTION` should match the MATLAB
`paramsStruct` exactly and should not be changed unless the simulation changes.

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
sbatch jobs/train_latent_surrogate.sh --latent_dim 128 --nll_weight 0.05
```

### Key hyperparameters

| Argument | Default | Meaning |
|----------|---------|---------|
| `--latent_dim` | `64` | Latent space dimension $d_z$ |
| `--enc_hidden / enc_depth` | `256 / 3` | Encoder width and depth |
| `--trans_hidden / trans_depth` | `128 / 3` | Transition MLP width and depth |
| `--dec_hidden / dec_depth` | `256 / 3` | Decoder width and depth |
| `--recon_st_weight` | `1.0` | $\lambda_{\text{rs}}$ |
| `--recon_st1_weight` | `1.0` | $\lambda_{\text{rs1}}$ |
| `--nll_weight` | `0.1` | $\lambda_{\text{nll}}$ |
| `--rollout_steps` | `12` | $k$ for rollout loss (0 = off) |
| `--rollout_weight` | `0.1` | $\lambda_{\text{ro}}$ |

### Outputs

```
surrogate_model_latent/runs/<timestamp>/
  latent_best.pt          ← best validation checkpoint
  latent_final.pt         ← final epoch checkpoint
  loss_curves.png         ← total train / val loss
  loss_components.png     ← per-component breakdown
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
| Per-layer mean $\sigma$ | Mean transition uncertainty across latent dims |
| Autoencoder RMSE [K] | Encoder → decoder round-trip error |
| Loss components | `recon_st`, `recon_st1`, `nll` on test set |

### Outputs

```
<run_dir>/
  per_layer_mae.png
  per_layer_rmse.png
  per_layer_uncertainty.png
  traj_000/
    layer_01_LP<p>W.png     ← GT | Pred | Error  (jet/hot colormap, MATLAB style)
    action_sequence.png
    sigma_per_layer.png
    actions.txt
```

---

## File Structure

```
surrogate_model_latent/
  model.py      ← Encoder, GaussianTransitionMLP, Decoder, LatentDynamicsModel
  dataset.py    ← LatentSurrogateDataset, LatentTrajectoryDataset,
                   build_normalizers, compute_roi_weights_table
  train.py      ← multi-term loss, training loop, load_latent_surrogate()
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
| Uncertainty | None | Per-dim $\sigma$ from transition MLP |
| Loss | MSE + rollout MSE | Recon-st + Recon-st1 + NLL + rollout |
| Physics weights | None | Square ROI, per-layer, growing |
| Parameters | ~1.5 M | ~1.5 M (split across 3 networks) |
