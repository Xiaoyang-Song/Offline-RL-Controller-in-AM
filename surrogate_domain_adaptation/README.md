# Few-Shot Domain Adaptation of the Latent Surrogate for Cooling Time Shift

This module extends `surrogate_model_latent` to handle a **cooling-time
domain shift**: the base surrogate is trained on source cooling times
(ct_050–ct_100, i.e. 0.05–0.10 s) and then adapted to a target domain
(ct_150, i.e. 0.15 s) using only **K adaptation trajectories**.

The base model architecture and all loss functions are identical to
`surrogate_model_latent`.  The only addition is three lightweight
`DomainAdapter` modules that are inserted at the encoder output, transition
output, and decoder input — trained while the base model is kept frozen.

---

## Motivation: Why Cooling Time Is a Domain Shift

Longer inter-layer cooling time lets the part cool down more between laser
passes, so the initial temperature for each subsequent layer is lower:

| Cooling time | Mean field at layer 12 |
|---|---|
| ct_050 (0.05 s) | ≈ 2274 K |
| ct_150 (0.15 s) | ≈  953 K |

This 2.4× temperature difference manifests as three compounding effects that
a fixed surrogate cannot handle without adaptation:

1. **Encoder shift** — states lie in a different region of the input manifold.
2. **Transition shift** — the thermal response (ΔT per laser pass) is smaller when starting from a cooler state.
3. **Decoder shift** — the latent-to-state mapping shifts with the temperature scale.

---

## Architecture

The base `EnsembleLatentDynamicsModel` from `surrogate_model_latent` is
**unchanged and frozen** during adaptation.  Three `DomainAdapter` modules
are grafted onto it:

```
s_t ──[Encoder]──► z_t ──[EncAdapter]──► z̃_t ──┬──[Trans_1]──► δz_1 ──┐
       (frozen)             (trained)              ├──[Trans_2]──► δz_2    │  mean → Δz
                                                   │      ⋮                 │
                                                   └──[Trans_K]──► δz_K ──┘
                                                                       │
                                               [TransAdapter] (trained)│
                                                                       ▼
s_{t+1} ◄──[Decoder]──[DecAdapter]──── z̃_{t+1} = z̃_t + Δz (adapted)
           (frozen)     (trained)
```

### DomainAdapter

Each adapter is a bottleneck residual MLP that learns only the **shift** away
from the base representation, initialised to the identity:

$$x \;\mapsto\; x + W_{\text{up}}\!\left(\sigma\!\left(W_{\text{down}}\,x\right)\right)$$

`W_up` is zero-initialised, so the adapted model is **identical to the base
at K=0** and only diverges as adaptation data is provided.

Default bottleneck: 32.  Parameter count ≈ 2.1 K per adapter, **6.3 K total**
— less than 0.4% of the 1.9 M base model.

### Adapter placement

| Adapter | Input dim | Where inserted |
|---------|-----------|----------------|
| `enc_adapter` | `latent_dim` | After `Encoder`, before transitions |
| `trans_adapter` | `latent_dim` | Applied to every ensemble member's output `δz_k` |
| `dec_adapter` | `latent_dim` | Before `Decoder`, after computing `z_{t+1}` |

The `trans_adapter` is shared across all K ensemble members.  Applying it to
each member's output preserves ensemble diversity (the std over members is
unaffected by a shared shift).

---

## Two-Stage Training

### Stage 1 — Base model (source domains)

Train a standard `EnsembleLatentDynamicsModel` on pooled data from all source
cooling times (ct_050 to ct_100, 6 domains × 300 trajectories each):

```
source pool: 1800 trajectories total
  train 80% / val 10% / test 10%  (trajectory-level split)
```

Loss is identical to `surrogate_model_latent`:

$$\mathcal{L} = \lambda_{\text{rs}}\,\mathcal{L}_{\text{recon-st}} + \lambda_{\text{rs1}}\,\mathcal{L}_{\text{recon-st1}} + \lambda_{\text{ro}}\,\mathcal{L}_{\text{rollout}}$$

with per-layer ROI weights inherited from the mesh geometry.

### Stage 2 — Adapter-only few-shot adaptation (target domain)

Freeze the base model.  For each K ∈ {5, 10, 20, 50, 100}:

1. Sample K trajectories from ct_150.
2. Train only the three `DomainAdapter` modules on those K trajectories.
3. Evaluate on a fixed held-out test set (30% of ct_150, never seen during adaptation).

At K=5 the adapter trains on just 60 transitions (5 traj × 12 layers), so
the rollout loss is disabled to prevent overfitting.  For K≥20 it is
optionally re-enabled.

Normalisation statistics (mean, std) always come from the **source training
data** — the adapter learns to bridge the domain gap rather than re-normalize.

---

## Running

### Step 1 — Train base surrogate on source domains

```bash
sbatch jobs/train_base_domain_surrogate.sh
# outputs → surrogate_domain_adaptation/runs/base/<timestamp>/base_best.pt
```

### Step 2 — Few-shot adaptation + evaluation plots

```bash
sbatch jobs/adapt_domain_surrogate.sh \
    --base_checkpoint surrogate_domain_adaptation/runs/base/<ts>/base_best.pt
```

This runs `adapt.py` (saves per-K checkpoints and metrics) then
`evaluate.py` (produces all comparison plots).

**Chain with SLURM dependency:**

```bash
JID=$(sbatch --parsable jobs/train_base_domain_surrogate.sh)
sbatch --dependency=afterok:$JID jobs/adapt_domain_surrogate.sh \
    --base_checkpoint surrogate_domain_adaptation/runs/base/<ts>/base_best.pt
```

---

## Outputs

```
surrogate_domain_adaptation/runs/base/<timestamp>/
  base_best.pt              ← best-val base checkpoint (source domains)
  base_final.pt             ← final epoch checkpoint
  loss_curves.png           ← train / val loss over epochs
  adapt_ct150/
    K005/
      adapted.pt            ← adapter weights for K=5 (base frozen)
      metrics.json          ← MAE, RMSE per layer for K=5
    K010/ ...
    K020/ ...
    K050/ ...
    K100/ ...
    metrics_summary.json    ← all K results side-by-side
    few_shot_learning_curve.png   ← MAE and RMSE vs K
    per_layer_mae.png             ← per-layer MAE for K=0 (base) and each K
    per_layer_rmse.png            ← per-layer RMSE for K=0 (base) and each K
    metrics_table.txt             ← numeric summary table
```

---

## Comparison with `surrogate_model_latent`

| | `surrogate_model_latent` | `surrogate_domain_adaptation` |
|---|---|---|
| Training data | Single cooling time (pkl) | 6 source domains (mat files) |
| Target domain | — | ct_150 (0.15 s, few-shot) |
| Extra parameters | — | 3 × DomainAdapter ≈ 6.3 K params |
| Base frozen at adaptation | — | Yes |
| Architecture | EnsembleLatentDynamicsModel | Same + 3 adapters |
| Loss functions | Identical | Identical (adapters pass through) |
| Uncertainty | Ensemble epistemic std | Same (preserved through adapters) |

---

## File Structure

```
surrogate_domain_adaptation/
  dataset.py      ← load_domain_trajectories, load_source_domains,
                     load_target_domain, sample_few_shot, build_normalizers,
                     MultiDomainFlatDataset, MultiDomainTrajectoryDataset
  model.py        ← DomainAdapter, AdaptedEnsembleLatentDynamicsModel,
                     save_adapted, load_adapted
  train_base.py   ← base model training loop + checkpoint helpers
  adapt.py        ← per-K few-shot adaptation loop + evaluation
  evaluate.py     ← plots: few-shot learning curve, per-layer MAE/RMSE
  __init__.py

jobs/
  train_base_domain_surrogate.sh   ← SLURM: Stage 1
  adapt_domain_surrogate.sh        ← SLURM: Stage 2 (adapt + evaluate)
```
