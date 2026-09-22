# surrogate_model_v3

The primary LPBF digital-twin surrogate, consolidated into a single
self-contained package. Design: **two-stage (heating → cooling), bootstrap
Gaussian ensemble, NO learned latent bottleneck.**

## Why this package exists

Earlier iterations (`surrogate_model/` → `surrogate_model_latent/` →
`surrogate_model_latent_uncertainty/` → `surrogate_model_latent_uncertainty_v2/`)
progressively added a two-stage heating/cooling split, then a learned
encoder/decoder latent space, then epistemic+aleatoric uncertainty via a
bootstrap Gaussian ensemble. Comparing the latent-space main surrogate
against its own ablation that removes just the latent bottleneck
(`baseline_surrogate/ablation_no_latent/`) showed the encoder/decoder
**hurting** accuracy badly:

| Regime | with learned latent | without latent (raw state) |
|---|---|---|
| Teacher-forced  | 13.68 / 26.87 K (MAE/RMSE) | **0.997 / 3.92 K** |
| Auto-regressive | 23.93 / 43.48 K            | **2.08 / 5.80 K** |

(see `baseline_surrogate/README.md`'s "Result: the latent encoder/decoder
underperforms" section for the full experiment). The project reverted to
**not** using a learned latent space. `surrogate_model_v3/` is that
no-latent design, extracted out of `baseline_surrogate/ablation_no_latent/`
(which depended on `surrogate_model_latent_uncertainty_v2` for its dataset
pipeline and `baseline_surrogate/common/` for training/eval plumbing) into
its own standalone package — nothing here imports from either of those.

## Design

  - **Stage 1 (heating):** `s_t --[a_t = laser power]--> u_heat_t` —
    action-DEPENDENT, this is where the controller acts.
  - **Stage 2 (cooling):** `u_heat_t --[cool_time_t]--> s_{t+1}` —
    action-INDEPENDENT; cooling physics is the same regardless of what
    laser power produced `u_heat_t`, so this stage is conditioned on
    `cool_time` instead.
  - **No encoder/decoder.** Both stages' K-member Gaussian ensembles act
    directly on the raw (z-scored) 1053-dim temperature field. Each member
    is a `GaussianTransitionMLP` predicting `(μ_Δ, log σ_Δ)` with a
    PETS-style (Chua et al., 2018) learnable soft-clamp on `log σ`.
  - **Bootstrap ensemble.** Each of the K members trains on its own
    bootstrap resample of the training set (per-sample multiplicity
    weights, not physically reordered data), decomposing uncertainty into
    epistemic (ensemble disagreement) and aleatoric (average member noise)
    via mixture-of-Gaussians moment matching (Lakshminarayanan et al., 2017).
  - **Teacher-forced training.** The cooling stage always trains on the
    ground-truth `u_heat_t`, not the heating stage's own prediction, so the
    two stages' losses are independent — matching the physical fact that
    cooling doesn't care how `u_heat_t` was produced.

See `model.py` for the full derivation (module + class docstrings) and
`train.py` for the loss terms.

## File layout

```
dataset.py           trajectory extraction/loading, normalizers, bootstrap
                      masks, TwoStageSurrogateDataset (flat, training),
                      TwoStageTrajectoryDataset (full-trajectory, eval/rollout)
extract_dataset.py    CLI: extract + pickle trajectories from the raw .mat sim output
model.py              GaussianTransitionMLP, TwoStageSurrogate, load_surrogate()
train.py              training CLI + loss functions (weighted MSE + Gaussian NLL)
evaluate.py           per-layer teacher-forced/rollout MAE/RMSE, uncertainty
                      plots, example field plots (train/val/test)
evaluate_ood.py        ID-vs-OOD epistemic-uncertainty stress test (narrow or
                      gapped checkpoint vs. the wider dataset it was filtered from)
evaluate_ood_ratio.py  coverage-normalized OOD check: epistemic-σ ratio of a
                      patchy checkpoint vs. a matched full-range checkpoint
jobs/                 SLURM submission scripts for all of the above
```

## Usage

All commands run from the repo root, `conda activate RL`.

### 0. Dataset

`Data/DatasetV2_layer_12_samples_5000.pkl` already exists (5000
trajectories × 12 layers) and works out of the box — `dataset.py`'s
`load_trajectories` can read it even though it was pickled by the sibling
`surrogate_model_latent_uncertainty_v2` package (pickle resolves each
object's class by the module recorded at pickle time, not by whichever
module calls `pickle.load`; see `dataset.py`'s docstring). Only extract more
trajectories if you need a larger dataset:

```bash
python -m surrogate_model_v3.extract_dataset --n 5000
# or: sbatch surrogate_model_v3/jobs/dataset.sh
```

### 1. Train

Full laser-power range (canonical checkpoint):

```bash
python -m surrogate_model_v3.train \
    --data_path Data/DatasetV2_layer_12_samples_5000.pkl \
    --out_dir   surrogate_model_v3/runs/full_range
# or: sbatch surrogate_model_v3/jobs/train_full_range.sh
```

Narrow 200-300W range (for the ID-vs-OOD uncertainty check):

```bash
python -m surrogate_model_v3.train \
    --data_path Data/DatasetV2_layer_12_samples_5000.pkl \
    --lp_filter_min 200 --lp_filter_max 300 \
    --out_dir   surrogate_model_v3/runs/narrow_200_300W
# or: sbatch surrogate_model_v3/jobs/train_narrow_200_300W.sh
```

Patchy/gapped coverage + target perturbation + decorrelated bootstrap (a
harder epistemic-uncertainty stress test — see `train.py`'s
`--lp_filter_ranges`/`--perturb_frac`/`--bootstrap_frac` docs):

```bash
python -m surrogate_model_v3.train \
    --data_path Data/DatasetV2_layer_12_samples_5000.pkl \
    --lp_filter_ranges "100-150,200-250,300-350" \
    --perturb_frac 0.1 --bootstrap_frac 0.5 \
    --out_dir   surrogate_model_v3/runs/patchy_100-150_200-250_300-350_perturb0.1
# or: sbatch surrogate_model_v3/jobs/train_gap_perturb.sh
```

...with its **matched** full-range control (same perturb/bootstrap knobs,
no `--lp_filter_*` — required by `evaluate_ood_ratio.py`):

```bash
python -m surrogate_model_v3.train \
    --data_path Data/DatasetV2_layer_12_samples_5000.pkl \
    --perturb_frac 0.1 --bootstrap_frac 0.5 \
    --out_dir   surrogate_model_v3/runs/full_matched_perturb0.1_bootstrap0.5
# or: sbatch surrogate_model_v3/jobs/train_full_matched.sh
```

### 2. Evaluate

Standard per-layer accuracy + uncertainty (single-step and auto-regressive
rollout, train/val/test) + example field plots:

```bash
python -m surrogate_model_v3.evaluate \
    --checkpoint surrogate_model_v3/runs/full_range/surrogate_best.pt \
    --data_path  Data/DatasetV2_layer_12_samples_5000.pkl
# or: sbatch surrogate_model_v3/jobs/evaluate.sh
```

ID-vs-OOD epistemic check (narrow checkpoint vs. the wider dataset):

```bash
python -m surrogate_model_v3.evaluate_ood \
    --checkpoint surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt \
    --data_path  Data/DatasetV2_layer_12_samples_5000.pkl \
    --id_action_min 200 --id_action_max 300
# or: sbatch surrogate_model_v3/jobs/evaluate_ood.sh
```

Same, for the gapped/patchy checkpoint (`--id_ranges` matches training):

```bash
python -m surrogate_model_v3.evaluate_ood \
    --checkpoint surrogate_model_v3/runs/patchy_100-150_200-250_300-350_perturb0.1/surrogate_best.pt \
    --data_path  Data/DatasetV2_layer_12_samples_5000.pkl \
    --id_ranges  "100-150,200-250,300-350"
# or: sbatch surrogate_model_v3/jobs/evaluate_ood_gap.sh
```

Coverage-normalized ratio check (patchy vs. matched full-range — divides
out the "higher power is intrinsically harder" confound; run as a batch
job, not on the login node — it loads the full dataset + two model forward
passes):

```bash
python -m surrogate_model_v3.evaluate_ood_ratio \
    --checkpoint_patchy surrogate_model_v3/runs/patchy_100-150_200-250_300-350_perturb0.1/surrogate_best.pt \
    --checkpoint_full   surrogate_model_v3/runs/full_matched_perturb0.1_bootstrap0.5/surrogate_best.pt \
    --data_path Data/DatasetV2_layer_12_samples_5000.pkl \
    --id_ranges "100-150,200-250,300-350"
# or: sbatch surrogate_model_v3/jobs/evaluate_ood_ratio.sh
```

## Checkpoint interface (for RL / MPC consumers)

`model.load_surrogate(checkpoint_path, device)` returns
`(model, state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std)`.
`TwoStageSurrogate.predict_unnorm(state_raw, lp_raw, cool_raw, layer_idx,
state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std)` accepts raw
[K]/[W]/[s] tensors and returns a dict with `heat_pred_raw` (needed for
reward — `meanDeviation` is computed from the end-of-heating field, not the
next state), `next_pred_raw`, and per-stage + combined
epistemic/aleatoric/total uncertainty (`combine_stage_uncertainties` in
`model.py` derives why the two stages' variances simply add).

**Not yet wired up:** `online_RL_ucpg`/`online_RL_ucpg_v2` still consume the
older latent-encoder surrogate
(`surrogate_model_latent_uncertainty_v2.train.load_two_stage_surrogate`).
Switching the RL side to this package is a separate, deliberately deferred
step.
