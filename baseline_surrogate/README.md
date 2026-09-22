# baseline_surrogate

Comparison points for the main uncertainty-aware surrogate
([`surrogate_model_latent_uncertainty_v2/`](../surrogate_model_latent_uncertainty_v2/)),
trained/tested on the same 200–300W laser-power range as that package's
`runs/narrow_200_300W` checkpoint. Nothing here reports uncertainty —
epistemic/aleatoric decomposition is the main surrogate's contribution, not
something these baselines need to also provide; every method below is
compared purely on next-state prediction accuracy (per-layer MAE/RMSE).

## Result: the latent encoder/decoder underperforms — reverted to no-latent

`results_latent256/leaderboard[_autoregressive].csv` shows the main
surrogate's learned latent bottleneck (`surrogate (main, two-stage+latent+
ensemble)`) losing badly to every other method, including its own
`ablation_no_latent` ablation (same two-stage + bootstrap + Gaussian-NLL
machinery, identity encoder/decoder instead):

| Regime | `surrogate (main)` next-MAE / RMSE [K] | `ablation_no_latent` next-MAE / RMSE [K] |
|---|---|---|
| Teacher-forced | 13.68 / 26.87 | **0.997 / 3.92** |
| Auto-regressive | 23.93 / 43.48 | **2.08 / 5.80** |

Given this, the project has reverted to **not** using a learned latent
encoder/decoder for the main surrogate — `ablation_no_latent`'s design
(two-stage heating→cooling, bootstrap ensemble, Gaussian NLL, but
transitions act directly on the raw 1053-dim state) is the current primary
comparison point going forward, not just an ablation. Regenerate the
comparison plots/leaderboard without the latent-encoder main surrogate by
omitting `--surrogate_checkpoint` from the `summarize_results.py` call (see
"Summarizing" below) — every other flag is unchanged.

## Design: single-stage, except the two ablations

The main surrogate's two big architectural bets are (1) splitting each
layer into an explicit heating-then-cooling transition and (2) doing that
transition in a learned latent space rather than on the raw 1053-dim field.
Both are part of what should make it *better* than a generic baseline — so
every **external** baseline here is single-stage
(`s_{t+1} = f(s_t, a_t, cool_time_t, layer_idx)`, no intermediate
`u_heat_t` target) and none get the latent bottleneck for free either,
except where noted:

| # | Method | Two-stage? | Latent space? | Ensemble? |
|---|---|---|---|---|
| 1 | `mlp/` | No | No (raw 1053-dim) | No |
| 2 | `lstm/` | No (recurrent over the build instead) | No (raw hidden state) | No |
| 3 | `kalman_filter/` | No | Yes — PCA (fixed, not learned) | No |
| 4 | `vanilla_ensemble/` | No | No (raw 1053-dim) | Yes, K=5, plain MSE, no bootstrap/NLL |
| 5 | `ablation_no_two_stage/` | No | Yes — learned | Yes, K=5, full bootstrap + Gaussian NLL (same as main model) |
| 6 | `ablation_no_latent/` | Yes | No (raw 1053-dim) | Yes, K=5, full bootstrap + Gaussian NLL (same as main model) |

Methods 5 and 6 are ablations of the main model itself — each restores
exactly one of its two architectural bets so its contribution can be
isolated one variable at a time. They keep the full bootstrap/Gaussian-NLL
machinery (imported directly from `surrogate_model_latent_uncertainty_v2.model`,
not reimplemented) since removing *that* isn't the point of these two
ablations; only the point-estimate (ensemble-mean) prediction is compared,
same as everywhere else in this package.

**No borrowed tricks.** Methods 1/2/4 (`mlp/`, `lstm/`, `vanilla_ensemble/`)
additionally do NOT use a learned per-layer embedding or predict a
residual/delta (`s_t + Δ`) — both are techniques `surrogate_model_v3/` (the
proposed no-latent two-stage surrogate) itself relies on. A baseline that
borrows the proposed method's own tricks isn't a fair floor to beat, so
these three regress `s_{t+1}` directly from `(s_t, a_t, cool_t)` with no
layer conditioning at all. `kalman_filter/` needed no change — its
per-layer `b_layer` term is the classical linear model's own natural
formulation (an intercept per layer in an OLS fit), not a borrowed neural
trick, and it never predicted a residual to begin with.

## Method summaries

**1. Plain MLP** (`mlp/`) — `s_{t+1} = MLP([s_t, a_t, cool_t])`, raw
1053-dim space, no ensemble, no layer conditioning, no residual prediction.
The floor.

**2. LSTM** (`lstm/`) — `LSTMCell` carries hidden state across the 12-layer
build; teacher-forced on the true `s_t` at every step (so predictions are
still single-step relative to ground truth — only the *hidden state* is
recurrent). Needs the true unbroken 12-layer chain, so `--lp_filter_min/max`
here doesn't drop transitions like every other baseline — it masks the
*loss* to only count in-range layers while still feeding the model every
real layer (see `lstm/train.py`'s docstring).

**3. Kalman filter** (`kalman_filter/`) — PCA(64) + a linear
`z_{t+1} = A z_t + B a_t + C cool_t + b_layer` fit once via ordinary least
squares (closed form, no gradient descent). No filtering *update* step —
a surrogate is always queried with the true current state, never a noisy
sensor reading of it, so only the linear-Gaussian *predict* equation is
exercised (see `kalman_filter/model.py`'s docstring for why that's still
fairly called "a Kalman filter's dynamics model").

**4. Vanilla deep ensemble** (`vanilla_ensemble/`) — K=5 independently
initialised copies of the plain-MLP architecture (raw 1053-dim, no layer
embedding, no residual prediction — same as `mlp/`), not bootstrap-resampled,
MSE-only, no Gaussian NLL. Isolates exactly one variable against the main
surrogate: does bootstrap resampling + a Gaussian-NLL-calibrated σ head buy
anything over plain ensembling (independent init + MSE)?

**5/6. Ablations** — see the table above; `ablation_no_two_stage/` and
`ablation_no_latent/` each remove exactly one piece of the main model.

## Mathematical formulation

Notation matches [`surrogate_model_latent_uncertainty_v2/README.md`](../surrogate_model_latent_uncertainty_v2/README.md):
`s_t` = pre-heat state (raw Kelvin, `state_dim = 1053`), `a_t` = laser power [W],
`c_t` = cool time [s], `l_t` = 0-indexed layer, `u_h` = end-of-heating field,
`s_{t+1}` = end-of-cooling field (next state). A tilde denotes z-scored
(normalised) values, e.g. `s̃_t = (s_t − μ_s)/σ_s`. `e(l_t) ∈ R^{d_e}` is a
learned per-layer embedding (a small `nn.Embedding`) — used only by methods
5/6 below (the main model's own ablations); methods 1/2/4 deliberately have
no layer conditioning at all (see "No borrowed tricks" above). Every
method's `μ_s, σ_s, μ_a, σ_a, μ_c, σ_c` are fit ONCE from
the training split's `{s_t, s_{t+1}}` (pooled — no `u_h` view, since only
`ablation_no_latent` ever sees one) via
`common/data.py:build_single_stage_normalizers`, mirroring the main
package's `build_normalizers` but two views instead of three.

### 1. Plain MLP (`mlp/`)

Raw 1053-dim space, no latent, no ensemble, single-stage, no layer
conditioning, no residual prediction — `s̃_{t+1}` is regressed directly. One
block is `Linear → LayerNorm → SiLU → Dropout`; `depth` blocks are stacked:

```
h_0 = block_0([s̃_t, ã_t, c̃_t])
h_i = block_i(h_{i-1})                    i = 1 .. depth-1
ŝ̃_{t+1} = Linear_head(h_{depth-1})
```

Loss: plain MSE, `L = E[(1/D) ‖ŝ̃_{t+1} − s̃_{t+1}‖²]`, `D = state_dim`.

### 2. LSTM (`lstm/`)

Same per-step inputs as the MLP (no layer embedding — the recurrent hidden
state is this baseline's own way of knowing "where" it is in the build), a
hidden state carries information across the 12-layer build instead of each
step being independent, and `s̃_{t+1}` is regressed directly (no residual):

```
(h_t, c^{cell}_t) = LSTMCell([s̃_t, ã_t, c̃_t],  (h_{t-1}, c^{cell}_{t-1}))
ŝ̃_{t+1} = Linear_head(Dropout(h_t))
```

with `(h_{-1}, c^{cell}_{-1}) = (0, 0)` at the start of every trajectory.
`s̃_t` fed in at each step is the TRUE previous state during training
(teacher-forced) — only `h_t` is recurrent, not the state input itself.
Loss is masked MSE, counting only layers whose action falls in the
training LP range (see `lstm/train.py`'s docstring for why the trajectory
still isn't truncated):

```
L = ( Σ_t Σ_b m_{t,b} · (1/D)‖ŝ̃_{t+1,b} − s̃_{t+1,b}‖² )  /  ( Σ_t Σ_b m_{t,b} )
m_{t,b} = 1[ a_{t,b} ∈ [LP_min, LP_max] ]
```

### 3. Kalman filter (`kalman_filter/`)

A fixed (non-learned) PCA basis stands in for the main model's learned
encoder/decoder, and the transition is a single global linear map instead
of a neural network:

```
PCA (fit once, pooled {s_t, s_{t+1}} raw states):
    z_t = W(s_t − μ_PCA)                      W ∈ R^{n_c × 1053}, rows orthonormal
    ŝ_{t+1} = Wᵀ ẑ_{t+1} + μ_PCA

Design vector:
    x_t = [ z_t ; a_t ; c_t ; onehot(l_t) ] ∈ R^{n_c + 2 + n_layers}

Linear-Gaussian process model, fit by ordinary least squares (closed form,
no gradient descent):
    Θ* = argmin_Θ Σ_i ‖x_iᵀ Θ − z_{i,next}‖²  =  (XᵀX)⁻¹ Xᵀ Z     (via np.linalg.lstsq)

Prediction (the KF PREDICT equation; no UPDATE step — see
kalman_filter/model.py's docstring for why: the true s_t is always known
exactly at query time, so there is no observation to fuse against):
    ẑ_{t+1} = x_tᵀ Θ*
```

`n_c = 64` by default (`--n_components`, matching the main model's default
`--latent_dim`). Note `Θ` folds what the main model would call `A` (the
`z_t` block of rows), `B` (the `a_t` row), `C` (the `c_t` row), and
`b_layer` (the one-hot rows) into one matrix fit jointly.

### 4. Vanilla deep ensemble (`vanilla_ensemble/`)

`K = 5` independently-initialised copies of the SAME plain-MLP architecture
as `mlp/` (no latent space, no layer embedding, no residual prediction),
trained on the SAME full dataset (no bootstrap resampling):

```
ŝ̃_{t+1}^{(k)} = MLP_k([s̃_t; ã_t; c̃_t])                      k = 1 .. K

ŝ̃_{t+1} = (1/K) Σ_k ŝ̃_{t+1}^{(k)}          ← ensemble MEAN of direct predictions
```

Loss (plain MSE, no NLL, no bootstrap weighting — every member sees every
sample with weight 1):

```
L = (1/K) Σ_k E[(1/D) ‖ŝ̃_{t+1}^{(k)} − s̃_{t+1}‖²]
```

### 5. Ablation: no two-stage (`ablation_no_two_stage/`)

Identical machinery to the main model — `Encoder`/`Decoder`,
`GaussianTransitionMLP` (PETS-clamped `log σ`), moment-matched `K`-member
mixture, bootstrap-resampled training (all imported directly from
`surrogate_model_latent_uncertainty_v2.model`/`.train`, not reimplemented)
— but ONE ensemble instead of two, conditioned on `a_t` and `c_t` TOGETHER
(`cond_dim = 2`) instead of splitting them across a heating and a cooling
stage:

```
z_t = Encoder(s̃_t),   cond_t = [ã_t; c̃_t]
(μ_k, log σ_k) = g_k(z_t, cond_t, e(l_t))                     k = 1 .. K

Moment matching (Lakshminarayanan et al., 2017 — identical to the main
model's _moment_match, reused unchanged):
    μ̄ = (1/K) Σ_k μ_k
    epistemic_var = Var_k[μ_k]              (population variance, ÷K)
    aleatoric_var = (1/K) Σ_k σ_k²

Point estimate (what's evaluated — uncertainty is computed but not
reported, same as everywhere else in this package):
    ẑ_{t+1} = z_t + μ̄,   ŝ̃_{t+1} = Decoder(ẑ_{t+1})
```

Loss (`bw_{k,b}` = sample b's bootstrap multiplicity for member k, from
`make_bootstrap_masks`; `sg(·)` = stop-gradient):

```
L_recon_s = E[(1/D)‖Decoder(z_t) − s̃_t‖²]

L_recon = ( Σ_k Σ_b bw_{k,b} · (1/D)‖Decoder(z_t+μ_k) − s̃_{t+1,b}‖² )  /  ( Σ_k Σ_b bw_{k,b} )

Δz_target = sg(Encoder(s̃_{t+1})) − z_t
L_NLL = ( Σ_k Σ_b bw_{k,b} · (1/D) Σ_d [ 0.5log(2π) + log σ_{k,d} + 0.5(Δz_{target,d} − μ_{k,d})²/σ_{k,d}² ] )  /  ( Σ_k Σ_b bw_{k,b} )

L = w_s·L_recon_s + w_r·L_recon + w_n·L_NLL          (defaults: w_s=w_r=1.0, w_n=0.1)
```

### 6. Ablation: no latent space (`ablation_no_latent/`)

Also identical `GaussianTransitionMLP`/moment-matching/bootstrap machinery,
and KEEPS the two-stage heating→cooling split — but `Encoder`/`Decoder`
are the IDENTITY, so `GaussianTransitionMLP` acts directly on the raw
1053-dim (normalised) field (`latent_dim := state_dim`) instead of a
learned bottleneck. Since `Decoder(Encoder(x)) ≡ x` exactly, the main
model's two autoencoder reconstruction terms (`L_recon_s`, `L_recon_heat_ae`)
are trivially zero here and are dropped from the loss entirely (not just
computed-and-ignored):

```
z_t := s̃_t                                    (identity — no encoder)

Heating stage:
    (μ_heat,k, log σ_heat,k) = g_heat,k(z_t, ã_t, e(l_t))       k = 1..K
    μ̄_heat = (1/K) Σ_k μ_heat,k
    ũ_h    = z_t + μ̄_heat                       (point-estimate heat prediction)

Cooling stage — teacher-forced on the GROUND-TRUTH ũ_h during training
(never the heating stage's own prediction — matches the main model's
single-step training regime exactly):
    (μ_cool,k, log σ_cool,k) = g_cool,k(ũ_h, c̃_t, e(l_t))       k = 1..K
    μ̄_cool = (1/K) Σ_k μ_cool,k
    ŝ̃_{t+1} = ũ_h + μ̄_cool
```

Loss (same bootstrap-weighted reconstruction + NLL pattern as ablation 5,
applied to both stages, target deltas computed directly against the raw
identity `z_t`/`ũ_h` — no `sg(Encoder(·))` needed since there's no encoder
to protect from collapsing):

```
L_recon_heat = ( Σ_k Σ_b bw_{k,b} · (1/D)‖z_t+μ_heat,k − u_{h,b}‖² ) / ( Σ_k Σ_b bw_{k,b} )
L_NLL_heat   = bootstrap-weighted Gaussian NLL,  target = u_h − z_t

L_recon_cool = ( Σ_k Σ_b bw_{k,b} · (1/D)‖ũ_h+μ_cool,k − s̃_{t+1,b}‖² ) / ( Σ_k Σ_b bw_{k,b} )
L_NLL_cool   = bootstrap-weighted Gaussian NLL,  target = s̃_{t+1} − ũ_h

L = w_rh·L_recon_heat + w_nh·L_NLL_heat + w_rc·L_recon_cool + w_nc·L_NLL_cool
    (defaults: w_rh=w_rc=1.0, w_nh=w_nc=0.1 — identical defaults to the main model)
```

At AUTO-REGRESSIVE evaluation time (`summarize_results.py`, not training),
the cooling stage instead consumes the heating stage's OWN prediction
`ũ_h` chained forward — never ground truth — exactly matching how
`evaluate_autoregressive_with_heat` and the main model's own
`model.rollout(...)` behave. This is precisely why teacher-forced and
auto-regressive numbers can diverge sharply for any two-stage method: the
cooling stage's `g_cool,k` was only ever trained on inputs of the form
`Encoder(true u_h)` (or, here, `true u_h` directly), never on its own
upstream stage's error.

## Training (same data/split convention throughout)

```bash
DATA=Data/DatasetV2_layer_12_samples_5000.pkl

python -m baseline_surrogate.mlp.train                   --data_path $DATA --lp_filter_min 200 --lp_filter_max 300 --out_dir baseline_surrogate/mlp/runs/narrow_200_300W
python -m baseline_surrogate.lstm.train                  --data_path $DATA --lp_filter_min 200 --lp_filter_max 300 --out_dir baseline_surrogate/lstm/runs/narrow_200_300W
python -m baseline_surrogate.kalman_filter.train          --data_path $DATA --lp_filter_min 200 --lp_filter_max 300 --out_dir baseline_surrogate/kalman_filter/runs/narrow_200_300W
python -m baseline_surrogate.vanilla_ensemble.train        --data_path $DATA --lp_filter_min 200 --lp_filter_max 300 --out_dir baseline_surrogate/vanilla_ensemble/runs/narrow_200_300W
python -m baseline_surrogate.ablation_no_two_stage.train   --data_path $DATA --lp_filter_min 200 --lp_filter_max 300 --out_dir baseline_surrogate/ablation_no_two_stage/runs/narrow_200_300W
python -m baseline_surrogate.ablation_no_latent.train       --data_path $DATA --lp_filter_min 200 --lp_filter_max 300 --out_dir baseline_surrogate/ablation_no_latent/runs/narrow_200_300W
```

All six trained/fit by `sbatch baseline_surrogate/jobs/train_all_baselines.sh`
in one job (~8h budgeted, GPU partition) — reuses the existing
`surrogate_model_latent_uncertainty_v2/runs/narrow_200_300W/two_stage_best.pt`
checkpoint rather than retraining the main model.

## Epistemic-uncertainty / patchy-coverage stress test (no-latent main surrogate)

Since `ablation_no_latent` is now the primary (no-latent) surrogate design,
it needs the same epistemic-uncertainty validation the encoder/decoder main
model has via `surrogate_model_latent_uncertainty_v2/evaluate_ood.py` /
`evaluate_ood_ratio.py`. `ablation_no_latent/train.py` now also accepts
`--lp_filter_ranges` (gapped/patchy, comma-separated `lo-hi` ranges),
`--perturb_frac` (target-noise), and `--bootstrap_frac` (decorrelated
bootstrap) — the same three flags `surrogate_model_latent_uncertainty_v2/
train.py` documents under "Harder / gapped-surrogate experiments" — passed
straight through to the same `TwoStageLatentSurrogateDataset` that package
uses, so the semantics are identical.

`ablation_no_latent/evaluate_ood_ratio.py` is the no-latent counterpart of
`surrogate_model_latent_uncertainty_v2/evaluate_ood_ratio.py`: same
`ood_uncertainty_summary_2x2.png` (raw epistemic σ | raw aleatoric σ on top,
epistemic ratio | RMSE on bottom, patchy vs. full-range overlaid) and
`ood_epistemic_ratio_vs_action.png`, reusing that package's
`collect_ood_samples`/`bin_by_action`/plotting code unchanged — only the
checkpoint loader differs.

```bash
DATA=Data/DatasetV2_layer_12_samples_5000.pkl

# 1. Full-range checkpoint (no lp_filter) — the "intrinsic difficulty" baseline
python -m baseline_surrogate.ablation_no_latent.train \
    --data_path $DATA \
    --out_dir baseline_surrogate/ablation_no_latent/runs/full_range

# 2. Patchy/gapped checkpoint — deliberately sparse laser-power coverage
python -m baseline_surrogate.ablation_no_latent.train \
    --data_path $DATA \
    --lp_filter_ranges "100-150,200-250,300-350" \
    --perturb_frac 0.1 --bootstrap_frac 0.5 \
    --out_dir baseline_surrogate/ablation_no_latent/runs/patchy_100-150_200-250_300-350_perturb0.1

# 3. 2x2 uncertainty summary: patchy vs. full-range, same wide dataset
python -m baseline_surrogate.ablation_no_latent.evaluate_ood_ratio \
    --checkpoint_patchy baseline_surrogate/ablation_no_latent/runs/patchy_100-150_200-250_300-350_perturb0.1/ablation_no_latent_best.pt \
    --checkpoint_full   baseline_surrogate/ablation_no_latent/runs/full_range/ablation_no_latent_best.pt \
    --data_path $DATA \
    --id_ranges "100-150,200-250,300-350"
```

A working epistemic channel shows `ood_uncertainty_summary_2x2.png`'s
epistemic-ratio panel sitting near 1 across the patchy model's ID ranges and
rising well above 1 in the interior gaps `(150,200)`/`(250,300)` and past
the top edge `(350,400)` — see the main package's `evaluate_ood_ratio.py`
docstring for the full rationale (identical here, model swapped).

## Summarizing

**Current (no latent encoder/decoder)** — omits `--surrogate_checkpoint`
(see "Result" section above for why):

```bash
python -m baseline_surrogate.summarize_results \
    --data_path Data/DatasetV2_layer_12_samples_5000.pkl \
    --mlp_checkpoint                   baseline_surrogate/mlp/runs/narrow_200_300W/mlp_best.pt \
    --lstm_checkpoint                  baseline_surrogate/lstm/runs/narrow_200_300W/lstm_best.pt \
    --kalman_checkpoint                baseline_surrogate/kalman_filter/runs/narrow_200_300W/kalman_filter_fitted.pt \
    --vanilla_ensemble_checkpoint      baseline_surrogate/vanilla_ensemble/runs/narrow_200_300W/vanilla_ensemble_best.pt \
    --ablation_no_two_stage_checkpoint baseline_surrogate/ablation_no_two_stage/runs/narrow_200_300W/ablation_no_two_stage_best.pt \
    --ablation_no_latent_checkpoint    baseline_surrogate/ablation_no_latent/runs/narrow_200_300W/ablation_no_latent_best.pt \
    --out_dir baseline_surrogate/results_no_latent
```

Legacy (includes the latent-encoder main surrogate, for reference — this is
what produced `results_latent256/`): add back
`--surrogate_checkpoint surrogate_model_latent_uncertainty_v2/runs/narrow_200_300W/two_stage_best.pt`.

Every `--*_checkpoint` flag is optional — pass only what you have. Evaluates
every method on the SAME held-out test split (re-derived from
`--data_path`/`--seed`, unfiltered — matches
`surrogate_model_latent_uncertainty_v2/evaluate.py`'s convention of
reporting on the full range rather than only the training-filtered region),
in TWO regimes:

- **teacher-forced** (single-step) — every layer's input `s_t` is the
  TRUE previous state. Isolates one-step accuracy, matches how every
  method was trained.
- **auto-regressive** (rollout) — only the true initial state `s_0` is
  given; every later `s_t` is the method's OWN previous prediction,
  chained across all 12 layers. Matches real deployment (only the
  action/cool-time schedule is known ahead of time, never the true
  intermediate states) and exposes compounding error for any
  stage/step that was only ever trained via teacher forcing.

Outputs, under `--out_dir` — every file comes in a teacher-forced version
and an `_autoregressive` version:

```
leaderboard[_autoregressive].csv     ← method, next_mae_mean_K, next_rmse_mean_K, heat_mae_mean_K, n_params
leaderboard[_autoregressive].png     ← horizontal bar chart, sorted by mean next-state MAE
per_layer_mae[_autoregressive].png   ← per-layer next-state MAE, one line per method
per_layer_rmse[_autoregressive].png  ← per-layer next-state RMSE, one line per method
rmse_vs_action[_autoregressive].png  ← next-state RMSE binned by laser power [W], training range shaded —
                                        shows whether error is concentrated outside the training range
                                        (extrapolation) or bad even inside it
```

`heat_mae_mean_K` is only populated for the main surrogate and
`ablation_no_latent` (the only two genuinely two-stage methods here) —
every other row reports `N/A` there, since predicting `s_{t+1}` directly
with no `u_heat_t` target is exactly what makes them single-stage. A large
gap between a method's teacher-forced and auto-regressive numbers is itself
informative — it means that method's later steps/stages were never
exposed to their own upstream errors during training (no analogue of
`surrogate_model_latent_uncertainty_v2/train.py`'s `--rollout_steps` was
used), not necessarily that the architecture is bad in isolation.

## File structure

```
baseline_surrogate/
  common/
    data.py        ← single-stage flat/trajectory datasets, raw-array extraction, normalizers
    eval.py         ← shared per-layer MAE/RMSE evaluator (predict_fn abstraction)
    train_loop.py   ← shared early-stopping training loop
  mlp/{model,train}.py
  lstm/{model,train}.py
  kalman_filter/{model,train}.py
  vanilla_ensemble/{model,train}.py
  ablation_no_two_stage/{model,train}.py
  ablation_no_latent/{model,train,evaluate_ood_ratio}.py
  summarize_results.py
  jobs/train_all_baselines.sh
  results_no_latent/  ← leaderboard.csv/.png, per_layer_{mae,rmse}.png — current (no latent encoder/decoder)
  results_latent256/  ← same, WITH the latent-encoder main surrogate — legacy, see "Result" section above
```
