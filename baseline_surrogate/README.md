# baseline_surrogate

Comparison points for the main uncertainty-aware surrogate
([`surrogate_model_latent_uncertainty_v2/`](../surrogate_model_latent_uncertainty_v2/)),
trained/tested on the same 200–300W laser-power range as that package's
`runs/narrow_200_300W` checkpoint. Nothing here reports uncertainty —
epistemic/aleatoric decomposition is the main surrogate's contribution, not
something these baselines need to also provide; every method below is
compared purely on next-state prediction accuracy (per-layer MAE/RMSE).

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
| 4 | `vanilla_ensemble/` | No | Yes — learned (same Encoder/Decoder as the main model) | Yes, K=5, plain MSE, no bootstrap/NLL |
| 5 | `ablation_no_two_stage/` | No | Yes — learned | Yes, K=5, full bootstrap + Gaussian NLL (same as main model) |
| 6 | `ablation_no_latent/` | Yes | No (raw 1053-dim) | Yes, K=5, full bootstrap + Gaussian NLL (same as main model) |

Methods 5 and 6 are ablations of the main model itself — each restores
exactly one of its two architectural bets so its contribution can be
isolated one variable at a time. They keep the full bootstrap/Gaussian-NLL
machinery (imported directly from `surrogate_model_latent_uncertainty_v2.model`,
not reimplemented) since removing *that* isn't the point of these two
ablations; only the point-estimate (ensemble-mean) prediction is compared,
same as everywhere else in this package.

## Method summaries

**1. Plain MLP** (`mlp/`) — `s_{t+1} = s_t + MLP([s_t, a_t, cool_t, layer_embed])`,
raw 1053-dim space, no ensemble. The floor.

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
initialised (not bootstrap-resampled) point-estimate transition heads
sharing one learned Encoder/Decoder; MSE-only, no Gaussian NLL. The most
direct comparison point: same latent architecture as the main model, minus
bootstrap resampling and the calibrated σ head.

**5/6. Ablations** — see the table above; `ablation_no_two_stage/` and
`ablation_no_latent/` each remove exactly one piece of the main model.

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

## Summarizing

```bash
python -m baseline_surrogate.summarize_results \
    --data_path Data/DatasetV2_layer_12_samples_5000.pkl \
    --surrogate_checkpoint             surrogate_model_latent_uncertainty_v2/runs/narrow_200_300W/two_stage_best.pt \
    --mlp_checkpoint                   baseline_surrogate/mlp/runs/narrow_200_300W/mlp_best.pt \
    --lstm_checkpoint                  baseline_surrogate/lstm/runs/narrow_200_300W/lstm_best.pt \
    --kalman_checkpoint                baseline_surrogate/kalman_filter/runs/narrow_200_300W/kalman_filter_fitted.pt \
    --vanilla_ensemble_checkpoint      baseline_surrogate/vanilla_ensemble/runs/narrow_200_300W/vanilla_ensemble_best.pt \
    --ablation_no_two_stage_checkpoint baseline_surrogate/ablation_no_two_stage/runs/narrow_200_300W/ablation_no_two_stage_best.pt \
    --ablation_no_latent_checkpoint    baseline_surrogate/ablation_no_latent/runs/narrow_200_300W/ablation_no_latent_best.pt \
    --out_dir baseline_surrogate/results
```

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
  ablation_no_latent/{model,train}.py
  summarize_results.py
  jobs/train_all_baselines.sh
  results/          ← leaderboard.csv/.png, per_layer_{mae,rmse}.png
```
