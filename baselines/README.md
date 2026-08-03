# Baselines for `online_RL_ucpg_v2`

Five comparison points for the Uncertainty-Constrained Policy Gradient (UCPG)
controller in [`online_RL_ucpg_v2/`](../online_RL_ucpg_v2/), ranging from "RL
minus the uncertainty constraint" down to classical, non-learning process
control. **None of these are expected to beat UCPG v2** — several (constant
power, Kalman/particle filter) are included specifically because they're
*supposed* to struggle, to make the case for RL (and for the uncertainty
constraint specifically) concrete rather than assumed. This package only
ever **imports** from `surrogate_model_latent_uncertainty_v2` and
`online_RL_ucpg_v2` — neither is modified.

| # | Method | Uses surrogate? | Uses offline dataset? | Learned? |
|---|---|---|---|---|
| 0 | [Naive policy gradient](#0-naive-policy-gradient) | ✅ (as the training env) | — | ✅ (RL) |
| 1 | [Offline Q-learning](#1-offline-q-learning-discrete) | ❌ (raw dataset only) | ✅ | ✅ (batch RL) |
| 2 | [Proportional controller](#2-proportional-controller) | ❌ (raw dataset only, for fitting) | ✅ | Fit via least squares, not RL |
| 3 | [Constant policy](#3-constant-policy) | — | — | No |
| 4 | [Kalman / particle filter](#4-kalman--particle-filter) | ❌ (raw dataset only, for fitting) | ✅ | Fit via least squares, not RL |

**Fair-comparison design.** Methods 1/2/4 are trained/fit purely from the
static pickled dataset and never touch the neural surrogate while learning —
that's the whole point of calling them "offline"/"traditional." But *all
five* baselines (plus, optionally, the real UCPG v2 checkpoint) are
***evaluated*** by rolling out through the exact same
`online_RL_ucpg_v2.env.TwoStageLatentLPBFEnv`, backed by the same surrogate
checkpoint — this is unavoidable (we need *some* stand-in for the real LPBF
process to score any policy) and is exactly how UCPG v2 itself is evaluated.
Using the surrogate as the *evaluation environment* is not the same as using
it as part of a *controller's decision-making* — only baseline 0 does that,
by construction (see below).

---

## 0. Naive policy gradient

Plain Monte Carlo REINFORCE maximising reward alone — the "before" picture
for UCPG v2's uncertainty constraint:

$$\max_{\pi_\theta} \mathbb{E}\Big[\sum_t \gamma_r^t r_t\Big]$$

Implemented by **reusing `online_RL_ucpg_v2` unmodified** (environment,
`ContinuousLatentPolicyNet`, `UCPGAgentV2`, `collect_batch`,
`discounted_returns`) with the uncertainty term simply never added to the
advantage — no $\lambda$, no dual ascent, no `G_u` at all. This isolates
exactly one variable: what the Lagrangian constraint buys you over plain
policy gradient on the identical architecture/environment.

```bash
python -m baselines.naive_pg.train \
    --surrogate surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt \
    --action_min 100 --action_max 400 \
    --n_iterations 2000
```

Saves `naive_pg_best.pt` / `naive_pg_final.pt` in the same format as UCPG
v2's checkpoints (both use `UCPGAgentV2.save`/`load`) — pass either straight
to `--naive_pg_checkpoint` below.

## 1. Offline Q-learning (discrete)

Standard offline / fitted Q-iteration: a Bellman target and a target network,
trained entirely over a **fixed** buffer built once from the pickled dataset
— no environment interaction, no exploration, no surrogate model anywhere in
training. Actions are the training data's own discrete laser-power grid
(100, 110, ..., 400 W — see `multilayer_random_v2.m`'s `LP_values`), so there
is no discretisation choice to make. The offline transition is:

$$s_t \xrightarrow{a_t} r_t,\ s_{t+1}\qquad s_t = \text{pre-heat field},\quad s_{t+1} = \text{this layer's post-cooling field}$$

reconstructed directly from the pickle
(`baselines/common/data_utils.build_offline_transitions`) using the same
state-chaining convention as `surrogate_model_latent_uncertainty_v2`'s own
normaliser, but with its own, independent standardisation (never the
surrogate's).

```bash
python -m baselines.offline_q.train \
    --data_path Data/DatasetV2_layer_12_samples_5000.pkl \
    --epochs 50
```

## 2. Proportional controller

$$a_t = K_p \cdot \big(T_{\text{mid}} - \overline{T}_{\text{ROI}}(s_t)\big) + b$$

$K_p$ and $b$ are fit **once**, via ordinary least squares against the
offline dataset's own (error, laser-power) pairs — no gradient descent, no
reward signal, no RL of any kind; a genuinely classical control law.

```bash
python -m baselines.proportional.controller \
    --data_path Data/DatasetV2_layer_12_samples_5000.pkl \
    --out baselines/proportional/fitted.pt
```

**Expected caveat, worth stating up front:** the training dataset's laser
powers were drawn *randomly* (for surrogate training), not from any
closed-loop feedback controller — there is no real causal error→power
relationship for the regression to recover. Don't be surprised if the fitted
$K_p$ comes out small or even the "wrong" sign; that is the regression
honestly reporting "this dataset carries near-zero information about a
proportional control law," not a bug. This is itself an informative result:
naive log-data regression cannot substitute for actually acting in
(or simulating) the closed loop, which is exactly what RL does.

## 3. Constant policy

No state feedback at all: one fixed laser power for the entire 12-layer
episode, swept across the full training grid {100, 110, ..., 400} W (31
values). Per the task spec, the headline number is the **mean return across
the whole sweep** (not the best-in-hindsight constant) — `evaluate_baselines.py`
also reports the single best constant as a secondary, clearly-labelled
reference point, plus a `constant_sweep.png` plot of return vs. power. No
training step exists for this one; it only runs inside
`evaluate_baselines.py`.

## 4. Kalman / particle filter

Included specifically to illustrate a state-estimation approach's limits, as
requested. Setup: let $x_t$ = mean ROI temperature of $s_t$. A controller
observes a *synthetically noised* sensor reading $z_t = x_t + v_t,\; v_t\sim
\mathcal N(0, R)$ — not the free, exact field a policy in this repo
otherwise gets — and must filter it before acting. The process model is a
**random walk** ($x_{t+1} = x_t + w_t,\; w_t \sim \mathcal N(0, Q)$, $Q$ fit
as the empirical variance of consecutive per-trajectory ROI-mean
differences) — deliberately *not* the true nonlinear PDE, since a controller
in this position wouldn't have access to a full physics model either. The
control law (certainty equivalence) inverts a once-fit linear regression

$$\overline{T}^{\text{heat}}_{\text{ROI}} \approx \alpha \cdot x_t + \beta \cdot a_t + \gamma_c$$

around the filter's current posterior mean $\mu_t$:
$a_t = (T_{\text{mid}} - \alpha\,\mu_t - \gamma_c)\,/\,\beta$.

Fit once (both filters share the same fitted $\alpha,\beta,\gamma_c,Q$ —
only the filtering algorithm differs):

```bash
python -m baselines.kalman_particle.filters \
    --data_path Data/DatasetV2_layer_12_samples_5000.pkl \
    --out baselines/kalman_particle/fitted.pt
```

$R$ (sensor noise) and the particle count are evaluation-time knobs, not fit
parameters (`--kalman_R`, `--particle_R`, `--particle_n` on
`evaluate_baselines.py`) — they represent a *scenario* choice (how bad is
the hypothetical sensor), not something the data determines.

**Why this is expected to underperform** (the point of including it): a
scalar, single-fixed-coefficient linear map cannot represent a
spatially-varying, layer-dependent nonlinear PDE — the coefficients $\alpha,
\beta$ are forced to be one global compromise across all 12 layers' very
different ROI sizes and heat-accumulation regimes, and the random-walk
process model has no idea the ROI is growing over the build. Adding sensor
noise on top only compounds this. If a Kalman/particle-filter controller
*did* control this process well, that would suggest the underlying dynamics
were much simpler than they actually are.

---

## Aggregating all results

`evaluate_baselines.py` rolls every requested method through the **same**
surrogate-backed environment and prints/saves one leaderboard. Every method
is optional — pass only the checkpoints/fitted-params you have; the constant
sweep always runs unless `--skip_constant` is given. Optionally include the
real UCPG v2 checkpoint too, for a genuine head-to-head:

```bash
python -m baselines.evaluate_baselines \
    --surrogate                surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt \
    --naive_pg_checkpoint      baselines/naive_pg/runs/<ts>/naive_pg_best.pt \
    --offline_q_checkpoint     baselines/offline_q/runs/<ts>/offline_q_best.pt \
    --proportional_fitted      baselines/proportional/fitted.pt \
    --kalman_particle_fitted   baselines/kalman_particle/fitted.pt \
    --ucpg_v2_checkpoint       online_RL_ucpg_v2/runs/<ts>/ucpg_best.pt \
    --n_episodes 50
```

Outputs, all under `--out_dir` (default `baselines/results/`):

```
leaderboard.csv     ← name, return_mean, return_std, uncertainty_mean, action_mean, action_std
leaderboard.png     ← horizontal bar chart, sorted by return
constant_sweep.png  ← baseline 3's per-constant return curve (if not --skip_constant)
```

`--n_episodes` controls every non-constant method; the constant sweep uses
its own (smaller) `--n_episodes_constant`, since it evaluates 31 separate
policies. All environment flags (`--T_l/--T_h/--mesh_path/...`) should match
whatever the surrogate and baselines were fit/trained with — defaults are
shared with `online_RL_ucpg_v2`.

## File structure

```
baselines/
  common/
    data_utils.py      ← offline transition reconstruction (no surrogate),
                          square-ROI mesh helpers, least-squares fit helper
    eval_harness.py    ← shared surrogate-env rollout, StepContext, metrics,
                          LatentAgentController (wraps any UCPGAgentV2 checkpoint)
  naive_pg/train.py            ← baseline 0
  offline_q/{model,train}.py   ← baseline 1
  proportional/controller.py   ← baseline 2
  constant/controller.py       ← baseline 3
  kalman_particle/filters.py   ← baseline 4 (both filters)
  evaluate_baselines.py        ← aggregation CLI
  results/                     ← leaderboard.csv / .png, constant_sweep.png
  README.md                    ← this file
```
