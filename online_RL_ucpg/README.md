# Uncertainty-Constrained Policy Gradient (UCPG) for LPBF Control

A Monte Carlo policy-gradient controller for the LPBF laser-power process
that maximises expected reward subject to a hard budget on expected
cumulative surrogate uncertainty — implemented as a Lagrangian-relaxed
constrained MDP, trained entirely against the
[`surrogate_model_latent_uncertainty`](../surrogate_model_latent_uncertainty/)
Gaussian bootstrap ensemble surrogate.

This is a sibling to [`online_RL/`](../online_RL/) (which trains a Double-DQN
controller against the deterministic-ensemble `surrogate_model_latent`). The
two differ in every load-bearing way:

| | `online_RL` (DQN) | `online_RL_ucpg` |
|---|---|---|
| Algorithm | Off-policy Double-DQN | On-policy Monte Carlo policy gradient (REINFORCE) |
| Policy input | Raw decoded state (D=1053 field) | **Latent state** z_t from the surrogate encoder |
| Uncertainty handling | Optional scalar penalty folded into reward | **Constrained** via a Lagrange multiplier, kept as a separate signal |
| Uncertainty source | Epistemic only (deterministic ensemble) | **Epistemic + aleatoric** (Gaussian bootstrap ensemble) |
| Data | Replay buffer, off-policy | Fresh on-policy trajectories every iteration, no replay |

## Method

For a transition $(s_t, a_t, s_{t+1})$, the uncertainty cost is

$$u_t = u_{\text{epi}}(s_t, a_t) + u_{\text{alea}}(s_t, a_t)$$

read directly off the surrogate's `predict_ensemble` (mean-over-latent-dims
epistemic and aleatoric std). The policy $\pi_\theta: \mathcal{Z} \to
\mathcal{A}$ maximises expected reward subject to a budget on expected
cumulative uncertainty:

$$\max_{\pi_\theta} J_r(\pi_\theta) = \mathbb{E}\Big[\sum_t \gamma_r^t r_t\Big]
\quad \text{s.t.} \quad
J_u(\pi_\theta) = \mathbb{E}\Big[\sum_t \gamma_u^t u_t\Big] \le \delta$$

Introducing a Lagrange multiplier $\lambda \ge 0$, the inner policy update
uses the penalized return $G_{\lambda,t} = G_{r,t} - \lambda G_{u,t}$:

$$\widehat{\nabla_\theta\mathcal{L}} = \frac{1}{N}\sum_{i=1}^N \sum_t
\nabla_\theta \log\pi_\theta(a_t^{(i)}|s_t^{(i)})\, \big[G_{r,t}^{(i)} - \lambda G_{u,t}^{(i)}\big]$$

and $\lambda$ is updated by projected dual ascent after each batch:

$$\lambda \leftarrow \big[\lambda + \alpha_\lambda(\widehat{J}_u - \delta)\big]_+$$

`train.py` implements this literally: no replay buffer, no baseline
subtraction, no entropy bonus — a fresh batch of $N$ trajectories is
collected with the current policy every iteration, used for exactly one
policy-gradient step and one dual-ascent step, then discarded (matches the
"Monte Carlo Uncertainty-Constrained Policy Gradient" algorithm box exactly).

## Why the policy operates in latent space

`LatentPolicyNet` never sees the 1053-dim temperature field — its input is
$z_t$ (the surrogate encoder's output) concatenated with a layer-index token,
matching the method's $\pi_\theta: \mathcal{Z}\to\mathcal{A}$ formulation.
`LatentLPBFEnv` also carries the *latent* state forward step-to-step
($z_{t+1} = z_t + \mu_{\Delta z}$, exactly like `EnsembleGaussianLatentDynamicsModel.rollout()`)
rather than round-tripping through the decoder and back through the encoder
every step — the raw field is only decoded as a side computation to score
the reward.

## Experiment this package was built for

Train the surrogate on a **narrow** action range (e.g. 150–300 W, see
[`surrogate_model_latent_uncertainty`](../surrogate_model_latent_uncertainty/)'s
narrow-checkpoint / `evaluate_ood.py` workflow), then let the UCPG policy
choose from a **wider** action grid (e.g. 150–400 W, via `--action_max 400`)
while the environment is still backed by the narrow-trained surrogate. The
300–400 W region is then genuinely out-of-distribution for the surrogate:
epistemic $\sigma$ (and typically aleatoric $\sigma$ too) rises there, so a
working uncertainty constraint should push $\lambda$ up whenever the policy
drifts into that band, discouraging it — **without the reward function ever
being told the boundary exists**. `--ood_threshold` (default 300 W) is a
pure logging/plotting threshold used to track "fraction of chosen actions
above this power" during training and evaluation; it plays no role in the
loss or constraint itself.

## Usage

### Training

```bash
python -m online_RL_ucpg.train \
    --surrogate surrogate_model_latent_uncertainty/runs/narrow_150_300W/latent_best.pt \
    --action_min 150 --action_max 400 --action_step 10 \
    --delta 0.05 --ood_threshold 300
```

`--delta` (the uncertainty budget) is required and problem-specific — it
depends on your surrogate's own uncertainty scale. Guidance: run a short
warm-start (e.g. `--n_iterations 20 --lambda_init 0`, effectively
unconstrained since $\lambda$ starts at 0 and moves slowly) and look at the
logged `J_u` values under the random/near-uniform initial policy to calibrate
a sensible $\delta$ — too high and the constraint never binds, too low and
the policy can't do anything useful.

### Evaluation

```bash
python -m online_RL_ucpg.evaluate \
    --checkpoint online_RL_ucpg/runs/<ts>/ucpg_best.pt \
    --surrogate  surrogate_model_latent_uncertainty/runs/narrow_150_300W/latent_best.pt \
    --ood_threshold 300 --n_episodes 50 --plot
```

Reports, for both the greedy and stochastic policy: mean return, mean
uncertainty, and — the key number — what fraction of chosen actions exceed
`--ood_threshold`. With `--plot`, saves `eval_action_histogram.png` (action
distribution with the OOD region shaded) and `eval_reward_and_action.png`
(per-layer reward/action trace).

### Outputs

```
online_RL_ucpg/runs/<timestamp>/
  ucpg_best.pt / ucpg_final.pt / ucpg_iter#####.pt
  return.png               ← undiscounted episode return per iteration
  j_u.png                  ← J_u_hat vs. δ per iteration
  lambda.png                ← Lagrange multiplier trajectory
  loss.png                  ← policy-gradient loss
  entropy.png                ← mean policy entropy (exploration diagnostic)
  ood_action_fraction.png   ← fraction of chosen actions above --ood_threshold
                               (the headline plot for this experiment)
```

## File Structure

```
online_RL_ucpg/
  env.py      ← LatentLPBFEnv (latent obs, separate reward/uncertainty)
  model.py    ← LatentPolicyNet (categorical policy over latent input)
  agent.py    ← UCPGAgent (policy-gradient update, dual ascent, checkpointing)
  train.py    ← Algorithm 1 main loop
  evaluate.py ← greedy/stochastic rollout, OOD action-fraction diagnostic
  README.md   ← this file

jobs/
  train_online_rl_ucpg.sh   ← SLURM job script
```
