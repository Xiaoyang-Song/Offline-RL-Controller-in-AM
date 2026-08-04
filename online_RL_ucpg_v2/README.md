# Uncertainty-Constrained Policy Gradient v2 (UCPG v2) for LPBF Control

A continuous-action Monte Carlo policy-gradient controller for the LPBF
laser-power process that maximises expected reward subject to a hard budget
on expected cumulative surrogate uncertainty — implemented as a
Lagrangian-relaxed constrained MDP, trained entirely against the
[`surrogate_model_latent_uncertainty_v2`](../surrogate_model_latent_uncertainty_v2/)
two-stage (heating/cooling) Gaussian bootstrap ensemble surrogate.

This is a sibling to [`online_RL_ucpg/`](../online_RL_ucpg/), which this
package is a direct port of. The algorithm (Monte Carlo REINFORCE with a
Lagrangian uncertainty constraint) is **unchanged**; three things about the
*process being controlled* changed and drove every code difference:

| | `online_RL_ucpg` (v1) | `online_RL_ucpg_v2` |
|---|---|---|
| Action space | Discrete grid (`--action_min/max/step`) | **Continuous** — Gaussian policy $\pi_\theta(\cdot\mid z_t) = \mathcal N(\mu_\theta(z_t), \sigma_\theta)$ |
| Surrogate | One-stage `EnsembleGaussianLatentDynamicsModel` | **Two-stage** `TwoStageEnsembleGaussianLatentDynamicsModel` (heating → cooling, chained in latent space) |
| Reward field | Post-cooling field $s_{t+1}$ | **End-of-heating field** $u_{\text{heat},t}$ — reward no longer waits for cooling to finish (matches the corrected v2 simulator/dataset) |
| Cooling input | N/A | `cool_time` — exogenous per-episode context, **observed but never chosen** by the policy |
| Uncertainty $u_t$ | Epistemic + aleatoric of the single transition | Epistemic + aleatoric **combined across both stages** via variance addition |

Everything else — the Lagrangian relaxation, the dual-ascent update, the
on-policy Monte Carlo data collection with no replay buffer, the optional
per-timestep baseline — is identical to v1 and is only summarized here; see
[`online_RL_ucpg/README.md`](../online_RL_ucpg/README.md) for the original
derivation this package inherits.

## 1. MDP formulation

**State.** $s_t\in\mathbb R^{1053}$, the temperature field just before layer
$t$'s laser pass ($s_0 \equiv 300\,\mathrm K$ everywhere). The policy never
sees $s_t$ directly — it observes the surrogate's latent encoding
$z_t = \mathrm{Encoder}(s_t)$, concatenated with a normalised layer index
**and** this episode's (normalised) cool time:

$$\mathrm{obs}_t = \Big[\,z_t \,\Big\Vert\, \tfrac{t}{T-1} \,\Big\Vert\, \tfrac{c - c_{\min}}{c_{\max}-c_{\min}}\,\Big] \in \mathbb R^{d_z+2}$$

**Action.** $a_t \in \mathbb R$, the laser power \[W\] applied during layer
$t$ — the *only* thing the policy controls.

**Exogenous context (observed, not controlled).** $c \in \mathbb R$, the
cooling duration \[s\] for this episode. Per `multilayer_random_v2.m`
("coolTime is randomly sampled once per trajectory... same coolTime for
every layer in this trajectory"), $c$ is drawn **once at episode reset**,
$c \sim \mathrm{Unif}(c_{\min}, c_{\max})$ (default $[0.05, 0.15]$ s,
matching the simulator's sampling range), and held fixed for all 12 layers.

It is **not** part of the action space — the policy never chooses it — but
it **is** part of $\mathrm{obs}_t$. In the real process, cooling duration is
a known build parameter, not something hidden from the controller, so
withholding it would only turn this into an unnecessary POMDP: the policy
would have to infer $c$'s effect indirectly through how $z_t$ has evolved so
far, which gives it *zero* information before the very first action — right
when knowing the cooling regime for planning the whole episode matters most.
Observing $c$ makes this a proper **contextual MDP**: the policy conditions
its laser-power choice on the actual cooling regime this episode drew (e.g.
compensating differently under fast vs. slow cooling), without controlling
it.

**Transition — two stages, chained in latent space.** For layer $t$:

$$
\begin{aligned}
z_{\text{heat}} &= z_t + \mu_{\Delta z}^{\text{heat}}(z_t, a_t, t) &&\text{(heating, action-dependent)}\\
z_{t+1} &= z_{\text{heat}} + \mu_{\Delta z}^{\text{cool}}(z_{\text{heat}}, c, t) &&\text{(cooling, action-INdependent)}
\end{aligned}
$$

using the ensemble means from
`surrogate.predict_heating_ensemble` / `predict_cooling_ensemble` — the same
two calls `TwoStageEnsembleGaussianLatentDynamicsModel.rollout()` chains
during surrogate evaluation. The cooling stage is conditioned on $c$, never
on $a_t$: physically, "no matter what laser power you applied previously,
the cooling mechanism is the same" (see
[`surrogate_model_latent_uncertainty_v2`](../surrogate_model_latent_uncertainty_v2/)'s
README for the derivation).

**Reward — computed at the END OF HEATING.**

$$
u_{\text{heat},t} = \mathrm{Decoder}(z_{\text{heat}})\cdot\sigma_s + \mu_s
\qquad\text{(denormalised end-of-heating field)}
$$

$$
r_t = -\,\overline{\mathrm{dev}}(u_{\text{heat},t})\,, \qquad
\mathrm{dev}_i = \max(0, T_l - u_i) + \max(0, u_i - T_h)\,,\qquad
\overline{\mathrm{dev}} = \frac{\mathrm{mean}_{i \in \text{ROI}}(\mathrm{dev}_i)}{T_h - T_l}
$$

evaluated only over nodes inside the square scan region (the same
$\mathrm{squareSideFraction}$ schedule $[0.4 \to 0.5]$ over 12 layers used by
the simulator), with default target window $(T_l, T_h) = (2000, 2800)\,$K.
This is a direct, literal port of `simulateHeatingCooling_v2.m`'s corrected
`meanDeviation` computation — which is now evaluated on `uHeatFinal`, not
`uFinal` — into the RL environment. **This is the one substantive physical
correction this package carries forward**: v1's environment scored the
*post-cooling* field; cooling has no bearing on how well the laser pass hit
its target, so scoring it there was measuring the wrong thing.

**Uncertainty — combined across both stages.**

$$u_t = \sigma^{\text{total}}_{\text{epi},t} + \sigma^{\text{total}}_{\text{ale},t}$$

where, since the full step is literally a sum of two latent increments
$z_{t+1} - z_t = \Delta z_{\text{heat}} + \Delta z_{\text{cool}}$, and each is
already a moment-matched Gaussian-mixture ensemble output, the two stages'
variances add (see `combine_stage_uncertainties` in
`surrogate_model_latent_uncertainty_v2/model.py` for the full derivation):

$$
(\sigma^{\text{total}}_{\text{epi},t})^2 = (\sigma^{\text{heat}}_{\text{epi},t})^2 + (\sigma^{\text{cool}}_{\text{epi},t})^2,\qquad
(\sigma^{\text{total}}_{\text{ale},t})^2 = (\sigma^{\text{heat}}_{\text{ale},t})^2 + (\sigma^{\text{cool}}_{\text{ale},t})^2
$$

Reward and uncertainty are always returned **separately** by `env.step()`,
never blended into one scalar — the training loop keeps independent reward
and uncertainty trajectories so it can discount them differently
($\gamma_r$, $\gamma_u$) and combine them only via the Lagrangian, matching
v1 exactly.

## 2. Continuous Gaussian policy

$\pi_\theta: \mathcal Z \to \mathcal N(\mu_\theta(z_t, t), \sigma_\theta(t))$,
implemented in `model.ContinuousLatentPolicyNet`:

$$
\mu_\theta(z_t, t) = c_a + r_a \cdot \tanh\!\big(f_\theta(z_t, t)\big), \qquad
c_a = \tfrac{a_{\min}+a_{\max}}{2},\ \ r_a = \tfrac{a_{\max}-a_{\min}}{2}
$$

where $f_\theta$ is the same trunk architecture as v1's `LatentPolicyNet`
(layer-embedding, residual MLP stem/trunk) with a scalar head instead of
$|\mathcal A|$ logits. $\sigma_\theta(t)$ is a **layer-specific** (one
learnable value per build layer $t$, looked up the same way as the layer
embedding — still state-independent *within* a layer), expressed as a
PETS-style soft-clamped log-fraction of the action half-range $r_a$, with
the soft-clamp bounds shared across layers so only the per-layer raw value
differs and the bound stays scale-invariant regardless of
`--action_min/--action_max`:

$$\sigma_\theta(t) = r_a \cdot \exp\big(\mathrm{softclamp}(\ell_\theta(t);\ \ell_{\min}, \ell_{\max})\big)$$

the exact soft-clamp construction (`max - softplus(max - x)`, then
`min + softplus(x - min)`) already used for the surrogate's own transition
heads (`GaussianTransitionMLP`) — reused here for consistency, not
reinvented.

**Why per-layer, not one global $\sigma_\theta$.** Layer 0 always sees the
identical latent input (the fixed initial 300K field encoded once), so
choosing its action is a stateless "find the peak of one fixed curve"
problem — it can and should sharpen (shrink $\sigma$) independently of
later, more state-variable layers whose optimal action genuinely depends on
history. A single global $\sigma_\theta$ couples every layer's exploration
together: noisy gradient signal from harder layers can keep an
already-solved layer's $\sigma$ artificially wide, and vice versa. Giving
every layer its own $\sigma_\theta(t)$ (still sharing the soft-clamp bounds,
so none of them can collapse to exactly 0 or blow up) removes that coupling
at essentially no extra cost — $n\_layers$ extra scalar parameters.

**Why $\tanh$ only touches the mean, never the sample.** $a_t \sim
\mathcal N(\mu_\theta, \sigma_\theta)$ is sampled **unclipped** — no
tanh-squash-then-Jacobian-correction on the sample itself (as SAC does), no
clipping at the environment boundary. `--action_min/--action_max` only give
the policy's mean a bounded, sensible place to *aim*; the actual chosen
action can and does land outside that range with some probability. This
keeps $\nabla_\theta \log\pi_\theta(a_t\mid z_t)$ exactly
`Normal.log_prob(a_t)` — no change-of-variables term to derive, differentiate,
or debug — and, more importantly, preserves the exact mechanism the parent
package's OOD experiment relied on: **nothing hard-stops the policy from
choosing an action the surrogate has never seen; the uncertainty constraint
is the only thing discouraging it.** A hard clip or tanh-squash on the
sample would silently make wide exploration impossible near the boundary and
defeat that purpose.

## 3. Objective — unchanged from v1

$$\max_{\pi_\theta} J_r(\pi_\theta) = \mathbb{E}\Big[\sum_t \gamma_r^t r_t\Big]
\quad \text{s.t.} \quad
J_u(\pi_\theta) = \mathbb{E}\Big[\sum_t \gamma_u^t u_t\Big] \le \delta$$

Lagrangian relaxation with multiplier $\lambda \ge 0$, penalized return
$G_{\lambda,t} = G_{r,t} - \lambda\, G_{u,t}$:

$$
\widehat{\nabla_\theta\mathcal{L}} = \frac{1}{N}\sum_{i=1}^N \sum_t
\nabla_\theta \log\pi_\theta(a_t^{(i)}\mid z_t^{(i)})\, \big[G_{r,t}^{(i)} - \lambda\, G_{u,t}^{(i)}\big]
$$

$$\lambda \leftarrow \big[\lambda + \alpha_\lambda(\widehat{J}_u - \delta)\big]_+$$

`train.py` implements this literally by default: no replay buffer, no
baseline subtraction, no entropy bonus — a fresh batch of $N$ trajectories
is collected with the current policy every iteration, used for exactly one
policy-gradient step and one dual-ascent step, then discarded.

### Optional per-timestep baseline (`--use_baseline`)

Identical rationale and mechanics to v1: subtracts the batch's empirical
mean return at each timestep from $G_{r,t}$ and $G_{u,t}$ before forming the
advantage (unbiased, reduces variance, repairs early-layer credit
assignment on the 12-step horizon). Never touches $\widehat J_u$ / the
$\lambda$ update, which always use the raw $G_{u,0}$. See
[`online_RL_ucpg/README.md`](../online_RL_ucpg/README.md#optional-per-timestep-baseline---use_baseline)
for the full derivation.

## 4. Why the policy operates in latent space

Same rationale as v1: `ContinuousLatentPolicyNet` never sees the 1053-dim
temperature field — its input is $z_t$ (the surrogate encoder's output)
concatenated with a layer-index token and the cool_time context token.
`TwoStageLatentLPBFEnv` also carries
the *latent* state forward step-to-step ($z_{t+1} = z_{\text{heat}} +
\mu_{\Delta z}^{\text{cool}}$, chained exactly like
`TwoStageEnsembleGaussianLatentDynamicsModel.rollout()`) rather than
round-tripping through the decoder and back through the encoder every step
— the raw field is only decoded twice per step (end-of-heating, for reward;
end-of-cooling, for logging), never fed back through the encoder.

## Usage

### Training

```bash
python -m online_RL_ucpg_v2.train \
    --surrogate surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt \
    --action_min 100 --action_max 400 \
    --delta 0.05
```

`--delta` (the uncertainty budget) is required and problem-specific — it
depends on your surrogate's own uncertainty scale. Guidance: run a short
warm-start (e.g. `--n_iterations 20 --lambda_init 0`, effectively
unconstrained since $\lambda$ starts at 0 and moves slowly) and look at the
logged `J_u` values under the near-random initial policy to calibrate a
sensible $\delta$ — too high and the constraint never binds, too low and the
policy can't do anything useful.

`--ood_threshold` is optional; pass it (e.g. `--ood_threshold 300`) to also
log "fraction of chosen actions above this power" if you want to reuse v1's
narrow-surrogate/wide-policy OOD-avoidance diagnostic here.

### Evaluation

```bash
python -m online_RL_ucpg_v2.evaluate \
    --checkpoint online_RL_ucpg_v2/runs/<ts>/ucpg_best.pt \
    --surrogate  surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt \
    --n_episodes 50 --plot
```

Reports, for both the greedy (deterministic $\mu_\theta$) and stochastic
(sampled) policy: mean return, mean uncertainty, and the chosen-action
distribution (mean/std/median/min/max). With `--plot`, saves
`eval_action_histogram.png`, `eval_reward_and_action.png`, and
`eval_policy_mean_std.png` (the policy's own $\mu_\theta \pm \sigma_\theta$
per layer along one greedy rollout — a diagnostic only a continuous policy
can produce; a categorical distribution has no single "mean ± std" to plot).

### Outputs

```
online_RL_ucpg_v2/runs/<timestamp>/
  ucpg_best.pt / ucpg_final.pt / ucpg_iter#####.pt
  return.png               ← undiscounted episode return per iteration
  j_u.png                  ← J_u_hat vs. δ per iteration
  lambda.png                ← Lagrange multiplier trajectory
  loss.png                  ← policy-gradient loss
  entropy.png                ← mean policy entropy (= f(sigma_theta), exploration diagnostic)
  action_stats.png          ← mean ± std of CHOSEN actions per iteration (continuous analogue
                               of watching a categorical distribution sharpen)
  ood_action_fraction.png   ← only if --ood_threshold given
```

## File Structure

```
online_RL_ucpg_v2/
  env.py      ← TwoStageLatentLPBFEnv (latent obs, two-stage transition,
                reward at end-of-heating, cool_time nuisance variable)
  model.py    ← ContinuousLatentPolicyNet (Gaussian policy over latent input)
  agent.py    ← UCPGAgentV2 (policy-gradient update, dual ascent, checkpointing)
  train.py    ← Algorithm 1 main loop (continuous action)
  evaluate.py ← greedy/stochastic rollout, action distribution, policy mu±sigma trace
  README.md   ← this file

jobs/
  train_online_rl_ucpg_v2.sh   ← SLURM job script
```
