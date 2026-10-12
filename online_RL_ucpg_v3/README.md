# UCPG v3 — uncertainty-shaped reward

v2 constrained the expected cumulative uncertainty with a Lagrange multiplier
and did not outperform naive policy gradient. v3 drops the constraint and
puts the uncertainty **inside the reward, per node**: the reward trusts the
surrogate less where its ensemble disagrees, and penalises that distrust.

## Uncertainty carried across layers: member rollouts

The K = 5 ensemble members are rolled out as K separate trajectories; member
*k* applies its own heating and cooling networks to its own previous state:

$$h^{(k)}_t = s^{(k)}_t + \Delta^{\mathrm{heat}}_k(s^{(k)}_t, a_t, t), \qquad
s^{(k)}_{t+1} = h^{(k)}_t + \Delta^{\mathrm{cool}}_k(h^{(k)}_t, c, t).$$

Disagreement from earlier layers therefore propagates into later ones through
the nonlinear model itself (PETS-style trajectory sampling), with the correct
correlation across layers. The policy observes the member-mean state
$\bar s_t$ in the same format as v2.

## Reward

For the reward field $f_t$ (post-cooling $s_{t+1}$ by default, or the
end-of-heating $h_t$), ROI nodes $R_t$, window $[T_l, T_h]$, $W = T_h - T_l$:

$$\mu_{t,i} = \tfrac1K\sum_k f^{(k)}_{t,i}, \qquad
\sigma_{t,i} = \mathrm{std}_k\, f^{(k)}_{t,i}\ \ \text{(epistemic, K)}, \qquad
d(u) = \max(0, T_l-u) + \max(0, u-T_h)$$

$$w_{t,i} = \frac{\tau^2}{\tau^2 + \sigma_{t,i}^2}, \qquad
\tilde d_{t,i} = d(\mu_{t,i}) + (1 - w_{t,i})\, c, \qquad
r_t = -\frac{1}{|R_t|}\sum_{i\in R_t} \frac{\tilde d_{t,i}}{W}.$$

* $w_{t,i}\in(0,1]$ is a precision weight — how much the model is trusted at node *i*.
* $(1-w)\,c$ adds a penalty of up to $c$ (default $W$) for a fully untrusted node.
  Because $\tilde d \ge d$, uncertainty can never *improve* the reward. (The
  shrinkage form $w\,d + (1-w)\,c$ — `--shaping shrink` — can, whenever
  $d > c$, which happens for the post-cooling field at low laser power.)
* $\tau = k\cdot\mathrm{median}\,\sigma$ over multi-step member rollouts with
  in-distribution actions (`--ood_min/--ood_max`, $k$ = `--tau_k` = 2), so
  in-distribution nodes keep $w\approx 0.8$.
* Only epistemic $\sigma$ is used: aleatoric noise is irreducible and would
  penalise every action roughly equally.
* `--reward_mode plain` gives $r_t = -\frac1{|R_t|}\sum_i d(\mu_{t,i})/W$ — the
  matched baseline with no uncertainty.

## Algorithms (same reward, two learners)

* `train_pg.py` — Monte Carlo REINFORCE, $\nabla_\theta \sum_t \log\pi_\theta(a_t|\bar s_t)\,(G_t - b_t)$,
  policy = v2's `UCPGAgentV2` with $\lambda = 0$.
* `train_q.py` — online DQN with replay and target network; same network,
  action grid and checkpoint format as `baselines/online_q`; best checkpoint
  chosen on greedy evaluation episodes.

## Evaluation in the real simulator

`evaluate_real.py` runs every method in MATLAB (one PDE solve per layer) and
scores both the **post-cooling** ROI deviation and MATLAB's **end-of-heating**
deviation, with a per-layer check that the Python ROI mask reproduces
MATLAB's value.

## Jobs

```bash
sbatch online_RL_ucpg_v3/jobs/train_narrow.sh            # 4 runs: PG/Q x plain/shaped
sbatch online_RL_ucpg_v3/jobs/eval_real_narrow.sh        # after training finishes
# heating-field variant:
sbatch --export=ALL,REWARD_FIELD=heat online_RL_ucpg_v3/jobs/train_narrow.sh
```
