"""
online_RL_ucpg_v2
--------------------
Monte Carlo Uncertainty-Constrained Policy Gradient (UCPG) controller for
the LPBF process — continuous laser-power action, trained by interacting
with the two-stage (heating/cooling) Gaussian ensemble latent surrogate
(surrogate_model_latent_uncertainty_v2).

Sibling to online_RL_ucpg/ (categorical action, one-stage surrogate); see
this package's README.md for the full method and everything that changed.

Package layout
--------------
  env.py      — TwoStageLatentLPBFEnv: obs = [z_t ‖ layer token ‖ cool_time
                token], two-stage (heating→cooling) latent transition per
                step, reward computed from the END-OF-HEATING field,
                cool_time drawn once per episode and observed but never
                chosen (contextual MDP, not an action). Reward and
                uncertainty (u = sigma_epi + sigma_alea, combined across
                both stages) returned separately (never blended into a
                single scalar).
  model.py    — ContinuousLatentPolicyNet: Gaussian policy pi_theta: Z -> N(mu, sigma)
  agent.py    — UCPGAgentV2: policy-gradient update + Lagrange dual ascent on lambda
  train.py    — Main training entry-point (python -m online_RL_ucpg_v2.train)
  evaluate.py — Policy evaluation: action distribution, per-layer reward,
                and the policy's own mu±sigma trace
"""
