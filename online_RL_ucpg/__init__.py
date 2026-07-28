"""
online_RL_ucpg
----------------
Monte Carlo Uncertainty-Constrained Policy Gradient (UCPG) controller for
the LPBF process, trained by interacting with the Gaussian bootstrap
ensemble latent surrogate (surrogate_model_latent_uncertainty).

Package layout
--------------
  env.py      — Latent-space LPBF environment: obs = [z_t ‖ layer token],
                reward and uncertainty (u = sigma_epi + sigma_alea) returned
                separately (never blended into a single scalar).
  model.py    — LatentPolicyNet: categorical policy pi_theta: Z -> A
  agent.py    — UCPGAgent: policy-gradient update + Lagrange dual ascent on lambda
  train.py    — Main training entry-point (python -m online_RL_ucpg.train)
  evaluate.py — Policy evaluation, focused on whether the constrained policy
                avoids laser powers the surrogate was never trained on
"""
