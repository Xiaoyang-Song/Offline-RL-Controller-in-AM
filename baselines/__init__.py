"""
baselines
-----------
Comparison baselines for online_RL_ucpg_v2, evaluated against the SAME
two-stage surrogate environment (surrogate_model_latent_uncertainty_v2 +
online_RL_ucpg_v2.env.TwoStageLatentLPBFEnv) so every method's reported
return is apples-to-apples. See baselines/README.md for the full write-up.

This package only ever IMPORTS from surrogate_model_latent_uncertainty_v2 /
online_RL_ucpg_v2 — it does not modify either.
"""
