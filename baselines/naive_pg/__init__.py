"""
baselines/naive_pg
---------------------
Baseline 0: naive (unconstrained) continuous policy gradient — REINFORCE on
reward alone, no uncertainty constraint. Everything else (env, policy
architecture, Monte Carlo trajectory collection) is reused unmodified from
online_RL_ucpg_v2, so this baseline isolates exactly one thing: what the
Lagrangian uncertainty constraint buys you over plain policy gradient.
"""
