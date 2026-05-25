"""
online_RL
---------
Online Deep Q-Network (Double-DQN) controller for the LPBF process,
trained by interacting with the trained surrogate dynamics model.

Package layout
--------------
  env.py           — LPBF Gym-like environment (surrogate + MATLAB reward)
  model.py         — Residual Q-network  Q(s) → R^A
  replay_buffer.py — Circular experience replay buffer
  agent.py         — Double-DQN agent (ε-greedy, target net, Huber loss)
  train.py         — Main training entry-point  (python -m online_RL.train)
"""
