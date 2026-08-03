"""
baselines/offline_q
----------------------
Baseline 1: offline (batch) Q-learning with a discrete action space, trained
purely from the static pickled dataset — no surrogate model, no environment
interaction, no online data collection at all. Standard fitted-Q / offline
DQN: a target network and Bellman backups computed entirely over the fixed
offline transition buffer.
"""
