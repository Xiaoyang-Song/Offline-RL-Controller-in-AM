"""
baselines/online_q
----------------------
Baseline 5: online (interactive) Q-learning with a discrete action space,
trained by stepping through the SAME surrogate environment naive_pg / UCPG v2
use — no fixed offline buffer, standard epsilon-greedy DQN with a replay
buffer and target network. See baselines/offline_q/ for the offline
(fixed-buffer) counterpart; this package exists to isolate one further
variable: does interacting with the environment (vs. a static dataset) let
value-based RL do any better here, independent of the reward-vs-uncertainty
question naive_pg/UCPG v2 already isolate.
"""
