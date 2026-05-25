"""
online_RL/agent.py
------------------
Double-DQN agent for discrete-action online RL on the LPBF environment.

Algorithm highlights
--------------------
  • Double DQN  — online net selects the next action, target net evaluates
                  its value → decouples selection from evaluation, reducing
                  overestimation bias  (van Hasselt et al., 2016).
  • Target network — hard copy every `target_update_freq` episodes; keeps
                  bootstrapping targets stable.
  • ε-greedy exploration — linear decay from ε_start → ε_end over a
                  configurable number of gradient-update steps.
  • Huber (smooth-L1) loss — less sensitive to outlier Q-value errors than
                  MSE, which matters early in training.
  • Gradient clipping — clips the L2 norm of gradients to `max_grad_norm`.

Checkpoint format
-----------------
  Saved by `save()` and restored by `load()`.  The file is a dict with:
    q_net_state_dict, target_net_state_dict, optimizer_state_dict,
    epsilon, update_count, agent_config, action_list  (+ optional extras).
"""

import copy
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from online_RL.model import QNet
from online_RL.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Double-DQN agent for the LPBF laser-power control task.

    Parameters
    ----------
    state_dim             : dimension of normalised temperature field
    n_actions             : number of discrete laser-power choices
    hidden                : Q-network hidden width
    depth                 : number of ResidualBlocks in Q-network
    dropout               : dropout inside each block (0 = off)
    lr                    : Adam learning rate
    gamma                 : discount factor
    epsilon_start         : initial exploration probability
    epsilon_end           : minimum exploration probability
    epsilon_decay_steps   : number of Q-updates over which ε decays linearly
    max_grad_norm         : gradient-norm clipping threshold
    double_dqn            : use Double-DQN (True recommended)
    device                : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        state_dim:           int   = 1053,
        n_actions:           int   = 26,
        hidden:              int   = 256,
        depth:               int   = 4,
        dropout:             float = 0.0,
        lr:                  float = 3e-4,
        gamma:               float = 0.99,
        epsilon_start:       float = 1.0,
        epsilon_end:         float = 0.05,
        epsilon_decay_steps: int   = 50_000,
        max_grad_norm:       float = 10.0,
        double_dqn:          bool  = True,
        device:              str   = "cpu",
    ) -> None:
        self.n_actions           = n_actions
        self.gamma               = gamma
        self.epsilon             = float(epsilon_start)
        self.epsilon_end         = float(epsilon_end)
        self.epsilon_step        = (epsilon_start - epsilon_end) / max(epsilon_decay_steps, 1)
        self.max_grad_norm       = max_grad_norm
        self.double_dqn          = double_dqn
        self.device              = device
        self._update_count       = 0
        self._hidden             = hidden
        self._depth              = depth

        # ── networks ──────────────────────────────────────────────────────
        self.q_net      = QNet(state_dim, n_actions, hidden, depth, dropout).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.target_net.eval()

        # ── optimiser ─────────────────────────────────────────────────────
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    # =========================================================================
    # Action selection
    # =========================================================================

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        ε-greedy action selection.

        Parameters
        ----------
        state   : (D,) normalised float32 numpy array
        explore : if False, always act greedily (evaluation mode)

        Returns
        -------
        action_idx : int   index into ACTION_LIST
        """
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(s)
        self.q_net.train()
        return int(q.argmax(dim=1).item())

    # =========================================================================
    # Learning
    # =========================================================================

    def update(
        self,
        buffer:     ReplayBuffer,
        batch_size: int,
    ) -> Optional[float]:
        """
        One gradient update step.

        Returns
        -------
        loss : float  Huber loss value, or None if buffer is too small.
        """
        if len(buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        s  = torch.tensor(states,      dtype=torch.float32, device=self.device)
        a  = torch.tensor(actions,     dtype=torch.long,    device=self.device)
        r  = torch.tensor(rewards,     dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d  = torch.tensor(dones,       dtype=torch.float32, device=self.device).unsqueeze(1)

        # ── compute TD target ────────────────────────────────────────────
        with torch.no_grad():
            if self.double_dqn:
                # online net selects best next action; target net evaluates it
                a_next   = self.q_net(s2).argmax(dim=1, keepdim=True)     # (B,1)
                q_next   = self.target_net(s2).gather(1, a_next)           # (B,1)
            else:
                q_next   = self.target_net(s2).max(dim=1, keepdim=True)[0]  # (B,1)

            target = r + self.gamma * (1.0 - d) * q_next                  # (B,1)

        # ── Q-prediction for taken actions ────────────────────────────────
        q_pred = self.q_net(s).gather(1, a.unsqueeze(1))                   # (B,1)

        loss = F.smooth_l1_loss(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # ── linear ε decay ────────────────────────────────────────────────
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_step)
        self._update_count += 1

        return float(loss.item())

    # =========================================================================
    # Target network
    # =========================================================================

    def update_target(self) -> None:
        """Hard copy Q-net weights → target net."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # =========================================================================
    # Checkpoint helpers
    # =========================================================================

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        """
        Save agent state to `path`.  Pass any extra metadata via `extra`.
        """
        data = {
            "q_net_state_dict":      self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict":  self.optimizer.state_dict(),
            "epsilon":               self.epsilon,
            "update_count":          self._update_count,
            "agent_config": {
                "state_dim":           self.q_net.state_dim,
                "n_actions":           self.n_actions,
                "hidden":              self._hidden,
                "depth":               self._depth,
                "double_dqn":          self.double_dqn,
            },
        }
        if extra:
            data.update(extra)
        torch.save(data, path)
        print(f"[agent] Checkpoint saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DQNAgent":
        """
        Restore a DQNAgent from a checkpoint produced by `save()`.

        Returns the agent in training mode with its exploration state intact.
        """
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        cfg   = ckpt["agent_config"]
        agent = cls(device=device, **cfg)
        agent.q_net.load_state_dict(ckpt["q_net_state_dict"])
        agent.target_net.load_state_dict(ckpt["target_net_state_dict"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        agent.epsilon       = float(ckpt.get("epsilon", agent.epsilon))
        agent._update_count = int(ckpt.get("update_count", 0))
        print(f"[agent] Loaded checkpoint from {path}  "
              f"(ε={agent.epsilon:.3f}, updates={agent._update_count:,})")
        return agent

    def select_action_greedy(self, state: np.ndarray) -> int:
        """Greedy action selection — for evaluation / deployment."""
        return self.select_action(state, explore=False)

    def __repr__(self) -> str:
        return (
            f"DQNAgent({self.q_net}  "
            f"double_dqn={self.double_dqn}, ε={self.epsilon:.3f}, "
            f"updates={self._update_count:,})"
        )
