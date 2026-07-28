"""
online_RL_ucpg/agent.py
--------------------------
Monte Carlo Uncertainty-Constrained Policy Gradient agent.

Implements the theta-update and lambda-update of Algorithm 1
("Monte Carlo Uncertainty-Constrained Policy Gradient"):

  theta <- theta + alpha_theta * (1/N) sum_i sum_t
               nabla_theta log pi_theta(a_t^(i)|s_t^(i)) * [G_r,t^(i) - lambda * G_u,t^(i)]

  lambda <- [lambda + alpha_lambda * (J_u_hat - delta)]_+

This class only owns the policy network, optimiser, and dual variable. Data
collection (rolling out trajectories through the environment) and return
computation (G_r, G_u, J_u_hat) live in train.py, matching the separation
already used elsewhere in this repo (online_RL/agent.py owns the Q-network
and TD update; online_RL/train.py owns the episode loop).

No replay buffer: this is an on-policy Monte Carlo method — every batch of N
trajectories is collected fresh with the CURRENT policy, used for exactly one
gradient step, and discarded (matches the algorithm box exactly; reusing old
trajectories for an on-policy log-prob gradient would be biased).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from online_RL_ucpg.model import LatentPolicyNet


class UCPGAgent:
    """
    Parameters
    ----------
    latent_dim      : surrogate latent space dimension
    n_actions       : number of discrete laser-power choices
    hidden          : policy network hidden width
    depth           : number of ResidualBlocks in the policy trunk
    dropout         : dropout inside each block (0 = off)
    n_layers        : number of LPBF build layers
    layer_embed_dim : dimension of the learned layer embedding
    lr_theta        : policy learning rate  (alpha_theta)
    lambda_init     : initial Lagrange multiplier (>= 0)
    max_grad_norm   : gradient-norm clipping threshold for the policy update
    device          : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        latent_dim:      int,
        n_actions:       int,
        hidden:          int   = 128,
        depth:           int   = 3,
        dropout:         float = 0.0,
        n_layers:        int   = 12,
        layer_embed_dim: int   = 8,
        lr_theta:        float = 3e-4,
        lambda_init:     float = 0.0,
        max_grad_norm:   float = 10.0,
        device:          str   = "cpu",
    ) -> None:
        self.latent_dim      = latent_dim
        self.n_actions       = n_actions
        self.device           = device
        self.max_grad_norm    = max_grad_norm
        self._hidden          = hidden
        self._depth           = depth
        self._dropout         = dropout
        self._n_layers        = n_layers
        self._layer_embed_dim = layer_embed_dim

        self.policy = LatentPolicyNet(
            latent_dim, n_actions, hidden, depth, dropout, n_layers, layer_embed_dim
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_theta)

        self.lam = float(lambda_init)   # Lagrange multiplier lambda >= 0

    # =========================================================================
    # Action selection (rollout — no grad)
    # =========================================================================

    @torch.no_grad()
    def select_action(self, obs, explore: bool = True) -> int:
        """
        Parameters
        ----------
        obs     : (obs_dim,) float32 numpy array — latent state + layer token
        explore : if True, sample a ~ pi_theta(.|s) (training rollout).
                  if False, take argmax logits (greedy evaluation).

        Returns
        -------
        action_idx : int  index into env.action_list
        """
        self.policy.eval()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(obs_t)
        if explore:
            action = Categorical(logits=logits).sample()
        else:
            action = logits.argmax(dim=-1)
        self.policy.train()
        return int(action.item())

    def select_action_greedy(self, obs) -> int:
        return self.select_action(obs, explore=False)

    # =========================================================================
    # Monte Carlo policy-gradient update
    # =========================================================================

    def policy_step(
        self,
        obs_flat:       torch.Tensor,   # (N*T, obs_dim)
        actions_flat:   torch.Tensor,   # (N*T,) long
        advantage_flat: torch.Tensor,   # (N*T,) float — G_r,t - lambda * G_u,t
        n_traj:         int,            # N — number of trajectories in this batch
    ) -> Tuple[float, float]:
        """
        One Monte Carlo policy-gradient ascent step:
            L(theta) = -(1/N) sum_i sum_t log pi_theta(a_t^(i)|s_t^(i)) * advantage_t^(i)
        (minimising L is equivalent to ascending the Lagrangian objective).

        Returns
        -------
        loss    : float — the scalar loss value optimised this step
        entropy : float — mean policy entropy over the batch (diagnostic only,
                  not part of the loss — the algorithm as specified has no
                  entropy bonus)
        """
        logits = self.policy(obs_flat)
        dist   = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_flat)          # (N*T,)

        loss = -(log_probs * advantage_flat).sum() / n_traj

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return float(loss.item()), float(dist.entropy().mean().item())

    # =========================================================================
    # Dual (Lagrange multiplier) update
    # =========================================================================

    def update_lambda(self, j_u_hat: float, delta: float, lr_lambda: float) -> float:
        """lambda <- [lambda + alpha_lambda * (J_u_hat - delta)]_+"""
        self.lam = max(0.0, self.lam + lr_lambda * (j_u_hat - delta))
        return self.lam

    # =========================================================================
    # Checkpoint helpers
    # =========================================================================

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        data = {
            "policy_state_dict":    self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lam":                  self.lam,
            "agent_config": {
                "latent_dim":      self.latent_dim,
                "n_actions":       self.n_actions,
                "hidden":          self._hidden,
                "depth":           self._depth,
                "dropout":         self._dropout,
                "n_layers":        self._n_layers,
                "layer_embed_dim": self._layer_embed_dim,
            },
        }
        if extra:
            data.update(extra)
        torch.save(data, path)
        print(f"[agent] Checkpoint saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "UCPGAgent":
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        cfg   = ckpt["agent_config"]
        agent = cls(device=device, **cfg)
        agent.policy.load_state_dict(ckpt["policy_state_dict"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        agent.lam = float(ckpt.get("lam", 0.0))
        print(f"[agent] Loaded checkpoint from {path}  (lambda={agent.lam:.5f})")
        return agent

    def __repr__(self) -> str:
        return f"UCPGAgent({self.policy}  lambda={self.lam:.5f})"
