"""
online_RL_ucpg_v2/agent.py
------------------------------
Monte Carlo Uncertainty-Constrained Policy Gradient agent — continuous
laser-power action.

Implements the theta-update and lambda-update of Algorithm 1
("Monte Carlo Uncertainty-Constrained Policy Gradient"), unchanged from
online_RL_ucpg/agent.py:

  theta <- theta + alpha_theta * (1/N) sum_i sum_t
               nabla_theta log pi_theta(a_t^(i)|s_t^(i)) * [G_r,t^(i) - lambda * G_u,t^(i)]

  lambda <- [lambda + alpha_lambda * (J_u_hat - delta)]_+

The only difference from v1 is the action distribution: pi_theta(.|s) is now
Normal(mu_theta(s), sigma_theta) (see model.ContinuousLatentPolicyNet)
instead of Categorical(logits_theta(s)) — nabla_theta log pi_theta is
Normal.log_prob instead of Categorical.log_prob, everything else (data
collection lives in train.py, no replay buffer, on-policy Monte Carlo) is
identical.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from online_RL_ucpg_v2.model import ContinuousLatentPolicyNet


class UCPGAgentV2:
    """
    Parameters
    ----------
    latent_dim      : surrogate latent space dimension
    action_min      : lower end of the policy mean's aim range [W]
    action_max      : upper end of the policy mean's aim range [W]
    hidden          : policy network hidden width
    depth           : number of ResidualBlocks in the policy trunk
    dropout         : dropout inside each block (0 = off)
    n_layers        : number of LPBF build layers
    layer_embed_dim : dimension of the learned layer embedding
    log_sigma_init, min_log_sigma, max_log_sigma : Gaussian std hyperparameters
                      (see model.ContinuousLatentPolicyNet)
    lr_theta        : policy learning rate  (alpha_theta)
    lambda_init     : initial Lagrange multiplier (>= 0)
    max_grad_norm   : gradient-norm clipping threshold for the policy update
    device          : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        latent_dim:      int,
        action_min:      float,
        action_max:      float,
        hidden:          int   = 128,
        depth:           int   = 3,
        dropout:         float = 0.0,
        n_layers:        int   = 12,
        layer_embed_dim: int   = 8,
        log_sigma_init:  float = -1.0,
        min_log_sigma:   float = -3.0,
        max_log_sigma:   float = 0.0,
        lr_theta:        float = 3e-4,
        lambda_init:     float = 0.0,
        max_grad_norm:   float = 10.0,
        device:          str   = "cpu",
    ) -> None:
        self.latent_dim       = latent_dim
        self.action_min       = action_min
        self.action_max       = action_max
        self.device           = device
        self.max_grad_norm    = max_grad_norm
        self._hidden          = hidden
        self._depth           = depth
        self._dropout         = dropout
        self._n_layers        = n_layers
        self._layer_embed_dim = layer_embed_dim
        self._log_sigma_init  = log_sigma_init
        self._min_log_sigma   = min_log_sigma
        self._max_log_sigma   = max_log_sigma

        self.policy = ContinuousLatentPolicyNet(
            latent_dim, action_min, action_max, hidden, depth, dropout,
            n_layers, layer_embed_dim, log_sigma_init, min_log_sigma, max_log_sigma,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_theta)

        self.lam = float(lambda_init)   # Lagrange multiplier lambda >= 0

    # =========================================================================
    # Action selection (rollout — no grad)
    # =========================================================================

    @torch.no_grad()
    def select_action(self, obs, explore: bool = True) -> float:
        """
        Parameters
        ----------
        obs     : (obs_dim,) float32 numpy array — latent state + layer token + cool_time token
        explore : if True, sample a ~ pi_theta(.|s) (training rollout).
                  if False, take the Gaussian mean mu (greedy evaluation).

        Returns
        -------
        action_W : float — continuous laser power [W], UNCLIPPED
        """
        self.policy.eval()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, sigma = self.policy(obs_t)
        if explore:
            action = Normal(mu, sigma).sample()
        else:
            action = mu
        self.policy.train()
        return float(action.item())

    def select_action_greedy(self, obs) -> float:
        return self.select_action(obs, explore=False)

    # =========================================================================
    # Monte Carlo policy-gradient update
    # =========================================================================

    def policy_step(
        self,
        obs_flat:       torch.Tensor,   # (N*T, obs_dim)
        actions_flat:   torch.Tensor,   # (N*T,) float — laser power [W]
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
                  entropy bonus). For a Gaussian, entropy = 0.5*log(2*pi*e*sigma^2),
                  a direct, monotonic readout of the (state-independent) sigma.
        """
        mu, sigma = self.policy(obs_flat)
        dist      = Normal(mu, sigma)
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
                "action_min":      self.action_min,
                "action_max":      self.action_max,
                "hidden":          self._hidden,
                "depth":           self._depth,
                "dropout":         self._dropout,
                "n_layers":        self._n_layers,
                "layer_embed_dim": self._layer_embed_dim,
                "log_sigma_init":  self._log_sigma_init,
                "min_log_sigma":   self._min_log_sigma,
                "max_log_sigma":   self._max_log_sigma,
            },
        }
        if extra:
            data.update(extra)
        torch.save(data, path)
        print(f"[agent] Checkpoint saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "UCPGAgentV2":
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        cfg   = ckpt["agent_config"]
        agent = cls(device=device, **cfg)
        agent.policy.load_state_dict(ckpt["policy_state_dict"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        agent.lam = float(ckpt.get("lam", 0.0))
        print(f"[agent] Loaded checkpoint from {path}  (lambda={agent.lam:.5f})")
        return agent

    def __repr__(self) -> str:
        return f"UCPGAgentV2({self.policy}  lambda={self.lam:.5f})"
