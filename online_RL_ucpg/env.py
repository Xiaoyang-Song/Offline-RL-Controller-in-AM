"""
online_RL_ucpg/env.py
------------------------
Latent-space LPBF environment for Uncertainty-Constrained Policy Gradient (UCPG).

Differs from online_RL/env.py in two ways
-------------------------------------------
  1. Observations are latent, not raw-field:
         obs = [z_t (latent_dim,) ‖ layer_token (1,)]
     The UCPG policy pi_theta: Z -> A operates entirely in the surrogate's
     latent space (per the method spec) — it never sees the 1053-dim
     temperature field directly. The internal transition is also carried
     forward in latent space (z_{t+1} = z_t + mu_mean), matching how the
     surrogate itself was trained/evaluated for rollout — the raw field is
     only decoded as a side computation to score the reward.

  2. Reward and uncertainty are returned SEPARATELY, never blended.
         u_t = u_epi(s_t, a_t) + u_alea(s_t, a_t)
     u_epi / u_alea are the mean-over-latent-dims epistemic / aleatoric std
     from the Gaussian bootstrap ensemble (surrogate_model_latent_uncertainty).
     The training loop keeps independent reward and uncertainty trajectories
     so it can form G_r and G_u with (possibly different) discount factors
     gamma_r, gamma_u, and combine them only via the Lagrangian
     G_lambda = G_r - lambda * G_u — never inside the environment itself.

Episode / reward mechanics (mirrors online_RL/env.py exactly)
-----------------------------------------------------------------
  Each episode = one 12-layer LPBF build.
  reward = -meanDeviation in (-inf, 0]  (0 = perfect thermal control),
  computed from the DECODED raw temperature field using the same square-ROI
  node mask and deviation formula as online_RL/env.py / MATLAB
  simulateHeatingCooling.m.

Action space
------------
  Discrete laser-power choices, configurable via action_min/action_max/
  action_step (default 150-400 W step 10 -> 26 actions). Unlike online_RL's
  fixed module-level ACTION_LIST, this is intentionally per-instance: the
  whole point of the UCPG experiment is to let the policy choose from a
  WIDER action range than the surrogate was trained on (e.g. surrogate
  trained on 150-300 W, policy allowed to pick up to 400 W) so the
  uncertainty constraint has something to actually avoid.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat


class LatentLPBFEnv:
    """
    LPBF process environment driven by a pre-trained Gaussian ensemble latent
    surrogate (surrogate_model_latent_uncertainty), exposing latent
    observations and separate reward / uncertainty signals for UCPG.

    Parameters
    ----------
    surrogate    : EnsembleGaussianLatentDynamicsModel — in eval mode, on `device`
    state_mean   : (D,) float32 tensor — surrogate normalisation mean
    state_std    : (D,) float32 tensor — surrogate normalisation std
    action_mean  : float — surrogate action norm mean [W]
    action_std   : float — surrogate action norm std  [W]
    temp_range   : (T_l, T_h) nominal temperature window [K]
    n_layers     : episode length (number of LPBF layers)
    initial_temp : initial uniform temperature [K]
    device       : torch device string
    mesh_path    : path to mesh.mat; falls back to all-nodes if missing
    width        : domain width  [same units as mesh X]
    height       : domain height [same units as mesh Y]
    sq_frac_start: squareSideFraction at layer 0
    sq_frac_end  : squareSideFraction at last layer
    action_min   : lower bound of the discrete action grid [W]
    action_max   : upper bound of the discrete action grid [W] (inclusive)
    action_step  : spacing of the discrete action grid [W]
    """

    def __init__(
        self,
        surrogate,                           # EnsembleGaussianLatentDynamicsModel, eval mode
        state_mean:    torch.Tensor,
        state_std:     torch.Tensor,
        action_mean:   float,
        action_std:    float,
        temp_range:    Tuple[float, float] = (2000.0, 2800.0),
        n_layers:      int   = 12,
        initial_temp:  float = 300.0,
        device:        str   = "cpu",
        mesh_path:     str   = "surrogate_model/mesh.mat",
        width:         float = 12.0,
        height:        float = 3.0,
        sq_frac_start: float = 0.4,
        sq_frac_end:   float = 0.5,
        action_min:    float = 150.0,
        action_max:    float = 400.0,
        action_step:   float = 10.0,
    ) -> None:
        self.surrogate     = surrogate
        self.state_mean    = state_mean.to(device)
        self.state_std     = state_std.to(device)
        self.action_mean   = action_mean
        self.action_std    = action_std
        self.T_l, self.T_h = temp_range
        self.n_layers       = n_layers
        self.initial_temp   = initial_temp
        self.device          = device
        self.width           = width
        self.height          = height

        self.sq_fracs = np.linspace(sq_frac_start, sq_frac_end, n_layers)

        self.state_dim  = int(state_mean.shape[0])   # raw temperature field dim (1053)
        self.latent_dim = int(surrogate.latent_dim)
        self.obs_dim    = self.latent_dim + 1          # +1 for normalised layer index

        self.action_list = torch.arange(
            action_min, action_max + action_step / 2.0, action_step, dtype=torch.float32
        )
        self.n_actions = len(self.action_list)

        # ── try to load mesh for exact node-mask computation ──────────────
        self._nodes_xy: Optional[np.ndarray] = None   # (N, 2)
        if os.path.exists(mesh_path):
            try:
                data  = loadmat(mesh_path)
                nodes = data["nodes"]                         # (2, N)  X, Y rows
                self._nodes_xy = (nodes.T if nodes.shape[0] == 2 else nodes).astype(np.float32)
                print(
                    f"[LatentLPBFEnv] Mesh loaded: {self._nodes_xy.shape[0]} nodes  "
                    f"X∈[{self._nodes_xy[:,0].min():.2f}, {self._nodes_xy[:,0].max():.2f}]  "
                    f"Y∈[{self._nodes_xy[:,1].min():.2f}, {self._nodes_xy[:,1].max():.2f}]"
                )
            except Exception as exc:
                print(f"[LatentLPBFEnv] Warning: could not load mesh from '{mesh_path}': {exc}")
                print("[LatentLPBFEnv] Falling back to all-nodes reward computation.")
        else:
            print(
                f"[LatentLPBFEnv] mesh.mat not found at '{mesh_path}'. "
                "Using all nodes for reward (approximation)."
            )

        self._masks: list = [self._build_mask(i) for i in range(n_layers)]

        # ── internal episode state (latent, carried forward like model.rollout) ──
        self._z:     Optional[torch.Tensor] = None   # (1, latent_dim)
        self._layer: int = 0

    # =========================================================================
    # Public API
    # =========================================================================

    def reset(self) -> np.ndarray:
        """
        Reset environment to the initial all-300 K state, encoded to latent.

        Returns
        -------
        obs : (obs_dim = latent_dim+1,) float32 numpy array
              Latent state concatenated with normalised layer index.
        """
        s0_raw = np.full(self.state_dim, self.initial_temp, dtype=np.float32)
        s0_t   = torch.tensor(s0_raw, dtype=torch.float32, device=self.device).unsqueeze(0)
        s0_norm = (s0_t - self.state_mean) / self.state_std
        with torch.no_grad():
            self._z = self.surrogate.encode(s0_norm)   # (1, latent_dim)
        self._layer = 0
        return self._make_obs(self._z, self._layer)

    def step(self, action_idx: int):
        """
        Apply one laser-layer step.

        Parameters
        ----------
        action_idx : int  index into self.action_list (0 … n_actions−1)

        Returns
        -------
        next_obs : (obs_dim,) float32 np.ndarray — latent state + layer token
        reward   : float  — -meanDeviation, UNPENALISED by uncertainty
        done     : bool   — True after n_layers steps
        info     : dict   {'layer', 'action_W', 'mean_temp_K', 'max_temp_K',
                            'epistemic_std', 'aleatoric_std', 'uncertainty'}
                   info['uncertainty'] = epistemic_std + aleatoric_std = u_t
        """
        if self._z is None:
            raise RuntimeError("Call reset() before step().")
        if not (0 <= action_idx < self.n_actions):
            raise ValueError(f"action_idx={action_idx} out of range [0, {self.n_actions})")

        action_W = float(self.action_list[action_idx].item())

        a_t = torch.tensor([[action_W]], dtype=torch.float32, device=self.device)
        a_norm = (a_t - self.action_mean) / self.action_std
        layer_idx_t = torch.tensor([self._layer], dtype=torch.long, device=self.device)

        with torch.no_grad():
            mu_mean, epi_std, ale_std, _tot_std = self.surrogate.predict_ensemble(
                self._z, a_norm, layer_idx_t
            )                                                    # each (1, latent_dim)
            z_next   = self._z + mu_mean
            s_next_n = self.surrogate.decode(z_next)              # (1, D) normalised

        s_next_raw = (s_next_n * self.state_std + self.state_mean).squeeze(0).cpu().numpy()

        reward = self._compute_reward(s_next_raw, self._layer)

        u_epi = float(epi_std.mean().item())
        u_ale = float(ale_std.mean().item())
        u_t   = u_epi + u_ale

        # ── advance internal latent state ─────────────────────────────────
        self._z      = z_next
        self._layer += 1
        done = self._layer >= self.n_layers

        info = {
            "layer":          self._layer,
            "action_W":       action_W,
            "mean_temp_K":    float(s_next_raw.mean()),
            "max_temp_K":     float(s_next_raw.max()),
            "epistemic_std":  u_epi,
            "aleatoric_std":  u_ale,
            "uncertainty":    u_t,
        }

        return self._make_obs(self._z, self._layer), reward, done, info

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _make_obs(self, z: torch.Tensor, layer_idx: int) -> np.ndarray:
        """
        Build the observation vector: [latent state ‖ layer token].

        layer token = layer_idx / (n_layers − 1)  ∈ [0, 1]
        """
        z_np = z.squeeze(0).detach().cpu().numpy()                       # (latent_dim,)
        layer_token = np.array(
            [min(layer_idx, self.n_layers - 1) / max(self.n_layers - 1, 1)],
            dtype=np.float32,
        )
        return np.concatenate([z_np, layer_token])                       # (latent_dim+1,)

    def _build_mask(self, layer_idx: int) -> Optional[np.ndarray]:
        """Precompute boolean mask for nodes inside the square scan region at layer_idx."""
        if self._nodes_xy is None:
            return None

        frac    = float(self.sq_fracs[layer_idx])
        sq_side = min(self.width, self.height) * frac
        half    = sq_side / 2.0
        cx      = self.width  / 2.0
        cy      = self.height / 2.0

        xs   = self._nodes_xy[:, 0]
        ys   = self._nodes_xy[:, 1]
        mask = (
            (xs >= cx - half) & (xs <= cx + half) &
            (ys >= cy - half) & (ys <= cy + half)
        )
        return mask

    def _compute_reward(self, state_raw: np.ndarray, layer_idx: int) -> float:
        """
        reward = −meanDeviation   (mirrors MATLAB simulateHeatingCooling.m)

        deviation_i = max(0, T_l − u_i) + max(0, u_i − T_h)
        meanDeviation = mean(deviation[inSquare]) / (T_h − T_l)
        reward = −meanDeviation  ∈ (−∞, 0]
        """
        mask = self._masks[layer_idx]
        u    = state_raw[mask] if mask is not None else state_raw

        T_l, T_h = self.T_l, self.T_h
        dev = np.zeros_like(u)
        dev[u < T_l] = T_l - u[u < T_l]
        dev[u > T_h] = u[u > T_h] - T_h

        mean_dev = dev.mean() / (T_h - T_l)
        return -float(mean_dev)
