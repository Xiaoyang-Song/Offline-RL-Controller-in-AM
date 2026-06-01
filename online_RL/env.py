"""
online_RL/env.py
----------------
Gym-like LPBF environment backed by the trained surrogate dynamics model.

Episode structure
-----------------
  Each episode = one 12-layer LPBF build.
  At every step the agent chooses a discrete laser power; the surrogate
  predicts the resulting temperature field; a reward is computed.

  state   : [normalised temperature field (D=1053,) ‖ normalised layer index (1,)]
            shape (D+1 = 1054,), float32
            The layer index is normalised to [0, 1]: layer_idx / (n_layers − 1).
            Including the layer index is critical: the Q-net must behave very
            differently on a cold layer-1 substrate vs. a hot layer-7 substrate.
  action  : int index into ACTION_LIST  (26 choices, 150→400 W)
  reward  : −meanDeviation  ∈ (−∞, 0]  (0 = perfect thermal control)
  done    : True after n_layers steps

Reward function — mirrors MATLAB simulateHeatingCooling.m
----------------------------------------------------------
  For nodes i inside the square scan region:
      deviation_i = max(0, T_l − u_i) + max(0, u_i − T_h)
  meanDeviation   = mean(deviation) / (T_h − T_l)
  reward          = −meanDeviation

Geometry defaults (from test.py)
---------------------------------
  width=12, height=3  [mm or dimensionless – same as mesh units]
  tempRange = [2000.0, 2800.0] K
  squareSideFraction linearly interpolated 0.40 → 0.50 across layers
  (matches the test.py evaluation protocol)

Mesh-based node masks
---------------------
  If surrogate_model/mesh.mat is present (produced by extract_mesh.m in
  MATLAB), the exact per-layer node masks are computed from the geometry.
  Otherwise all nodes are used as an approximation (warning is printed).
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat

# ── Discrete action space (matches offline RL model.py) ─────────────────────
ACTION_LIST = torch.arange(150, 410, 10, dtype=torch.float32)   # 26 actions
NUM_ACTIONS = len(ACTION_LIST)


class LPBFEnv:
    """
    LPBF process environment driven by a pre-trained surrogate model.

    Parameters
    ----------
    surrogate       : ResidualDynamicsModel — in eval mode, on `device`
    state_mean      : (D,) float32 tensor  — surrogate normalisation mean
    state_std       : (D,) float32 tensor  — surrogate normalisation std
    action_mean     : float                — surrogate action norm mean [W]
    action_std      : float                — surrogate action norm std  [W]
    temp_range      : (T_l, T_h) nominal temperature window [K]
    n_layers        : episode length (number of LPBF layers)
    initial_temp    : initial uniform temperature [K]
    device          : torch device string
    mesh_path       : path to mesh.mat; falls back to all-nodes if missing
    width           : domain width  [same units as mesh X]
    height          : domain height [same units as mesh Y]
    sq_frac_start   : squareSideFraction at layer 0  (matches test.py: 0.4)
    sq_frac_end     : squareSideFraction at last layer (matches test.py: 0.5)
    """

    def __init__(
        self,
        surrogate,                           # EnsembleLatentDynamicsModel, eval mode
        state_mean:               torch.Tensor,
        state_std:                torch.Tensor,
        action_mean:              float,
        action_std:               float,
        temp_range:               Tuple[float, float] = (2000.0, 2800.0),
        n_layers:                 int   = 12,
        initial_temp:             float = 300.0,
        device:                   str   = "cpu",
        mesh_path:                str   = "surrogate_model/mesh.mat",
        width:                    float = 12.0,
        height:                   float = 3.0,
        sq_frac_start:            float = 0.4,
        sq_frac_end:              float = 0.5,
        uncertainty_penalty_weight: float = 0.0,
    ) -> None:
        self.surrogate                  = surrogate
        self.state_mean                 = state_mean.to(device)
        self.state_std                  = state_std.to(device)
        self.action_mean                = action_mean
        self.action_std                 = action_std
        self.T_l, self.T_h              = temp_range
        self.n_layers                   = n_layers
        self.initial_temp               = initial_temp
        self.device                     = device
        self.width                      = width
        self.height                     = height
        self.uncertainty_penalty_weight = uncertainty_penalty_weight

        # Running EMA of epistemic_std — normalises the penalty so that λ is
        # interpretable as "reward units lost per standard unit of excess uncertainty"
        # rather than depending on the absolute scale of the latent-space std.
        # Initialised to None; set on the first step and updated every step.
        # Not reset between episodes: we want the lifetime average across the
        # entire training run, not per-episode.
        self._epist_ema: Optional[float] = None
        self._ema_alpha: float = 0.995   # slow decay → stable baseline

        # squareSideFraction per layer (layer 0 → sq_frac_start, last → sq_frac_end)
        self.sq_fracs = np.linspace(sq_frac_start, sq_frac_end, n_layers)

        self.state_dim  = int(state_mean.shape[0])   # raw temperature field dim (1053)
        self.obs_dim    = self.state_dim + 1          # +1 for normalised layer index
        self.n_actions  = NUM_ACTIONS
        self.action_list = ACTION_LIST  # raw laser powers [W]

        # ── try to load mesh for exact node-mask computation ──────────────
        self._nodes_xy: Optional[np.ndarray] = None   # (N, 2)
        if os.path.exists(mesh_path):
            try:
                data  = loadmat(mesh_path)
                nodes = data["nodes"]                         # (2, N)  X, Y rows
                # MATLAB stores as (2, N); transpose to (N, 2)
                self._nodes_xy = (nodes.T if nodes.shape[0] == 2 else nodes).astype(np.float32)
                print(
                    f"[LPBFEnv] Mesh loaded: {self._nodes_xy.shape[0]} nodes  "
                    f"X∈[{self._nodes_xy[:,0].min():.2f}, {self._nodes_xy[:,0].max():.2f}]  "
                    f"Y∈[{self._nodes_xy[:,1].min():.2f}, {self._nodes_xy[:,1].max():.2f}]"
                )
            except Exception as exc:
                print(f"[LPBFEnv] Warning: could not load mesh from '{mesh_path}': {exc}")
                print("[LPBFEnv] Falling back to all-nodes reward computation.")
        else:
            print(
                f"[LPBFEnv] mesh.mat not found at '{mesh_path}'. "
                "Using all nodes for reward (approximation). "
                "Run surrogate_model/extract_mesh.m in MATLAB for exact masks."
            )

        # ── per-layer node masks (precomputed once if mesh is available) ──
        # Each entry is a boolean (D,) array or None (→ use all nodes).
        self._masks: list = [self._build_mask(i) for i in range(n_layers)]

        # ── internal episode state ────────────────────────────────────────
        self._state_raw: Optional[np.ndarray] = None   # (D,) float32 Kelvin
        self._layer: int = 0

    # =========================================================================
    # Public API
    # =========================================================================

    def reset(self) -> np.ndarray:
        """
        Reset environment to the initial all-300 K state.

        Returns
        -------
        obs : (obs_dim = D+1,) float32 numpy array
              Normalised temperature field concatenated with normalised layer index.
        """
        self._state_raw = np.full(self.state_dim, self.initial_temp, dtype=np.float32)
        self._layer = 0
        return self._make_obs(self._state_raw, self._layer)

    def step(self, action_idx: int):
        """
        Apply one laser-layer step.

        Parameters
        ----------
        action_idx : int  index into ACTION_LIST (0 … n_actions−1)

        Returns
        -------
        next_obs : (obs_dim = D+1,) float32 np.ndarray — normalised field + layer token
        reward   : float                               — −meanDeviation
        done     : bool                                — True after n_layers steps
        info     : dict  {'layer', 'action_W', 'mean_temp_K', 'max_temp_K'}
        """
        if self._state_raw is None:
            raise RuntimeError("Call reset() before step().")
        if not (0 <= action_idx < self.n_actions):
            raise ValueError(f"action_idx={action_idx} out of range [0, {self.n_actions})")

        action_W = float(ACTION_LIST[action_idx].item())

        # ── run surrogate dynamics ────────────────────────────────────────
        s_t = torch.tensor(self._state_raw, dtype=torch.float32,
                           device=self.device).unsqueeze(0)       # (1, D)
        a_t = torch.tensor([[action_W]], dtype=torch.float32,
                           device=self.device)                     # (1, 1)

        s_norm = (s_t - self.state_mean) / self.state_std
        a_norm = (a_t - self.action_mean) / self.action_std

        # ── ensemble latent surrogate forward pass ────────────────────────
        with torch.no_grad():
            layer_idx_t    = torch.tensor([self._layer], dtype=torch.long,
                                          device=self.device)          # (1,)
            z_t            = self.surrogate.encode(s_norm)             # (1, latent_dim)
            mu_mean, mu_std = self.surrogate.predict_ensemble(
                z_t, a_norm, layer_idx_t
            )                                                           # both (1, latent_dim)
            z_t1           = z_t + mu_mean
            s2_norm        = self.surrogate.decode(z_t1)               # (1, D)
            epist_std      = float(mu_std.mean().item())               # scalar

        # Denormalise → raw Kelvin for reward
        s2_raw = (s2_norm * self.state_std + self.state_mean).squeeze(0).cpu().numpy()

        # ── compute reward ────────────────────────────────────────────────
        reward_base = self._compute_reward(s2_raw, self._layer)

        # Uncertainty penalty — EMA-normalised so λ is scale-invariant.
        # Update running EMA of epistemic_std (persists across episodes).
        if self._epist_ema is None:
            self._epist_ema = epist_std
        else:
            self._epist_ema = (self._ema_alpha * self._epist_ema
                               + (1.0 - self._ema_alpha) * epist_std)

        # normalised_std ≈ 1.0 for typical in-distribution steps,
        #                 >> 1.0 for OOD steps.
        # λ is now directly interpretable: penalty = λ at typical uncertainty,
        # 2λ at twice-typical uncertainty, etc.
        normalised_std = epist_std / max(self._epist_ema, 1e-8)
        reward = reward_base - self.uncertainty_penalty_weight * normalised_std

        # ── advance internal state ────────────────────────────────────────
        self._state_raw = s2_raw
        self._layer    += 1
        done = self._layer >= self.n_layers

        info = {
            "layer":           self._layer,
            "action_W":        action_W,
            "mean_temp_K":     float(s2_raw.mean()),
            "max_temp_K":      float(s2_raw.max()),
            "epistemic_std":   epist_std,        # raw ensemble std (latent space)
            "normalised_std":  normalised_std,   # epist_std / ema  (scale-free)
            "epist_ema":       self._epist_ema,  # current EMA baseline
            "reward_base":     reward_base,      # reward before uncertainty penalty
        }

        return self._make_obs(s2_raw, self._layer), reward, done, info

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _normalise(self, state_raw: np.ndarray) -> np.ndarray:
        """z-score normalise a raw temperature field → (D,) float32 numpy."""
        s = torch.tensor(state_raw, dtype=torch.float32, device=self.device)
        return ((s - self.state_mean) / self.state_std).cpu().numpy()

    def _make_obs(self, state_raw: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Build the observation vector: [normalised temperature field ‖ layer token].

        The layer token = layer_idx / (n_layers − 1)  ∈ [0, 1], giving the
        Q-network explicit context about where in the build it currently is.
        Without this, the Q-net must learn the same weights for cold layer-1
        and hot layer-7 substrates — an impossible compromise.

        Returns
        -------
        obs : (D+1,) float32 numpy array
        """
        state_norm = self._normalise(state_raw)                          # (D,)
        layer_token = np.array(
            [min(layer_idx, self.n_layers - 1) / max(self.n_layers - 1, 1)],
            dtype=np.float32,   # (1,) clamped to [0, 1]
        )
        return np.concatenate([state_norm, layer_token])                 # (D+1,)

    def _build_mask(self, layer_idx: int) -> Optional[np.ndarray]:
        """
        Precompute boolean mask for nodes inside the square scan region at
        layer `layer_idx`.  Returns None if mesh is not available.
        """
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
