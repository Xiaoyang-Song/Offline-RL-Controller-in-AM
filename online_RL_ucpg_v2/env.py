"""
online_RL_ucpg_v2/env.py
----------------------------
Latent-space, two-stage (heating/cooling) LPBF environment for Uncertainty-
Constrained Policy Gradient (UCPG), driven by the
surrogate_model_latent_uncertainty_v2 two-stage Gaussian ensemble surrogate.

Differs from online_RL_ucpg/env.py in three ways
---------------------------------------------------
  1. Two-stage transition, not one:
         z_t --[laser_power a_t]--> z_heat --[cool_time c_t]--> z_{t+1}
     The heating stage is action-DEPENDENT (the controller acts here); the
     cooling stage is action-INDEPENDENT (conditioned on cool_time instead —
     "no matter what laser power you applied previously, the cooling
     mechanism is the same"). Both stages are chained in latent space every
     step via surrogate.predict_heating_ensemble / predict_cooling_ensemble,
     exactly mirroring TwoStageEnsembleGaussianLatentDynamicsModel.rollout().

  2. Reward is computed from the END-OF-HEATING field, not the post-cooling
     field. This matches the v2 dataset/simulator correction (see
     surrogate_model_latent_uncertainty_v2/dataset_v2.py and
     ../LPBF-Simulation/simulation_v2/simulateHeatingCooling_v2.m, whose
     meanDeviation is now computed from uHeatFinal, not uFinal): the reward
     signal is about how well the LASER PASS itself hit the target window,
     which cooling has no influence over.

  3. cool_time is NOT an action, but IS part of the observation — it is
     exogenous context, randomly drawn ONCE per episode (matching
     multilayer_random_v2.m: "coolTime is randomly sampled once per
     trajectory ... same coolTime for every layer in this trajectory") and
     held fixed for all 12 layers. The policy observes it (normalised, as
     the last observation component — see _make_obs) but never chooses it:
     in the real process cooling duration is a known build parameter, not
     something hidden from the controller, so there is no reason to make
     this a POMDP by withholding it. This is a contextual-MDP setup — the
     policy conditions its laser-power choice on the cooling regime this
     episode drew, without controlling it.

  Reward and uncertainty are still returned SEPARATELY, never blended — see
  online_RL_ucpg/env.py's docstring for the rationale (unchanged here).

Action space
------------
  Continuous laser power a_t in R (no clipping — see model.py's docstring
  for why). --action_min/--action_max only set the policy's mean "aim
  range" and the (optional) OOD diagnostic threshold; they are not hard
  bounds enforced by the environment.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat

from surrogate_model_latent_uncertainty_v2.model import combine_stage_uncertainties


class TwoStageLatentLPBFEnv:
    """
    LPBF process environment driven by a pre-trained two-stage Gaussian
    ensemble latent surrogate (surrogate_model_latent_uncertainty_v2),
    exposing latent observations and separate reward / uncertainty signals
    for UCPG.

    Parameters
    ----------
    surrogate      : TwoStageEnsembleGaussianLatentDynamicsModel — eval mode, on `device`
    state_mean, state_std       : (D,) float32 tensors — surrogate state normalisation
    lp_mean, lp_std             : float — surrogate laser-power norm stats [W]
    cool_mean, cool_std         : float — surrogate cool-time norm stats [s]
    temp_range     : (T_l, T_h) nominal temperature window [K] (reward target,
                     applied to the END-OF-HEATING field — matches
                     paramsStruct.tempRange in simulateHeatingCooling_v2.m)
    n_layers       : episode length (number of LPBF layers)
    initial_temp   : initial uniform temperature [K]
    device         : torch device string
    mesh_path      : path to mesh.mat; falls back to all-nodes if missing
    width, height  : domain size (same units as mesh X/Y)
    sq_frac_start, sq_frac_end : squareSideFraction schedule endpoints (layer 0 / last)
    action_min, action_max     : policy "aim range" for laser power [W] (soft, not enforced)
    cool_time_min, cool_time_max : range cool_time is drawn from at reset() [s]
                     (matches multilayer_random_v2.m's coolTime_values = 0.05:0.01:0.15,
                     sampled continuously here since the surrogate is a continuous
                     regressor over cool_time)
    """

    def __init__(
        self,
        surrogate,                           # TwoStageEnsembleGaussianLatentDynamicsModel, eval mode
        state_mean:     torch.Tensor,
        state_std:      torch.Tensor,
        lp_mean:        float,
        lp_std:         float,
        cool_mean:      float,
        cool_std:       float,
        temp_range:     Tuple[float, float] = (2000.0, 2800.0),
        n_layers:       int   = 12,
        initial_temp:   float = 300.0,
        device:         str   = "cpu",
        mesh_path:      str   = "surrogate_model/mesh.mat",
        width:          float = 12.0,
        height:         float = 3.0,
        sq_frac_start:  float = 0.4,
        sq_frac_end:    float = 0.5,
        action_min:     float = 100.0,
        action_max:     float = 400.0,
        cool_time_min:  float = 0.05,
        cool_time_max:  float = 0.15,
    ) -> None:
        self.surrogate    = surrogate
        self.state_mean   = state_mean.to(device)
        self.state_std    = state_std.to(device)
        self.lp_mean      = lp_mean
        self.lp_std       = lp_std
        self.cool_mean    = cool_mean
        self.cool_std     = cool_std
        self.T_l, self.T_h = temp_range
        self.n_layers      = n_layers
        self.initial_temp  = initial_temp
        self.device        = device
        self.width         = width
        self.height        = height

        self.action_min    = action_min
        self.action_max    = action_max
        self.cool_time_min = cool_time_min
        self.cool_time_max = cool_time_max

        self.sq_fracs = np.linspace(sq_frac_start, sq_frac_end, n_layers)

        self.state_dim  = int(state_mean.shape[0])   # raw temperature field dim (1053)
        self.latent_dim = int(surrogate.latent_dim)
        self.obs_dim    = self.latent_dim + 2          # +1 layer index, +1 cool_time token

        # ── try to load mesh for exact node-mask computation ──────────────
        self._nodes_xy: Optional[np.ndarray] = None   # (N, 2)
        if os.path.exists(mesh_path):
            try:
                data  = loadmat(mesh_path)
                nodes = data["nodes"]                         # (2, N)  X, Y rows
                self._nodes_xy = (nodes.T if nodes.shape[0] == 2 else nodes).astype(np.float32)
                print(
                    f"[TwoStageLatentLPBFEnv] Mesh loaded: {self._nodes_xy.shape[0]} nodes  "
                    f"X∈[{self._nodes_xy[:,0].min():.2f}, {self._nodes_xy[:,0].max():.2f}]  "
                    f"Y∈[{self._nodes_xy[:,1].min():.2f}, {self._nodes_xy[:,1].max():.2f}]"
                )
            except Exception as exc:
                print(f"[TwoStageLatentLPBFEnv] Warning: could not load mesh from '{mesh_path}': {exc}")
                print("[TwoStageLatentLPBFEnv] Falling back to all-nodes reward computation.")
        else:
            print(
                f"[TwoStageLatentLPBFEnv] mesh.mat not found at '{mesh_path}'. "
                "Using all nodes for reward (approximation)."
            )

        self._masks: list = [self._build_mask(i) for i in range(n_layers)]

        # ── internal episode state (latent, carried forward like model.rollout) ──
        self._z:         Optional[torch.Tensor] = None   # (1, latent_dim)
        self._layer:      int = 0
        self._cool_time:  float = 0.0                     # fixed for the whole episode

    # =========================================================================
    # Public API
    # =========================================================================

    def reset(self) -> np.ndarray:
        """
        Reset environment to the initial all-300 K state, encoded to latent,
        and draw a fresh per-episode cool_time (held fixed for all layers,
        matching how the training data was generated).

        Returns
        -------
        obs : (obs_dim = latent_dim+2,) float32 numpy array
              Latent state concatenated with normalised layer index and
              normalised cool_time (this episode's fixed value).
        """
        s0_raw = np.full(self.state_dim, self.initial_temp, dtype=np.float32)
        s0_t   = torch.tensor(s0_raw, dtype=torch.float32, device=self.device).unsqueeze(0)
        s0_norm = (s0_t - self.state_mean) / self.state_std
        with torch.no_grad():
            self._z = self.surrogate.encode(s0_norm)   # (1, latent_dim)
        self._layer     = 0
        self._cool_time = float(np.random.uniform(self.cool_time_min, self.cool_time_max))
        return self._make_obs(self._z, self._layer)

    def step(self, action_W: float):
        """
        Apply one laser-layer step: heating (action-dependent) then cooling
        (action-independent, using this episode's fixed cool_time).

        Parameters
        ----------
        action_W : float — continuous laser power [W], UNCLIPPED (see model.py)

        Returns
        -------
        next_obs : (obs_dim,) float32 np.ndarray — latent state + layer token + cool_time token
        reward   : float  — -meanDeviation of the END-OF-HEATING field,
                   UNPENALISED by uncertainty
        done     : bool   — True after n_layers steps
        info     : dict   {'layer', 'action_W', 'cool_time_s', 'mean_heat_temp_K',
                            'max_heat_temp_K', 'mean_next_temp_K', 'max_next_temp_K',
                            'epistemic_std', 'aleatoric_std', 'uncertainty'}
                   info['uncertainty'] = combined (heating+cooling) epistemic_std
                   + aleatoric_std = u_t (see
                   surrogate_model_latent_uncertainty_v2.model.combine_stage_uncertainties)
        """
        if self._z is None:
            raise RuntimeError("Call reset() before step().")

        a_t = torch.tensor([[action_W]], dtype=torch.float32, device=self.device)
        c_t = torch.tensor([[self._cool_time]], dtype=torch.float32, device=self.device)
        a_norm = (a_t - self.lp_mean)   / self.lp_std
        c_norm = (c_t - self.cool_mean) / self.cool_std
        layer_idx_t = torch.tensor([self._layer], dtype=torch.long, device=self.device)

        with torch.no_grad():
            mu_heat, heat_epi, heat_ale, _ = self.surrogate.predict_heating_ensemble(
                self._z, a_norm, layer_idx_t
            )
            z_heat = self._z + mu_heat
            heat_pred_n = self.surrogate.decode(z_heat)          # (1, D) normalised

            mu_cool, cool_epi, cool_ale, _ = self.surrogate.predict_cooling_ensemble(
                z_heat, c_norm, layer_idx_t
            )
            z_next = z_heat + mu_cool
            next_pred_n = self.surrogate.decode(z_next)          # (1, D) normalised

        heat_pred_raw = (heat_pred_n * self.state_std + self.state_mean).squeeze(0).cpu().numpy()
        next_pred_raw = (next_pred_n * self.state_std + self.state_mean).squeeze(0).cpu().numpy()

        # Reward is computed from the END-OF-HEATING field — see module docstring.
        reward = self._compute_reward(heat_pred_raw, self._layer)

        total_epi, total_ale, _total_std = combine_stage_uncertainties(
            heat_epi, heat_ale, cool_epi, cool_ale
        )
        u_epi = float(total_epi.mean().item())
        u_ale = float(total_ale.mean().item())
        u_t   = u_epi + u_ale

        # ── advance internal latent state (post-cooling, becomes s_{t+1}) ──
        self._z      = z_next
        self._layer += 1
        done = self._layer >= self.n_layers

        info = {
            "layer":           self._layer,
            "action_W":        action_W,
            "cool_time_s":     self._cool_time,
            "mean_heat_temp_K": float(heat_pred_raw.mean()),
            "max_heat_temp_K":  float(heat_pred_raw.max()),
            "mean_next_temp_K": float(next_pred_raw.mean()),
            "max_next_temp_K":  float(next_pred_raw.max()),
            "epistemic_std":   u_epi,
            "aleatoric_std":   u_ale,
            "uncertainty":     u_t,
        }

        return self._make_obs(self._z, self._layer), reward, done, info

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _make_obs(self, z: torch.Tensor, layer_idx: int) -> np.ndarray:
        """
        Build the observation vector: [latent state ‖ layer token ‖ cool_time token].

        layer token      = layer_idx / (n_layers − 1)  ∈ [0, 1]
        cool_time token  = (cool_time − cool_time_min) / (cool_time_max − cool_time_min)  ∈ [0, 1]

        cool_time is fixed for the whole episode (see reset()) but is NOT an
        action — it is exogenous context, known at reset time (in the real
        process, cooling duration is a set build parameter, not something
        hidden from the controller). Including it in the observation turns
        this into a proper (contextual) MDP: the policy can condition its
        laser-power choice on the actual cooling regime this episode drew,
        rather than only inferring its effect indirectly through how z_t has
        evolved so far — which gives it zero information before the very
        first action, exactly when planning the whole episode matters most.
        """
        z_np = z.squeeze(0).detach().cpu().numpy()                       # (latent_dim,)
        layer_token = min(layer_idx, self.n_layers - 1) / max(self.n_layers - 1, 1)
        cool_range  = max(self.cool_time_max - self.cool_time_min, 1e-8)
        cool_token  = (self._cool_time - self.cool_time_min) / cool_range
        tail = np.array([layer_token, cool_token], dtype=np.float32)
        return np.concatenate([z_np, tail])                              # (latent_dim+2,)

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

    def _compute_reward(self, heat_field_raw: np.ndarray, layer_idx: int) -> float:
        """
        reward = −meanDeviation   (mirrors simulateHeatingCooling_v2.m, computed
        from uHeatFinal / heat_field_raw — the end-of-heating field, NOT the
        post-cooling field)

        deviation_i = max(0, T_l − u_i) + max(0, u_i − T_h)
        meanDeviation = mean(deviation[inSquare]) / (T_h − T_l)
        reward = −meanDeviation  ∈ (−∞, 0]
        """
        mask = self._masks[layer_idx]
        u    = heat_field_raw[mask] if mask is not None else heat_field_raw

        T_l, T_h = self.T_l, self.T_h
        dev = np.zeros_like(u)
        dev[u < T_l] = T_l - u[u < T_l]
        dev[u > T_h] = u[u > T_h] - T_h

        mean_dev = dev.mean() / (T_h - T_l)
        return -float(mean_dev)
