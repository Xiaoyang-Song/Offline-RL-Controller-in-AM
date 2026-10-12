"""
online_RL_ucpg_v3/env.py
----------------------------
LPBF surrogate environment with an UNCERTAINTY-SHAPED, per-node reward and
uncertainty carried ACROSS layers by rolling out every ensemble member as its
own trajectory (see README.md for the full math).

Differences from online_RL_ucpg_v2.env.TwoStageLatentLPBFEnv
----------------------------------------------------------------
1. Member rollouts. The K members of the two-stage surrogate are rolled out as
   K separate trajectories (member k uses ITS OWN heating and cooling network
   on ITS OWN previous state):
       h^(k)_t      = s^(k)_t + Δheat_k(s^(k)_t, a_t, t)
       s^(k)_{t+1}  = h^(k)_t + Δcool_k(h^(k)_t, c, t)
   so ensemble disagreement accumulates across layers exactly as the model
   would produce it (no linearisation, correct correlation across layers).
   The policy observes the MEMBER-MEAN state, in the same format as v2
   ([normalised field ‖ layer token ‖ cool token]) — so v2 policies, the
   online-Q network and all real-simulator evaluators work unchanged.

2. Reward field: post-cooling s_{t+1} by default (--reward_field cool), or the
   end-of-heating field h_t (--reward_field heat, v2 behaviour).

3. Reward shaping (reward_mode='shaped'). Per ROI node i, with member-mean
   temperature mu_i, epistemic std sigma_i (std across members, Kelvin), hinge
   deviation d(u) = max(0, T_l - u) + max(0, u - T_h) and window W = T_h - T_l:
       w_i      = tau^2 / (tau^2 + sigma_i^2)            (trust in the model at node i)
       shaping='additive' (default):  d_tilde = d(mu_i) + (1 - w_i) * c
       shaping='shrink':              d_tilde = w_i * d(mu_i) + (1 - w_i) * c
       r_t      = -(1/|R_t|) * sum_i d_tilde_i / W
   'additive' guarantees d_tilde >= d, so uncertainty can never LOWER the
   penalty. 'shrink' does whenever d(mu_i) > c (observed for the post-cooling
   field at low laser power) — it is kept only for comparison.
   reward_mode='plain' gives r_t = -(1/|R_t|) sum_i d(mu_i) / W (no uncertainty).
   info always reports BOTH, plus the plain deviation of the other field.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat


def hinge_deviation(u: np.ndarray, T_l: float, T_h: float) -> np.ndarray:
    """d(u) = max(0, T_l - u) + max(0, u - T_h), elementwise [K]."""
    return np.maximum(0.0, T_l - u) + np.maximum(0.0, u - T_h)


def build_roi_masks(mesh_path: str, n_layers: int, width: float, height: float,
                    sq_frac_start: float, sq_frac_end: float):
    """Per-layer boolean in-square masks — identical construction to
    online_RL_ucpg_v2.env._build_mask (and simulateHeatingCooling_v2.m's
    inSquare). Returns a list of (N,) bool arrays, or Nones if no mesh."""
    if not os.path.exists(mesh_path):
        print(f"[v3.env] mesh.mat not found at '{mesh_path}' — reward uses ALL nodes.")
        return [None] * n_layers
    nodes = loadmat(mesh_path)["nodes"]
    xy = (nodes.T if nodes.shape[0] == 2 else nodes).astype(np.float32)
    masks = []
    for frac in np.linspace(sq_frac_start, sq_frac_end, n_layers):
        half = min(width, height) * frac / 2.0
        cx, cy = width / 2.0, height / 2.0
        masks.append((xy[:, 0] >= cx - half) & (xy[:, 0] <= cx + half) &
                     (xy[:, 1] >= cy - half) & (xy[:, 1] <= cy + half))
    return masks


class UncertaintyShapedLPBFEnv:
    """
    Parameters (beyond v2's, which keep their meaning)
    --------------------------------------------------
    reward_field : 'cool' (post-cooling s_{t+1}, default) or 'heat' (end-of-heating h_t)
    reward_mode  : 'shaped' (uncertainty-shaped, default) or 'plain'
    tau          : trust scale [K] — node trusted at weight 1/2 when sigma_i = tau.
                   Must be set (directly or via calibrate_tau) before reward_mode='shaped' is used.
    pess_dev     : penalty c [K] for a fully untrusted node (default: W = T_h - T_l)
    shaping      : 'additive' (default, d + (1-w) c) or 'shrink' (w d + (1-w) c)
    """

    def __init__(
        self,
        surrogate,
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
        reward_field:   str   = "cool",
        reward_mode:    str   = "shaped",
        tau:            Optional[float] = None,
        pess_dev:       Optional[float] = None,
        shaping:        str   = "additive",
    ) -> None:
        assert shaping in ("additive", "shrink"), shaping
        assert reward_field in ("cool", "heat"), reward_field
        assert reward_mode in ("shaped", "plain"), reward_mode
        self.surrogate = surrogate
        self.K = int(surrogate.n_ensemble)
        self.state_mean = state_mean.to(device)
        self.state_std = state_std.to(device)
        self.lp_mean, self.lp_std = lp_mean, lp_std
        self.cool_mean, self.cool_std = cool_mean, cool_std
        self.T_l, self.T_h = temp_range
        self.W = self.T_h - self.T_l
        self.n_layers = n_layers
        self.initial_temp = initial_temp
        self.device = device
        self.action_min, self.action_max = action_min, action_max
        self.cool_time_min, self.cool_time_max = cool_time_min, cool_time_max
        self.reward_field = reward_field
        self.reward_mode = reward_mode
        self.tau = tau
        self.pess_dev = self.W if pess_dev is None else pess_dev
        self.shaping = shaping

        self.state_dim = int(state_mean.shape[0])
        self.latent_dim = self.state_dim                  # name kept for v2 compatibility
        self.obs_dim = self.state_dim + 2                 # + layer token + cool token
        self._masks = build_roi_masks(mesh_path, n_layers, width, height, sq_frac_start, sq_frac_end)
        self._std_np = self.state_std.detach().cpu().numpy()
        self._mean_np = self.state_mean.detach().cpu().numpy()

        self._s: Optional[torch.Tensor] = None            # (K, D) normalised member states
        self._layer = 0
        self._cool_time = 0.0

    # ------------------------------------------------------------------ API
    def reset(self, cool_time: Optional[float] = None) -> np.ndarray:
        s0 = torch.full((1, self.state_dim), self.initial_temp, dtype=torch.float32, device=self.device)
        s0 = (s0 - self.state_mean) / self.state_std
        self._s = s0.expand(self.K, -1).clone()            # every member starts from the known IC
        self._layer = 0
        self._cool_time = (float(cool_time) if cool_time is not None
                           else float(np.random.uniform(self.cool_time_min, self.cool_time_max)))
        return self._make_obs()

    @torch.no_grad()
    def step(self, action_W: float):
        if self._s is None:
            raise RuntimeError("Call reset() before step().")
        K = self.K
        t = self._layer
        a_n = torch.full((1, 1), (action_W - self.lp_mean) / self.lp_std, device=self.device)
        c_n = torch.full((1, 1), (self._cool_time - self.cool_mean) / self.cool_std, device=self.device)
        li = torch.tensor([t], dtype=torch.long, device=self.device)

        heat, nxt = [], []
        for k in range(K):                                  # member k on its own state
            s_k = self._s[k:k + 1]
            h_k = s_k + self.surrogate.heating_transitions[k](s_k, a_n, li)[0]
            n_k = h_k + self.surrogate.cooling_transitions[k](h_k, c_n, li)[0]
            heat.append(h_k); nxt.append(n_k)
        heat = torch.cat(heat, 0)                           # (K, D) normalised
        nxt = torch.cat(nxt, 0)

        heat_raw = (heat * self.state_std + self.state_mean).cpu().numpy()   # (K, D) Kelvin
        next_raw = (nxt * self.state_std + self.state_mean).cpu().numpy()
        r = self._rewards(heat_raw, next_raw, t)

        self._s = nxt
        self._layer += 1
        done = self._layer >= self.n_layers

        heat_mean = heat_raw.mean(0)
        next_mean = next_raw.mean(0)
        info = dict(
            layer=self._layer, action_W=action_W, cool_time_s=self._cool_time,
            mean_heat_temp_K=float(heat_mean.mean()), max_heat_temp_K=float(heat_mean.max()),
            mean_next_temp_K=float(next_mean.mean()), max_next_temp_K=float(next_mean.max()),
            **r,
        )
        reward = r["reward_shaped"] if self.reward_mode == "shaped" else r["reward_plain"]
        return self._make_obs(), float(reward), done, info

    # ------------------------------------------------------------- rewards
    def _rewards(self, heat_raw: np.ndarray, next_raw: np.ndarray, t: int) -> dict:
        m = self._masks[t]
        sel = (lambda f: f[:, m]) if m is not None else (lambda f: f)
        field = next_raw if self.reward_field == "cool" else heat_raw
        other = heat_raw if self.reward_field == "cool" else next_raw

        f = sel(field)                                      # (K, |R|)
        mu = f.mean(0)
        sigma = f.std(0)                                    # epistemic, carried across layers
        d = hinge_deviation(mu, self.T_l, self.T_h)
        r_plain = -float(d.mean() / self.W)

        out = dict(
            reward_plain=r_plain,
            reward_plain_other_field=-float(hinge_deviation(sel(other).mean(0), self.T_l, self.T_h).mean() / self.W),
            sigma_epi_roi_K=float(sigma.mean()),
            epistemic_std=float(sigma.mean()),              # v2-compatible key
        )
        if self.tau is not None:
            w = self.tau ** 2 / (self.tau ** 2 + sigma ** 2)
            if self.shaping == "additive":
                d_tilde = d + (1.0 - w) * self.pess_dev
            else:
                d_tilde = w * d + (1.0 - w) * self.pess_dev
            out["reward_shaped"] = -float(d_tilde.mean() / self.W)
            out["trust_roi"] = float(w.mean())
        else:
            if self.reward_mode == "shaped":
                raise RuntimeError("reward_mode='shaped' needs tau (pass --tau or call calibrate_tau).")
            out["reward_shaped"] = float("nan")
            out["trust_roi"] = float("nan")
        return out

    # ----------------------------------------------------------------- obs
    def _make_obs(self) -> np.ndarray:
        z = self._s.mean(0).detach().cpu().numpy()          # member-mean normalised state
        layer_token = min(self._layer, self.n_layers - 1) / max(self.n_layers - 1, 1)
        cool_token = (self._cool_time - self.cool_time_min) / max(self.cool_time_max - self.cool_time_min, 1e-8)
        return np.concatenate([z, np.array([layer_token, cool_token], dtype=np.float32)]).astype(np.float32)


def calibrate_tau(env: UncertaintyShapedLPBFEnv, id_min: float, id_max: float, k: float = 2.0,
                  n_episodes: int = 64, seed: int = 0) -> Tuple[float, float]:
    """tau = k x median ROI epistemic sigma [K] of MULTI-STEP member rollouts whose
    actions are drawn uniformly from the in-distribution range [id_min, id_max]
    (every layer, every ROI node pooled). Returns (tau, median_sigma)."""
    rng = np.random.default_rng(seed)
    saved_mode, saved_tau = env.reward_mode, env.tau
    env.reward_mode, env.tau = "plain", None
    sig = []
    for _ in range(n_episodes):
        env.reset()
        for _t in range(env.n_layers):
            _o, _r, _d, info = env.step(float(rng.uniform(id_min, id_max)))
            sig.append(info["sigma_epi_roi_K"])
    env.reward_mode, env.tau = saved_mode, saved_tau
    med = float(np.median(sig))
    return k * med, med
