"""
online_RL_ucpg_v3/env.py
----------------------------
LPBF surrogate environment with an UNCERTAINTY-SHAPED, per-node reward and
uncertainty carried ACROSS layers (see README.md for the full math).

uncertainty_mode (how the per-node epistemic sigma is carried across layers):
  'propagated' (DEFAULT) — scheme A: the state mean is chained through the
      ensemble mean (as TwoStageSurrogate.rollout), and the epistemic covariance
      is propagated BETWEEN layers through both stages' Jacobians
      (surrogate_model_v3.multistep_propagation; P_0 = 0).
  'member_rollout' (LEGACY, scheme B) — every ensemble member rolled out as its
      own trajectory; kept only so runs trained before this change reproduce.

Differences from online_RL_ucpg_v2.env.TwoStageLatentLPBFEnv
----------------------------------------------------------------
1. (legacy 'member_rollout' mode) Member rollouts. The K members of the two-stage surrogate are rolled out as
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

from surrogate_model_v3.multistep_propagation import _members, _apply_A


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
        uncertainty_mode: str = "propagated",
    ) -> None:
        assert shaping in ("additive", "shrink"), shaping
        assert uncertainty_mode in ("propagated", "member_rollout"), uncertainty_mode
        self.uncertainty_mode = uncertainty_mode
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

        self._s: Optional[torch.Tensor] = None            # member_rollout: (K, D) member states; propagated: (1, D) mean
        self._F: Optional[torch.Tensor] = None            # propagated: (1, D, m) epistemic covariance factor
        self._layer = 0
        self._cool_time = 0.0

    # ------------------------------------------------------------------ API
    def reset(self, cool_time: Optional[float] = None) -> np.ndarray:
        s0 = torch.full((1, self.state_dim), self.initial_temp, dtype=torch.float32, device=self.device)
        s0 = (s0 - self.state_mean) / self.state_std
        if self.uncertainty_mode == "member_rollout":
            self._s = s0.expand(self.K, -1).clone()        # every member starts from the known IC
        else:
            self._s = s0.clone()                           # mean state
            self._F = torch.zeros(1, self.state_dim, 0, device=self.device)   # P_0 = 0 (known IC)
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

        if self.uncertainty_mode == "member_rollout":     # legacy scheme B
            heat, nxt = [], []
            for k in range(K):                              # member k on its own state
                s_k = self._s[k:k + 1]
                h_k = s_k + self.surrogate.heating_transitions[k](s_k, a_n, li)[0]
                n_k = h_k + self.surrogate.cooling_transitions[k](h_k, c_n, li)[0]
                heat.append(h_k); nxt.append(n_k)
            heat_raw = (torch.cat(heat, 0) * self.state_std + self.state_mean).cpu().numpy()   # (K, D) Kelvin
            next_raw = (torch.cat(nxt, 0) * self.state_std + self.state_mean).cpu().numpy()
            mu_h, sig_h = heat_raw.mean(0), heat_raw.std(0)
            mu_n, sig_n = next_raw.mean(0), next_raw.std(0)
            nxt = torch.cat(nxt, 0)
        else:                                               # scheme A + between-layer propagation
            m, F = self._s, self._F
            with torch.enable_grad():
                dh, Fe, _ = _members(self.surrogate.heating_transitions, m, a_n, li)
                F = torch.cat([_apply_A(self.surrogate.heating_transitions, m, a_n, li, F), Fe], -1)
                h = m + dh
                sig_h_n = (F ** 2).sum(-1).sqrt()
                dc, Fe, _ = _members(self.surrogate.cooling_transitions, h, c_n, li)
                F = torch.cat([_apply_A(self.surrogate.cooling_transitions, h, c_n, li, F), Fe], -1)
                nxt = h + dc
                sig_n_n = (F ** 2).sum(-1).sqrt()
            self._F = F.detach()
            nxt = nxt.detach()
            mu_h = (h.detach() * self.state_std + self.state_mean)[0].cpu().numpy()
            mu_n = (nxt * self.state_std + self.state_mean)[0].cpu().numpy()
            sig_h = (sig_h_n.detach() * self.state_std)[0].cpu().numpy()       # Kelvin
            sig_n = (sig_n_n.detach() * self.state_std)[0].cpu().numpy()
        r = self._rewards(mu_h, sig_h, mu_n, sig_n, t)

        self._s = nxt
        self._layer += 1
        done = self._layer >= self.n_layers

        info = dict(
            layer=self._layer, action_W=action_W, cool_time_s=self._cool_time,
            mean_heat_temp_K=float(mu_h.mean()), max_heat_temp_K=float(mu_h.max()),
            mean_next_temp_K=float(mu_n.mean()), max_next_temp_K=float(mu_n.max()),
            **r,
        )
        reward = r["reward_shaped"] if self.reward_mode == "shaped" else r["reward_plain"]
        return self._make_obs(), float(reward), done, info

    # --------------------------------------------------------- batched API
    # Steps B independent episodes in lock-step (same layer index, own action /
    # cool time each). Propagated mode only. The per-call cost of the Jacobian
    # products is dominated by fixed kernel-launch overhead, so batching B
    # trajectories costs about the same as one.
    def reset_batch(self, B: int, cool_times=None) -> np.ndarray:
        assert self.uncertainty_mode == "propagated", "batched API is for uncertainty_mode='propagated'"
        s0 = torch.full((B, self.state_dim), self.initial_temp, dtype=torch.float32, device=self.device)
        self._bs = (s0 - self.state_mean) / self.state_std
        self._bF = torch.zeros(B, self.state_dim, 0, device=self.device)
        self._bcool = (np.asarray(cool_times, dtype=float) if cool_times is not None
                       else np.random.uniform(self.cool_time_min, self.cool_time_max, B))
        self._blayer = 0
        return self._make_obs_batch()

    @torch.no_grad()
    def step_batch(self, actions_W):
        B = self._bs.shape[0]
        t = self._blayer
        a_n = torch.as_tensor((np.asarray(actions_W, dtype=float) - self.lp_mean) / self.lp_std,
                              dtype=torch.float32, device=self.device).reshape(B, 1)
        c_n = torch.as_tensor((self._bcool - self.cool_mean) / self.cool_std,
                              dtype=torch.float32, device=self.device).reshape(B, 1)
        li = torch.full((B,), t, dtype=torch.long, device=self.device)
        m, F = self._bs, self._bF
        with torch.enable_grad():
            dh, Fe, _ = _members(self.surrogate.heating_transitions, m, a_n, li)
            F = torch.cat([_apply_A(self.surrogate.heating_transitions, m, a_n, li, F, chunk=256), Fe], -1)
            h = m + dh
            sig_h = (F ** 2).sum(-1).sqrt()
            dc, Fe, _ = _members(self.surrogate.cooling_transitions, h, c_n, li)
            F = torch.cat([_apply_A(self.surrogate.cooling_transitions, h, c_n, li, F, chunk=256), Fe], -1)
            nxt = h + dc
            sig_n = (F ** 2).sum(-1).sqrt()
        self._bF, self._bs = F.detach(), nxt.detach()
        std, mean = self.state_std, self.state_mean
        mu_h = (h.detach() * std + mean).cpu().numpy(); mu_n = (self._bs * std + mean).cpu().numpy()
        sg_h = (sig_h.detach() * std).cpu().numpy(); sg_n = (sig_n.detach() * std).cpu().numpy()
        rows = [self._rewards(mu_h[b], sg_h[b], mu_n[b], sg_n[b], t) for b in range(B)]
        info = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        self._blayer += 1
        done = self._blayer >= self.n_layers
        reward = info["reward_shaped"] if self.reward_mode == "shaped" else info["reward_plain"]
        return self._make_obs_batch(), reward.astype(np.float32), done, info

    def _make_obs_batch(self) -> np.ndarray:
        z = self._bs.detach().cpu().numpy()
        B = z.shape[0]
        layer_token = np.full((B, 1), min(self._blayer, self.n_layers - 1) / max(self.n_layers - 1, 1))
        cool_token = ((self._bcool - self.cool_time_min) /
                      max(self.cool_time_max - self.cool_time_min, 1e-8)).reshape(B, 1)
        return np.concatenate([z, layer_token, cool_token], 1).astype(np.float32)

    # ------------------------------------------------------------- rewards
    def _rewards(self, mu_h, sig_h, mu_n, sig_n, t: int) -> dict:
        """mu_*: per-node mean temperature [K]; sig_*: per-node epistemic std [K], carried across layers."""
        m = self._masks[t]
        sel = (lambda f: f[m]) if m is not None else (lambda f: f)
        mu, sigma, mu_other = ((mu_n, sig_n, mu_h) if self.reward_field == "cool" else (mu_h, sig_h, mu_n))
        mu, sigma = sel(mu), sel(sigma)
        d = hinge_deviation(mu, self.T_l, self.T_h)
        r_plain = -float(d.mean() / self.W)

        out = dict(
            reward_plain=r_plain,
            reward_plain_other_field=-float(hinge_deviation(sel(mu_other), self.T_l, self.T_h).mean() / self.W),
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
        z = self._s.mean(0).detach().cpu().numpy()          # mean normalised state (both modes)
        layer_token = min(self._layer, self.n_layers - 1) / max(self.n_layers - 1, 1)
        cool_token = (self._cool_time - self.cool_time_min) / max(self.cool_time_max - self.cool_time_min, 1e-8)
        return np.concatenate([z, np.array([layer_token, cool_token], dtype=np.float32)]).astype(np.float32)


def parse_ranges(spec):
    """'100-150,350-400' -> [(100.0, 150.0), (350.0, 400.0)]."""
    out = []
    for part in str(spec).split(","):
        lo, hi = part.strip().split("-")
        out.append((float(lo), float(hi)))
    return out


def in_ranges(actions, ranges) -> np.ndarray:
    """Boolean mask: action inside ANY of the (lo, hi) ranges (inclusive)."""
    a = np.asarray(actions, dtype=float)
    m = np.zeros(a.shape, bool)
    for lo, hi in ranges:
        m |= (a >= lo) & (a <= hi)
    return m


def calibrate_tau(env: UncertaintyShapedLPBFEnv, id_ranges, k: float = 2.0,
                  n_episodes: int = 64, seed: int = 0) -> Tuple[float, float]:
    """tau = k x median ROI epistemic sigma [K] of MULTI-STEP member rollouts whose
    actions are drawn uniformly from the in-distribution laser-power set — the
    union of `id_ranges` [(lo, hi), ...], each range picked with probability
    proportional to its width (every layer, every ROI node pooled).
    Returns (tau, median_sigma)."""
    rng = np.random.default_rng(seed)
    widths = np.array([hi - lo for lo, hi in id_ranges], dtype=float)
    probs = widths / widths.sum()
    saved_mode, saved_tau = env.reward_mode, env.tau
    env.reward_mode, env.tau = "plain", None
    sig = []
    if env.uncertainty_mode == "propagated":                 # batched: all episodes in lock-step
        env.reset_batch(n_episodes)
        for _t in range(env.n_layers):
            idx = rng.choice(len(id_ranges), p=probs, size=n_episodes)
            acts = [rng.uniform(*id_ranges[i]) for i in idx]
            _o, _r, _d, info = env.step_batch(acts)
            sig.extend(info["sigma_epi_roi_K"].tolist())
    else:
        for _ in range(n_episodes):
            env.reset()
            for _t in range(env.n_layers):
                lo, hi = id_ranges[rng.choice(len(id_ranges), p=probs)]
                _o, _r, _d, info = env.step(float(rng.uniform(lo, hi)))
                sig.append(info["sigma_epi_roi_K"])
    env.reward_mode, env.tau = saved_mode, saved_tau
    med = float(np.median(sig))
    return k * med, med
