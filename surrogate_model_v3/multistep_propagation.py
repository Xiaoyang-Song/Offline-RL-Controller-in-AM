"""
surrogate_model_v3/multistep_propagation.py
--------------------------------------------------
BETWEEN-LAYER uncertainty propagation for autoregressive rollouts — a standalone
add-on; model.py is not modified and every existing (one-step) result is
unchanged.

Rollout scheme ("scheme A", moment matching): the state MEAN is chained through
the ensemble mean (exactly as TwoStageSurrogate.rollout does), and a state
covariance P_t is carried along and propagated through BOTH stages' local
Jacobians:

    heating : P_heat = A_h P_t A_h^T + Sigma_heat,   A_h = I + d mu_heat / d s
    cooling : P_{t+1} = A_c P_heat A_c^T + Sigma_cool, A_c = I + d mu_cool / d h
    P_0 = 0 (known initial state)

Representation (D = 1053 nodes, so P is never formed densely):
  * epistemic part — exact low-rank factor F (P_epi = F F^T). Each stage appends
    its K member deviations (mu_k - mu_bar)/sqrt(K) as new columns; existing
    columns are pushed through A = I + J with batched forward-mode JVPs
    (torch.func.jvp). After T layers F has 2 K T columns.
  * aleatoric part — diagonal d. Propagating diag through A uses the unbiased
    probe estimator diag(A D A^T) = E_v[(A (sqrt(d) * v))^2], v Rademacher
    (n_probes JVPs); the off-diagonal aleatoric covariance this creates is not
    carried to the next layer (documented approximation).
  * Each stage's new uncertainty is treated as independent of the propagated
    part (standard EKF assumption).

reset_each_layer=True gives the ONE-STEP version (P reset to 0 before every
layer) computed with the same machinery, for like-for-like comparison.

All variances are in normalised units (multiply by state_std^2 for K^2).
"""

from typing import Dict

import torch


def _members(transitions, x, c, layer):
    """-> mu_bar (N,D), epistemic factor (N,D,K), mean aleatoric variance (N,D)."""
    mus, logs = [], []
    for t in transitions:
        mu, log_sigma, _u = t(x, c, layer)
        mus.append(mu); logs.append(log_sigma)
    mus = torch.stack(mus)                                   # (K,N,D)
    K = mus.shape[0]
    mu_bar = mus.mean(0)
    factor = ((mus - mu_bar) / K ** 0.5).permute(1, 2, 0)    # (N,D,K)
    ale = (2.0 * torch.stack(logs)).exp().mean(0)            # (N,D)
    return mu_bar, factor, ale


def _apply_A(transitions, x, c, layer, cols, chunk=64):
    """(I + J) @ cols for every column; J = Jacobian of the ensemble-mean Delta of
    `transitions` w.r.t. its input state x, at (x, c, layer). cols: (N,D,M)."""
    N, D, M = cols.shape
    if M == 0:
        return cols
    out = []
    for m0 in range(0, M, chunk):                            # bound memory: chunk columns
        V = cols[..., m0:m0 + chunk]
        m = V.shape[-1]
        X = x.repeat(m, 1); C = c.repeat(m, 1); L = layer.repeat(m)
        Vf = V.permute(2, 0, 1).reshape(m * N, D)
        fn = lambda z: torch.stack([t(z, C, L)[0] for t in transitions]).mean(0)
        _, jv = torch.func.jvp(fn, (X,), (Vf,))
        out.append(V + jv.reshape(m, N, D).permute(1, 2, 0))
    return torch.cat(out, -1)


def _propagate_diag(transitions, x, c, layer, d, n_probes):
    """Unbiased estimate of diag((I+J) diag(d) (I+J)^T) with Rademacher probes."""
    if n_probes <= 0 or not torch.any(d > 0):
        return d
    v = torch.randint(0, 2, (*d.shape, n_probes), device=d.device, dtype=d.dtype) * 2 - 1
    cols = d.clamp_min(0).sqrt().unsqueeze(-1) * v
    return (_apply_A(transitions, x, c, layer, cols) ** 2).mean(-1)


@torch.no_grad()
def rollout_propagated(model, s0, actions, cool_times, reset_each_layer: bool = False,
                       track_aleatoric: bool = True, n_probes: int = 16) -> Dict[str, torch.Tensor]:
    """
    s0 (N,D) normalised; actions, cool_times (N,T,1) normalised.
    Returns per-layer tensors (N,T,D): heat_mean, next_mean, and per-node variances
    heat_epi, heat_ale, next_epi, next_ale (normalised units).
    """
    N, T, _ = actions.shape
    D = s0.shape[-1]
    dev = s0.device
    m = s0
    F = torch.zeros(N, D, 0, device=dev)
    d = torch.zeros(N, D, device=dev)
    out = {k: [] for k in ("heat_mean", "next_mean", "heat_epi", "heat_ale", "next_epi", "next_ale")}
    for t in range(T):
        a, c = actions[:, t], cool_times[:, t]
        layer = torch.full((N,), t, dtype=torch.long, device=dev)
        if reset_each_layer:
            F = torch.zeros(N, D, 0, device=dev); d = torch.zeros(N, D, device=dev)
        with torch.enable_grad():                           # forward-mode JVPs need autograd enabled
            # ---- heating: s -> h
            mu_h, Fe, ale_h = _members(model.heating_transitions, m, a, layer)
            F = _apply_A(model.heating_transitions, m, a, layer, F)
            if track_aleatoric:
                d = _propagate_diag(model.heating_transitions, m, a, layer, d, n_probes)
            F = torch.cat([F, Fe], -1)
            d = d + (ale_h if track_aleatoric else 0.0)
            h = m + mu_h
            out["heat_mean"].append(h); out["heat_epi"].append((F ** 2).sum(-1)); out["heat_ale"].append(d.clone())
            # ---- cooling: h -> s_{t+1}
            mu_c, Fe, ale_c = _members(model.cooling_transitions, h, c, layer)
            F = _apply_A(model.cooling_transitions, h, c, layer, F)
            if track_aleatoric:
                d = _propagate_diag(model.cooling_transitions, h, c, layer, d, n_probes)
            F = torch.cat([F, Fe], -1)
            d = d + (ale_c if track_aleatoric else 0.0)
            m = h + mu_c
            out["next_mean"].append(m); out["next_epi"].append((F ** 2).sum(-1)); out["next_ale"].append(d.clone())
        F = F.detach(); d = d.detach(); m = m.detach()
    return {k: torch.stack([x.detach() for x in v], 1) for k, v in out.items()}
