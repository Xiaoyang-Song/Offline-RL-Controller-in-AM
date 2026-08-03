"""
baselines/kalman_particle/filters.py
----------------------------------------
Baseline 4: Kalman-filter and particle-filter controllers.

Setup
-----
Let x_t = mean ROI temperature of s_t (the true pre-heat field mean this
layer, over the same square scan region used for the reward). A real
deployed controller would not read x_t off a full 1053-node field for free
— it would read it off SOME sensor (e.g. a pyrometer / thermal camera) with
measurement noise. We synthesise that: at evaluation time, the controller
only observes

    z_t = x_t + v_t,   v_t ~ N(0, R)

and must filter it. The (deliberately simple) process model is a random
walk in x_t with process noise Q, fit from the offline dataset as the
empirical variance of consecutive per-trajectory ROI-mean differences — NOT
the true nonlinear PDE, which is exactly why this baseline is expected to
struggle (see package docstring / README).

Control law (certainty equivalence): fit, once, via ordinary least squares
on the offline dataset,

    mean_ROI(u_heat_t) ~= alpha * x_t + beta * a_t + gamma_c

then at each step invert it around the filtered estimate mu_t (Kalman
posterior mean, or the particle filter's weighted particle mean) to aim the
predicted end-of-heating ROI mean at T_mid = (T_l+T_h)/2:

    a_t = (T_mid - alpha * mu_t - gamma_c) / beta

Usage (fit alpha/beta/gamma_c/Q once, from the offline dataset)
-----------------------------------------------------------------
    python -m baselines.kalman_particle.filters \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --out baselines/kalman_particle/fitted.pt

R (sensor noise) and n_particles are evaluation-time choices, not fit
parameters — see load_kalman_controller / load_particle_controller.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.dataset_v2 import load_trajectories, split_trajectories
from baselines.common.data_utils import (
    build_offline_heat_transitions, load_mesh_nodes, roi_masks_per_layer, roi_mean,
)


# =============================================================================
# Fitting (offline dataset only — no surrogate)
# =============================================================================

def fit_process_and_control_model(
    trajectories, mesh_path: str, width: float, height: float,
    sq_frac_start: float, sq_frac_end: float, n_layers: int = 12, initial_temp: float = 300.0,
) -> dict:
    data = build_offline_heat_transitions(trajectories, initial_temp=initial_temp)
    nodes_xy = load_mesh_nodes(mesh_path)
    masks = roi_masks_per_layer(nodes_xy, width, height, sq_frac_start, sq_frac_end, n_layers)

    x  = np.array([roi_mean(data["s"][i],    masks[data["layer"][i]]) for i in range(data["s"].shape[0])])
    y  = np.array([roi_mean(data["heat"][i], masks[data["layer"][i]]) for i in range(data["heat"].shape[0])])
    a  = data["a"]

    # control model: y ~= alpha*x + beta*a + gamma_c   (multivariate OLS)
    design = np.stack([x, a, np.ones_like(x)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    alpha, beta, gamma_c = (float(c) for c in coeffs)

    # process noise Q: empirical variance of consecutive per-trajectory ROI-mean
    # differences (x_{t+1} - x_t) — a random-walk process model, deliberately
    # not the true nonlinear PDE (see module docstring).
    diffs = []
    for traj in trajectories:
        prev_x = initial_temp
        for t, step in enumerate(traj):
            cur_x = roi_mean(np.asarray(step.u_final, dtype=np.float32).reshape(-1), masks[t])
            diffs.append(cur_x - prev_x)
            prev_x = cur_x
    Q = float(np.var(diffs))

    return dict(alpha=alpha, beta=beta, gamma_c=gamma_c, Q=Q,
               mesh_path=mesh_path, width=width, height=height,
               sq_frac_start=sq_frac_start, sq_frac_end=sq_frac_end, n_layers=n_layers)


def save_fitted(path: str, fit: dict, T_mid: float, initial_temp: float) -> None:
    d = dict(fit); d["T_mid"] = T_mid; d["initial_temp"] = initial_temp
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(d, path)


def _load_fit(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


# =============================================================================
# Kalman filter controller
# =============================================================================

class KalmanController:
    def __init__(self, alpha, beta, gamma_c, T_mid, Q, R, masks, initial_temp=300.0, seed=0):
        self.alpha, self.beta, self.gamma_c, self.T_mid = alpha, beta, gamma_c, T_mid
        self.Q, self.R = Q, R
        self.masks = masks
        self.initial_temp = initial_temp
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.mu  = self.initial_temp
        self.var = 1e-3   # the initial 300 K field is known essentially exactly

    def act(self, ctx) -> float:
        mask   = self.masks[min(ctx.layer, len(self.masks) - 1)]
        x_true = roi_mean(ctx.raw_state, mask)
        z = x_true + self.rng.normal(0.0, np.sqrt(self.R))

        mu_pred, var_pred = self.mu, self.var + self.Q          # predict (random walk)
        K = var_pred / (var_pred + self.R)                       # update
        self.mu  = mu_pred + K * (z - mu_pred)
        self.var = (1.0 - K) * var_pred

        return (self.T_mid - self.alpha * self.mu - self.gamma_c) / self.beta


def load_kalman_controller(path: str, R: float = 2500.0, seed: int = 0) -> KalmanController:
    d = _load_fit(path)
    nodes_xy = load_mesh_nodes(d["mesh_path"])
    masks = roi_masks_per_layer(nodes_xy, d["width"], d["height"], d["sq_frac_start"], d["sq_frac_end"], d["n_layers"])
    return KalmanController(d["alpha"], d["beta"], d["gamma_c"], d["T_mid"], d["Q"], R,
                            masks, d["initial_temp"], seed=seed)


# =============================================================================
# Particle filter controller (bootstrap filter, same generative model)
# =============================================================================

class ParticleController:
    def __init__(self, alpha, beta, gamma_c, T_mid, Q, R, masks, initial_temp=300.0,
                n_particles=200, seed=0):
        self.alpha, self.beta, self.gamma_c, self.T_mid = alpha, beta, gamma_c, T_mid
        self.Q, self.R = Q, R
        self.masks = masks
        self.initial_temp = initial_temp
        self.n_particles = n_particles
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.particles = np.full(self.n_particles, self.initial_temp, dtype=np.float64)
        self.weights   = np.full(self.n_particles, 1.0 / self.n_particles)

    @staticmethod
    def _systematic_resample(weights: np.ndarray, rng) -> np.ndarray:
        n = len(weights)
        positions = (np.arange(n) + rng.uniform()) / n
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0
        return np.searchsorted(cumsum, positions)

    def act(self, ctx) -> float:
        mask   = self.masks[min(ctx.layer, len(self.masks) - 1)]
        x_true = roi_mean(ctx.raw_state, mask)
        z = x_true + self.rng.normal(0.0, np.sqrt(self.R))

        self.particles = self.particles + self.rng.normal(0.0, np.sqrt(self.Q), size=self.n_particles)
        likelihood = np.exp(-0.5 * (z - self.particles) ** 2 / self.R)
        self.weights = self.weights * likelihood
        wsum = self.weights.sum()
        self.weights = self.weights / wsum if wsum > 1e-300 else np.full(self.n_particles, 1.0 / self.n_particles)

        idx = self._systematic_resample(self.weights, self.rng)
        self.particles = self.particles[idx]
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

        mu_est = float(self.particles.mean())
        return (self.T_mid - self.alpha * mu_est - self.gamma_c) / self.beta


def load_particle_controller(path: str, R: float = 2500.0, n_particles: int = 200, seed: int = 0) -> ParticleController:
    d = _load_fit(path)
    nodes_xy = load_mesh_nodes(d["mesh_path"])
    masks = roi_masks_per_layer(nodes_xy, d["width"], d["height"], d["sq_frac_start"], d["sq_frac_end"], d["n_layers"])
    return ParticleController(d["alpha"], d["beta"], d["gamma_c"], d["T_mid"], d["Q"], R,
                              masks, d["initial_temp"], n_particles=n_particles, seed=seed)


# =============================================================================
# CLI — fit once from the offline dataset
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline 4: fit the Kalman/particle-filter control model from data.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--T_l", type=float, default=2000.0)
    p.add_argument("--T_h", type=float, default=2800.0)
    p.add_argument("--mesh_path", type=str, default="surrogate_model/mesh.mat")
    p.add_argument("--width",  type=float, default=12.0)
    p.add_argument("--height", type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end",   type=float, default=0.5)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--out", type=str, default="baselines/kalman_particle/fitted.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_trajs = load_trajectories(args.data_path)
    train_trajs, _val, _test = split_trajectories(
        all_trajs, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
    )
    fit = fit_process_and_control_model(
        train_trajs, args.mesh_path, args.width, args.height,
        args.sq_frac_start, args.sq_frac_end, args.n_layers, args.initial_temp,
    )
    T_mid = (args.T_l + args.T_h) / 2.0
    print(f"[kalman_particle] Fit on {len(train_trajs)} trajectories: "
          f"alpha={fit['alpha']:.4f}  beta={fit['beta']:.4f}  gamma_c={fit['gamma_c']:.2f}  "
          f"Q={fit['Q']:.2f}  T_mid={T_mid:.1f}")
    save_fitted(args.out, fit, T_mid, args.initial_temp)
    print(f"[kalman_particle] Saved → {args.out}")


if __name__ == "__main__":
    main()
