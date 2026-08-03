"""
baselines/common/data_utils.py
----------------------------------
Pure-dataset (NO surrogate model) utilities shared by the offline/traditional
baselines: reconstructing (s_t, a_t, r_t, s_{t+1}) transitions directly from
the pickled TrajectoryV2 lists, and square-ROI mesh helpers for computing
scalar mean-temperature features (used by the proportional and
Kalman/particle-filter controllers to fit their parameters from data,
without ever calling the neural surrogate).

State-chaining convention (mirrors
surrogate_model_latent_uncertainty_v2.dataset_v2.build_normalizers exactly,
duplicated here — not imported — since that module also builds latent
normalisation statistics we don't want baselines/ implicitly depending on):
  s_0 = all-initial_temp field
  s_{t+1} = step[t].u_final   (post-cooling field)
  reward_t = step[t].reward   (already computed by the simulator/extraction
             pipeline as -meanDeviation of u_heat_final — the corrected,
             end-of-heating reward)
"""

import os
from typing import List, Optional

import numpy as np
from scipy.io import loadmat

from surrogate_model_latent_uncertainty_v2.dataset_v2 import TrajectoryV2


# =============================================================================
# Offline transition reconstruction (no surrogate)
# =============================================================================

def build_offline_transitions(trajectories: List[TrajectoryV2], initial_temp: float = 300.0) -> dict:
    """
    Flatten every trajectory into single-step (s_t, a_t, r_t, s_{t+1}) offline
    RL transitions, reconstructed purely from the raw pickled dataset.

    Returns a dict of numpy arrays, all first-dim aligned (N = n_traj * n_layers):
      s          : (N, D) float32   — state BEFORE this layer's laser pass
      a          : (N,)   float32   — laser power [W] actually applied
      r          : (N,)   float32   — reward (already -meanDeviation of u_heat_final)
      s2         : (N, D) float32   — state BEFORE the NEXT layer (= this layer's u_final)
      layer      : (N,)   int64     — 0-indexed layer within the trajectory
      done       : (N,)   bool      — True on the last layer of its trajectory
      cool_time  : (N,)   float32   — this trajectory's (fixed) cool time [s]
    """
    sample = np.asarray(trajectories[0][0].u_final, dtype=np.float32).reshape(-1)
    state_dim = sample.shape[0]
    init_s    = np.full(state_dim, initial_temp, dtype=np.float32)

    s_l, a_l, r_l, s2_l, layer_l, done_l, cool_l = [], [], [], [], [], [], []

    for traj in trajectories:
        prev = init_s.copy()
        T = len(traj)
        for t, step in enumerate(traj):
            nxt = np.asarray(step.u_final, dtype=np.float32).reshape(-1)
            s_l.append(prev)
            a_l.append(step.lp_action)
            r_l.append(step.reward)
            s2_l.append(nxt)
            layer_l.append(t)
            done_l.append(t == T - 1)
            cool_l.append(step.cool_time)
            prev = nxt

    return dict(
        s=np.stack(s_l, axis=0).astype(np.float32),
        a=np.array(a_l, dtype=np.float32),
        r=np.array(r_l, dtype=np.float32),
        s2=np.stack(s2_l, axis=0).astype(np.float32),
        layer=np.array(layer_l, dtype=np.int64),
        done=np.array(done_l, dtype=bool),
        cool_time=np.array(cool_l, dtype=np.float32),
    )


def build_offline_heat_transitions(trajectories: List[TrajectoryV2], initial_temp: float = 300.0) -> dict:
    """
    Same as build_offline_transitions, but ALSO includes the end-of-heating
    field u_heat_final for each step — needed by the proportional and
    Kalman/particle-filter baselines, which regress against the mean
    end-of-heating temperature rather than a full Bellman target.

    Adds one extra key to the returned dict vs. build_offline_transitions:
      heat : (N, D) float32 — end-of-heating field (reward input)
    """
    base = build_offline_transitions(trajectories, initial_temp)
    heat_l = []
    for traj in trajectories:
        for step in traj:
            heat_l.append(np.asarray(step.u_heat_final, dtype=np.float32).reshape(-1))
    base["heat"] = np.stack(heat_l, axis=0).astype(np.float32)
    return base


# =============================================================================
# Square-ROI mesh helpers (standalone re-implementation of
# online_RL_ucpg_v2.env.TwoStageLatentLPBFEnv's private _build_mask, so that
# baselines/ never needs to instantiate a full environment just to compute a
# scalar ROI-mean feature from a raw dataset field)
# =============================================================================

def load_mesh_nodes(mesh_path: str) -> Optional[np.ndarray]:
    """Returns (N, 2) node X/Y coordinates, or None if mesh_path doesn't exist."""
    if not os.path.exists(mesh_path):
        return None
    data  = loadmat(mesh_path)
    nodes = data["nodes"]
    return (nodes.T if nodes.shape[0] == 2 else nodes).astype(np.float32)


def square_roi_mask(nodes_xy: np.ndarray, width: float, height: float, frac: float) -> np.ndarray:
    """Boolean mask of nodes inside the centred square scan region of side
    fraction `frac` (see simulateHeatingCooling_v2.m's squareSideFraction)."""
    sq_side = min(width, height) * frac
    half    = sq_side / 2.0
    cx, cy  = width / 2.0, height / 2.0
    xs, ys  = nodes_xy[:, 0], nodes_xy[:, 1]
    return (xs >= cx - half) & (xs <= cx + half) & (ys >= cy - half) & (ys <= cy + half)


def roi_masks_per_layer(
    nodes_xy: np.ndarray, width: float, height: float,
    sq_frac_start: float, sq_frac_end: float, n_layers: int,
) -> List[np.ndarray]:
    """One boolean mask per layer, matching the squareSideFraction schedule
    linspace(sq_frac_start, sq_frac_end, n_layers)."""
    fracs = np.linspace(sq_frac_start, sq_frac_end, n_layers)
    return [square_roi_mask(nodes_xy, width, height, f) for f in fracs]


def roi_mean(field: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """Mean temperature inside the ROI mask (or over all nodes if mask is None)."""
    return float(field[mask].mean()) if mask is not None else float(field.mean())


# =============================================================================
# Simple least-squares fit helper (no gradient descent, no RL — the whole
# point of the "traditional" baselines is that they are NOT learned)
# =============================================================================

def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Ordinary least squares y ~= slope * x + intercept.
    Returns (slope, intercept, residual_std) — residual_std is the fitted
    process-noise scale used to seed the Kalman/particle filter's Q.
    """
    slope, intercept = np.polyfit(x, y, deg=1)
    resid = y - (slope * x + intercept)
    return float(slope), float(intercept), float(resid.std())
