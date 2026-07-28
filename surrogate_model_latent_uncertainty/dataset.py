"""
surrogate_model_latent_uncertainty/dataset.py
------------------------------------------------
Data loading for the latent-space LPBF surrogate with a bootstrap-trained
Gaussian ensemble. Extends surrogate_model_latent/dataset.py by:

  1. make_bootstrap_masks — precomputes, for each of the K ensemble members,
     an independent bootstrap resample of the training set, expressed as a
     per-sample integer multiplicity mask rather than a physically reordered
     copy of the data. A mask lets every member's forward pass stay batched
     together (same (s, a, s') rows, same shared decoder call across K
     members) while reproducing the statistics of drawing N samples with
     replacement per member: samples the resample skipped get weight 0,
     samples drawn twice get weight 2, etc. This is the standard "online
     bootstrap" / bagging-by-reweighting trick (Oza & Russell, 2001) used in
     bootstrapped ensembles (e.g. Osband et al.'s Bootstrapped DQN).

  2. LatentSurrogateDataset  — bootstraps at the transition level (each row
     is one (s, a, s') sample, the natural resampling unit for single-step
     training).
     LatentTrajectoryDataset — bootstraps at the trajectory level (each row
     is one full trajectory, the natural resampling unit for rollout
     training, since the transitions inside one trajectory are not
     independent draws from the population).

Everything else — build-layer index tracking, physics-aware ROI weights — is
unchanged from surrogate_model_latent/dataset.py.

MDP convention (unchanged)
---------------------------
  state      = temperature field BEFORE the laser pass
               (layer 0 → all-300 K initial field)
  action     = laser power applied during this layer [W]
  next_state = temperature field AFTER the laser pass
"""

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

Trajectory = List[List]


# =============================================================================
# I/O helpers  (identical to surrogate_model_latent)
# =============================================================================

def load_trajectories(pkl_path: str) -> List[Trajectory]:
    """Load a pickled trajectory dataset produced by dataset.py."""
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"[dataset] Loaded {len(dataset)} trajectories "
          f"× {len(dataset[0])} layers from {os.path.basename(pkl_path)}")
    return dataset


def split_trajectories(
    trajectories:  List[Trajectory],
    val_fraction:  float = 0.10,
    test_fraction: float = 0.10,
    seed:          int   = 42,
) -> Tuple[List[Trajectory], List[Trajectory], List[Trajectory]]:
    """
    Trajectory-level split (no data leakage between sets).
    Returns (train_trajs, val_trajs, test_trajs).
    """
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(trajectories))
    n    = len(trajectories)
    n_te = max(1, int(n * test_fraction))
    n_va = max(1, int(n * val_fraction))

    test_idx  = idx[:n_te]
    val_idx   = idx[n_te : n_te + n_va]
    train_idx = idx[n_te + n_va :]

    train = [trajectories[i] for i in train_idx]
    val   = [trajectories[i] for i in val_idx]
    test  = [trajectories[i] for i in test_idx]

    print(f"[dataset] Split → train {len(train)} | val {len(val)} | test {len(test)}")
    return train, val, test


def build_normalizers(
    trajectories: List[Trajectory],
    initial_temp: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Compute z-score stats for states and actions from training trajectories only.

    Returns
    -------
    state_mean  : (state_dim,) float32
    state_std   : (state_dim,) float32 (clamped ≥ 1e-6)
    action_mean : float
    action_std  : float
    """
    all_states, all_actions = [], []

    sample_u  = np.asarray(trajectories[0][0][0], dtype=np.float32).reshape(-1)
    state_dim = sample_u.shape[0]
    init_s    = np.full(state_dim, initial_temp, dtype=np.float32)

    for traj in trajectories:
        prev = init_s.copy()
        for step in traj:
            u, lp, _ = step
            nxt = np.asarray(u, dtype=np.float32).reshape(-1)
            all_states.extend([prev, nxt])
            all_actions.append(float(lp))
            prev = nxt

    states_arr  = np.stack(all_states, axis=0)
    actions_arr = np.array(all_actions, dtype=np.float32)

    state_mean  = torch.tensor(states_arr.mean(axis=0), dtype=torch.float32)
    state_std   = torch.tensor(states_arr.std(axis=0),  dtype=torch.float32).clamp_min(1e-6)
    action_mean = float(actions_arr.mean())
    action_std  = float(actions_arr.std()) if actions_arr.std() > 1e-6 else 1.0

    print(f"[dataset] State  mean ∈ [{state_mean.min():.1f}, {state_mean.max():.1f}], "
          f"std  ∈ [{state_std.min():.4f}, {state_std.max():.1f}]")
    print(f"[dataset] Action mean = {action_mean:.2f} W, std = {action_std:.2f} W")
    return state_mean, state_std, action_mean, action_std


# =============================================================================
# Physics-aware ROI weight helpers  (identical to surrogate_model_latent)
# =============================================================================

def compute_roi_weights_table(
    nodes:            np.ndarray,   # (2, N_nodes)  X, Y coordinates
    n_layers:         int   = 12,
    initial_fraction: float = 0.4,
    final_fraction:   float = 0.5,
    roi_boost:        float = 5.0,
    edge_sigma_frac:  float = 0.05,
) -> np.ndarray:
    """
    Precompute per-node weight vectors for each LPBF build layer.

    The laser scans a SQUARE region centred at the domain centre (width/2, height/2).
    The square GROWS from layer 0 to layer n_layers-1:

        side[l] = min(width, height) * fractions[l]
        fractions = linspace(initial_fraction, final_fraction, n_layers)

    Nodes inside the square receive roi_boost × higher weight. The square
    boundary is softened using a sigmoid with width edge_sigma_frac ×
    min(width, height).

    Returns
    -------
    table : (n_layers, N_nodes) float32, each row normalised to mean = 1
    """
    x = nodes[0]
    y = nodes[1]

    width   = x.max() - x.min()
    height  = y.max() - y.min()
    cx      = x.min() + width  / 2.0
    cy      = y.min() + height / 2.0
    min_dim = min(width, height)
    sigma   = edge_sigma_frac * min_dim

    fractions = np.linspace(initial_fraction, final_fraction, n_layers)

    table = np.zeros((n_layers, len(x)), dtype=np.float32)
    for l, frac in enumerate(fractions):
        half_side = min_dim * frac / 2.0

        box_dist = np.maximum(np.abs(x - cx) - half_side,
                              np.abs(y - cy) - half_side)
        smooth_inside = 1.0 / (1.0 + np.exp(box_dist / sigma))

        w        = 1.0 + (roi_boost - 1.0) * smooth_inside
        table[l] = (w / w.mean()).astype(np.float32)

    print(f"[dataset] ROI weight table: {table.shape}, "
          f"range [{table.min():.3f}, {table.max():.3f}]  "
          f"(square side {initial_fraction:.2f}→{final_fraction:.2f} × min_dim)")
    return table


def uniform_weights_table(n_layers: int, state_dim: int) -> np.ndarray:
    """Fallback: uniform per-node weights when mesh is unavailable."""
    return np.ones((n_layers, state_dim), dtype=np.float32)


# =============================================================================
# Bootstrap resampling
# =============================================================================

def make_bootstrap_masks(
    n_samples:  int,
    n_ensemble: int,
    seed:       int = 0,
) -> np.ndarray:
    """
    Precompute per-member bootstrap multiplicity masks.

    For each of the K ensemble members, draws n_samples indices with
    replacement from {0, ..., n_samples-1} and counts how many times each
    original sample was drawn. The resulting integer multiplicity acts as a
    per-sample loss weight: multiplicity 0 excludes that sample from the
    member's effective training set, multiplicity 2+ means it's seen more
    than once — reproducing the statistics of an actual bootstrap resample
    while keeping every member's rows aligned in one shared mini-batch (so
    the batched K-member decoder call in train.py still works unchanged).

    Parameters
    ----------
    n_samples  : size of the dataset being resampled (transitions or trajectories)
    n_ensemble : number of ensemble members K
    seed       : RNG seed (independent per-member resamples derived from it)

    Returns
    -------
    masks : (n_samples, n_ensemble) float32, each column has mean ≈ 1.0
    """
    rng   = np.random.default_rng(seed)
    masks = np.zeros((n_samples, n_ensemble), dtype=np.float32)
    for k in range(n_ensemble):
        draws  = rng.integers(0, n_samples, size=n_samples)
        counts = np.bincount(draws, minlength=n_samples)
        masks[:, k] = counts.astype(np.float32)

    frac_unseen = (masks == 0).mean()
    print(f"[dataset] Bootstrap masks: {masks.shape}, "
          f"~{frac_unseen*100:.1f}% of (sample, member) pairs unseen "
          f"(expected ≈ {np.exp(-1)*100:.1f}% for large N)")
    return masks


# =============================================================================
# Single-step dataset with layer indices  (+ per-transition bootstrap masks)
# =============================================================================

class LatentSurrogateDataset(Dataset):
    """
    Flat (state, action, next_state, layer_idx, bootstrap_mask) transitions.

    layer_idx is 0-indexed: 0 = first laser pass (layer 1), n-1 = last pass.
    bootstrap_mask is (n_ensemble,): member k's per-sample loss weight, drawn
    once at construction time (see make_bootstrap_masks). All state / action
    tensors are z-score normalised.
    """

    def __init__(
        self,
        trajectories:    List[Trajectory],
        state_mean:      torch.Tensor,
        state_std:       torch.Tensor,
        action_mean:     float,
        action_std:      float,
        initial_temp:    float = 300.0,
        n_ensemble:      int   = 5,
        bootstrap_seed:  int   = 0,
    ):
        super().__init__()
        state_mean = state_mean.cpu()
        state_std  = state_std.cpu()

        sample_u  = np.asarray(trajectories[0][0][0], dtype=np.float32).reshape(-1)
        state_dim = sample_u.shape[0]
        init_s    = np.full(state_dim, initial_temp, dtype=np.float32)

        states, actions, next_states, layer_indices = [], [], [], []

        for traj in trajectories:
            prev = init_s.copy()
            for layer_idx, step in enumerate(traj):
                u, lp, _ = step
                nxt = np.asarray(u, dtype=np.float32).reshape(-1)
                states.append(prev.copy())
                actions.append(float(lp))
                next_states.append(nxt)
                layer_indices.append(layer_idx)
                prev = nxt

        S  = torch.tensor(np.stack(states),     dtype=torch.float32)
        A  = torch.tensor(actions,               dtype=torch.float32).unsqueeze(-1)
        S2 = torch.tensor(np.stack(next_states), dtype=torch.float32)

        self.states        = (S  - state_mean) / state_std
        self.actions       = (A  - action_mean) / action_std
        self.next_states   = (S2 - state_mean) / state_std
        self.layer_indices = torch.tensor(layer_indices, dtype=torch.long)

        self.n_ensemble = n_ensemble
        masks = make_bootstrap_masks(len(self.states), n_ensemble, seed=bootstrap_seed)
        self.bootstrap_masks = torch.tensor(masks, dtype=torch.float32)  # (N, K)

        print(f"[LatentSurrogateDataset] {len(self.states):,} transitions, "
              f"state_dim={state_dim}, layers 0–{max(layer_indices)}, "
              f"bootstrap K={n_ensemble} (transition-level resampling)")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx],
            self.layer_indices[idx],
            self.bootstrap_masks[idx],
        )


# =============================================================================
# Full-trajectory dataset  (for rollout loss; + per-trajectory bootstrap masks)
# =============================================================================

class LatentTrajectoryDataset(Dataset):
    """
    Each sample is one full trajectory. Layer indices are implicit (step t →
    layer t, 0-indexed). bootstrap_mask is (n_ensemble,): member k's
    per-trajectory loss weight, drawn once at construction time — trajectories
    are the natural resampling unit here since the transitions within one
    trajectory are temporally correlated, not independent population draws.

    Returns
    -------
    states  : (T+1, state_dim) normalised  — s_0 … s_T
    actions : (T,   1)         normalised  — a_0 … a_{T-1}
    bootstrap_mask : (n_ensemble,)
    """

    def __init__(
        self,
        trajectories:    List[Trajectory],
        state_mean:      torch.Tensor,
        state_std:       torch.Tensor,
        action_mean:     float,
        action_std:      float,
        initial_temp:    float = 300.0,
        n_ensemble:      int   = 5,
        bootstrap_seed:  int   = 0,
    ):
        super().__init__()
        state_mean = state_mean.cpu()
        state_std  = state_std.cpu()
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        sample_u  = np.asarray(trajectories[0][0][0], dtype=np.float32).reshape(-1)
        state_dim = sample_u.shape[0]
        init_raw  = np.full(state_dim, initial_temp, dtype=np.float32)

        T_last = 0
        for traj in trajectories:
            T      = len(traj)
            T_last = T
            s_list = [init_raw.copy()]
            a_list = []
            for step in traj:
                u, lp, _ = step
                s_list.append(np.asarray(u, dtype=np.float32).reshape(-1))
                a_list.append(float(lp))

            S_raw = torch.tensor(np.stack(s_list), dtype=torch.float32)   # (T+1, D)
            A_raw = torch.tensor(a_list,            dtype=torch.float32).unsqueeze(-1)  # (T, 1)
            self.samples.append(
                ((S_raw - state_mean) / state_std,
                 (A_raw - action_mean) / action_std)
            )

        self.n_ensemble = n_ensemble
        masks = make_bootstrap_masks(len(self.samples), n_ensemble, seed=bootstrap_seed)
        self.bootstrap_masks = torch.tensor(masks, dtype=torch.float32)  # (N_traj, K)

        print(f"[LatentTrajectoryDataset] {len(self.samples):,} trajectories, "
              f"T={T_last}, state_dim={state_dim}, "
              f"bootstrap K={n_ensemble} (trajectory-level resampling)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s, a = self.samples[idx]
        return s, a, self.bootstrap_masks[idx]
