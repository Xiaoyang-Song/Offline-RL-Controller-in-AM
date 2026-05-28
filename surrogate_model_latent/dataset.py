"""
surrogate_model_latent/dataset.py
----------------------------------
Data loading for the latent-space LPBF surrogate.

Extends surrogate_model/dataset.py by:
  1. Tracking the build-layer index (0-indexed, 0…11) for each transition,
     enabling physics-aware loss weighting per layer.
  2. Providing compute_roi_weights_table to precompute per-node Gaussian
     weights centred on each layer's heat-source region.

Two Dataset classes
-------------------
  LatentSurrogateDataset   – flat (s, a, s', layer_idx) transitions → single-step training
  LatentTrajectoryDataset  – full trajectories (states, actions)     → rollout training
     Layer indices in trajectory mode are implicit: step t → layer t.

MDP convention (unchanged from surrogate_model)
------------------------------------------------
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
# I/O helpers  (identical to surrogate_model)
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
# Physics-aware ROI weight helpers
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

    This mirrors the MATLAB simulation:
        squareSide = min(width,height) * paramsStruct.squareSideFraction
        fractions  = linspace(0.4, 0.5, 12)   % grows each layer

    Nodes inside the square receive roi_boost × higher weight.
    The square boundary is softened using a sigmoid with width
    edge_sigma_frac × min(width, height).

    Parameters
    ----------
    nodes            : (2, N) [X, Y] node coordinates from mesh.mat
    n_layers         : number of build layers (12)
    initial_fraction : squareSideFraction at layer 0  (0.4)
    final_fraction   : squareSideFraction at layer n-1 (0.5)
    roi_boost        : weight inside the square (background = 1.0)
    edge_sigma_frac  : soft-edge width as fraction of min(width, height)

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

        # Box signed distance: positive = outside the square, negative = inside.
        # Uses the L-inf (Chebyshev) norm distance from the square boundary.
        box_dist = np.maximum(np.abs(x - cx) - half_side,
                              np.abs(y - cy) - half_side)

        # Smooth step: approaches 1 deep inside (box_dist → -∞),
        #              approaches 0 far outside (box_dist → +∞).
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
# Single-step dataset with layer indices
# =============================================================================

class LatentSurrogateDataset(Dataset):
    """
    Flat (state, action, next_state, layer_idx) transitions.

    layer_idx is 0-indexed: 0 = first laser pass (layer 1), n-1 = last pass.
    All state / action tensors are z-score normalised.
    """

    def __init__(
        self,
        trajectories: List[Trajectory],
        state_mean:   torch.Tensor,
        state_std:    torch.Tensor,
        action_mean:  float,
        action_std:   float,
        initial_temp: float = 300.0,
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

        print(f"[LatentSurrogateDataset] {len(self.states):,} transitions, "
              f"state_dim={state_dim}, layers 0–{max(layer_indices)}")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx],
            self.layer_indices[idx],
        )


# =============================================================================
# Full-trajectory dataset  (for rollout loss)
# =============================================================================

class LatentTrajectoryDataset(Dataset):
    """
    Each sample is one full trajectory.
    Layer indices are implicit (step t → layer t, 0-indexed).

    Returns
    -------
    states  : (T+1, state_dim) normalised  — s_0 … s_T
    actions : (T,   1)         normalised  — a_0 … a_{T-1}
    """

    def __init__(
        self,
        trajectories: List[Trajectory],
        state_mean:   torch.Tensor,
        state_std:    torch.Tensor,
        action_mean:  float,
        action_std:   float,
        initial_temp: float = 300.0,
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

        print(f"[LatentTrajectoryDataset] {len(self.samples):,} trajectories, "
              f"T={T_last}, state_dim={state_dim}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
