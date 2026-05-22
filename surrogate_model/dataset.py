"""
surrogate_model/dataset.py
--------------------------
Data loading and preprocessing for the LPBF surrogate model.

Raw data format (from dataset.py / .mat files)
-----------------------------------------------
  dataset : list[trajectory]
  trajectory : list of 12 steps, each step = [u, lp, r]
    u  : np.ndarray  – temperature field after this layer, shape varies
                       (flattened to 1053-D float32)
    lp : float       – laser power applied this layer [W]
    r  : float       – reward = -meanDeviation

MDP convention (mirrors existing model.py flatten_dataset)
-----------------------------------------------------------
  state      = temperature field BEFORE the laser pass
               (layer 0 → all-300 K initial field)
  action     = laser power applied during this layer
  next_state = temperature field AFTER the laser pass  (= u)

Two Dataset classes are provided:

  SurrogateDataset   – flat (s, a, s') transitions  → single-step training
  TrajectoryDataset  – full 12-step trajectories     → multi-step rollout training
"""

import pickle
import sys
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ── type aliases ─────────────────────────────────────────────────────────────
Trajectory = List[List]   # list of [u, lp, r] steps


# =============================================================================
# I/O helpers
# =============================================================================

def load_trajectories(pkl_path: str) -> List[Trajectory]:
    """Load a pickled trajectory dataset produced by dataset.py."""
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"[dataset] Loaded {len(dataset)} trajectories "
          f"× {len(dataset[0])} layers from {os.path.basename(pkl_path)}")
    return dataset


def split_trajectories(
    trajectories: List[Trajectory],
    val_fraction:  float = 0.10,
    test_fraction: float = 0.10,
    seed:          int   = 42,
) -> Tuple[List[Trajectory], List[Trajectory], List[Trajectory]]:
    """
    Split *at the trajectory level* (not transition level) to prevent
    data leakage between train / val / test sets.

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


# =============================================================================
# Normalisation helpers
# =============================================================================

def build_normalizers(
    trajectories:    List[Trajectory],
    initial_temp:    float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Compute per-node state normalisation (mean / std) and action
    normalisation statistics from the *training* trajectories only.

    Returns
    -------
    state_mean  : (state_dim,)  float32 tensor
    state_std   : (state_dim,)  float32 tensor   (clamped ≥ 1e-6)
    action_mean : float
    action_std  : float
    """
    all_states  = []
    all_actions = []

    sample_u = np.asarray(trajectories[0][0][0], dtype=np.float32).reshape(-1)
    state_dim = sample_u.shape[0]
    initial_state = np.full(state_dim, initial_temp, dtype=np.float32)

    for traj in trajectories:
        prev = initial_state.copy()
        for step in traj:
            u, lp, _ = step
            next_s = np.asarray(u, dtype=np.float32).reshape(-1)
            all_states.append(prev)
            all_states.append(next_s)
            all_actions.append(float(lp))
            prev = next_s

    states_arr  = np.stack(all_states,  axis=0)   # (N, 1053)
    actions_arr = np.array(all_actions, dtype=np.float32)

    state_mean = torch.tensor(states_arr.mean(axis=0),  dtype=torch.float32)
    state_std  = torch.tensor(states_arr.std(axis=0),   dtype=torch.float32).clamp_min(1e-6)
    action_mean = float(actions_arr.mean())
    action_std  = float(actions_arr.std()) if actions_arr.std() > 1e-6 else 1.0

    print(f"[dataset] State  mean ∈ [{state_mean.min():.1f}, {state_mean.max():.1f}], "
          f"std ∈ [{state_std.min():.4f}, {state_std.max():.1f}]")
    print(f"[dataset] Action mean = {action_mean:.2f} W, std = {action_std:.2f} W")
    return state_mean, state_std, action_mean, action_std


# =============================================================================
# Single-step transition dataset
# =============================================================================

class SurrogateDataset(Dataset):
    """
    Flat dataset of (state, action, next_state) transitions.
    One sample = one (layer t → layer t+1) transition.

    All tensors are returned *normalised* (z-score for state, z-score for action).
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
        # Always keep dataset tensors on CPU; move to device in the training loop
        state_mean = state_mean.cpu()
        state_std  = state_std.cpu()
        states, actions, next_states = [], [], []

        sample_u   = np.asarray(trajectories[0][0][0], dtype=np.float32).reshape(-1)
        state_dim  = sample_u.shape[0]
        init_state = np.full(state_dim, initial_temp, dtype=np.float32)

        for traj in trajectories:
            prev = init_state.copy()
            for step in traj:
                u, lp, _ = step
                nxt = np.asarray(u, dtype=np.float32).reshape(-1)
                states.append(prev.copy())
                actions.append(float(lp))
                next_states.append(nxt)
                prev = nxt

        S  = torch.tensor(np.stack(states),      dtype=torch.float32)
        A  = torch.tensor(actions,                dtype=torch.float32).unsqueeze(-1)
        S2 = torch.tensor(np.stack(next_states),  dtype=torch.float32)

        # Normalise
        self.states      = (S  - state_mean)  / state_std
        self.actions     = (A  - action_mean) / action_std
        self.next_states = (S2 - state_mean)  / state_std

        print(f"[SurrogateDataset] {len(self.states):,} transitions, "
              f"state_dim={state_dim}")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return self.states[idx], self.actions[idx], self.next_states[idx]


# =============================================================================
# Full-trajectory dataset  (for multi-step rollout loss)
# =============================================================================

class TrajectoryDataset(Dataset):
    """
    Dataset where each sample is one full trajectory of length T.

    Returns
    -------
    states      : (T+1, state_dim)  — s_0 … s_T  (normalised)
    actions     : (T,   1)          — a_0 … a_{T-1}  (normalised)

    s_0 is the all-300 K initial state; s_1…s_T are the post-layer fields.
    The training loop can then unroll any prefix of length k ≤ T.
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
        # Always keep dataset tensors on CPU; move to device in the training loop
        state_mean = state_mean.cpu()
        state_std  = state_std.cpu()
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        sample_u  = np.asarray(trajectories[0][0][0], dtype=np.float32).reshape(-1)
        state_dim = sample_u.shape[0]
        init_raw  = np.full(state_dim, initial_temp, dtype=np.float32)

        for traj in trajectories:
            T      = len(traj)
            s_list = [init_raw.copy()]
            a_list = []
            for step in traj:
                u, lp, _ = step
                s_list.append(np.asarray(u, dtype=np.float32).reshape(-1))
                a_list.append(float(lp))

            S_raw = torch.tensor(np.stack(s_list), dtype=torch.float32)  # (T+1, D)
            A_raw = torch.tensor(a_list,            dtype=torch.float32).unsqueeze(-1)  # (T, 1)

            S_norm = (S_raw - state_mean) / state_std
            A_norm = (A_raw - action_mean) / action_std
            self.samples.append((S_norm, A_norm))

        print(f"[TrajectoryDataset] {len(self.samples):,} trajectories, "
              f"T={T}, state_dim={state_dim}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # states: (T+1, D), actions: (T, 1)
        return self.samples[idx]
