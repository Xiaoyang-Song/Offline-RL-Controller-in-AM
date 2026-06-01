"""
surrogate_domain_adaptation/dataset.py
---------------------------------------
Data loading for the multi-cooling-time domain adaptation surrogate.

Data layout
-----------
  /nfs/turbo/coe-sunwbgt/xysong/LPBF-Simulation/
    RL_Dataset_ct_050/   ← cooling time 0.05 s  (source)
    RL_Dataset_ct_060/   ← cooling time 0.06 s  (source)
    ...
    RL_Dataset_ct_100/   ← cooling time 0.10 s  (source, last)
    RL_Dataset_ct_150/   ← cooling time 0.15 s  (target)

  Each domain has ~300 trajectories, each with 12 layer .mat files:
    trajectory_NNN/
        layer_1_data.mat  → {uFinal (1053,1), LP_action (1,1), ...}
        layer_2_data.mat
        ...
        layer_12_data.mat
        trajectory_summary.mat

MDP convention (unchanged from surrogate_model_latent)
------------------------------------------------------
  state      = temperature field BEFORE the laser pass  (layer 0 → 300 K uniform)
  action     = LP_action (laser power [W]) for this layer
  next_state = uFinal (1053,) AFTER laser + cooling period

Domain shift
------------
  Longer cooling time → lower inter-layer temperatures:
    ct_050 mean field at layer 12 ≈ 2274 K
    ct_150 mean field at layer 12 ≈  953 K
  The adapter modules in model.py are designed to bridge this gap.

Classes exported
----------------
  load_domain_trajectories   — single domain from mat files → Trajectory list
  load_source_domains        — pool all source cooling times
  load_target_domain         — load ct_150
  split_trajectories         — trajectory-level train/val/test split
  sample_few_shot            — K adaptation shots + evaluation remainder
  build_normalizers          — z-score stats from training trajectories
  MultiDomainFlatDataset     — flat (s, a, s', layer_idx) from multiple domains
  MultiDomainTrajectoryDataset — full trajectories from multiple domains
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

DATA_ROOT = "/nfs/turbo/coe-sunwbgt/xysong/LPBF-Simulation"

SOURCE_COOLING_TIMES = [50, 60, 70, 80, 90, 100]   # 0.05 → 0.10 s
TARGET_COOLING_TIME  = 150                           # 0.15 s

Trajectory = List[List]   # each step: [uFinal (1053,), LP_action (float), ct (float)]


# =============================================================================
# I/O helpers
# =============================================================================

def load_domain_trajectories(
    domain_dir:   str,
    n_trajs:      Optional[int] = None,
    cooling_time: float = 0.0,
) -> List[Trajectory]:
    """
    Load trajectories from one RL_Dataset_ct_XXX folder.

    Returns a list of Trajectories in the standard format:
        step = [uFinal (1053,), LP_action (float), cooling_time (float)]
    where uFinal is the state AFTER the laser pass (= s_{t+1}).
    The initial state (before layer 1) is handled externally as 300 K.
    """
    traj_dirs = sorted([
        os.path.join(domain_dir, d)
        for d in os.listdir(domain_dir)
        if d.startswith("trajectory")
        and os.path.isdir(os.path.join(domain_dir, d))
    ])
    if n_trajs is not None:
        traj_dirs = traj_dirs[:n_trajs]

    trajectories = []
    for tdir in traj_dirs:
        steps = []
        for l in range(1, 13):
            mat_path = os.path.join(tdir, f"layer_{l}_data.mat")
            if not os.path.exists(mat_path):
                break
            mat = loadmat(mat_path)
            u  = mat["uFinal"].flatten().astype(np.float32)
            lp = float(mat["LP_action"].flatten()[0])
            steps.append([u, lp, cooling_time])
        if len(steps) == 12:
            trajectories.append(steps)

    print(f"[dataset] {os.path.basename(domain_dir)}: {len(trajectories)} trajectories")
    return trajectories


def load_source_domains(
    data_root:     str = DATA_ROOT,
    ct_list:       List[int] = SOURCE_COOLING_TIMES,
    n_trajs_each:  Optional[int] = None,
) -> List[Trajectory]:
    """Pool trajectories from all source cooling times."""
    all_trajs = []
    for ct in ct_list:
        domain_dir = os.path.join(data_root, f"RL_Dataset_ct_{ct:03d}")
        trajs = load_domain_trajectories(
            domain_dir, n_trajs=n_trajs_each, cooling_time=ct / 1000.0
        )
        all_trajs.extend(trajs)
    ct_labels = [f"ct={ct/10:.1f}" for ct in ct_list]
    print(f"[dataset] Source pool: {len(all_trajs)} trajectories  ({', '.join(ct_labels)})")
    return all_trajs


def load_target_domain(
    data_root: str = DATA_ROOT,
    ct:        int = TARGET_COOLING_TIME,
    n_trajs:   Optional[int] = None,
) -> List[Trajectory]:
    """Load target domain (default ct_150) trajectories."""
    domain_dir = os.path.join(data_root, f"RL_Dataset_ct_{ct:03d}")
    return load_domain_trajectories(domain_dir, n_trajs=n_trajs, cooling_time=ct / 1000.0)


# =============================================================================
# Splits
# =============================================================================

def split_trajectories(
    trajectories:  List[Trajectory],
    val_fraction:  float = 0.10,
    test_fraction: float = 0.10,
    seed:          int   = 42,
) -> Tuple[List[Trajectory], List[Trajectory], List[Trajectory]]:
    """Trajectory-level train / val / test split."""
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(trajectories))
    n    = len(trajectories)
    n_te = max(1, int(n * test_fraction))
    n_va = max(1, int(n * val_fraction))

    train = [trajectories[i] for i in idx[n_te + n_va:]]
    val   = [trajectories[i] for i in idx[n_te:n_te + n_va]]
    test  = [trajectories[i] for i in idx[:n_te]]

    print(f"[dataset] Split → train {len(train)} | val {len(val)} | test {len(test)}")
    return train, val, test


def sample_few_shot(
    trajectories: List[Trajectory],
    K:            int,
    seed:         int = 42,
) -> Tuple[List[Trajectory], List[Trajectory]]:
    """
    Split into K adaptation trajectories + remaining evaluation trajectories.
    The evaluation set is always the same regardless of K (seeded permutation).
    """
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(trajectories))
    adapt = [trajectories[i] for i in idx[:K]]
    eval_ = [trajectories[i] for i in idx[K:]]
    return adapt, eval_


# =============================================================================
# Normalizers
# =============================================================================

def build_normalizers(
    trajectories: List[Trajectory],
    initial_temp: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Compute z-score statistics for states and actions from training trajectories.
    Should always be called on SOURCE training data only — never on target data.
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

    state_mean  = torch.tensor(states_arr.mean(0), dtype=torch.float32)
    state_std   = torch.tensor(states_arr.std(0),  dtype=torch.float32).clamp_min(1e-6)
    action_mean = float(actions_arr.mean())
    action_std  = float(actions_arr.std()) if actions_arr.std() > 1e-6 else 1.0

    print(f"[dataset] State mean ∈ [{state_mean.min():.1f}, {state_mean.max():.1f}]  "
          f"std ∈ [{state_std.min():.4f}, {state_std.max():.1f}]")
    print(f"[dataset] Action mean = {action_mean:.2f} W  std = {action_std:.2f} W")
    return state_mean, state_std, action_mean, action_std


# =============================================================================
# Dataset classes  (same format as surrogate_model_latent/dataset.py)
# =============================================================================

class MultiDomainFlatDataset(Dataset):
    """
    Flat (state, action, next_state, layer_idx) transitions pooled from
    multiple domains.  Compatible with compute_single_step_losses() in
    surrogate_model_latent/train.py.
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

        print(f"[MultiDomainFlatDataset] {len(self.states):,} transitions, "
              f"state_dim={state_dim}")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return (self.states[idx], self.actions[idx],
                self.next_states[idx], self.layer_indices[idx])


class MultiDomainTrajectoryDataset(Dataset):
    """
    Full trajectories (states, actions) for rollout training.
    Compatible with the rollout loss in surrogate_model_latent/train.py.
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
        self.samples = []

        sample_u  = np.asarray(trajectories[0][0][0], dtype=np.float32).reshape(-1)
        state_dim = sample_u.shape[0]
        init_raw  = np.full(state_dim, initial_temp, dtype=np.float32)

        for traj in trajectories:
            s_list = [init_raw.copy()]
            a_list = []
            for step in traj:
                u, lp, _ = step
                s_list.append(np.asarray(u, dtype=np.float32).reshape(-1))
                a_list.append(float(lp))

            S_raw = torch.tensor(np.stack(s_list), dtype=torch.float32)
            A_raw = torch.tensor(a_list,            dtype=torch.float32).unsqueeze(-1)
            self.samples.append(
                ((S_raw - state_mean) / state_std,
                 (A_raw - action_mean) / action_std)
            )

        print(f"[MultiDomainTrajectoryDataset] {len(self.samples):,} trajectories, "
              f"T=12, state_dim={state_dim}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
