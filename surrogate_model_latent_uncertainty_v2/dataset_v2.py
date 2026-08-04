"""
surrogate_model_latent_uncertainty_v2/dataset_v2.py
------------------------------------------------------
Data extraction + loading for the two-stage (heating / cooling) LPBF
surrogate, built on the v2 simulation dataset at
../LPBF-Simulation/simulation_v2/RL_Dataset_v2/.

What changed vs. the v1 pipeline (dataset.py + surrogate_model_latent_uncertainty/dataset.py)
------------------------------------------------------------------------------------------------
  - The reward (meanDeviation) is now computed by the simulator at the END OF
    HEATING, not after cooling (see simulateHeatingCooling_v2.m). Each layer's
    .mat file exposes BOTH fields separately:
        uHeatFinal  - temperature field at the end of heating (reward input)
        uFinal      - temperature field at the end of cooling (next state)
    coolTime is randomized once per trajectory and stored per layer as
    coolTime_step.
  - The surrogate is accordingly split into two stages:
        Stage 1 (heating): (s_t, laser_power_t)      -> u_heat_t   [action-dependent]
        Stage 2 (cooling): (u_heat_t, cool_time_t)    -> s_{t+1}    [NOT action-dependent
                                                                      — cooling physics is
                                                                      the same regardless
                                                                      of the laser power
                                                                      that produced u_heat_t]

This file has two parts:
  1. Extraction — reads the raw .mat trajectories into TrajectoryV2 lists
     (see extract_dataset_v2.py for the CLI that pickles them).
  2. Consumer utilities — load/split trajectories, build normalizers, ROI
     weight tables, bootstrap masks, and the two PyTorch Dataset classes used
     by train.py / evaluate.py.

MDP convention
--------------
  state (s_t)     = temperature field BEFORE the laser pass on layer t
                     (layer 0 -> all-300 K initial field)
  laser_power (a_t)= laser power applied during layer t [W]
  u_heat_t        = temperature field AFTER heating, BEFORE cooling (reward input)
  cool_time_t     = cooling duration used for layer t [s]
  next_state(s_t+1)= temperature field AFTER cooling -> becomes s_{t+1}
"""

import os
import pickle
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from tqdm import tqdm

RL_DATASET_V2_DIR = os.path.join("..", "LPBF-Simulation", "simulation_v2", "RL_Dataset_v2")


class StepV2(NamedTuple):
    u_final:      np.ndarray  # (1053, 1) end-of-cooling field  = s_{t+1}
    u_heat_final: np.ndarray  # (1053, 1) end-of-heating field  (reward input, stage-1 target)
    lp_action:    float       # laser power [W]                (stage-1 input)
    ss_action:    float       # scan speed [mm/s]               (recorded, unused by the model)
    cool_time:    float       # cooling duration [s]            (stage-2 input)
    reward:       float       # -meanDeviation, computed from u_heat_final


TrajectoryV2 = List[StepV2]


# =============================================================================
# Extraction / pickling
# =============================================================================

def extract_single_trajectory_v2(trajectory_id: int, trajectory_length: int = 12) -> TrajectoryV2:
    trajectory = []
    for j in range(trajectory_length):
        filename = os.path.join(
            RL_DATASET_V2_DIR, f"trajectory_{trajectory_id:03d}", f"layer_{j + 1}_data.mat"
        )
        data = loadmat(filename)
        trajectory.append(StepV2(
            u_final=np.asarray(data["uFinal"], dtype=np.float32),
            u_heat_final=np.asarray(data["uHeatFinal"], dtype=np.float32),
            lp_action=float(data["LP_action"][0][0]),
            ss_action=float(data["SS_action"][0][0]),
            cool_time=float(data["coolTime_step"][0][0]),
            reward=-float(data["meanDeviation"][0][0]),
        ))
    return trajectory


def gather_dataset_v2(id_list, trajectory_length: int = 12) -> List[TrajectoryV2]:
    dataset = []
    for trajectory_id in tqdm(id_list):
        dataset.append(extract_single_trajectory_v2(trajectory_id, trajectory_length))
    return dataset


# =============================================================================
# I/O helpers
# =============================================================================

def load_trajectories(pkl_path: str) -> List[TrajectoryV2]:
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"[dataset_v2] Loaded {len(dataset)} trajectories "
          f"× {len(dataset[0])} layers from {os.path.basename(pkl_path)}")
    return dataset


def split_trajectories(
    trajectories:  List[TrajectoryV2],
    val_fraction:  float = 0.10,
    test_fraction: float = 0.10,
    seed:          int   = 42,
) -> Tuple[List[TrajectoryV2], List[TrajectoryV2], List[TrajectoryV2]]:
    """Trajectory-level split (no data leakage between sets)."""
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

    print(f"[dataset_v2] Split → train {len(train)} | val {len(val)} | test {len(test)}")
    return train, val, test


def build_normalizers(
    trajectories: List[TrajectoryV2],
    initial_temp: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    """
    Compute z-score stats from training trajectories only.

    State stats are pooled across all three field "views" (pre-heat s_t,
    end-of-heat u_heat_t, end-of-cool s_{t+1}) since the encoder/decoder is
    shared across all of them — they are literally the same kind of object
    (a temperature field over the same mesh).

    Returns
    -------
    state_mean, state_std : (state_dim,) float32
    lp_mean,   lp_std      : laser power stats [W]
    cool_mean, cool_std    : cool time stats [s]
    """
    all_states, all_lp, all_cool = [], [], []

    sample_u  = np.asarray(trajectories[0][0].u_final, dtype=np.float32).reshape(-1)
    state_dim = sample_u.shape[0]
    init_s    = np.full(state_dim, initial_temp, dtype=np.float32)

    for traj in trajectories:
        prev = init_s.copy()
        for step in traj:
            heat = np.asarray(step.u_heat_final, dtype=np.float32).reshape(-1)
            nxt  = np.asarray(step.u_final,      dtype=np.float32).reshape(-1)
            all_states.extend([prev, heat, nxt])
            all_lp.append(step.lp_action)
            all_cool.append(step.cool_time)
            prev = nxt

    states_arr = np.stack(all_states, axis=0)
    lp_arr     = np.array(all_lp,   dtype=np.float32)
    cool_arr   = np.array(all_cool, dtype=np.float32)

    state_mean = torch.tensor(states_arr.mean(axis=0), dtype=torch.float32)
    state_std  = torch.tensor(states_arr.std(axis=0),  dtype=torch.float32).clamp_min(1e-6)
    lp_mean    = float(lp_arr.mean())
    lp_std     = float(lp_arr.std()) if lp_arr.std() > 1e-6 else 1.0
    cool_mean  = float(cool_arr.mean())
    cool_std   = float(cool_arr.std()) if cool_arr.std() > 1e-6 else 1.0

    print(f"[dataset_v2] State mean ∈ [{state_mean.min():.1f}, {state_mean.max():.1f}], "
          f"std ∈ [{state_std.min():.4f}, {state_std.max():.1f}]")
    print(f"[dataset_v2] Laser power mean = {lp_mean:.2f} W, std = {lp_std:.2f} W")
    print(f"[dataset_v2] Cool time    mean = {cool_mean:.4f} s, std = {cool_std:.4f} s")
    return state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std


# =============================================================================
# Physics-aware ROI weight helpers  (unchanged from surrogate_model_latent_uncertainty)
# =============================================================================

def compute_roi_weights_table(
    nodes:            np.ndarray,   # (2, N_nodes)  X, Y coordinates
    n_layers:         int   = 12,
    initial_fraction: float = 0.4,
    final_fraction:   float = 0.5,
    roi_boost:        float = 5.0,
    edge_sigma_frac:  float = 0.05,
) -> np.ndarray:
    """Per-layer node weight table — see surrogate_model_latent_uncertainty/dataset.py
    for the full derivation. The scan-region square grows from layer 0 to
    layer n_layers-1; nodes inside it get roi_boost× higher weight."""
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

    print(f"[dataset_v2] ROI weight table: {table.shape}, "
          f"range [{table.min():.3f}, {table.max():.3f}]")
    return table


def uniform_weights_table(n_layers: int, state_dim: int) -> np.ndarray:
    return np.ones((n_layers, state_dim), dtype=np.float32)


# =============================================================================
# Bootstrap resampling  (unchanged from surrogate_model_latent_uncertainty)
# =============================================================================

def make_bootstrap_masks(
    n_samples:  int,
    n_ensemble: int,
    seed:       int = 0,
) -> np.ndarray:
    """Per-member bootstrap multiplicity masks — see
    surrogate_model_latent_uncertainty/dataset.py for the full derivation."""
    rng   = np.random.default_rng(seed)
    masks = np.zeros((n_samples, n_ensemble), dtype=np.float32)
    for k in range(n_ensemble):
        draws  = rng.integers(0, n_samples, size=n_samples)
        counts = np.bincount(draws, minlength=n_samples)
        masks[:, k] = counts.astype(np.float32)

    frac_unseen = (masks == 0).mean()
    print(f"[dataset_v2] Bootstrap masks: {masks.shape}, "
          f"~{frac_unseen*100:.1f}% of (sample, member) pairs unseen "
          f"(expected ≈ {np.exp(-1)*100:.1f}% for large N)")
    return masks


# =============================================================================
# Single-step dataset with layer indices  (+ per-transition bootstrap masks)
# =============================================================================

class TwoStageLatentSurrogateDataset(Dataset):
    """
    Flat (s_t, lp_t, cool_t, u_heat_t, s_{t+1}, layer_idx, bootstrap_mask) transitions.

    All state-like tensors (s_t, u_heat_t, s_{t+1}) are z-score normalised
    with the SAME (shared) state_mean/state_std, since they share one
    encoder/decoder. lp_t and cool_t are normalised independently.

    lp_filter (optional): restrict the TRANSITIONS the heating/cooling
    networks actually train on to those whose laser power falls in
    [lo, hi] — e.g. for deliberately training a "narrow" surrogate (see
    experiments comparing it to a surrogate trained on the full range).
    State-chaining above still walks every original trajectory in full
    (so s_t is always the true, physically-correct predecessor state,
    regardless of what LP produced it) — filtering only drops which
    resulting transitions are kept for training, it never fabricates or
    skips over states. Only usable with this flat dataset (single-step
    training): TwoStageLatentTrajectoryDataset (rollout loss) needs
    unbroken 12-layer trajectories, which this filter would leave with
    gaps in — it deliberately has no equivalent parameter.
    """

    def __init__(
        self,
        trajectories:    List[TrajectoryV2],
        state_mean:      torch.Tensor,
        state_std:       torch.Tensor,
        lp_mean:         float,
        lp_std:          float,
        cool_mean:       float,
        cool_std:        float,
        initial_temp:    float = 300.0,
        n_ensemble:      int   = 5,
        bootstrap_seed:  int   = 0,
        lp_filter:       Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        state_mean = state_mean.cpu()
        state_std  = state_std.cpu()

        sample_u  = np.asarray(trajectories[0][0].u_final, dtype=np.float32).reshape(-1)
        state_dim = sample_u.shape[0]
        init_s    = np.full(state_dim, initial_temp, dtype=np.float32)

        states, lp_actions, cool_times, heat_states, next_states, layer_indices = (
            [], [], [], [], [], []
        )

        for traj in trajectories:
            prev = init_s.copy()
            for layer_idx, step in enumerate(traj):
                heat = np.asarray(step.u_heat_final, dtype=np.float32).reshape(-1)
                nxt  = np.asarray(step.u_final,      dtype=np.float32).reshape(-1)
                states.append(prev.copy())
                lp_actions.append(step.lp_action)
                cool_times.append(step.cool_time)
                heat_states.append(heat)
                next_states.append(nxt)
                layer_indices.append(layer_idx)
                prev = nxt

        if lp_filter is not None:
            lo, hi = lp_filter
            keep = [i for i, lp in enumerate(lp_actions) if lo <= lp <= hi]
            n_total = len(lp_actions)
            states        = [states[i]        for i in keep]
            lp_actions    = [lp_actions[i]     for i in keep]
            cool_times    = [cool_times[i]     for i in keep]
            heat_states   = [heat_states[i]    for i in keep]
            next_states   = [next_states[i]    for i in keep]
            layer_indices = [layer_indices[i]  for i in keep]
            print(f"[TwoStageLatentSurrogateDataset] lp_filter=[{lo}, {hi}]W kept "
                  f"{len(keep):,}/{n_total:,} transitions")

        S  = torch.tensor(np.stack(states),      dtype=torch.float32)
        A  = torch.tensor(lp_actions,             dtype=torch.float32).unsqueeze(-1)
        C  = torch.tensor(cool_times,             dtype=torch.float32).unsqueeze(-1)
        H  = torch.tensor(np.stack(heat_states),  dtype=torch.float32)
        S2 = torch.tensor(np.stack(next_states),  dtype=torch.float32)

        self.states        = (S  - state_mean) / state_std
        self.lp_actions     = (A  - lp_mean)   / lp_std
        self.cool_times      = (C  - cool_mean) / cool_std
        self.heat_states     = (H  - state_mean) / state_std
        self.next_states    = (S2 - state_mean) / state_std
        self.layer_indices = torch.tensor(layer_indices, dtype=torch.long)

        self.n_ensemble = n_ensemble
        masks = make_bootstrap_masks(len(self.states), n_ensemble, seed=bootstrap_seed)
        self.bootstrap_masks = torch.tensor(masks, dtype=torch.float32)  # (N, K)

        print(f"[TwoStageLatentSurrogateDataset] {len(self.states):,} transitions, "
              f"state_dim={state_dim}, layers 0–{max(layer_indices)}, "
              f"bootstrap K={n_ensemble} (transition-level resampling)")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return (
            self.states[idx],
            self.lp_actions[idx],
            self.cool_times[idx],
            self.heat_states[idx],
            self.next_states[idx],
            self.layer_indices[idx],
            self.bootstrap_masks[idx],
        )


# =============================================================================
# Full-trajectory dataset  (for rollout loss; + per-trajectory bootstrap masks)
# =============================================================================

class TwoStageLatentTrajectoryDataset(Dataset):
    """
    Each sample is one full trajectory.

    Returns
    -------
    states      : (T+1, state_dim) normalised  — s_0 … s_T
    heat_states : (T,   state_dim) normalised  — u_heat_0 … u_heat_{T-1}
    lp_actions   : (T,   1)         normalised  — a_0 … a_{T-1}
    cool_times    : (T,   1)         normalised  — cool_time_0 … cool_time_{T-1}
    bootstrap_mask : (n_ensemble,)
    """

    def __init__(
        self,
        trajectories:    List[TrajectoryV2],
        state_mean:      torch.Tensor,
        state_std:       torch.Tensor,
        lp_mean:         float,
        lp_std:          float,
        cool_mean:       float,
        cool_std:        float,
        initial_temp:    float = 300.0,
        n_ensemble:      int   = 5,
        bootstrap_seed:  int   = 0,
    ):
        super().__init__()
        state_mean = state_mean.cpu()
        state_std  = state_std.cpu()
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        sample_u  = np.asarray(trajectories[0][0].u_final, dtype=np.float32).reshape(-1)
        state_dim = sample_u.shape[0]
        init_raw  = np.full(state_dim, initial_temp, dtype=np.float32)

        T_last = 0
        for traj in trajectories:
            T      = len(traj)
            T_last = T
            s_list, h_list, a_list, c_list = [init_raw.copy()], [], [], []
            for step in traj:
                h_list.append(np.asarray(step.u_heat_final, dtype=np.float32).reshape(-1))
                s_list.append(np.asarray(step.u_final,      dtype=np.float32).reshape(-1))
                a_list.append(step.lp_action)
                c_list.append(step.cool_time)

            S_raw = torch.tensor(np.stack(s_list), dtype=torch.float32)   # (T+1, D)
            H_raw = torch.tensor(np.stack(h_list), dtype=torch.float32)   # (T,   D)
            A_raw = torch.tensor(a_list, dtype=torch.float32).unsqueeze(-1)  # (T, 1)
            C_raw = torch.tensor(c_list, dtype=torch.float32).unsqueeze(-1)  # (T, 1)
            self.samples.append((
                (S_raw - state_mean) / state_std,
                (H_raw - state_mean) / state_std,
                (A_raw - lp_mean)   / lp_std,
                (C_raw - cool_mean) / cool_std,
            ))

        self.n_ensemble = n_ensemble
        masks = make_bootstrap_masks(len(self.samples), n_ensemble, seed=bootstrap_seed)
        self.bootstrap_masks = torch.tensor(masks, dtype=torch.float32)  # (N_traj, K)

        print(f"[TwoStageLatentTrajectoryDataset] {len(self.samples):,} trajectories, "
              f"T={T_last}, state_dim={state_dim}, "
              f"bootstrap K={n_ensemble} (trajectory-level resampling)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s, h, a, c = self.samples[idx]
        return s, h, a, c, self.bootstrap_masks[idx]


# NOTE: no `if __name__ == "__main__":` CLI block here on purpose. `python -m
# surrogate_model_latent_uncertainty_v2.dataset_v2` re-executes this file as
# `__main__`, which rebinds StepV2.__module__ to "__main__" — pickles written
# under that run then fail to unpickle from train.py/evaluate.py (which
# import this module normally, so they look for StepV2 in
# surrogate_model_latent_uncertainty_v2.dataset_v2, not __main__). The CLI
# entry point lives in extract_dataset_v2.py instead, which only *imports*
# StepV2 from here — see that file for the "python -m ..." usage.
