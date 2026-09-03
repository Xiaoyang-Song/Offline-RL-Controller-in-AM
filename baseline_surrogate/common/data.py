"""
baseline_surrogate/common/data.py
------------------------------------
Shared data utilities for baseline_surrogate/. Every baseline here predicts
the NEXT STATE directly:

    s_{t+1} = f(s_t, a_t, cool_time_t, layer_idx)

collapsing surrogate_model_latent_uncertainty_v2's explicit
heating-then-cooling decomposition (s_t -[a_t]-> u_heat_t -[cool_time_t]->
s_{t+1}) into one step. That two-stage split is part of THAT model's
contribution, not something an external baseline should get for free — so
every baseline in this package is single-stage EXCEPT the two ablation
studies (ablation_no_two_stage/, ablation_no_latent/), which deliberately
restore one piece of the full architecture at a time to isolate its
contribution; ablation_no_latent specifically needs the true two-stage
target (u_heat_t) since "keep two-stage, remove latent" is exactly what it
tests, so it imports TwoStageLatentSurrogateDataset from the main package
directly instead of anything here.

Reuses surrogate_model_latent_uncertainty_v2.dataset_v2 for trajectory
loading/splitting/extraction (TrajectoryV2, load_trajectories,
split_trajectories) — no need to duplicate those, they don't encode any
two-stage-specific assumption. Only the normalizer + Dataset classes below
are new, since they pool over {s_t, s_{t+1}} only (no u_heat_t).
"""

import os
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.dataset_v2 import TrajectoryV2, make_bootstrap_masks


# =============================================================================
# Normalizers (pooled over {s_t, s_{t+1}} only — no u_heat_t)
# =============================================================================

def build_single_stage_normalizers(
    trajectories: List[TrajectoryV2],
    initial_temp: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    all_states, all_lp, all_cool = [], [], []

    sample_u  = np.asarray(trajectories[0][0].u_final, dtype=np.float32).reshape(-1)
    state_dim = sample_u.shape[0]
    init_s    = np.full(state_dim, initial_temp, dtype=np.float32)

    for traj in trajectories:
        prev = init_s.copy()
        for step in traj:
            nxt = np.asarray(step.u_final, dtype=np.float32).reshape(-1)
            all_states.extend([prev, nxt])
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

    print(f"[baseline_surrogate.data] State mean ∈ [{state_mean.min():.1f}, {state_mean.max():.1f}], "
          f"std ∈ [{state_std.min():.4f}, {state_std.max():.1f}]")
    print(f"[baseline_surrogate.data] Laser power mean = {lp_mean:.2f} W, std = {lp_std:.2f} W")
    print(f"[baseline_surrogate.data] Cool time    mean = {cool_mean:.4f} s, std = {cool_std:.4f} s")
    return state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std


def extract_single_stage_raw_arrays(
    trajectories: List[TrajectoryV2],
    initial_temp: float = 300.0,
    lp_filter:    Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
) -> dict:
    """
    Flat (s_t, a_t, c_t, s_{t+1}, layer_idx) transitions as RAW numpy arrays
    (no normalisation) — used by the Kalman filter baseline, which fits its
    own PCA + linear map directly on raw Kelvin/W/s values rather than
    z-scored ones. Same state-chaining + lp_filter semantics as
    SingleStageFlatDataset (see its docstring).
    """
    sample_u  = np.asarray(trajectories[0][0].u_final, dtype=np.float32).reshape(-1)
    state_dim = sample_u.shape[0]
    init_s    = np.full(state_dim, initial_temp, dtype=np.float32)

    states, lp_actions, cool_times, next_states, layer_indices = [], [], [], [], []
    for traj in trajectories:
        prev = init_s.copy()
        for layer_idx, step in enumerate(traj):
            nxt = np.asarray(step.u_final, dtype=np.float32).reshape(-1)
            states.append(prev.copy())
            lp_actions.append(step.lp_action)
            cool_times.append(step.cool_time)
            next_states.append(nxt)
            layer_indices.append(layer_idx)
            prev = nxt

    if lp_filter is not None:
        ranges = [lp_filter] if isinstance(lp_filter[0], (int, float)) else list(lp_filter)
        keep = [i for i, lp in enumerate(lp_actions)
                if any(lo <= lp <= hi for lo, hi in ranges)]
        n_total = len(lp_actions)
        states        = [states[i]        for i in keep]
        lp_actions    = [lp_actions[i]     for i in keep]
        cool_times    = [cool_times[i]     for i in keep]
        next_states   = [next_states[i]    for i in keep]
        layer_indices = [layer_indices[i]  for i in keep]
        range_str = ", ".join(f"[{lo}, {hi}]" for lo, hi in ranges)
        print(f"[extract_single_stage_raw_arrays] lp_filter={range_str}W kept "
              f"{len(keep):,}/{n_total:,} transitions")

    return dict(
        s=np.stack(states, axis=0).astype(np.float32),
        a=np.array(lp_actions, dtype=np.float32),
        c=np.array(cool_times, dtype=np.float32),
        s2=np.stack(next_states, axis=0).astype(np.float32),
        layer=np.array(layer_indices, dtype=np.int64),
    )


# =============================================================================
# Flat single-stage dataset  (s_t, a_t, c_t, layer_idx) -> s_{t+1}
# =============================================================================

class SingleStageFlatDataset(Dataset):
    """
    Flat (s_t, lp_t, cool_t, s_{t+1}, layer_idx[, bootstrap_mask]) transitions.
    State chaining always walks the true, unfiltered trajectory (s_t is
    always the real predecessor state); lp_filter only restricts which
    resulting transitions are KEPT for training — same semantics as
    surrogate_model_latent_uncertainty_v2's TwoStageLatentSurrogateDataset,
    duplicated here rather than imported since that class also carries a
    u_heat_t target this package doesn't use.

    n_ensemble > 1 attaches per-transition bootstrap multiplicity masks
    (make_bootstrap_masks, reused from the main package) — only used by
    ablation_no_two_stage's bootstrap-trained ensemble; every other baseline
    here ignores the mask column.
    """

    def __init__(
        self,
        trajectories: List[TrajectoryV2],
        state_mean:   torch.Tensor,
        state_std:    torch.Tensor,
        lp_mean:      float,
        lp_std:       float,
        cool_mean:    float,
        cool_std:     float,
        initial_temp: float = 300.0,
        lp_filter:    Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
        n_ensemble:   int = 1,
        bootstrap_seed: int = 0,
    ):
        super().__init__()
        state_mean = state_mean.cpu()
        state_std  = state_std.cpu()

        sample_u  = np.asarray(trajectories[0][0].u_final, dtype=np.float32).reshape(-1)
        state_dim = sample_u.shape[0]
        init_s    = np.full(state_dim, initial_temp, dtype=np.float32)

        states, lp_actions, cool_times, next_states, layer_indices = [], [], [], [], []
        for traj in trajectories:
            prev = init_s.copy()
            for layer_idx, step in enumerate(traj):
                nxt = np.asarray(step.u_final, dtype=np.float32).reshape(-1)
                states.append(prev.copy())
                lp_actions.append(step.lp_action)
                cool_times.append(step.cool_time)
                next_states.append(nxt)
                layer_indices.append(layer_idx)
                prev = nxt

        if lp_filter is not None:
            ranges = [lp_filter] if isinstance(lp_filter[0], (int, float)) else list(lp_filter)
            keep = [i for i, lp in enumerate(lp_actions)
                    if any(lo <= lp <= hi for lo, hi in ranges)]
            n_total = len(lp_actions)
            states        = [states[i]        for i in keep]
            lp_actions    = [lp_actions[i]     for i in keep]
            cool_times    = [cool_times[i]     for i in keep]
            next_states   = [next_states[i]    for i in keep]
            layer_indices = [layer_indices[i]  for i in keep]
            range_str = ", ".join(f"[{lo}, {hi}]" for lo, hi in ranges)
            print(f"[SingleStageFlatDataset] lp_filter={range_str}W kept "
                  f"{len(keep):,}/{n_total:,} transitions")

        S  = torch.tensor(np.stack(states),      dtype=torch.float32)
        A  = torch.tensor(lp_actions,             dtype=torch.float32).unsqueeze(-1)
        C  = torch.tensor(cool_times,             dtype=torch.float32).unsqueeze(-1)
        S2 = torch.tensor(np.stack(next_states),  dtype=torch.float32)

        self.states       = (S  - state_mean) / state_std
        self.lp_actions   = (A  - lp_mean)    / lp_std
        self.cool_times   = (C  - cool_mean)  / cool_std
        self.next_states  = (S2 - state_mean) / state_std
        self.layer_indices = torch.tensor(layer_indices, dtype=torch.long)

        self.n_ensemble = n_ensemble
        if n_ensemble > 1:
            masks = make_bootstrap_masks(len(self.states), n_ensemble, seed=bootstrap_seed)
            self.bootstrap_masks = torch.tensor(masks, dtype=torch.float32)
        else:
            self.bootstrap_masks = torch.ones(len(self.states), 1, dtype=torch.float32)

        print(f"[SingleStageFlatDataset] {len(self.states):,} transitions, "
              f"state_dim={state_dim}, layers 0–{max(layer_indices)}")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return (
            self.states[idx], self.lp_actions[idx], self.cool_times[idx],
            self.next_states[idx], self.layer_indices[idx], self.bootstrap_masks[idx],
        )


# =============================================================================
# Full-trajectory dataset  (for the LSTM baseline)
# =============================================================================

class SingleStageTrajectoryDataset(Dataset):
    """
    Each sample is one full (unfiltered) trajectory: states (T+1, D),
    actions (T, 1), cool_times (T, 1), plus a per-step lp_mask (T,) bool
    marking which steps' laser power falls inside lp_filter.

    The LSTM baseline needs the TRUE, unbroken state chain to carry
    meaningful recurrent context (dropping filtered-out layers would
    desynchronise every later state from its real predecessor — s_{t+1}
    already reflects whatever action layer t actually used, filtered or
    not). So lp_filter here does NOT drop transitions like
    SingleStageFlatDataset does; it only marks which steps' loss should
    count during training (masked loss — see lstm/train.py), giving the
    LSTM the same "only ever supervised on the 200-300W regime" property
    as every other baseline while still seeing physically real inputs.
    """

    def __init__(
        self,
        trajectories: List[TrajectoryV2],
        state_mean:   torch.Tensor,
        state_std:    torch.Tensor,
        lp_mean:      float,
        lp_std:       float,
        cool_mean:    float,
        cool_std:     float,
        initial_temp: float = 300.0,
        lp_filter:    Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
    ):
        super().__init__()
        state_mean = state_mean.cpu()
        state_std  = state_std.cpu()
        self.samples = []

        sample_u  = np.asarray(trajectories[0][0].u_final, dtype=np.float32).reshape(-1)
        state_dim = sample_u.shape[0]
        init_raw  = np.full(state_dim, initial_temp, dtype=np.float32)

        ranges = None
        if lp_filter is not None:
            ranges = [lp_filter] if isinstance(lp_filter[0], (int, float)) else list(lp_filter)

        n_kept_steps, n_total_steps = 0, 0
        for traj in trajectories:
            s_list, a_list, c_list, mask_list = [init_raw.copy()], [], [], []
            for step in traj:
                s_list.append(np.asarray(step.u_final, dtype=np.float32).reshape(-1))
                a_list.append(step.lp_action)
                c_list.append(step.cool_time)
                in_range = ranges is None or any(lo <= step.lp_action <= hi for lo, hi in ranges)
                mask_list.append(in_range)
                n_total_steps += 1
                n_kept_steps  += int(in_range)

            S_raw = torch.tensor(np.stack(s_list), dtype=torch.float32)
            A_raw = torch.tensor(a_list, dtype=torch.float32).unsqueeze(-1)
            C_raw = torch.tensor(c_list, dtype=torch.float32).unsqueeze(-1)
            self.samples.append((
                (S_raw - state_mean) / state_std,
                (A_raw - lp_mean)   / lp_std,
                (C_raw - cool_mean) / cool_std,
                torch.tensor(mask_list, dtype=torch.bool),
            ))

        if ranges is not None:
            range_str = ", ".join(f"[{lo}, {hi}]" for lo, hi in ranges)
            print(f"[SingleStageTrajectoryDataset] lp_filter={range_str}W marks "
                  f"{n_kept_steps:,}/{n_total_steps:,} steps loss-eligible "
                  f"(full trajectories still walked in full)")

        print(f"[SingleStageTrajectoryDataset] {len(self.samples):,} trajectories, "
              f"state_dim={state_dim}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
