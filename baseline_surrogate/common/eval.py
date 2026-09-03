"""
baseline_surrogate/common/eval.py
------------------------------------
Shared single-step (teacher-forced) per-layer MAE/RMSE evaluator, mirroring
surrogate_model_latent_uncertainty_v2.evaluate.evaluate_single_step but for
a single next-state target only (no heat-field term, since every baseline
here predicts s_{t+1} directly — see common/data.py's module docstring).

Every baseline exposes one function of this signature:

    predict_fn(s_norm, a_norm, c_norm, layer_idx) -> s2_pred_norm

operating entirely in that baseline's OWN normalised space (each baseline
fits its own state_mean/state_std from the same train split via
build_single_stage_normalizers, so per-node scales are consistent across
baselines even though the tensors themselves are never shared) — this lets
summarize_results.py evaluate every architecturally different baseline
(MLP / LSTM / KF / ensemble / ablations) through one identical loop.
"""

from typing import Callable, Optional

import numpy as np
import torch


@torch.no_grad()
def evaluate_single_stage(
    predict_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    traj_loader,
    state_std: torch.Tensor,
    device:    str,
    traj_len:  int = 12,
    on_batch_start: Optional[Callable[[int], None]] = None,
):
    """
    traj_loader yields (states, actions, cool_times, ...) per
    SingleStageTrajectoryDataset (extra trailing tensors, e.g. lp_mask, are
    ignored). on_batch_start(B), if given, is called once per batch BEFORE
    the per-layer loop — used by the LSTM baseline to reset its recurrent
    hidden state to the right batch size at the start of each trajectory
    batch; every other (stateless) baseline can leave it as None.

    Returns (mae_per_layer, rmse_per_layer), each (traj_len,) numpy arrays,
    in raw Kelvin via the *state_std diff trick (normalised-space
    differences rescale to raw Kelvin exactly by the multiplicative std
    alone — the additive mean cancels in the subtraction — see
    surrogate_model_latent_uncertainty_v2.evaluate for the full derivation).
    """
    ss = state_std.to(device)
    abs_err = [[] for _ in range(traj_len)]
    sq_err  = [[] for _ in range(traj_len)]

    for batch in traj_loader:
        traj_s, traj_a, traj_c = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        B, T1, D = traj_s.shape
        T = min(T1 - 1, traj_len)

        if on_batch_start is not None:
            on_batch_start(B)

        for t in range(T):
            s_in = traj_s[:, t, :]
            a_in = traj_a[:, t, :]
            c_in = traj_c[:, t, :]
            s_gt = traj_s[:, t + 1, :]
            layer_idx = torch.full((s_in.shape[0],), t, dtype=torch.long, device=device)

            s2_pred = predict_fn(s_in, a_in, c_in, layer_idx)
            diff = (s2_pred - s_gt) * ss
            abs_err[t].extend(diff.abs().mean(dim=-1).cpu().numpy().tolist())
            sq_err[t].extend((diff ** 2).mean(dim=-1).cpu().numpy().tolist())

    mae  = np.array([np.mean(abs_err[t])        for t in range(traj_len)])
    rmse = np.array([np.sqrt(np.mean(sq_err[t])) for t in range(traj_len)])
    return mae, rmse


@torch.no_grad()
def evaluate_single_stage_binned(
    predict_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    traj_loader,
    state_std: torch.Tensor,
    lp_mean:   float,
    lp_std:    float,
    device:    str,
    traj_len:  int = 12,
    on_batch_start: Optional[Callable[[int], None]] = None,
):
    """Same single-step evaluation as evaluate_single_stage, but ALSO
    returns flat (raw laser-power action, per-sample squared error) arrays
    for every (trajectory, layer) — used to bin accuracy by laser-power
    range (e.g. comparing in-distribution vs. out-of-distribution error,
    see summarize_results.py's plot_rmse_vs_action). lp_mean/lp_std
    denormalise the action back to raw Watts for binning; every other
    argument matches evaluate_single_stage exactly.

    Returns (mae_per_layer, rmse_per_layer, actions_raw, sq_err_per_sample),
    the first two identical to evaluate_single_stage's return values.
    """
    ss = state_std.to(device)
    abs_err = [[] for _ in range(traj_len)]
    sq_err  = [[] for _ in range(traj_len)]
    actions_flat, sq_flat = [], []

    for batch in traj_loader:
        traj_s, traj_a, traj_c = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        B, T1, D = traj_s.shape
        T = min(T1 - 1, traj_len)

        if on_batch_start is not None:
            on_batch_start(B)

        for t in range(T):
            s_in = traj_s[:, t, :]
            a_in = traj_a[:, t, :]
            c_in = traj_c[:, t, :]
            s_gt = traj_s[:, t + 1, :]
            layer_idx = torch.full((s_in.shape[0],), t, dtype=torch.long, device=device)

            s2_pred = predict_fn(s_in, a_in, c_in, layer_idx)
            diff = (s2_pred - s_gt) * ss
            abs_per_sample = diff.abs().mean(dim=-1)
            sq_per_sample  = (diff ** 2).mean(dim=-1)

            abs_err[t].extend(abs_per_sample.cpu().numpy().tolist())
            sq_err[t].extend(sq_per_sample.cpu().numpy().tolist())

            a_raw = (a_in[:, 0] * lp_std + lp_mean).cpu().numpy()
            actions_flat.extend(a_raw.tolist())
            sq_flat.extend(sq_per_sample.cpu().numpy().tolist())

    mae  = np.array([np.mean(abs_err[t])        for t in range(traj_len)])
    rmse = np.array([np.sqrt(np.mean(sq_err[t])) for t in range(traj_len)])
    return mae, rmse, np.array(actions_flat), np.array(sq_flat)


@torch.no_grad()
def evaluate_autoregressive(
    predict_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    traj_loader,
    state_std: torch.Tensor,
    lp_mean:   float,
    lp_std:    float,
    device:    str,
    traj_len:  int = 12,
    on_batch_start: Optional[Callable[[int], None]] = None,
):
    """Auto-regressive (rollout) evaluation: chains each method's OWN
    predicted s_{t+1} as the NEXT step's input s_t, starting only from the
    true initial state s_0 (layer 0's input — deterministic, e.g. an
    all-300K field, not itself a prediction). This is what every method
    actually faces at real deployment: only the action/cool-time schedule
    is known ahead of time, never the true intermediate states — unlike
    evaluate_single_stage[_binned], which feeds the TRUE s_t at every layer
    (teacher forcing) and so can hide compounding error. Error at layer t
    reflects (t+1) chained prediction steps, not one isolated step.

    Same predict_fn(s_norm, a_norm, c_norm, layer_idx) -> s2_pred_norm
    interface as evaluate_single_stage — no changes needed to any
    baseline's predict_fn to use this; only which `s` is passed in differs.

    Returns (mae_per_layer, rmse_per_layer, actions_raw, sq_err_per_sample) —
    same shapes/semantics as evaluate_single_stage_binned.
    """
    ss = state_std.to(device)
    abs_err = [[] for _ in range(traj_len)]
    sq_err  = [[] for _ in range(traj_len)]
    actions_flat, sq_flat = [], []

    for batch in traj_loader:
        traj_s, traj_a, traj_c = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        B, T1, D = traj_s.shape
        T = min(T1 - 1, traj_len)

        if on_batch_start is not None:
            on_batch_start(B)

        s_pred = traj_s[:, 0, :]   # true initial state — the only "free" ground truth
        for t in range(T):
            a_in = traj_a[:, t, :]
            c_in = traj_c[:, t, :]
            s_gt = traj_s[:, t + 1, :]
            layer_idx = torch.full((s_pred.shape[0],), t, dtype=torch.long, device=device)

            s_pred = predict_fn(s_pred, a_in, c_in, layer_idx)   # chain OWN prediction forward
            diff = (s_pred - s_gt) * ss
            abs_per_sample = diff.abs().mean(dim=-1)
            sq_per_sample  = (diff ** 2).mean(dim=-1)

            abs_err[t].extend(abs_per_sample.cpu().numpy().tolist())
            sq_err[t].extend(sq_per_sample.cpu().numpy().tolist())

            a_raw = (a_in[:, 0] * lp_std + lp_mean).cpu().numpy()
            actions_flat.extend(a_raw.tolist())
            sq_flat.extend(sq_per_sample.cpu().numpy().tolist())

    mae  = np.array([np.mean(abs_err[t])        for t in range(traj_len)])
    rmse = np.array([np.sqrt(np.mean(sq_err[t])) for t in range(traj_len)])
    return mae, rmse, np.array(actions_flat), np.array(sq_flat)


@torch.no_grad()
def evaluate_autoregressive_with_heat(
    predict_with_heat_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                                   "tuple[torch.Tensor, torch.Tensor]"],
    traj_loader,
    state_std: torch.Tensor,
    device:    str,
    traj_len:  int = 12,
):
    """Auto-regressive variant of evaluate_autoregressive for the two
    genuinely two-stage methods (main surrogate, ablation_no_latent), which
    also expose a heat-field prediction. predict_with_heat_fn(s, a, c,
    layer_idx) -> (next_pred_norm, heat_pred_norm); traj_loader must yield
    (s, h, a, c, ...) per layer (e.g.
    surrogate_model_latent_uncertainty_v2.dataset_v2.TwoStageLatentTrajectoryDataset).
    Only next_pred is chained forward as the next step's s_t — heat_pred is
    a same-layer intermediate, not a next-state input.

    Returns (next_mae, next_rmse, heat_mae), each (traj_len,).
    """
    ss = state_std.to(device)
    next_abs = [[] for _ in range(traj_len)]
    next_sq  = [[] for _ in range(traj_len)]
    heat_abs = [[] for _ in range(traj_len)]

    for batch in traj_loader:
        traj_s, traj_h, traj_a, traj_c = (
            batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        )
        B, T1, D = traj_s.shape
        T = min(T1 - 1, traj_len)

        s_pred = traj_s[:, 0, :]
        for t in range(T):
            a_in = traj_a[:, t, :]
            c_in = traj_c[:, t, :]
            h_gt = traj_h[:, t, :]
            s_gt = traj_s[:, t + 1, :]
            layer_idx = torch.full((s_pred.shape[0],), t, dtype=torch.long, device=device)

            next_pred, heat_pred = predict_with_heat_fn(s_pred, a_in, c_in, layer_idx)

            dh = (heat_pred - h_gt) * ss
            heat_abs[t].extend(dh.abs().mean(dim=-1).cpu().numpy().tolist())

            dn = (next_pred - s_gt) * ss
            next_abs[t].extend(dn.abs().mean(dim=-1).cpu().numpy().tolist())
            next_sq[t].extend((dn ** 2).mean(dim=-1).cpu().numpy().tolist())

            s_pred = next_pred   # chain OWN prediction forward

    next_mae  = np.array([np.mean(next_abs[t])        for t in range(traj_len)])
    next_rmse = np.array([np.sqrt(np.mean(next_sq[t])) for t in range(traj_len)])
    heat_mae  = np.array([np.mean(heat_abs[t])         for t in range(traj_len)])
    return next_mae, next_rmse, heat_mae
