"""
baseline_surrogate/summarize_results.py
--------------------------------------------
Aggregation CLI: loads whichever checkpoints you pass (every one is
optional — pass only what you have), evaluates each in TWO regimes on the
SAME held-out test split (re-derived from --data_path/--seed, matching how
surrogate_model_latent_uncertainty_v2/evaluate.py re-derives its own split
rather than storing indices), and prints/plots one side-by-side comparison
for each regime:

  - teacher-forced (single-step): every layer's input s_t is the TRUE
    previous state — isolates one-step accuracy, matches training.
  - auto-regressive (rollout): only the true initial state s_0 is given;
    every later s_t is the method's OWN previous prediction, chained
    forward through all 12 layers — matches what every method actually
    faces at real deployment, where only the action/cool-time schedule is
    known ahead of time. This is the harder, more realistic number, and
    can look much worse than the teacher-forced one for methods whose
    later stages/steps were only ever trained on ground-truth inputs (a
    known compounding-error failure mode for multi-stage/multi-step
    dynamics models trained via teacher forcing only).

Every baseline predicts s_{t+1} directly (no intermediate u_heat_t) except
the main surrogate and the ablation_no_latent ablation (both genuinely
two-stage) — for those two, `heat_mae` is ALSO reported in both regimes;
every other method reports "N/A" there, since predicting s_{t+1} directly
with no u_heat_t target is exactly what makes them single-stage.

Usage — pass any subset of the checkpoint flags:
    python -m baseline_surrogate.summarize_results \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --surrogate_checkpoint             surrogate_model_latent_uncertainty_v2/runs/narrow_200_300W/two_stage_best.pt \\
        --mlp_checkpoint                   baseline_surrogate/mlp/runs/narrow_200_300W/mlp_best.pt \\
        --lstm_checkpoint                  baseline_surrogate/lstm/runs/narrow_200_300W/lstm_best.pt \\
        --kalman_checkpoint                baseline_surrogate/kalman_filter/runs/narrow_200_300W/kalman_filter_fitted.pt \\
        --vanilla_ensemble_checkpoint      baseline_surrogate/vanilla_ensemble/runs/narrow_200_300W/vanilla_ensemble_best.pt \\
        --ablation_no_two_stage_checkpoint baseline_surrogate/ablation_no_two_stage/runs/narrow_200_300W/ablation_no_two_stage_best.pt \\
        --ablation_no_latent_checkpoint    baseline_surrogate/ablation_no_latent/runs/narrow_200_300W/ablation_no_latent_best.pt \\
        --out_dir baseline_surrogate/results
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_latent_uncertainty_v2.dataset_v2 import (
    load_trajectories, split_trajectories, TwoStageLatentTrajectoryDataset,
)
from surrogate_model_latent_uncertainty_v2.train import load_two_stage_surrogate

from baseline_surrogate.common.data import SingleStageTrajectoryDataset
from baseline_surrogate.common.eval import (
    evaluate_single_stage_binned, evaluate_autoregressive, evaluate_autoregressive_with_heat,
)

from baseline_surrogate.mlp.model import PlainMLPSurrogate
from baseline_surrogate.lstm.model import LSTMSurrogate, LSTMPredictor
from baseline_surrogate.kalman_filter.model import load_kalman_surrogate, KalmanPredictor
from baseline_surrogate.vanilla_ensemble.model import VanillaDeepEnsembleSurrogate
from baseline_surrogate.ablation_no_two_stage.model import NoTwoStageSurrogate
from baseline_surrogate.ablation_no_latent.model import NoLatentTwoStageSurrogate
from surrogate_model_latent_uncertainty_v2.model import _moment_match


# =============================================================================
# Main-surrogate predictor adapter
# =============================================================================

class MainSurrogatePredictor:
    """Adapts TwoStageEnsembleGaussianLatentDynamicsModel.predict_unnorm
    (raw in/out) to the predict_fn(s_norm, a_norm, c_norm, layer_idx) ->
    s2_pred_norm interface — denormalises, chains heat->cool, renormalises.
    predict() returns (next_pred_norm, heat_pred_norm)."""

    def __init__(self, model, state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std, device):
        self.model = model
        self.state_mean, self.state_std = state_mean.to(device), state_std.to(device)
        self.lp_mean, self.lp_std = lp_mean, lp_std
        self.cool_mean, self.cool_std = cool_mean, cool_std
        self.device = device

    def predict(self, s_norm, a_norm, c_norm, layer_idx):
        s_raw = s_norm * self.state_std + self.state_mean
        a_raw = a_norm * self.lp_std   + self.lp_mean
        c_raw = c_norm * self.cool_std + self.cool_mean
        out = self.model.predict_unnorm(
            s_raw, a_raw, c_raw, layer_idx,
            self.state_mean, self.state_std, self.lp_mean, self.lp_std, self.cool_mean, self.cool_std,
        )
        return (out["next_pred_raw"] - self.state_mean) / self.state_std, \
               (out["heat_pred_raw"] - self.state_mean) / self.state_std


# =============================================================================
# Teacher-forced / auto-regressive eval wrappers
# =============================================================================

def _eval_teacher_forced(name, predict_fn, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len,
                         on_batch_start=None):
    """Returns (mae_per_layer, rmse_per_layer, actions_raw, sq_err_per_sample)."""
    ds = SingleStageTrajectoryDataset(test_trajs, state_mean=sm.cpu(), state_std=ss.cpu(),
                                      lp_mean=lm, lp_std=ls, cool_mean=cm, cool_std=cs)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    mae, rmse, act, sq = evaluate_single_stage_binned(predict_fn, loader, ss, lm, ls, device,
                                                       traj_len=traj_len, on_batch_start=on_batch_start)
    print(f"[summarize_results] [teacher-forced ] {name}: "
          f"MAE(mean)={mae.mean():.3f} K, RMSE(mean)={rmse.mean():.3f} K")
    return mae, rmse, act, sq


def _eval_autoregressive(name, predict_fn, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len,
                         on_batch_start=None):
    """Returns (mae_per_layer, rmse_per_layer, actions_raw, sq_err_per_sample)."""
    ds = SingleStageTrajectoryDataset(test_trajs, state_mean=sm.cpu(), state_std=ss.cpu(),
                                      lp_mean=lm, lp_std=ls, cool_mean=cm, cool_std=cs)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    mae, rmse, act, sq = evaluate_autoregressive(predict_fn, loader, ss, lm, ls, device,
                                                 traj_len=traj_len, on_batch_start=on_batch_start)
    print(f"[summarize_results] [auto-regressive] {name}: "
          f"MAE(mean)={mae.mean():.3f} K, RMSE(mean)={rmse.mean():.3f} K")
    return mae, rmse, act, sq


def _eval_two_stage_heat(name, predict_with_heat_fn, sm, ss, lm, ls, cm, cs, test_trajs, device,
                         traj_len, n_ensemble, autoregressive: bool):
    """Teacher-forced OR auto-regressive next-state + heat-stage MAE for a
    genuinely two-stage method (main surrogate / ablation_no_latent)."""
    ds = TwoStageLatentTrajectoryDataset(test_trajs, state_mean=sm.cpu(), state_std=ss.cpu(),
                                         lp_mean=lm, lp_std=ls, cool_mean=cm, cool_std=cs,
                                         n_ensemble=n_ensemble)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    if autoregressive:
        next_mae, next_rmse, heat_mae = evaluate_autoregressive_with_heat(
            predict_with_heat_fn, loader, ss, device, traj_len=traj_len,
        )
        tag = "auto-regressive"
    else:
        # Teacher-forced: feed the TRUE s_t every layer (drop-in loader wrapper).
        ss_dev = ss.to(device)
        next_abs = [[] for _ in range(traj_len)]
        next_sq  = [[] for _ in range(traj_len)]
        heat_abs = [[] for _ in range(traj_len)]
        with torch.no_grad():
            for traj_s, traj_h, traj_a, traj_c, _bmask in loader:
                traj_s, traj_h = traj_s.to(device), traj_h.to(device)
                traj_a, traj_c = traj_a.to(device), traj_c.to(device)
                B, T1, D = traj_s.shape
                T = min(T1 - 1, traj_len)
                for t in range(T):
                    layer_idx = torch.full((B,), t, dtype=torch.long, device=device)
                    next_pred, heat_pred = predict_with_heat_fn(
                        traj_s[:, t, :], traj_a[:, t, :], traj_c[:, t, :], layer_idx,
                    )
                    dh = (heat_pred - traj_h[:, t, :]) * ss_dev
                    heat_abs[t].extend(dh.abs().mean(dim=-1).cpu().numpy().tolist())
                    dn = (next_pred - traj_s[:, t + 1, :]) * ss_dev
                    next_abs[t].extend(dn.abs().mean(dim=-1).cpu().numpy().tolist())
                    next_sq[t].extend((dn ** 2).mean(dim=-1).cpu().numpy().tolist())
        next_mae = np.array([np.mean(next_abs[t]) for t in range(traj_len)])
        next_rmse = np.array([np.sqrt(np.mean(next_sq[t])) for t in range(traj_len)])
        heat_mae = np.array([np.mean(heat_abs[t]) for t in range(traj_len)])
        tag = "teacher-forced "

    print(f"[summarize_results] [{tag}] {name}: next MAE(mean)={next_mae.mean():.3f} K, "
          f"next RMSE(mean)={next_rmse.mean():.3f} K, heat MAE(mean)={heat_mae.mean():.3f} K")
    return next_mae, next_rmse, heat_mae


# =============================================================================
# Plotting
# =============================================================================

def plot_rmse_vs_action(binned_results, out_path, title, id_range=(200.0, 300.0), bin_width=20.0):
    """binned_results: {name: (actions_raw, sq_err_per_sample)}. Bins the
    test set by raw laser power [W] and plots RMSE per bin per method, with
    the training (in-distribution) range shaded."""
    lo = min(a.min() for a, _ in binned_results.values())
    hi = max(a.max() for a, _ in binned_results.values())
    edges = np.arange(np.floor(lo / bin_width) * bin_width,
                      np.ceil(hi / bin_width) * bin_width + bin_width, bin_width)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axvspan(id_range[0], id_range[1], color="tab:green", alpha=0.12,
              label=f"Training range [{id_range[0]:.0f}, {id_range[1]:.0f}]W")

    for name, (actions, sq) in binned_results.items():
        bin_idx = np.digitize(actions, edges) - 1
        rmse_per_bin = np.full(len(centers), np.nan)
        n_per_bin    = np.zeros(len(centers), dtype=int)
        for b in range(len(centers)):
            mask = bin_idx == b
            n_per_bin[b] = mask.sum()
            if n_per_bin[b] > 0:
                rmse_per_bin[b] = np.sqrt(np.mean(sq[mask]))
        valid = n_per_bin >= 5
        ax.plot(centers[valid], rmse_per_bin[valid], marker="o", linewidth=1.5, label=name)

    ax.set_xlabel("Laser power [W]")
    ax.set_ylabel("Next-state RMSE [K]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[summarize_results] Saved → {out_path}")


def _report(results, binned, out_dir, tag, traj_len, id_range, bin_width):
    """results: {name: (mae, rmse, heat_mae_or_None, n_params)}.
    binned: {name: (actions_raw, sq_err_per_sample)}.
    tag: "" for teacher-forced, "_autoregressive" for auto-regressive —
    used as both a filename suffix and a title annotation."""
    label = "Teacher-Forced" if tag == "" else "Auto-Regressive"

    print(f"\n[summarize_results] {'='*90}")
    print(f"  [{label}] {'Method':<43}{'next_MAE[K]':>13}{'next_RMSE[K]':>14}{'heat_MAE[K]':>13}{'#params':>14}")
    print(f"[summarize_results] {'='*90}")
    for name, (mae, rmse, heat_mae, n_params) in results.items():
        heat_str = f"{heat_mae.mean():>13.3f}" if heat_mae is not None else f"{'N/A':>13}"
        print(f"  {name:<52}{mae.mean():>13.3f}{rmse.mean():>14.3f}{heat_str}{n_params:>14,}")
    print(f"[summarize_results] {'='*90}\n")

    plot_rmse_vs_action(
        binned, os.path.join(out_dir, f"rmse_vs_action{tag}.png"),
        title=f"baseline_surrogate — RMSE vs. Laser-Power Range ({label}, test set)",
        id_range=id_range, bin_width=bin_width,
    )

    layers = np.arange(1, traj_len + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, (mae, _rmse, _heat, _n) in results.items():
        ax.plot(layers, mae, marker="o", linewidth=1.5, label=name)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Next-state MAE [K]")
    ax.set_title(f"baseline_surrogate — Per-Layer Next-State MAE ({label}, test set)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(layers)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"per_layer_mae{tag}.png")
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[summarize_results] Saved → {out_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, (_mae, rmse, _heat, _n) in results.items():
        ax.plot(layers, rmse, marker="s", linewidth=1.5, label=name)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Next-state RMSE [K]")
    ax.set_title(f"baseline_surrogate — Per-Layer Next-State RMSE ({label}, test set)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(layers)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"per_layer_rmse{tag}.png")
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[summarize_results] Saved → {out_path}")

    names_sorted = sorted(results.keys(), key=lambda n: results[n][0].mean())
    means = [results[n][0].mean() for n in names_sorted]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names_sorted) + 1.5))
    ax.barh(names_sorted, means, color="steelblue")
    ax.set_xlabel("Mean next-state MAE [K] (lower is better)")
    ax.set_title(f"baseline_surrogate — Leaderboard ({label}, test set)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"leaderboard{tag}.png")
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[summarize_results] Saved → {out_path}")

    csv_path = os.path.join(out_dir, f"leaderboard{tag}.csv")
    with open(csv_path, "w") as f:
        f.write("method,next_mae_mean_K,next_rmse_mean_K,heat_mae_mean_K,n_params\n")
        for name, (mae, rmse, heat_mae, n_params) in results.items():
            heat_str = f"{heat_mae.mean():.4f}" if heat_mae is not None else "NA"
            f.write(f"{name},{mae.mean():.4f},{rmse.mean():.4f},{heat_str},{n_params}\n")
    print(f"[summarize_results] Saved → {csv_path}")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise baseline_surrogate/ results against the main surrogate.")
    p.add_argument("--data_path",     type=str, required=True)
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--device",        type=str,   default="")

    p.add_argument("--surrogate_checkpoint",             type=str, default=None)
    p.add_argument("--mlp_checkpoint",                   type=str, default=None)
    p.add_argument("--lstm_checkpoint",                  type=str, default=None)
    p.add_argument("--kalman_checkpoint",                type=str, default=None)
    p.add_argument("--vanilla_ensemble_checkpoint",      type=str, default=None)
    p.add_argument("--ablation_no_two_stage_checkpoint", type=str, default=None)
    p.add_argument("--ablation_no_latent_checkpoint",    type=str, default=None)

    p.add_argument("--id_range_min",     type=float, default=200.0,
                   help="Training (in-distribution) laser-power range lower bound [W], for rmse_vs_action*.png shading.")
    p.add_argument("--id_range_max",     type=float, default=300.0,
                   help="Training (in-distribution) laser-power range upper bound [W], for rmse_vs_action*.png shading.")
    p.add_argument("--action_bin_width", type=float, default=20.0,
                   help="Laser-power bin width [W] for rmse_vs_action*.png.")

    p.add_argument("--out_dir", type=str, default="baseline_surrogate/results")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[summarize_results] Output dir : {args.out_dir}")
    print(f"[summarize_results] Device     : {device}")

    all_trajs = load_trajectories(args.data_path)
    _train, _val, test_trajs = split_trajectories(
        all_trajs, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
    )
    traj_len = len(test_trajs[0])

    results,    binned    = {}, {}   # teacher-forced
    results_ar, binned_ar = {}, {}   # auto-regressive

    # ── main surrogate (two-stage) ───────────────────────────────────────────
    if args.surrogate_checkpoint:
        (model, sm, ss, lm, ls, cm, cs, _roi) = load_two_stage_surrogate(args.surrogate_checkpoint, device)
        pred = MainSurrogatePredictor(model, sm, ss, lm, ls, cm, cs, device)
        name = "surrogate (main, two-stage+latent+ensemble)"

        next_mae, next_rmse, heat_mae = _eval_two_stage_heat(
            name, pred.predict, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len,
            model.n_ensemble, autoregressive=False)
        results["surrogate (main)"] = (next_mae, next_rmse, heat_mae, model.count_parameters())

        def _predict_fn_main(s, a, c, layer_idx, pred=pred):
            next_n, _heat_n = pred.predict(s, a, c, layer_idx)
            return next_n

        _, _, act_raw, sq_err = _eval_teacher_forced(
            name, _predict_fn_main, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len)
        binned["surrogate (main)"] = (act_raw, sq_err)

        next_mae_ar, next_rmse_ar, heat_mae_ar = _eval_two_stage_heat(
            name, pred.predict, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len,
            model.n_ensemble, autoregressive=True)
        results_ar["surrogate (main)"] = (next_mae_ar, next_rmse_ar, heat_mae_ar, model.count_parameters())

        _, _, act_raw_ar, sq_err_ar = _eval_autoregressive(
            name, _predict_fn_main, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len)
        binned_ar["surrogate (main)"] = (act_raw_ar, sq_err_ar)

    # ── MLP ───────────────────────────────────────────────────────────────
    if args.mlp_checkpoint:
        ckpt = torch.load(args.mlp_checkpoint, map_location=device, weights_only=False)
        model = PlainMLPSurrogate(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        sm, ss = ckpt["state_mean"], ckpt["state_std"]
        lm, ls, cm, cs = ckpt["lp_mean"], ckpt["lp_std"], ckpt["cool_mean"], ckpt["cool_std"]

        def _predict_fn_mlp(s, a, c, layer_idx, model=model):
            with torch.no_grad():
                return model(s, a, c, layer_idx)

        mae, rmse, act, sq = _eval_teacher_forced("mlp", _predict_fn_mlp, sm, ss, lm, ls, cm, cs,
                                                  test_trajs, device, traj_len)
        results["mlp"] = (mae, rmse, None, model.count_parameters())
        binned["mlp"] = (act, sq)

        mae_ar, rmse_ar, act_ar, sq_ar = _eval_autoregressive("mlp", _predict_fn_mlp, sm, ss, lm, ls, cm, cs,
                                                              test_trajs, device, traj_len)
        results_ar["mlp"] = (mae_ar, rmse_ar, None, model.count_parameters())
        binned_ar["mlp"] = (act_ar, sq_ar)

    # ── LSTM ─────────────────────────────────────────────────────────────
    if args.lstm_checkpoint:
        ckpt = torch.load(args.lstm_checkpoint, map_location=device, weights_only=False)
        model = LSTMSurrogate(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        sm, ss = ckpt["state_mean"], ckpt["state_std"]
        lm, ls, cm, cs = ckpt["lp_mean"], ckpt["lp_std"], ckpt["cool_mean"], ckpt["cool_std"]

        predictor = LSTMPredictor(model, device)
        mae, rmse, act, sq = _eval_teacher_forced(
            "lstm", predictor.predict, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len,
            on_batch_start=predictor.on_batch_start)
        results["lstm"] = (mae, rmse, None, model.count_parameters())
        binned["lstm"] = (act, sq)

        predictor_ar = LSTMPredictor(model, device)   # fresh hidden-state tracker
        mae_ar, rmse_ar, act_ar, sq_ar = _eval_autoregressive(
            "lstm", predictor_ar.predict, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len,
            on_batch_start=predictor_ar.on_batch_start)
        results_ar["lstm"] = (mae_ar, rmse_ar, None, model.count_parameters())
        binned_ar["lstm"] = (act_ar, sq_ar)

    # ── Kalman filter ────────────────────────────────────────────────────
    if args.kalman_checkpoint:
        kf, sm, ss, lm, ls, cm, cs = load_kalman_surrogate(args.kalman_checkpoint)
        n_params = kf.coeffs.size + kf.components.size
        predictor = KalmanPredictor(kf, sm, ss, lm, ls, cm, cs, device)

        mae, rmse, act, sq = _eval_teacher_forced("kalman_filter", predictor.predict, sm, ss, lm, ls, cm, cs,
                                                  test_trajs, device, traj_len)
        results["kalman_filter"] = (mae, rmse, None, n_params)
        binned["kalman_filter"] = (act, sq)

        mae_ar, rmse_ar, act_ar, sq_ar = _eval_autoregressive(
            "kalman_filter", predictor.predict, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len)
        results_ar["kalman_filter"] = (mae_ar, rmse_ar, None, n_params)
        binned_ar["kalman_filter"] = (act_ar, sq_ar)

    # ── Vanilla deep ensemble ───────────────────────────────────────────
    if args.vanilla_ensemble_checkpoint:
        ckpt = torch.load(args.vanilla_ensemble_checkpoint, map_location=device, weights_only=False)
        model = VanillaDeepEnsembleSurrogate(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        sm, ss = ckpt["state_mean"], ckpt["state_std"]
        lm, ls, cm, cs = ckpt["lp_mean"], ckpt["lp_std"], ckpt["cool_mean"], ckpt["cool_std"]

        def _predict_fn_ve(s, a, c, layer_idx, model=model):
            return model.predict_mean(s, a, c, layer_idx)

        mae, rmse, act, sq = _eval_teacher_forced("vanilla_ensemble", _predict_fn_ve, sm, ss, lm, ls, cm, cs,
                                                  test_trajs, device, traj_len)
        results["vanilla_ensemble"] = (mae, rmse, None, model.count_parameters())
        binned["vanilla_ensemble"] = (act, sq)

        mae_ar, rmse_ar, act_ar, sq_ar = _eval_autoregressive(
            "vanilla_ensemble", _predict_fn_ve, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len)
        results_ar["vanilla_ensemble"] = (mae_ar, rmse_ar, None, model.count_parameters())
        binned_ar["vanilla_ensemble"] = (act_ar, sq_ar)

    # ── Ablation: no two-stage ───────────────────────────────────────────
    if args.ablation_no_two_stage_checkpoint:
        ckpt = torch.load(args.ablation_no_two_stage_checkpoint, map_location=device, weights_only=False)
        model = NoTwoStageSurrogate(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        sm, ss = ckpt["state_mean"], ckpt["state_std"]
        lm, ls, cm, cs = ckpt["lp_mean"], ckpt["lp_std"], ckpt["cool_mean"], ckpt["cool_std"]

        def _predict_fn_nots(s, a, c, layer_idx, model=model):
            return model.predict_mean(s, a, c, layer_idx)

        mae, rmse, act, sq = _eval_teacher_forced("ablation_no_two_stage", _predict_fn_nots,
                                                  sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len)
        results["ablation_no_two_stage"] = (mae, rmse, None, model.count_parameters())
        binned["ablation_no_two_stage"] = (act, sq)

        mae_ar, rmse_ar, act_ar, sq_ar = _eval_autoregressive(
            "ablation_no_two_stage", _predict_fn_nots, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len)
        results_ar["ablation_no_two_stage"] = (mae_ar, rmse_ar, None, model.count_parameters())
        binned_ar["ablation_no_two_stage"] = (act_ar, sq_ar)

    # ── Ablation: no latent space ────────────────────────────────────────
    if args.ablation_no_latent_checkpoint:
        ckpt = torch.load(args.ablation_no_latent_checkpoint, map_location=device, weights_only=False)
        model = NoLatentTwoStageSurrogate(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        sm, ss = ckpt["state_mean"], ckpt["state_std"]
        lm, ls, cm, cs = ckpt["lp_mean"], ckpt["lp_std"], ckpt["cool_mean"], ckpt["cool_std"]
        name = "ablation_no_latent (main minus latent bottleneck)"

        def _predict_with_heat_nl(s, a, c, layer_idx, model=model):
            z_t = s
            mu_heat, log_sigma_heat = model._run(model.heating_transitions, z_t, a, layer_idx)
            mu_heat_mean, *_ = _moment_match(mu_heat, log_sigma_heat)
            z_heat = z_t + mu_heat_mean
            mu_cool, log_sigma_cool = model._run(model.cooling_transitions, z_heat, c, layer_idx)
            mu_cool_mean, *_ = _moment_match(mu_cool, log_sigma_cool)
            return z_heat + mu_cool_mean, z_heat

        next_mae, next_rmse, heat_mae = _eval_two_stage_heat(
            name, _predict_with_heat_nl, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len,
            model.n_ensemble, autoregressive=False)
        results["ablation_no_latent"] = (next_mae, next_rmse, heat_mae, model.count_parameters())

        def _predict_fn_nl(s, a, c, layer_idx, model=model):
            return model.predict_mean(s, a, c, layer_idx)

        _, _, act, sq = _eval_teacher_forced(name, _predict_fn_nl, sm, ss, lm, ls, cm, cs,
                                             test_trajs, device, traj_len)
        binned["ablation_no_latent"] = (act, sq)

        next_mae_ar, next_rmse_ar, heat_mae_ar = _eval_two_stage_heat(
            name, _predict_with_heat_nl, sm, ss, lm, ls, cm, cs, test_trajs, device, traj_len,
            model.n_ensemble, autoregressive=True)
        results_ar["ablation_no_latent"] = (next_mae_ar, next_rmse_ar, heat_mae_ar, model.count_parameters())

        _, _, act_ar, sq_ar = _eval_autoregressive(name, _predict_fn_nl, sm, ss, lm, ls, cm, cs,
                                                    test_trajs, device, traj_len)
        binned_ar["ablation_no_latent"] = (act_ar, sq_ar)

    if not results:
        print("[summarize_results] No checkpoints given — nothing to summarise.")
        return

    id_range = (args.id_range_min, args.id_range_max)
    _report(results,    binned,    args.out_dir, "",                traj_len, id_range, args.action_bin_width)
    _report(results_ar, binned_ar, args.out_dir, "_autoregressive", traj_len, id_range, args.action_bin_width)

    print(f"\n[summarize_results] Complete. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
