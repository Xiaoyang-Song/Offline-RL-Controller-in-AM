"""
surrogate_model_latent_uncertainty/evaluate_ood.py
-----------------------------------------------------
Out-of-distribution (OOD) validation for the epistemic uncertainty channel.

Motivation
----------
Evaluating a checkpoint on the train/val/test split of the SAME dataset it
was trained on can't show whether epistemic uncertainty is doing its job:
all three splits are drawn from the same action-power distribution, so
there's no real distribution shift for the K bootstrap-diversified members
to disagree about — everything looks "in-distribution" by construction.

This script instead evaluates a model trained on a NARROW action range
(e.g. 150-300 W) against a WIDER dataset (e.g. 150-400 W) that contains
genuinely unseen laser-power values with real ground-truth next-states
(no new simulation needed — the wider dataset already exists on disk).
Transitions are split into:

  ID  (in-distribution) : action ∈ [id_action_min, id_action_max]  — the
                           range the checkpoint was actually trained on
  OOD (out-of-distribution) : action outside that range — never seen

A working epistemic-uncertainty ensemble should show BOTH epistemic σ and
true rollout error climbing together as action moves into the OOD region,
and epistemic σ should positively correlate with per-sample error overall.
Aleatoric σ is plotted alongside as a control — it should NOT show the same
sharp OOD-boundary jump (see per-layer discussion in evaluate.py's docstring
and the package README: aleatoric collapses to a near-constant floor on this
near-deterministic simulator, so it's not the signal that should move here).

Usage
-----
    python -m surrogate_model_latent_uncertainty.evaluate_ood \\
        --checkpoint surrogate_model_latent_uncertainty/runs/narrow_150_300W/latent_best.pt \\
        --data_path  "Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl" \\
        --id_action_min 150 --id_action_max 300

Outputs (--out_dir, defaults to checkpoint directory)
------------------------------------------------------
  ood_uncertainty_vs_action.png   — epistemic | aleatoric | RMSE vs action, OOD region shaded
  ood_epistemic_vs_error.png      — scatter: epistemic σ vs per-sample error, ID vs OOD coloring
  Console summary: per-region (ID/OOD) mean σ, RMSE, and epistemic-vs-error correlation
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_latent_uncertainty.dataset import load_trajectories, LatentSurrogateDataset
from surrogate_model_latent_uncertainty.train import load_latent_surrogate


# =============================================================================
# Core evaluation
# =============================================================================

@torch.no_grad()
def collect_ood_samples(model, loader: DataLoader, state_mean, state_std,
                         action_mean, action_std, device: str):
    """
    Run every transition through the model and collect, per sample:
    raw action [W], mean epistemic/aleatoric/total σ (latent, averaged over
    dims), and single-step RMSE-per-sample [K] against ground truth.

    Returns a dict of 1-D numpy arrays, all length N.
    """
    model.eval()
    ss = state_std.to(device)
    sm = state_mean.to(device)

    actions, epi, ale, tot, sq_err, abs_err = [], [], [], [], [], []

    for s, a, s2, layer_idx, _bmask in loader:
        s         = s.to(device)
        a         = a.to(device)
        s2        = s2.to(device)
        layer_idx = layer_idx.to(device)

        z_t = model.encode(s)
        mu_mean, epi_std, ale_std, tot_std = model.predict_ensemble(z_t, a, layer_idx)
        s_pred = model.decode(z_t + mu_mean)

        diff_k = (s_pred - s2) * ss                      # (B, D) raw Kelvin
        a_raw  = (a.squeeze(-1) * action_std + action_mean)  # (B,) raw Watts

        actions.append(a_raw.cpu().numpy())
        epi.append(epi_std.mean(dim=-1).cpu().numpy())
        ale.append(ale_std.mean(dim=-1).cpu().numpy())
        tot.append(tot_std.mean(dim=-1).cpu().numpy())
        sq_err.append((diff_k ** 2).mean(dim=-1).cpu().numpy())
        abs_err.append(diff_k.abs().mean(dim=-1).cpu().numpy())

    return {
        "action":  np.concatenate(actions),
        "epi":     np.concatenate(epi),
        "ale":     np.concatenate(ale),
        "tot":     np.concatenate(tot),
        "sq_err":  np.concatenate(sq_err),
        "abs_err": np.concatenate(abs_err),
    }


def bin_by_action(data: dict, n_bins: int) -> dict:
    """
    Equal-width bins spanning [action.min(), action.max()].
    Returns per-bin arrays: centers, counts, mean epi/ale/tot, RMSE, MAE.
    """
    action = data["action"]
    edges  = np.linspace(action.min(), action.max(), n_bins + 1)
    idx    = np.clip(np.digitize(action, edges[1:-1]), 0, n_bins - 1)

    centers = 0.5 * (edges[:-1] + edges[1:])
    counts  = np.zeros(n_bins, dtype=int)
    epi_m   = np.full(n_bins, np.nan)
    ale_m   = np.full(n_bins, np.nan)
    tot_m   = np.full(n_bins, np.nan)
    rmse    = np.full(n_bins, np.nan)
    mae     = np.full(n_bins, np.nan)

    for b in range(n_bins):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] == 0:
            continue
        epi_m[b] = data["epi"][mask].mean()
        ale_m[b] = data["ale"][mask].mean()
        tot_m[b] = data["tot"][mask].mean()
        rmse[b]  = np.sqrt(data["sq_err"][mask].mean())
        mae[b]   = data["abs_err"][mask].mean()

    return {"centers": centers, "counts": counts, "epi": epi_m, "ale": ale_m,
            "tot": tot_m, "rmse": rmse, "mae": mae, "edges": edges}


def summarize_region(data: dict, mask: np.ndarray, label: str) -> dict:
    n = int(mask.sum())
    if n == 0:
        print(f"  {label:<12} n=0 (empty region)")
        return {}
    epi, sq_err = data["epi"][mask], data["sq_err"][mask]
    rmse = float(np.sqrt(sq_err.mean()))
    r_pearson, _  = pearsonr(epi, sq_err)  if n > 2 else (float("nan"), None)
    r_spearman, _ = spearmanr(epi, sq_err) if n > 2 else (float("nan"), None)
    print(f"  {label:<12} n={n:6d}  epist σ={data['epi'][mask].mean():.5f}  "
          f"aleat σ={data['ale'][mask].mean():.5f}  RMSE={rmse:7.2f} K  "
          f"corr(epi,err) pearson={r_pearson:.3f} spearman={r_spearman:.3f}")
    return {"n": n, "epi_mean": data["epi"][mask].mean(), "ale_mean": data["ale"][mask].mean(),
            "rmse": rmse, "pearson": r_pearson, "spearman": r_spearman}


# =============================================================================
# Plotting
# =============================================================================

def plot_uncertainty_vs_action(binned: dict, id_action_min: float, id_action_max: float,
                                out_path: str) -> None:
    centers = binned["centers"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [
        ("epi", "Epistemic σ (ensemble disagreement)", "tab:orange"),
        ("ale", "Aleatoric σ (avg. member noise)",       "tab:purple"),
        ("rmse", "Single-step RMSE [K]",                 "tab:red"),
    ]
    for ax, (key, title, color) in zip(axes, panels):
        ax.plot(centers, binned[key], marker="o", color=color, linewidth=1.5)
        ax.axvspan(id_action_min, id_action_max, color="tab:green", alpha=0.08,
                   label="Training range (ID)")
        ax.axvline(id_action_max, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("Laser power [W]")
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("OOD Stress Test — Uncertainty & Error vs. Laser Power "
                f"(shaded = training range [{id_action_min:.0f}, {id_action_max:.0f}] W)",
                fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate_ood] Saved → {out_path}")


def plot_epistemic_vs_error_scatter(data: dict, id_mask: np.ndarray, out_path: str,
                                    max_points: int = 20000) -> None:
    rng = np.random.default_rng(0)
    n   = len(data["epi"])
    idx = rng.choice(n, size=min(n, max_points), replace=False) if n > max_points else np.arange(n)

    err = np.sqrt(data["sq_err"][idx])
    epi = data["epi"][idx]
    id_m = id_mask[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(epi[id_m],  err[id_m],  s=6, alpha=0.35, color="tab:blue",   label="ID (training range)")
    ax.scatter(epi[~id_m], err[~id_m], s=6, alpha=0.35, color="tab:red",    label="OOD (unseen power)")
    ax.set_xlabel("Epistemic σ (latent, mean over dims)")
    ax.set_ylabel("Per-sample RMSE [K]")
    ax.set_title("Epistemic Uncertainty vs. Actual Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate_ood] Saved → {out_path}")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OOD stress test: evaluate a narrow-action-range checkpoint "
                    "on a wider dataset to validate epistemic uncertainty."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Checkpoint trained on a NARROW action range.")
    p.add_argument("--data_path",  type=str, required=True,
                   help="WIDER dataset (must cover a broader action range than the checkpoint saw).")
    p.add_argument("--id_action_min", type=float, default=150.0,
                   help="Lower bound of the checkpoint's training action range [W].")
    p.add_argument("--id_action_max", type=float, default=300.0,
                   help="Upper bound of the checkpoint's training action range [W]. "
                        "Actions above this are treated as OOD.")
    p.add_argument("--n_bins",      type=int, default=12)
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir",     type=str, default="",
                   help="Defaults to the checkpoint's directory.")
    p.add_argument("--device",      type=str, default="")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[evaluate_ood] Output dir : {out_dir}")
    print(f"[evaluate_ood] Device     : {device}")

    model, state_mean, state_std, action_mean, action_std, _roi = \
        load_latent_surrogate(args.checkpoint, device)
    print(f"[evaluate_ood] {model}")

    trajs = load_trajectories(args.data_path)
    all_actions = np.array([float(step[1]) for traj in trajs for step in traj])
    print(f"[evaluate_ood] Wide dataset action range: "
          f"[{all_actions.min():.1f}, {all_actions.max():.1f}] W  "
          f"(checkpoint trained on [{args.id_action_min:.1f}, {args.id_action_max:.1f}] W)")
    if all_actions.max() <= args.id_action_max:
        print("[evaluate_ood] WARNING: this dataset does not actually extend past "
              "--id_action_max — there is no real OOD region to test.")

    ds = LatentSurrogateDataset(
        trajs, state_mean=state_mean.cpu(), state_std=state_std.cpu(),
        action_mean=action_mean, action_std=action_std,
        n_ensemble=model.n_ensemble, bootstrap_seed=0,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    print("[evaluate_ood] Running forward pass over all transitions ...")
    data = collect_ood_samples(model, loader, state_mean, state_std,
                               action_mean, action_std, device)

    id_mask = (data["action"] >= args.id_action_min) & (data["action"] <= args.id_action_max)

    print(f"\n[evaluate_ood] {'═'*90}")
    print("[evaluate_ood] SUMMARY  (n, mean epistemic σ, mean aleatoric σ, RMSE, "
          "corr(epistemic σ, per-sample squared error))")
    print(f"[evaluate_ood] {'═'*90}")
    summarize_region(data, id_mask,  "ID")
    summarize_region(data, ~id_mask, "OOD")
    summarize_region(data, np.ones_like(id_mask), "ALL")
    print(f"[evaluate_ood] {'═'*90}\n")

    binned = bin_by_action(data, args.n_bins)
    print(f"  {'Power bin [W]':>16}  {'n':>7}  {'Epist σ':>10}  {'Aleat σ':>10}  {'RMSE [K]':>10}")
    for i in range(args.n_bins):
        lo, hi = binned["edges"][i], binned["edges"][i + 1]
        print(f"  {lo:7.1f}-{hi:7.1f}  {binned['counts'][i]:7d}  "
              f"{binned['epi'][i]:10.5f}  {binned['ale'][i]:10.5f}  {binned['rmse'][i]:10.2f}")

    plot_uncertainty_vs_action(binned, args.id_action_min, args.id_action_max,
                               os.path.join(out_dir, "ood_uncertainty_vs_action.png"))
    plot_epistemic_vs_error_scatter(data, id_mask,
                                    os.path.join(out_dir, "ood_epistemic_vs_error.png"))

    print(f"\n[evaluate_ood] Complete. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
