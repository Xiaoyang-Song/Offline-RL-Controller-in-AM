"""
surrogate_domain_adaptation/evaluate.py
-----------------------------------------
Evaluation and plotting for the few-shot domain adaptation experiment.

Loads:
  - Base model (K=0)
  - Adapted models for each K in K_LIST
  - Held-out ct_150 test trajectories

Produces:
  few_shot_learning_curve.png   — mean MAE and RMSE vs K
  per_layer_mae.png             — per-layer MAE for base + each K
  per_layer_rmse.png            — per-layer RMSE for base + each K
  metrics_table.txt             — numeric summary

Usage
-----
    python -m surrogate_domain_adaptation.evaluate \\
        --base_checkpoint surrogate_domain_adaptation/runs/base/<ts>/base_best.pt \\
        --adapt_dir       surrogate_domain_adaptation/runs/base/<ts>/adapt_ct150
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_domain_adaptation.dataset import (
    load_target_domain,
    sample_few_shot,
    MultiDomainTrajectoryDataset,
    DATA_ROOT,
    TARGET_COOLING_TIME,
)
from surrogate_domain_adaptation.model import (
    AdaptedEnsembleLatentDynamicsModel,
    load_adapted,
)
from surrogate_domain_adaptation.adapt import evaluate_on_loader
from surrogate_domain_adaptation.train_base import load_base_checkpoint


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate and plot few-shot domain adaptation results."
    )
    p.add_argument("--base_checkpoint", type=str, required=True)
    p.add_argument("--adapt_dir",       type=str, required=True,
                   help="Directory produced by adapt.py (contains K005/, K010/, ...).")
    p.add_argument("--data_root",   type=str, default=DATA_ROOT)
    p.add_argument("--target_ct",   type=int, default=TARGET_COOLING_TIME)
    p.add_argument("--n_test",      type=int, default=90,
                   help="Number of held-out test trajectories.")
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--out_dir",      type=str,   default="",
                   help="Where to save plots. Defaults to --adapt_dir.")
    p.add_argument("--device",       type=str,   default="")
    return p.parse_args()


# =============================================================================
# Colour palette — consistent across plots
# =============================================================================

def _k_color(k_val):
    """Colour gradient from grey (K=0) to deep blue (large K)."""
    palette = {
        0:   "#888888",
        5:   "#d62728",
        10:  "#ff7f0e",
        20:  "#2ca02c",
        50:  "#1f77b4",
        100: "#9467bd",
    }
    return palette.get(k_val, "#333333")


# =============================================================================
# Plotting
# =============================================================================

def plot_learning_curve(
    k_vals:  list,   # [0, 5, 10, 20, 50, 100]
    maes:    list,   # mean MAE per K
    rmses:   list,   # mean RMSE per K
    out_path: str,
) -> None:
    """Few-shot learning curve: MAE and RMSE vs K."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, vals, metric, color in [
        (axes[0], maes,  "Mean MAE [K]",  "tab:blue"),
        (axes[1], rmses, "Mean RMSE [K]", "tab:orange"),
    ]:
        ax.plot(k_vals, vals, marker="o", color=color, linewidth=2.0, markersize=7)
        ax.axhline(vals[0], color="grey", linestyle=":", linewidth=1,
                   label=f"Base (K=0): {vals[0]:.2f} K")
        for k, v in zip(k_vals, vals):
            ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xlabel("K  (adaptation shots)"); ax.set_ylabel(metric)
        ax.set_title(f"Few-shot adaptation — {metric}")
        ax.set_xticks(k_vals); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Domain adaptation: ct_050–100 → ct_150", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[evaluate] → {out_path}")


def plot_per_layer(
    results: dict,   # {label: {"per_layer_mae": [...], "per_layer_rmse": [...]}}
    metric:  str,    # "per_layer_mae" or "per_layer_rmse"
    ylabel:  str,
    title:   str,
    out_path: str,
) -> None:
    """Per-layer metric for base + each K on one axes."""
    layers = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(11, 5))

    for label, data in results.items():
        k_val = data["K"]
        color = _k_color(k_val)
        ls    = "--" if k_val == 0 else "-"
        vals  = np.array(data[metric])
        ax.plot(layers, vals, marker="o", color=color, linestyle=ls,
                linewidth=1.5, markersize=5,
                label=f"K={k_val} (mean={vals.mean():.1f} K)")

    ax.set_xlabel("Layer index"); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(layers)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[evaluate] → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or args.adapt_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── load base model ───────────────────────────────────────────────────────
    print(f"[evaluate] Loading base: {args.base_checkpoint}")
    base_model, state_mean, state_std, action_mean, action_std, _ = \
        load_base_checkpoint(args.base_checkpoint, device=device)

    # ── fixed test set from target domain ─────────────────────────────────────
    target_trajs = load_target_domain(
        data_root=args.data_root, ct=args.target_ct
    )
    rng        = np.random.default_rng(args.seed)
    perm       = rng.permutation(len(target_trajs))
    test_trajs = [target_trajs[i] for i in perm[:args.n_test]]
    print(f"[evaluate] Test set: {len(test_trajs)} ct_{args.target_ct:03d} trajectories")

    ds_kw = dict(state_mean=state_mean, state_std=state_std,
                 action_mean=action_mean, action_std=action_std,
                 initial_temp=args.initial_temp)
    ev_loader = DataLoader(
        MultiDomainTrajectoryDataset(test_trajs, **ds_kw),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    # ── K=0 baseline ──────────────────────────────────────────────────────────
    base_mae, base_rmse = evaluate_on_loader(base_model, ev_loader, device, state_std)
    results = {
        "K000_base": {
            "K": 0,
            "per_layer_mae":  base_mae.tolist(),
            "per_layer_rmse": base_rmse.tolist(),
        }
    }

    # ── load adapted models from adapt_dir ────────────────────────────────────
    k_subdirs = sorted([
        d for d in os.listdir(args.adapt_dir)
        if d.startswith("K") and os.path.isdir(os.path.join(args.adapt_dir, d))
    ])

    for kdir in k_subdirs:
        ckpt_path = os.path.join(args.adapt_dir, kdir, "adapted.pt")
        if not os.path.exists(ckpt_path):
            continue

        K = int(kdir[1:])
        adapted = load_adapted(base_model, ckpt_path, device=device)
        mae, rmse = evaluate_on_loader(adapted, ev_loader, device, state_std)
        results[kdir] = {
            "K": K,
            "per_layer_mae":  mae.tolist(),
            "per_layer_rmse": rmse.tolist(),
        }
        print(f"[evaluate] K={K:3d} | MAE={mae.mean():.3f} K | RMSE={rmse.mean():.3f} K")

    # ── build sorted K lists ──────────────────────────────────────────────────
    sorted_keys = sorted(results.keys())
    k_vals = [results[k]["K"] for k in sorted_keys]
    maes   = [np.mean(results[k]["per_layer_mae"])  for k in sorted_keys]
    rmses  = [np.mean(results[k]["per_layer_rmse"]) for k in sorted_keys]

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_learning_curve(
        k_vals, maes, rmses,
        os.path.join(out_dir, "few_shot_learning_curve.png"),
    )
    plot_per_layer(
        results, "per_layer_mae", "MAE [K]",
        "Per-layer MAE — base vs K-shot adapted (ct_150 test set)",
        os.path.join(out_dir, "per_layer_mae.png"),
    )
    plot_per_layer(
        results, "per_layer_rmse", "RMSE [K]",
        "Per-layer RMSE — base vs K-shot adapted (ct_150 test set)",
        os.path.join(out_dir, "per_layer_rmse.png"),
    )

    # ── text summary ──────────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "metrics_table.txt")
    with open(summary_path, "w") as f:
        f.write(f"Domain adaptation: ct_050-100 → ct_{args.target_ct:03d}\n")
        f.write(f"Test set: {len(test_trajs)} held-out trajectories\n\n")
        f.write(f"{'K':>6}  {'Mean MAE [K]':>14}  {'Mean RMSE [K]':>14}\n")
        f.write("-" * 40 + "\n")
        for k, m, r in zip(k_vals, maes, rmses):
            f.write(f"{k:>6}  {m:>14.3f}  {r:>14.3f}\n")
    print(f"[evaluate] Metrics table → {summary_path}")

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\n{'K':>6}  {'Mean MAE [K]':>14}  {'Mean RMSE [K]':>14}")
    for k, m, r in zip(k_vals, maes, rmses):
        print(f"{k:>6}  {m:>14.3f}  {r:>14.3f}")

    print(f"\n[evaluate] All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
