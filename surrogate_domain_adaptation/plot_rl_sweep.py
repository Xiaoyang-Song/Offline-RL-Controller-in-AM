"""
surrogate_domain_adaptation/plot_rl_sweep.py
--------------------------------------------
Aggregate per-K MATLAB evaluation results from the domain-adaptation + online
RL pipeline and produce comparison plots.

Pipeline recap
--------------
  For each K ∈ {0, 5, 10, 20, 50, 100}:
    1. adapt.py produces  <adapt_dir>/K<K>/adapted.pt
    2. online_RL/train.py produces  <adapt_dir>/K<K>/online_rl/dqn_best.pt
    3. test.py --results_out produces  <adapt_dir>/K<K>/online_rl/matlab_eval.json

  K=0 (base, no adaptation):
    online_rl/ and matlab_eval.json live at <adapt_dir>/K000/online_rl/

  This script reads all available matlab_eval.json files, then produces:
    return_vs_K.png         — total MATLAB return per K
    per_layer_rewards.png   — per-layer reward curves for each K
    per_layer_actions.png   — per-layer laser-power curves for each K
    metrics_table.txt       — numeric summary

Usage
-----
    python -m surrogate_domain_adaptation.plot_rl_sweep \\
        --adapt_dir surrogate_domain_adaptation/runs/base/<ts>/adapt_ct150

    # Include a custom K list and save plots elsewhere:
    python -m surrogate_domain_adaptation.plot_rl_sweep \\
        --adapt_dir <adapt_dir> \\
        --k_values 0 5 10 20 50 100 \\
        --out_dir  <some_dir>
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Colour palette (consistent with adapt.py evaluate plots)
# =============================================================================

_PALETTE = {
    0:   "#888888",
    5:   "#d62728",
    10:  "#ff7f0e",
    20:  "#2ca02c",
    50:  "#1f77b4",
    100: "#9467bd",
}

def _color(k: int) -> str:
    return _PALETTE.get(k, "#333333")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-K online RL MATLAB evaluation results and plot."
    )
    p.add_argument("--adapt_dir", type=str, required=True,
                   help="Directory produced by adapt.py (contains K005/, K010/, ...).")
    p.add_argument("--k_values", type=int, nargs="*", default=None,
                   help="K values to include. Defaults to all K<XXX>/ subdirs found.")
    p.add_argument("--out_dir", type=str, default="",
                   help="Output directory for plots (defaults to --adapt_dir).")
    return p.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def discover_results(adapt_dir: str, k_values) -> dict:
    """
    Return {K: result_dict} for every K that has a matlab_eval.json.
    result_dict keys: per_layer_rewards, per_layer_actions, total_return, ...
    """
    if k_values is None:
        # Auto-discover from K<XXX>/ subdirectories
        k_values = []
        for name in sorted(os.listdir(adapt_dir)):
            if name.startswith("K") and os.path.isdir(os.path.join(adapt_dir, name)):
                try:
                    k_values.append(int(name[1:]))
                except ValueError:
                    pass

    results = {}
    for K in sorted(k_values):
        json_path = os.path.join(adapt_dir, f"K{K:03d}", "online_rl", "matlab_eval.json")
        if not os.path.exists(json_path):
            print(f"[sweep] K={K:3d}: matlab_eval.json not found — skipping  ({json_path})")
            continue
        with open(json_path) as f:
            data = json.load(f)
        results[K] = data
        print(f"[sweep] K={K:3d}: total_return = {data['total_return']:+.4f}  "
              f"(cool_time={data.get('cool_time', '?')} s)")

    if not results:
        print("[sweep] No matlab_eval.json files found. "
              "Run the full pipeline first (see jobs/train_online_rl_adapted.sh).")
    return results


# =============================================================================
# Plots
# =============================================================================

def plot_return_vs_K(results: dict, out_path: str) -> None:
    """Bar chart of total MATLAB return per K."""
    k_vals = sorted(results.keys())
    returns = [results[k]["total_return"] for k in k_vals]
    colors  = [_color(k) for k in k_vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([str(k) for k in k_vals], returns, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, returns):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="green", linestyle="--", linewidth=1,
               label="Ideal (0 deviation)")
    ax.set_xlabel("K  (adaptation shots)")
    ax.set_ylabel("Total MATLAB return  (−mean deviation, 12 layers)")
    ax.set_title("Online RL performance vs. few-shot adaptation K\n"
                 "(evaluated on real MATLAB simulator, ct=0.15 s)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[sweep] → {out_path}")


def plot_per_layer(results: dict, key: str, ylabel: str, title: str, out_path: str) -> None:
    """Line plot of a per-layer metric for every K."""
    layers = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(11, 5))

    for K in sorted(results.keys()):
        vals = np.array(results[K][key])
        ls   = "--" if K == 0 else "-"
        mean = float(np.mean(vals))
        ax.plot(layers, vals, marker="o", color=_color(K), linestyle=ls,
                linewidth=1.8, markersize=5,
                label=f"K={K}  (mean={mean:.3f})")

    if key == "per_layer_rewards":
        ax.axhline(0, color="grey", linestyle=":", linewidth=1, alpha=0.6,
                   label="Ideal (0)")

    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[sweep] → {out_path}")


# =============================================================================
# Text summary
# =============================================================================

def write_table(results: dict, out_path: str) -> None:
    with open(out_path, "w") as f:
        f.write("Online RL sweep — MATLAB evaluation (ct=0.15 s)\n")
        f.write("=" * 55 + "\n")
        f.write(f"{'K':>6}  {'Total return':>14}  {'Mean/layer':>12}  "
                f"{'Min layer':>10}  {'Max layer':>10}\n")
        f.write("-" * 55 + "\n")
        for K in sorted(results.keys()):
            r  = np.array(results[K]["per_layer_rewards"])
            f.write(f"{K:>6}  {r.sum():>+14.4f}  {r.mean():>+12.4f}  "
                    f"{r.min():>+10.4f}  {r.max():>+10.4f}\n")
    print(f"[sweep] → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args    = parse_args()
    out_dir = args.out_dir or args.adapt_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[sweep] adapt_dir : {args.adapt_dir}")
    print(f"[sweep] out_dir   : {out_dir}")

    results = discover_results(args.adapt_dir, args.k_values)
    if not results:
        return

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_return_vs_K(
        results,
        os.path.join(out_dir, "return_vs_K.png"),
    )
    plot_per_layer(
        results,
        key      = "per_layer_rewards",
        ylabel   = "Reward (−meanDeviation)",
        title    = "Per-layer reward — MATLAB sim at ct=0.15 s  (K-shot adapted DQN)",
        out_path = os.path.join(out_dir, "per_layer_rewards.png"),
    )
    plot_per_layer(
        results,
        key      = "per_layer_actions",
        ylabel   = "Laser power [W]",
        title    = "Per-layer action sequence — MATLAB sim at ct=0.15 s",
        out_path = os.path.join(out_dir, "per_layer_actions.png"),
    )
    write_table(results, os.path.join(out_dir, "metrics_table.txt"))

    print(f"\n[sweep] All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
