"""
surrogate_model_latent_uncertainty_v2/evaluate_ood_ratio.py
----------------------------------------------------------
Coverage-normalized OOD check: ratio of a PATCHY-trained checkpoint's
epistemic sigma to a FULL-range checkpoint's epistemic sigma, both evaluated
on the same wide dataset, plotted against laser power.

Motivation
----------
evaluate_ood.py shows epistemic sigma climbing with laser power even for a
GAPPED/patchy checkpoint's own OOD region -- but epistemic sigma also climbs
with laser power in absolute terms for the boring reason that higher power
is intrinsically harder to model (more nonlinearity/variance in the heating
physics), independent of whether that power was covered in training. A
single checkpoint's epistemic-vs-power curve can't separate these two
effects.

This script disentangles them by dividing that confound out: evaluate a
FULL-range checkpoint (trained on the same data_path with no
--lp_filter_min/--lp_filter_max/--lp_filter_ranges, so it has seen every
power level) on the identical wide dataset, and take

    ratio(power) = epistemic_sigma_patchy(power) / epistemic_sigma_full(power)

The full-range model's curve is the "intrinsic difficulty" baseline (it
still slopes upward with power -- physics is physics -- but has no
distribution-shift excuse, since it trained on everything). If the patchy
model's epistemic signal is really tracking data coverage rather than just
riding the same intrinsic-difficulty trend, `ratio` should sit near 1 across
the patchy model's ID range(s) and rise well above 1 in its OOD range(s) /
interior gaps -- a flatter, coverage-specific signal than either raw curve
shows alone.

Does not modify evaluate_ood.py or re-run any existing checkpoint/results;
imports its collect_ood_samples/bin_by_action helpers directly so the two
scripts stay numerically consistent on the parts they share.

Usage
-----
    python -m surrogate_model_latent_uncertainty_v2.evaluate_ood_ratio \\
        --checkpoint_patchy surrogate_model_latent_uncertainty_v2/runs/patchy_100-150_200-250_300-350_perturb0.1/two_stage_best.pt \\
        --checkpoint_full   surrogate_model_latent_uncertainty_v2/runs/full_range/two_stage_best.pt \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --id_ranges "100-150,200-250,300-350"

`--checkpoint_full` must exist already or be trained first -- same
architecture/K/bootstrap settings as the patchy run, same --data_path, just
WITHOUT any --lp_filter_* flag, e.g.:

    python -m surrogate_model_latent_uncertainty_v2.train \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --out_dir surrogate_model_latent_uncertainty_v2/runs/full_range

For a single-range (non-gapped) patchy checkpoint, use --id_action_min/--id_action_max
instead of --id_ranges (matches evaluate_ood.py's own flags).

Outputs (--out_dir, defaults to the patchy checkpoint's directory)
--------------------------------------------------------------------
  ood_epistemic_ratio_vs_action.png  -- epi_patchy/epi_full ratio vs. laser
                                        power, ID range(s) shaded, ratio=1
                                        reference line
  ood_uncertainty_summary_2x2.png    -- 2x2: raw epistemic sigma | raw
                                        aleatoric sigma (top, patchy vs.
                                        full-range overlaid), epistemic ratio
                                        | RMSE (bottom, RMSE also overlaid)
  Console table: per-bin epi_patchy, epi_full, epi ratio, aleatoric ratio
                 (control -- aleatoric sigma isn't a coverage signal, so it
                 shouldn't show the same ID/OOD split the epistemic ratio does)
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_latent_uncertainty_v2.dataset_v2 import load_trajectories, TwoStageLatentSurrogateDataset
from surrogate_model_latent_uncertainty_v2.train import load_two_stage_surrogate, _parse_lp_filter_ranges
from surrogate_model_latent_uncertainty_v2.evaluate_ood import collect_ood_samples, bin_by_action


# =============================================================================
# Per-checkpoint pass
# =============================================================================

def _run_one(checkpoint: str, trajs, args, device: str) -> dict:
    (model, state_mean, state_std, lp_mean, lp_std,
     cool_mean, cool_std, _roi) = load_two_stage_surrogate(checkpoint, device)
    print(f"[evaluate_ood_ratio]   {checkpoint}")
    print(f"[evaluate_ood_ratio]   {model}")

    ds = TwoStageLatentSurrogateDataset(
        trajs, state_mean=state_mean.cpu(), state_std=state_std.cpu(),
        lp_mean=lp_mean, lp_std=lp_std, cool_mean=cool_mean, cool_std=cool_std,
        initial_temp=args.initial_temp, n_ensemble=model.n_ensemble, bootstrap_seed=0,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    return collect_ood_samples(model, loader, state_mean, state_std, lp_mean, lp_std, device)


# =============================================================================
# Plotting
# =============================================================================

def plot_ratio(binned_patchy: dict, binned_full: dict, id_ranges, out_path: str,
              eps: float = 1e-8) -> np.ndarray:
    centers = binned_patchy["centers"]
    ratio = binned_patchy["epi"] / np.clip(binned_full["epi"], eps, None)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, ratio, marker="o", color="tab:orange", linewidth=1.5,
            label="epistemic σ ratio (patchy / full-range)")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1, label="ratio = 1")
    for i, (lo, hi) in enumerate(id_ranges):
        ax.axvspan(lo, hi, color="tab:green", alpha=0.08,
                  label="Patchy model's training range (ID)" if i == 0 else None)
        ax.axvline(lo, color="grey", linestyle="--", linewidth=1)
        ax.axvline(hi, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Laser power [W]")
    ax.set_ylabel("Epistemic σ ratio (patchy / full-range)")
    ax.set_title("Coverage-normalized epistemic signal\n"
                "(intrinsic-difficulty confound divided out -- want ~1 in ID, >>1 in OOD/gaps)",
                fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate_ood_ratio] Saved → {out_path}")
    return ratio


def _shade_id_ranges(ax, id_ranges, labeled: bool = True) -> None:
    for i, (lo, hi) in enumerate(id_ranges):
        ax.axvspan(lo, hi, color="tab:green", alpha=0.08,
                  label="ID (training range)" if (labeled and i == 0) else None)
        ax.axvline(lo, color="grey", linestyle="--", linewidth=1)
        ax.axvline(hi, color="grey", linestyle="--", linewidth=1)


def plot_summary_2x2(binned_patchy: dict, binned_full: dict, id_ranges, out_path: str,
                     eps: float = 1e-8) -> None:
    """2x2: raw epistemic | raw aleatoric (top), epistemic ratio | RMSE (bottom).
    Top row and bottom-right overlay patchy vs. full-range so it's visible
    WHERE the two curves diverge, not just the divided-out ratio."""
    centers = binned_patchy["centers"]
    ratio = binned_patchy["epi"] / np.clip(binned_full["epi"], eps, None)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # top-left: raw epistemic sigma
    ax = axes[0, 0]
    ax.plot(centers, binned_patchy["epi"], marker="o", color="tab:orange", linewidth=1.5, label="patchy")
    ax.plot(centers, binned_full["epi"],   marker="o", color="tab:blue",   linewidth=1.5, label="full-range")
    _shade_id_ranges(ax, id_ranges)
    ax.set_xlabel("Laser power [W]")
    ax.set_ylabel("Epistemic σ")
    ax.set_title("Raw epistemic σ (heating stage)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # top-right: raw aleatoric sigma
    ax = axes[0, 1]
    ax.plot(centers, binned_patchy["ale"], marker="o", color="tab:orange", linewidth=1.5, label="patchy")
    ax.plot(centers, binned_full["ale"],   marker="o", color="tab:blue",   linewidth=1.5, label="full-range")
    _shade_id_ranges(ax, id_ranges)
    ax.set_xlabel("Laser power [W]")
    ax.set_ylabel("Aleatoric σ")
    ax.set_title("Raw aleatoric σ (heating stage) -- control, not a coverage signal", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # bottom-left: epistemic ratio
    ax = axes[1, 0]
    ax.plot(centers, ratio, marker="o", color="tab:purple", linewidth=1.5, label="epi ratio (patchy/full)")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1, label="ratio = 1")
    _shade_id_ranges(ax, id_ranges)
    ax.set_xlabel("Laser power [W]")
    ax.set_ylabel("Epistemic σ ratio")
    ax.set_title("Coverage-normalized epistemic ratio", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # bottom-right: RMSE
    ax = axes[1, 1]
    ax.plot(centers, binned_patchy["rmse"], marker="o", color="tab:orange", linewidth=1.5, label="patchy")
    ax.plot(centers, binned_full["rmse"],   marker="o", color="tab:blue",   linewidth=1.5, label="full-range")
    _shade_id_ranges(ax, id_ranges)
    ax.set_xlabel("Laser power [W]")
    ax.set_ylabel("Single-step heating RMSE [K]")
    ax.set_title("Actual error (ground-truth vs. prediction)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Patchy vs. full-range surrogate -- uncertainty & error vs. laser power", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate_ood_ratio] Saved → {out_path}")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Coverage-normalized OOD check: ratio of a patchy-trained "
                    "checkpoint's epistemic sigma to a full-range checkpoint's, "
                    "vs. laser power, to separate the data-coverage (OOD) effect "
                    "from the intrinsic-difficulty-increases-with-power confound."
    )
    p.add_argument("--checkpoint_patchy", type=str, required=True,
                   help="Checkpoint trained on a restricted/gapped laser-power "
                        "range (train.py's --lp_filter_min/--lp_filter_max or "
                        "--lp_filter_ranges).")
    p.add_argument("--checkpoint_full", type=str, required=True,
                   help="Checkpoint trained on the SAME --data_path with NO "
                        "--lp_filter_min/--lp_filter_max/--lp_filter_ranges "
                        "(has seen the full power range).")
    p.add_argument("--data_path", type=str, required=True,
                   help="The wide dataset both checkpoints are evaluated on "
                        "(should be the dataset the patchy checkpoint's "
                        "training data was filtered from).")
    p.add_argument("--id_action_min", type=float, default=150.0,
                   help="Lower bound of the patchy checkpoint's training range "
                        "[W]. Ignored if --id_ranges is given.")
    p.add_argument("--id_action_max", type=float, default=300.0,
                   help="Upper bound of the patchy checkpoint's training range "
                        "[W]. Ignored if --id_ranges is given.")
    p.add_argument("--id_ranges", type=str, default=None,
                   help="For a GAPPED patchy checkpoint: comma-separated "
                        "'lo-hi' ranges matching its --lp_filter_ranges exactly, "
                        "e.g. '100-150,200-250,300-350'.")
    p.add_argument("--n_bins", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--out_dir", type=str, default="",
                   help="Defaults to the patchy checkpoint's directory.")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint_patchy))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[evaluate_ood_ratio] Output dir : {out_dir}")
    print(f"[evaluate_ood_ratio] Device     : {device}")

    id_ranges = (_parse_lp_filter_ranges(args.id_ranges) if args.id_ranges is not None
                else [(args.id_action_min, args.id_action_max)])
    range_str = ", ".join(f"[{lo}, {hi}]" for lo, hi in id_ranges)
    print(f"[evaluate_ood_ratio] Patchy checkpoint's ID range(s): {range_str} W")

    trajs = load_trajectories(args.data_path)
    all_actions = np.array([step.lp_action for traj in trajs for step in traj])
    print(f"[evaluate_ood_ratio] Wide dataset laser-power range: "
          f"[{all_actions.min():.1f}, {all_actions.max():.1f}] W")

    print("[evaluate_ood_ratio] Running PATCHY checkpoint forward pass ...")
    data_patchy = _run_one(args.checkpoint_patchy, trajs, args, device)
    print("[evaluate_ood_ratio] Running FULL-RANGE checkpoint forward pass ...")
    data_full = _run_one(args.checkpoint_full, trajs, args, device)

    binned_patchy = bin_by_action(data_patchy, args.n_bins)
    binned_full   = bin_by_action(data_full,   args.n_bins)

    eps = 1e-8
    epi_ratio = binned_patchy["epi"] / np.clip(binned_full["epi"], eps, None)
    ale_ratio = binned_patchy["ale"] / np.clip(binned_full["ale"], eps, None)

    print(f"\n[evaluate_ood_ratio] {'═' * 100}")
    print(f"  {'Power bin [W]':>16}  {'epi_patchy':>11}  {'epi_full':>10}  "
          f"{'epi ratio':>10}  {'ale ratio':>10}  {'region':>6}")
    for i in range(args.n_bins):
        lo, hi = binned_patchy["edges"][i], binned_patchy["edges"][i + 1]
        center = binned_patchy["centers"][i]
        is_id = any(rlo <= center <= rhi for rlo, rhi in id_ranges)
        print(f"  {lo:7.1f}-{hi:7.1f}  {binned_patchy['epi'][i]:11.5f}  "
              f"{binned_full['epi'][i]:10.5f}  {epi_ratio[i]:10.3f}  "
              f"{ale_ratio[i]:10.3f}  {'ID' if is_id else 'OOD':>6}")
    print(f"[evaluate_ood_ratio] {'═' * 100}")
    print("[evaluate_ood_ratio] 'ale ratio' is a control: aleatoric σ isn't a "
          "coverage signal, so it should NOT show the same ID/OOD split the "
          "epistemic ratio does. If it does, the effect is not coverage-specific.\n")

    plot_ratio(binned_patchy, binned_full, id_ranges,
              os.path.join(out_dir, "ood_epistemic_ratio_vs_action.png"))
    plot_summary_2x2(binned_patchy, binned_full, id_ranges,
                     os.path.join(out_dir, "ood_uncertainty_summary_2x2.png"))

    print(f"\n[evaluate_ood_ratio] Complete. Output in: {out_dir}")


if __name__ == "__main__":
    main()
