"""
surrogate_model_v3/evaluate_ood.py
------------------------------------
Out-of-distribution (OOD) validation for the epistemic uncertainty channel.

Motivation
----------
Evaluating a checkpoint on the train/val/test split of the SAME dataset it
was trained on can't show whether epistemic uncertainty is doing its job:
all three splits are drawn from the same action-power distribution, so
there's no real distribution shift for the K bootstrap-diversified members
to disagree about — everything looks "in-distribution" by construction.

This script instead evaluates a model trained on a NARROW laser-power range
(e.g. 150-300 W, via train.py's --lp_filter_min/--lp_filter_max) against the
WIDER dataset it was filtered FROM (e.g. the full 100-400 W data) — no new
simulation needed, the wider coverage already exists on disk. Transitions
are split into:

  ID  (in-distribution)     : laser power in [id_action_min, id_action_max]
                               — the range the checkpoint actually trained on
  OOD (out-of-distribution) : laser power outside that range — never seen

For a checkpoint trained with --lp_filter_ranges (a GAPPED surrogate, e.g.
[150,200] U [300,350]), pass the same ranges via --id_ranges instead of
--id_action_min/--id_action_max — ID becomes the UNION of those ranges, and
OOD includes both the interior gap (e.g. (200, 300), bracketed by ID data on
both sides — an interpolation-uncertainty test) and the outer edges.

Why this checks the HEATING stage specifically
---------------------------------------------------
Laser power only conditions the HEATING transition
(predict_heating_ensemble) — the cooling stage is conditioned on cool_time
instead (no matter what laser power you applied previously, the cooling
mechanism is the same). So an LP-range OOD stress test is inherently a
heating-stage question: this script predicts the heating ensemble's Δ and
compares against the ground-truth end-of-heating field u_heat_t.

A working epistemic-uncertainty ensemble should show BOTH epistemic σ and
true single-step error climbing together as laser power moves into the OOD
region, and epistemic σ should positively correlate with per-sample error
overall. Aleatoric σ is plotted alongside as a control.

Usage
-----
    python -m surrogate_model_v3.evaluate_ood \\
        --checkpoint surrogate_model_v3/runs/narrow_150_300W/surrogate_best.pt \\
        --data_path  Data/DatasetV2_layer_12_samples_5000.pkl \\
        --id_action_min 150 --id_action_max 300

Outputs (--out_dir, defaults to checkpoint directory)
------------------------------------------------------
  ood_uncertainty_vs_action.png   — epistemic | aleatoric | RMSE vs laser power, OOD region shaded
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

from surrogate_model_v3.dataset import load_trajectories, TwoStageSurrogateDataset
from surrogate_model_v3.model import (
    load_surrogate, combine_stage_uncertainties, combine_stage_uncertainties_naive_full,
)
from surrogate_model_v3.train import _parse_lp_filter_ranges


# =============================================================================
# Core evaluation
# =============================================================================

@torch.no_grad()
def collect_ood_samples(model, loader: DataLoader, state_mean, state_std,
                        lp_mean, lp_std, device: str):
    """
    Run every transition's HEATING stage through the model and collect, per
    sample: raw laser power [W], mean epistemic/aleatoric/total σ (averaged
    over nodes), and single-step heating RMSE-per-sample [K] against the
    ground-truth end-of-heating field.

    Returns a dict of 1-D numpy arrays, all length N.
    """
    model.eval()
    ss = state_std.to(device)

    actions, epi, ale, tot, sq_err, abs_err = [], [], [], [], [], []

    for s, a, _c, h, _s2, layer_idx, _bmask in loader:
        s         = s.to(device)
        a         = a.to(device)
        h         = h.to(device)
        layer_idx = layer_idx.to(device)

        mu_heat, epi_std, ale_std, tot_std = model.predict_heating_ensemble(s, a, layer_idx)
        heat_pred = s + mu_heat

        diff_k = (heat_pred - h) * ss                         # (B, D) raw Kelvin
        a_raw  = (a.squeeze(-1) * lp_std + lp_mean)            # (B,) raw Watts

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


@torch.no_grad()
def collect_ood_samples_combined(
    model, loader: DataLoader, state_mean, state_std, lp_mean, lp_std, device: str,
    propagate_uncertainty: bool = False, num_probes: int = 4,
):
    """
    Like `collect_ood_samples`, but chains heating THEN cooling and reports
    the COMBINED post-cooling uncertainty (epistemic/aleatoric/total) and
    the final next-state RMSE against the ground-truth s_{t+1} — not just
    the heating stage alone. Laser power only conditions heating directly,
    but s_heat (cooling's input) depends on it, so the combined uncertainty
    RL/MPC code actually reads (see model.py's `combine_stage_uncertainties`
    docstring) still varies with laser power indirectly through the chain —
    this is the number relevant to "how uncertain is the surrogate about
    this transition," not just "how uncertain is the heating stage."

    propagate_uncertainty=False (default): naive combination (still uses
    the `_full` path so a low-rank checkpoint's aleatoric factor is
    correctly included — see combine_stage_uncertainties_naive_full).
    propagate_uncertainty=True: EKF-style Jacobian propagation through
    cooling (see model.py's `combine_stage_uncertainties_propagated`).

    Returns a dict of 1-D numpy arrays, all length N (same keys as
    `collect_ood_samples`).
    """
    model.eval()
    ss = state_std.to(device)
    use_full = propagate_uncertainty or (model.rank > 0)

    actions, epi, ale, tot, sq_err, abs_err = [], [], [], [], [], []

    for s, a, c, _h, s2, layer_idx, _bmask in loader:
        s, a, c, s2 = s.to(device), a.to(device), c.to(device), s2.to(device)
        layer_idx   = layer_idx.to(device)

        if use_full:
            heat_full = model.predict_heating_ensemble_full(s, a, layer_idx)
            s_heat    = s + heat_full["mu_mean"]
            cool_full = model.predict_cooling_ensemble_full(s_heat, c, layer_idx)
            mu_cool   = cool_full["mu_mean"]
            if propagate_uncertainty:
                combo = model.combine_stage_uncertainties_propagated(
                    s_heat, c, layer_idx, heat_full, cool_full, num_probes=num_probes)
            else:
                combo = combine_stage_uncertainties_naive_full(heat_full, cool_full)
            epi_std, ale_std, tot_std = combo["epistemic_std"], combo["aleatoric_std"], combo["total_std"]
        else:
            mu_heat, heat_epi, heat_ale, _ = model.predict_heating_ensemble(s, a, layer_idx)
            s_heat = s + mu_heat
            mu_cool, cool_epi, cool_ale, _ = model.predict_cooling_ensemble(s_heat, c, layer_idx)
            epi_std, ale_std, tot_std = combine_stage_uncertainties(heat_epi, heat_ale, cool_epi, cool_ale)

        s_next_pred = s_heat + mu_cool
        diff_k = (s_next_pred - s2) * ss                       # (B, D) raw Kelvin
        a_raw  = (a.squeeze(-1) * lp_std + lp_mean)             # (B,) raw Watts

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

def plot_uncertainty_vs_action(binned: dict, id_ranges, out_path: str, stage_label: str = "Heating") -> None:
    """id_ranges: list of (lo, hi) ID ranges — a single-range checkpoint just
    passes a one-element list; each range gets its own shaded axvspan, so a
    GAPPED checkpoint's interior OOD gap shows up unshaded BETWEEN two shaded
    ID bands rather than as a single contiguous span. `stage_label` (default
    "Heating", unchanged) is swapped to e.g. "Combined" by the caller that
    reports the full post-cooling uncertainty instead of heating alone."""
    centers = binned["centers"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [
        ("epi", f"{stage_label} epistemic σ (ensemble disagreement)", "tab:orange"),
        ("ale", f"{stage_label} aleatoric σ (avg. member noise)",      "tab:purple"),
        ("rmse", f"Single-step {stage_label.lower()} RMSE [K]",       "tab:red"),
    ]
    for ax, (key, title, color) in zip(axes, panels):
        ax.plot(centers, binned[key], marker="o", color=color, linewidth=1.5)
        for i, (lo, hi) in enumerate(id_ranges):
            ax.axvspan(lo, hi, color="tab:green", alpha=0.08,
                      label="Training range (ID)" if i == 0 else None)
            ax.axvline(hi, color="grey", linestyle="--", linewidth=1)
            ax.axvline(lo, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("Laser power [W]")
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    range_str = ", ".join(f"[{lo:.0f}, {hi:.0f}]" for lo, hi in id_ranges)
    fig.suptitle(f"OOD Stress Test ({stage_label.lower()} stage) — Uncertainty & Error vs. Laser Power "
                f"(shaded = training range(s) {range_str} W)",
                fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate_ood] Saved → {out_path}")


def plot_epistemic_vs_error_scatter(data: dict, id_mask: np.ndarray, out_path: str,
                                    max_points: int = 20000, stage_label: str = "Heating") -> None:
    rng = np.random.default_rng(0)
    n   = len(data["epi"])
    idx = rng.choice(n, size=min(n, max_points), replace=False) if n > max_points else np.arange(n)

    err = np.sqrt(data["sq_err"][idx])
    epi = data["epi"][idx]
    id_m = id_mask[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(epi[id_m],  err[id_m],  s=6, alpha=0.35, color="tab:blue",   label="ID (training range)")
    ax.scatter(epi[~id_m], err[~id_m], s=6, alpha=0.35, color="tab:red",    label="OOD (unseen power)")
    ax.set_xlabel(f"{stage_label} epistemic σ (mean over nodes) [K]")
    ax.set_ylabel(f"Per-sample {stage_label.lower()} RMSE [K]")
    ax.set_title(f"Epistemic Uncertainty vs. Actual Error ({stage_label.lower()} stage)")
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
        description="OOD stress test: evaluate a narrow-laser-power-range checkpoint "
                    "on the wider dataset it was filtered from, to validate the heating "
                    "stage's epistemic uncertainty."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Checkpoint trained on a NARROW laser-power range "
                        "(train.py's --lp_filter_min/--lp_filter_max).")
    p.add_argument("--data_path",  type=str, required=True,
                   help="The WIDER dataset the narrow checkpoint's training data was "
                        "filtered FROM (e.g. Data/DatasetV2_layer_12_samples_5000.pkl).")
    p.add_argument("--id_action_min", type=float, default=150.0,
                   help="Lower bound of the checkpoint's training laser-power range [W] "
                        "(should match its --lp_filter_min).")
    p.add_argument("--id_action_max", type=float, default=300.0,
                   help="Upper bound of the checkpoint's training laser-power range [W] "
                        "(should match its --lp_filter_max). Actions outside "
                        "[id_action_min, id_action_max] are treated as OOD. Ignored if "
                        "--id_ranges is given.")
    p.add_argument("--id_ranges", type=str, default=None,
                   help="For a GAPPED checkpoint trained with train.py's --lp_filter_ranges: "
                        "comma-separated 'lo-hi' ranges, e.g. '150-200,300-350' (should match "
                        "the checkpoint's --lp_filter_ranges exactly). ID = union of these "
                        "ranges; overrides --id_action_min/--id_action_max when given.")
    p.add_argument("--n_bins",      type=int, default=12)
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--propagate_uncertainty", action="store_true",
                   help="Use EKF-style Jacobian propagation (see model.py's "
                        "combine_stage_uncertainties_propagated) for the COMBINED "
                        "(post-cooling) analysis instead of the default naive sum. "
                        "The heating-only analysis is unaffected by this flag.")
    p.add_argument("--num_probes", type=int, default=4,
                   help="Hutchinson probe count for --propagate_uncertainty's residual diagonal.")
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

    model, state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std = load_surrogate(
        args.checkpoint, device
    )
    print(f"[evaluate_ood] {model}")

    id_ranges = (_parse_lp_filter_ranges(args.id_ranges) if args.id_ranges is not None
                else [(args.id_action_min, args.id_action_max)])
    range_str = ", ".join(f"[{lo}, {hi}]" for lo, hi in id_ranges)

    trajs = load_trajectories(args.data_path)
    all_actions = np.array([step.lp_action for traj in trajs for step in traj])
    print(f"[evaluate_ood] Wide dataset laser-power range: "
          f"[{all_actions.min():.1f}, {all_actions.max():.1f}] W  "
          f"(checkpoint trained on {range_str} W)")
    all_id = np.array([any(lo <= a <= hi for lo, hi in id_ranges) for a in all_actions])
    if all_id.all():
        print("[evaluate_ood] WARNING: this dataset does not actually extend past "
              "the training range(s) — there is no real OOD region to test.")

    ds = TwoStageSurrogateDataset(
        trajs, state_mean=state_mean.cpu(), state_std=state_std.cpu(),
        lp_mean=lp_mean, lp_std=lp_std, cool_mean=cool_mean, cool_std=cool_std,
        initial_temp=args.initial_temp, n_ensemble=model.n_ensemble, bootstrap_seed=0,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    print("[evaluate_ood] Running forward pass over all transitions (heating stage) ...")
    data = collect_ood_samples(model, loader, state_mean, state_std, lp_mean, lp_std, device)

    id_mask = np.zeros_like(data["action"], dtype=bool)
    for lo, hi in id_ranges:
        id_mask |= (data["action"] >= lo) & (data["action"] <= hi)

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

    plot_uncertainty_vs_action(binned, id_ranges,
                               os.path.join(out_dir, "ood_uncertainty_vs_action.png"))
    plot_epistemic_vs_error_scatter(data, id_mask,
                                    os.path.join(out_dir, "ood_epistemic_vs_error.png"))

    # ── COMBINED (post-cooling) analysis — what RL/MPC actually reads ───────
    print(f"\n[evaluate_ood] Running forward pass over all transitions (combined heat+cool, "
          f"propagate_uncertainty={args.propagate_uncertainty}) ...")
    data_c = collect_ood_samples_combined(
        model, loader, state_mean, state_std, lp_mean, lp_std, device,
        propagate_uncertainty=args.propagate_uncertainty, num_probes=args.num_probes,
    )

    print(f"\n[evaluate_ood] {'═'*90}")
    print("[evaluate_ood] SUMMARY — COMBINED (post-cooling)")
    print(f"[evaluate_ood] {'═'*90}")
    summarize_region(data_c, id_mask,  "ID")
    summarize_region(data_c, ~id_mask, "OOD")
    summarize_region(data_c, np.ones_like(id_mask), "ALL")
    print(f"[evaluate_ood] {'═'*90}\n")

    binned_c = bin_by_action(data_c, args.n_bins)
    print(f"  {'Power bin [W]':>16}  {'n':>7}  {'Epist σ':>10}  {'Aleat σ':>10}  {'RMSE [K]':>10}")
    for i in range(args.n_bins):
        lo, hi = binned_c["edges"][i], binned_c["edges"][i + 1]
        print(f"  {lo:7.1f}-{hi:7.1f}  {binned_c['counts'][i]:7d}  "
              f"{binned_c['epi'][i]:10.5f}  {binned_c['ale'][i]:10.5f}  {binned_c['rmse'][i]:10.2f}")

    plot_uncertainty_vs_action(binned_c, id_ranges,
                               os.path.join(out_dir, "ood_uncertainty_vs_action_combined.png"),
                               stage_label="Combined")
    plot_epistemic_vs_error_scatter(data_c, id_mask,
                                    os.path.join(out_dir, "ood_epistemic_vs_error_combined.png"),
                                    stage_label="Combined")

    print(f"\n[evaluate_ood] Complete. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
