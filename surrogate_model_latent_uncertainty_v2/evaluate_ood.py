"""
surrogate_model_latent_uncertainty_v2/evaluate_ood.py
----------------------------------------------------------
Out-of-distribution (OOD) validation for the epistemic uncertainty channel
— the v2 (two-stage) analogue of surrogate_model_latent_uncertainty/evaluate_ood.py.

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
both sides — an interpolation-uncertainty test) and the outer edges. See
--id_ranges' help for the exact syntax.

Why this checks the HEATING stage specifically
---------------------------------------------------
Laser power only conditions the HEATING transition
(predict_heating_ensemble) — the cooling stage is conditioned on cool_time
instead (see model.py's docstring: "no matter what laser power you applied
previously, the cooling mechanism is the same"). So an LP-range OOD stress
test is inherently a heating-stage question: this script encodes s_t,
predicts the heating ensemble's Δz, decodes it, and compares against the
ground-truth end-of-heating field u_heat_t — exactly analogous to v1's
single-stage next-state check, just for the stage that actually depends on
the action being stressed here.

A working epistemic-uncertainty ensemble should show BOTH epistemic σ and
true single-step error climbing together as laser power moves into the OOD
region, and epistemic σ should positively correlate with per-sample error
overall. Aleatoric σ is plotted alongside as a control.

Usage
-----
    python -m surrogate_model_latent_uncertainty_v2.evaluate_ood \\
        --checkpoint surrogate_model_latent_uncertainty_v2/runs/narrow_150_300W/two_stage_best.pt \\
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

from surrogate_model_latent_uncertainty_v2.dataset_v2 import load_trajectories, TwoStageLatentSurrogateDataset
from surrogate_model_latent_uncertainty_v2.train import load_two_stage_surrogate, _parse_lp_filter_ranges


# =============================================================================
# Core evaluation
# =============================================================================

@torch.no_grad()
def collect_ood_samples(model, loader: DataLoader, state_mean, state_std,
                        lp_mean, lp_std, device: str):
    """
    Run every transition's HEATING stage through the model and collect, per
    sample: raw laser power [W], mean epistemic/aleatoric/total σ (latent,
    averaged over dims), and single-step heating RMSE-per-sample [K] against
    the ground-truth end-of-heating field.

    Returns a dict of 1-D numpy arrays, all length N.
    """
    model.eval()
    ss = state_std.to(device)
    sm = state_mean.to(device)

    actions, epi, ale, tot, sq_err, abs_err = [], [], [], [], [], []

    for s, a, _c, h, _s2, layer_idx, _bmask in loader:
        s         = s.to(device)
        a         = a.to(device)
        h         = h.to(device)
        layer_idx = layer_idx.to(device)

        z_t = model.encode(s)
        mu_heat, epi_std, ale_std, tot_std = model.predict_heating_ensemble(z_t, a, layer_idx)
        heat_pred = model.decode(z_t + mu_heat)

        diff_k = (heat_pred - h) * ss                        # (B, D) raw Kelvin
        a_raw  = (a.squeeze(-1) * lp_std + lp_mean)           # (B,) raw Watts

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

def plot_uncertainty_vs_action(binned: dict, id_ranges, out_path: str) -> None:
    """id_ranges: list of (lo, hi) ID ranges — a single-range checkpoint just
    passes a one-element list; each range gets its own shaded axvspan, so a
    GAPPED checkpoint's interior OOD gap shows up unshaded BETWEEN two shaded
    ID bands rather than as a single contiguous span."""
    centers = binned["centers"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [
        ("epi", "Heating epistemic σ (ensemble disagreement)", "tab:orange"),
        ("ale", "Heating aleatoric σ (avg. member noise)",      "tab:purple"),
        ("rmse", "Single-step heating RMSE [K]",                "tab:red"),
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
    fig.suptitle("OOD Stress Test (heating stage) — Uncertainty & Error vs. Laser Power "
                f"(shaded = training range(s) {range_str} W)",
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
    ax.set_xlabel("Heating epistemic σ (latent, mean over dims)")
    ax.set_ylabel("Per-sample heating RMSE [K]")
    ax.set_title("Epistemic Uncertainty vs. Actual Error (heating stage)")
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
        description="OOD stress test: evaluate a narrow-laser-power-range v2 checkpoint "
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

    (model, state_mean, state_std, lp_mean, lp_std,
     cool_mean, cool_std, _roi) = load_two_stage_surrogate(args.checkpoint, device)
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

    ds = TwoStageLatentSurrogateDataset(
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

    print(f"\n[evaluate_ood] Complete. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
