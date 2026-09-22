"""
surrogate_model_v3/compare_uncertainty_methods.py
--------------------------------------------------------
Compares the uncertainty QUALITY produced by the different
diagonal-vs-low-rank / naive-vs-propagated combinations described in
model.py's module docstring, on the SAME test trajectories. This is not a
next-state accuracy comparison (see evaluate.py / summarize_results.py for
that) — every config here comes from the SAME underlying point predictions
whenever the checkpoint is shared; what differs is only how much
uncertainty each config reports and how it's structured.

Two independent tags, both optional to provide:
  --checkpoint_diag     : a checkpoint trained with --rank 0 (or omitted,
                           since 0 is train.py's default) — pure diagonal
                           per-member covariance.
  --checkpoint_lowrank  : a checkpoint trained with --rank R>0 — also has a
                           low-rank aleatoric factor per member.

For EACH checkpoint given, both propagate_uncertainty settings are
evaluated (naive-sum vs. EKF-style Jacobian propagation — see model.py's
`rollout`/`combine_stage_uncertainties_propagated`), so up to 4 configs are
compared: {diag, low-rank} x {naive, propagated}. Pass just one checkpoint
to compare only its 2 propagate settings; pass both to get the full 2x2.

Outputs (--out_dir)
--------------------
  per_layer_uncertainty_comparison.png  — epistemic | aleatoric | total,
      mean over test trajectories, per layer, all available configs overlaid.
      Shows whether propagation/low-rank makes uncertainty compound more
      realistically over a 12-layer rollout.
  uncertainty_vs_laser_power.png        — total/epistemic std vs. laser
      power (binned), all configs overlaid. Point this at a NARROW-trained
      checkpoint (e.g. surrogate_model_v3/runs/narrow_200_300W) and a wide
      --data_path to see whether propagation/low-rank sharpens the OOD
      uncertainty signal outside the training range.
  jacobian_amplification_map.png        — mean Jacobian diagonal
      (1 + d(mu_cool)/d(s_heat)) vs. layer, propagated configs only: >1
      means the cooling stage on average AMPLIFIES upstream heating
      uncertainty at that point in the build, <1 means it DAMPS it. A novel
      diagnostic this design makes available for free.
  uncertainty_summary.csv               — per-config, per-layer numbers.

Usage
-----
    python -m surrogate_model_v3.compare_uncertainty_methods \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --checkpoint_diag    surrogate_model_v3/runs/full_range/surrogate_best.pt \\
        --checkpoint_lowrank surrogate_model_v3/runs/full_range_rank8/surrogate_best.pt \\
        --out_dir surrogate_model_v3/results_uncertainty_comparison
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.dataset import load_trajectories, split_trajectories, TwoStageTrajectoryDataset
from surrogate_model_v3.model import load_surrogate


# =============================================================================
# Professional plot styling — consistent color per (checkpoint, propagate) config.
# =============================================================================

plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif",
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "legend.fontsize": 9, "legend.frameon": True, "legend.framealpha": 0.9,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

CONFIG_STYLE = {
    ("diag",     False): dict(color="#7f7f7f", linestyle="--", marker="o", linewidth=1.6),
    ("diag",     True):  dict(color="#1f77b4", linestyle="-",  marker="o", linewidth=2.0),
    ("lowrank",  False): dict(color="#bcbd22", linestyle="--", marker="s", linewidth=1.6),
    ("lowrank",  True):  dict(color="#d62728", linestyle="-",  marker="s", linewidth=2.4),
}
CONFIG_LABEL = {
    ("diag",    False): "diagonal, naive-sum (original)",
    ("diag",    True):  "diagonal, propagated (Jacobian)",
    ("lowrank", False): "low-rank, naive-sum",
    ("lowrank", True):  "low-rank, propagated (Jacobian)",
}


# =============================================================================
# Core evaluation: per-layer uncertainty + optional laser-power binning
# =============================================================================

@torch.no_grad()
def collect_rollout_uncertainty(model, traj_loader, device, propagate_uncertainty, num_probes, traj_len):
    """Runs model.rollout over every test trajectory batch with the given
    propagate_uncertainty setting. Returns per-layer mean epistemic/
    aleatoric/total std ((traj_len,) arrays), the amplification map's mean
    Jacobian diagonal per layer (None if propagate_uncertainty=False), and
    flat per-sample (laser_power_raw, total_std_mean_over_nodes) arrays for
    the laser-power binning plot.
    """
    model.eval()
    heat_epi_l = [[] for _ in range(traj_len)]
    heat_ale_l = [[] for _ in range(traj_len)]
    total_epi_l = [[] for _ in range(traj_len)]
    total_ale_l = [[] for _ in range(traj_len)]
    total_std_l = [[] for _ in range(traj_len)]
    jac_l       = [[] for _ in range(traj_len)] if propagate_uncertainty else None
    actions_flat, total_std_flat = [], []

    for traj_s, traj_h, traj_a, traj_c, _bmask in traj_loader:
        traj_s, traj_a, traj_c = traj_s.to(device), traj_a.to(device), traj_c.to(device)
        B, T1, D = traj_s.shape
        T = min(T1 - 1, traj_len)

        out = model.rollout(traj_s[:, 0, :], traj_a[:, :T, :], traj_c[:, :T, :],
                            propagate_uncertainty=propagate_uncertainty, num_probes=num_probes)

        for t in range(T):
            heat_epi_l[t].extend(out["heat_epistemic"][:, t].cpu().numpy().tolist())
            heat_ale_l[t].extend(out["heat_aleatoric"][:, t].cpu().numpy().tolist())
            total_epi_l[t].extend(out["total_epistemic"][:, t].cpu().numpy().tolist())
            total_ale_l[t].extend(out["total_aleatoric"][:, t].cpu().numpy().tolist())
            total_std_l[t].extend(out["total_std"][:, t].cpu().numpy().tolist())

            a_raw = traj_a[:, t, 0].cpu().numpy()  # normalised here; caller rescales if needed
            actions_flat.extend(a_raw.tolist())
            total_std_flat.extend(out["total_std"][:, t].cpu().numpy().tolist())

        if propagate_uncertainty:
            # Re-derive the mean Jacobian diagonal per layer by re-running the
            # combination once per layer on the rollout's own predicted
            # s_heat trajectory (rollout() doesn't expose jacobian_diag
            # per-layer directly to keep its return dict stable/simple).
            s_t = traj_s[:, 0, :]
            for t in range(T):
                a_t, c_t = traj_a[:, t, :], traj_c[:, t, :]
                layer_idx = torch.full((B,), t, dtype=torch.long, device=device)
                heat_full = model.predict_heating_ensemble_full(s_t, a_t, layer_idx)
                s_heat = s_t + heat_full["mu_mean"]
                cool_full = model.predict_cooling_ensemble_full(s_heat, c_t, layer_idx)
                combo = model.combine_stage_uncertainties_propagated(
                    s_heat, c_t, layer_idx, heat_full, cool_full, num_probes=num_probes)
                jac_l[t].extend(combo["jacobian_diag"].mean(dim=-1).cpu().numpy().tolist())
                s_t = s_heat + cool_full["mu_mean"]

    result = dict(
        heat_epi=np.array([np.mean(x) for x in heat_epi_l]),
        heat_ale=np.array([np.mean(x) for x in heat_ale_l]),
        total_epi=np.array([np.mean(x) for x in total_epi_l]),
        total_ale=np.array([np.mean(x) for x in total_ale_l]),
        total_std=np.array([np.mean(x) for x in total_std_l]),
        jacobian_diag=(np.array([np.mean(x) for x in jac_l]) if jac_l is not None else None),
        actions_norm=np.array(actions_flat),
        total_std_flat=np.array(total_std_flat),
    )
    return result


def bin_by_action(actions_raw: np.ndarray, values: np.ndarray, n_bins: int):
    edges = np.linspace(actions_raw.min(), actions_raw.max(), n_bins + 1)
    idx = np.clip(np.digitize(actions_raw, edges[1:-1]), 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() > 0:
            means[b] = values[mask].mean()
    return centers, means


# =============================================================================
# Plotting
# =============================================================================

def plot_per_layer_comparison(results: dict, traj_len: int, out_path: str):
    layers = np.arange(1, traj_len + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panels = [("total_epi", "Combined epistemic σ"), ("total_ale", "Combined aleatoric σ"),
              ("total_std", "Combined total σ (read by RL)")]
    for ax, (key, title) in zip(axes, panels):
        for cfg, res in results.items():
            style = CONFIG_STYLE.get(cfg, dict(color="grey", linestyle="-", marker="o", linewidth=1.4))
            ax.plot(layers, res[key], label=CONFIG_LABEL.get(cfg, str(cfg)), markersize=6, **style)
        ax.set_xlabel("Layer index"); ax.set_ylabel("Mean σ (normalised space)")
        ax.set_title(title); ax.set_xticks(layers)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Two-stage combined uncertainty per layer — diagonal vs. low-rank, naive vs. propagated")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[compare_uncertainty] Saved → {out_path}")


def plot_vs_laser_power(results: dict, lp_mean: float, lp_std: float, out_path: str,
                        id_range=None, n_bins: int = 12):
    fig, ax = plt.subplots(figsize=(10, 6))
    if id_range is not None:
        ax.axvspan(id_range[0], id_range[1], color="#2ca02c", alpha=0.10, zorder=0,
                  label=f"Training range [{id_range[0]:.0f}, {id_range[1]:.0f}] W")
    for cfg, res in results.items():
        actions_raw = res["actions_norm"] * lp_std + lp_mean
        centers, means = bin_by_action(actions_raw, res["total_std_flat"], n_bins)
        style = CONFIG_STYLE.get(cfg, dict(color="grey", linestyle="-", marker="o", linewidth=1.4))
        ax.plot(centers, means, label=CONFIG_LABEL.get(cfg, str(cfg)), markersize=6.5, **style)
    ax.set_xlabel("Laser power [W]")
    ax.set_ylabel("Mean combined total σ (normalised space)")
    ax.set_title("Combined uncertainty vs. laser power — diagonal vs. low-rank, naive vs. propagated")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[compare_uncertainty] Saved → {out_path}")


def plot_amplification_map(results: dict, traj_len: int, out_path: str):
    propagated = {cfg: res for cfg, res in results.items() if res["jacobian_diag"] is not None}
    if not propagated:
        print("[compare_uncertainty] No propagated configs available — skipping amplification map.")
        return
    layers = np.arange(1, traj_len + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.2, label="J = 1 (unchanged)")
    for cfg, res in propagated.items():
        style = CONFIG_STYLE.get(cfg, dict(color="grey", linestyle="-", marker="o", linewidth=1.4))
        ax.plot(layers, res["jacobian_diag"], label=CONFIG_LABEL.get(cfg, str(cfg)), markersize=6, **style)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean Jacobian diagonal  1 + ∂μ_cool/∂s_heat")
    ax.set_title("Cooling-stage uncertainty amplification map\n"
                "(>1: cooling amplifies upstream heating error; <1: cooling damps it)", fontsize=11)
    ax.set_xticks(layers)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[compare_uncertainty] Saved → {out_path}")


# =============================================================================
# Argument parsing / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare uncertainty produced by diagonal vs. low-rank covariance and "
                    "naive-sum vs. Jacobian-propagated two-stage combination."
    )
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--checkpoint_diag", type=str, default=None,
                   help="Checkpoint trained with --rank 0 (or default).")
    p.add_argument("--checkpoint_lowrank", type=str, default=None,
                   help="Checkpoint trained with --rank > 0.")
    p.add_argument("--val_fraction", type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--num_probes", type=int, default=4,
                   help="Hutchinson probe count for the propagated configs' residual diagonal.")
    p.add_argument("--n_bins", type=int, default=12, help="Laser-power bins for uncertainty_vs_laser_power.png.")
    p.add_argument("--id_range_min", type=float, default=None,
                   help="Shade this training range on uncertainty_vs_laser_power.png (e.g. 200).")
    p.add_argument("--id_range_max", type=float, default=None, help="See --id_range_min (e.g. 300).")
    p.add_argument("--out_dir", type=str, default="surrogate_model_v3/results_uncertainty_comparison")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[compare_uncertainty] Output dir : {args.out_dir}")
    print(f"[compare_uncertainty] Device     : {device}")

    if not args.checkpoint_diag and not args.checkpoint_lowrank:
        raise ValueError("Pass at least one of --checkpoint_diag / --checkpoint_lowrank.")

    all_trajs = load_trajectories(args.data_path)
    _train, _val, test_trajs = split_trajectories(
        all_trajs, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
    )
    traj_len = len(test_trajs[0])

    results = {}
    lp_mean_ref, lp_std_ref = None, None

    checkpoints = [("diag", args.checkpoint_diag), ("lowrank", args.checkpoint_lowrank)]
    for tag, ckpt_path in checkpoints:
        if not ckpt_path:
            continue
        model, sm, ss, lm, ls, cm, cs = load_surrogate(ckpt_path, device)
        print(f"[compare_uncertainty] Loaded [{tag}] {ckpt_path}")
        print(f"[compare_uncertainty]   {model}")
        if model.rank == 0 and tag == "lowrank":
            print(f"[compare_uncertainty] WARNING: --checkpoint_lowrank has rank=0 — "
                  f"naive/propagated will still differ (free epistemic-factor propagation) "
                  f"but there's no low-rank aleatoric term.")
        lp_mean_ref, lp_std_ref = lm, ls

        ds = TwoStageTrajectoryDataset(
            test_trajs, state_mean=sm.cpu(), state_std=ss.cpu(), lp_mean=lm, lp_std=ls,
            cool_mean=cm, cool_std=cs, initial_temp=args.initial_temp, n_ensemble=model.n_ensemble,
            bootstrap_seed=args.seed,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        for propagate in (False, True):
            print(f"[compare_uncertainty] Running rollout: config=({tag}, propagate={propagate}) ...")
            res = collect_rollout_uncertainty(model, loader, device, propagate, args.num_probes, traj_len)
            results[(tag, propagate)] = res
            print(f"[compare_uncertainty]   mean total σ (last layer) = {res['total_std'][-1]:.5f}")

    id_range = (args.id_range_min, args.id_range_max) if args.id_range_min is not None else None

    plot_per_layer_comparison(results, traj_len, os.path.join(args.out_dir, "per_layer_uncertainty_comparison.png"))
    plot_vs_laser_power(results, lp_mean_ref, lp_std_ref,
                        os.path.join(args.out_dir, "uncertainty_vs_laser_power.png"),
                        id_range=id_range, n_bins=args.n_bins)
    plot_amplification_map(results, traj_len, os.path.join(args.out_dir, "jacobian_amplification_map.png"))

    csv_path = os.path.join(args.out_dir, "uncertainty_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "layer", "heat_epi", "heat_ale", "total_epi", "total_ale", "total_std", "jacobian_diag"])
        for (tag, propagate), res in results.items():
            for t in range(traj_len):
                jac = res["jacobian_diag"][t] if res["jacobian_diag"] is not None else ""
                writer.writerow([f"{tag}_{'propagated' if propagate else 'naive'}", t + 1,
                                 res["heat_epi"][t], res["heat_ale"][t],
                                 res["total_epi"][t], res["total_ale"][t], res["total_std"][t], jac])
    print(f"[compare_uncertainty] Saved → {csv_path}")

    print(f"\n[compare_uncertainty] Complete. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
