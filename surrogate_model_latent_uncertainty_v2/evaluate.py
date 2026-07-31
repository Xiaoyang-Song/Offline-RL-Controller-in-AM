"""
surrogate_model_latent_uncertainty_v2/evaluate.py
------------------------------------------------------
Evaluation and visualisation for the trained two-stage (heating/cooling)
LPBF Latent-Space Gaussian Ensemble surrogate.

Plot style matches surrogate_model_latent_uncertainty/evaluate.py:
  jet colormap, caxis [300, 5000], Gouraud shading, triplot mesh overlay,
  hot colormap for error field.

Two kinds of per-layer metrics are reported for EACH stage (heating, cooling):
  - single-step (teacher-forced): each stage is fed its GROUND-TRUTH input
    (matches how the model is trained — the cooling stage is always
    conditioned on the true u_heat_t, never on the heating stage's own
    prediction, during single-step training). Isolates each stage's own error.
  - rollout (auto-regressive): the two stages are chained using the model's
    OWN predictions (model.rollout), matching actual inference-time behaviour.

Metrics
-------
  1. Single-step MAE / RMSE per layer, heating stage and cooling stage
  2. Auto-regressive rollout RMSE per layer, heating stage and cooling stage
  3. Per-layer mean epistemic / aleatoric σ for both stages
  4. Autoencoder round-trip RMSE for s_t and for u_heat_t
  5. Per-component test-set loss (recon_s, recon_heat_ae, recon_heat, nll_heat, recon_cool, nll_cool)

Plots saved to --out_dir (defaults to checkpoint directory)
------------------------------------------------------------
  per_layer_rmse.png             — 2 panels: heating | cooling
  per_layer_mae.png              — 2 panels: heating | cooling
  per_layer_uncertainty.png      — 2 rows (heating/cooling) × 3 cols (epistemic/aleatoric/total)
  traj_<i>/actions.txt
  traj_<i>/action_sequence.png
  traj_<i>/sigma_per_layer.png
  traj_<i>/layer_<l>_LP<p>W.png  — 2×3 grid: [heat GT|Pred|Err] / [cool GT|Pred|Err]

Usage
-----
    python -m surrogate_model_latent_uncertainty_v2.evaluate \\
        --checkpoint surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt \\
        --data_path  Data/DatasetV2_layer_12_samples_200.pkl
"""

import argparse
import os
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from collections import defaultdict
from scipy.io import loadmat
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_latent_uncertainty_v2.dataset_v2 import (
    load_trajectories,
    split_trajectories,
    TwoStageLatentTrajectoryDataset,
    TwoStageLatentSurrogateDataset,
)
from surrogate_model_latent_uncertainty_v2.train import (
    load_two_stage_surrogate, compute_single_step_losses, weighted_mse,
)


# =============================================================================
# Mesh loading  (identical to surrogate_model_latent_uncertainty/evaluate.py)
# =============================================================================

def load_mesh(mesh_path: str):
    if not os.path.exists(mesh_path):
        print(f"[evaluate] mesh.mat not found at {mesh_path}. "
              "Run surrogate_model/extract_mesh.m in MATLAB first for 2-D plots. "
              "Falling back to 1-D node-index plots.")
        return None
    data     = loadmat(mesh_path)
    nodes    = data["nodes"]
    elements = data["elements"] - 1
    x   = nodes[0, :].astype(np.float64)
    y   = nodes[1, :].astype(np.float64)
    tri = elements[:3, :].T.astype(np.int32)
    triang = mtri.Triangulation(x, y, tri)
    print(f"[evaluate] Mesh loaded: {len(x)} nodes, {len(tri)} triangles. "
          f"X∈[{x.min():.2f},{x.max():.2f}], Y∈[{y.min():.2f},{y.max():.2f}]")
    return triang


class _DropBootstrapMask:
    """Wraps a DataLoader yielding (s, h, a, c, bmask) and drops bmask, so
    metric functions written for (s, h, a, c) can be reused unchanged."""
    def __init__(self, loader: DataLoader):
        self._loader = loader

    def __iter__(self):
        for s, h, a, c, _bmask in self._loader:
            yield s, h, a, c

    def __len__(self):
        return len(self._loader)


# =============================================================================
# Core evaluation
# =============================================================================

@torch.no_grad()
def evaluate_single_step(
    model,
    traj_loader,
    device:   str,
    traj_len: int = 12,
):
    """
    Teacher-forced 1-step evaluation. Each stage is fed its own GROUND-TRUTH
    input (cooling stage conditions on encode(h_gt), not the heating stage's
    prediction) — matches the training regime and isolates each stage's error.
    """
    model.eval()
    heat_abs = [[] for _ in range(traj_len)]
    heat_sq  = [[] for _ in range(traj_len)]
    cool_abs = [[] for _ in range(traj_len)]
    cool_sq  = [[] for _ in range(traj_len)]

    for traj_s, traj_h, traj_a, traj_c in traj_loader:
        traj_s = traj_s.to(device)
        traj_h = traj_h.to(device)
        traj_a = traj_a.to(device)
        traj_c = traj_c.to(device)
        B, T1, D = traj_s.shape
        T = min(T1 - 1, traj_len)

        for t in range(T):
            s_in = traj_s[:, t,     :]
            a_in = traj_a[:, t,     :]
            c_in = traj_c[:, t,     :]
            h_gt = traj_h[:, t,     :]
            s_gt = traj_s[:, t + 1, :]
            layer_idx = torch.full((s_in.shape[0],), t, dtype=torch.long, device=device)

            z_t = model.encode(s_in)
            mu_heat, _, _, _ = model.predict_heating_ensemble(z_t, a_in, layer_idx)
            heat_pred = model.decode(z_t + mu_heat)
            diff_heat = heat_pred - h_gt
            heat_abs[t].extend(diff_heat.abs().mean(dim=-1).cpu().numpy().tolist())
            heat_sq[t].extend((diff_heat ** 2).mean(dim=-1).cpu().numpy().tolist())

            z_heat_gt = model.encode(h_gt)
            mu_cool, _, _, _ = model.predict_cooling_ensemble(z_heat_gt, c_in, layer_idx)
            next_pred = model.decode(z_heat_gt + mu_cool)
            diff_cool = next_pred - s_gt
            cool_abs[t].extend(diff_cool.abs().mean(dim=-1).cpu().numpy().tolist())
            cool_sq[t].extend((diff_cool ** 2).mean(dim=-1).cpu().numpy().tolist())

    return (
        np.array([np.mean(heat_abs[t])         for t in range(traj_len)]),
        np.array([np.sqrt(np.mean(heat_sq[t]))  for t in range(traj_len)]),
        np.array([np.mean(cool_abs[t])         for t in range(traj_len)]),
        np.array([np.sqrt(np.mean(cool_sq[t]))  for t in range(traj_len)]),
    )


@torch.no_grad()
def evaluate_rollout(
    model,
    traj_loader,
    action_mean: float, action_std: float,
    cool_mean:   float, cool_std:   float,
    device:      str,
    traj_len:    int = 12,
):
    """Auto-regressive rollout via model.rollout — the two stages chained
    using the model's OWN predictions, matching inference-time behaviour."""
    model.eval()
    heat_sq, heat_abs = [[] for _ in range(traj_len)], [[] for _ in range(traj_len)]
    cool_sq, cool_abs = [[] for _ in range(traj_len)], [[] for _ in range(traj_len)]

    all_heat_pred, all_heat_gt = [], []
    all_next_pred, all_next_gt = [], []
    all_actions, all_cooltimes = [], []
    all_heat_epi, all_heat_ale, all_cool_epi, all_cool_ale = [], [], [], []
    all_total_epi, all_total_ale, all_total_std = [], [], []

    for traj_s, traj_h, traj_a, traj_c in traj_loader:
        traj_s = traj_s.to(device)
        traj_h = traj_h.to(device)
        traj_a = traj_a.to(device)
        traj_c = traj_c.to(device)
        B, T1, D = traj_s.shape
        T = min(T1 - 1, traj_len)

        out = model.rollout(traj_s[:, 0, :], traj_a[:, :T, :], traj_c[:, :T, :])
        pred_heat, pred_next = out["pred_heat_states"], out["pred_next_states"]
        heat_epi, heat_ale   = out["heat_epistemic"], out["heat_aleatoric"]
        cool_epi, cool_ale   = out["cool_epistemic"], out["cool_aleatoric"]
        total_epi, total_ale, total_std = out["total_epistemic"], out["total_aleatoric"], out["total_std"]

        for t in range(T):
            dh = pred_heat[:, t, :] - traj_h[:, t, :]
            heat_sq[t].extend(dh.pow(2).mean(dim=-1).cpu().numpy().tolist())
            heat_abs[t].extend(dh.abs().mean(dim=-1).cpu().numpy().tolist())

            dc = pred_next[:, t, :] - traj_s[:, t + 1, :]
            cool_sq[t].extend(dc.pow(2).mean(dim=-1).cpu().numpy().tolist())
            cool_abs[t].extend(dc.abs().mean(dim=-1).cpu().numpy().tolist())

        heat_pred_np = pred_heat.cpu().numpy()
        heat_gt_np   = traj_h[:, :T, :].cpu().numpy()
        next_pred_np = pred_next.cpu().numpy()
        next_gt_np   = traj_s[:, 1:T+1, :].cpu().numpy()
        actions_np   = traj_a[:, :T, 0].cpu().numpy() * action_std + action_mean
        cooltime_np  = traj_c[:, :T, 0].cpu().numpy() * cool_std   + cool_mean
        heat_epi_np, heat_ale_np = heat_epi.cpu().numpy(), heat_ale.cpu().numpy()
        cool_epi_np, cool_ale_np = cool_epi.cpu().numpy(), cool_ale.cpu().numpy()
        total_epi_np, total_ale_np, total_std_np = (
            total_epi.cpu().numpy(), total_ale.cpu().numpy(), total_std.cpu().numpy()
        )

        for b in range(B):
            all_heat_pred.append(heat_pred_np[b]); all_heat_gt.append(heat_gt_np[b])
            all_next_pred.append(next_pred_np[b]); all_next_gt.append(next_gt_np[b])
            all_actions.append(actions_np[b]);     all_cooltimes.append(cooltime_np[b])
            all_heat_epi.append(heat_epi_np[b]);   all_heat_ale.append(heat_ale_np[b])
            all_cool_epi.append(cool_epi_np[b]);   all_cool_ale.append(cool_ale_np[b])
            all_total_epi.append(total_epi_np[b]); all_total_ale.append(total_ale_np[b])
            all_total_std.append(total_std_np[b])

    def _agg(sq, ab):
        rmse = np.array([np.sqrt(np.mean(sq[t])) for t in range(traj_len)])
        mae  = np.array([np.mean(ab[t])          for t in range(traj_len)])
        return rmse, mae

    heat_rmse, heat_mae = _agg(heat_sq, heat_abs)
    cool_rmse, cool_mae = _agg(cool_sq, cool_abs)
    heat_epi_pl  = np.array([np.mean([s[t] for s in all_heat_epi])  for t in range(traj_len)])
    heat_ale_pl  = np.array([np.mean([s[t] for s in all_heat_ale])  for t in range(traj_len)])
    cool_epi_pl  = np.array([np.mean([s[t] for s in all_cool_epi])  for t in range(traj_len)])
    cool_ale_pl  = np.array([np.mean([s[t] for s in all_cool_ale])  for t in range(traj_len)])
    total_epi_pl = np.array([np.mean([s[t] for s in all_total_epi]) for t in range(traj_len)])
    total_ale_pl = np.array([np.mean([s[t] for s in all_total_ale]) for t in range(traj_len)])
    total_std_pl = np.array([np.mean([s[t] for s in all_total_std]) for t in range(traj_len)])

    return dict(
        heat_rmse=heat_rmse, heat_mae=heat_mae, cool_rmse=cool_rmse, cool_mae=cool_mae,
        heat_epi=heat_epi_pl, heat_ale=heat_ale_pl, cool_epi=cool_epi_pl, cool_ale=cool_ale_pl,
        total_epi=total_epi_pl, total_ale=total_ale_pl, total_std=total_std_pl,
        all_heat_pred=all_heat_pred, all_heat_gt=all_heat_gt,
        all_next_pred=all_next_pred, all_next_gt=all_next_gt,
        all_actions=all_actions, all_cooltimes=all_cooltimes,
        all_heat_epi=all_heat_epi, all_heat_ale=all_heat_ale,
        all_cool_epi=all_cool_epi, all_cool_ale=all_cool_ale,
        all_total_std=all_total_std,
    )


@torch.no_grad()
def evaluate_loss_components(model, flat_loader, device, roi_table) -> dict:
    """Mean per-component losses on the test set (flat transition dataset)."""
    model.eval()
    agg = defaultdict(float)
    n   = 0
    for s, a, c, h, s2, layer_idx, bmask in flat_loader:
        s, a, c, h, s2 = (t.to(device) for t in (s, a, c, h, s2))
        layer_idx = layer_idx.to(device)
        bmask     = bmask.to(device)
        L = compute_single_step_losses(model, s, a, c, h, s2, layer_idx, roi_table, bmask)
        for k, v in L.items():
            agg[k] += v.item()
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


@torch.no_grad()
def evaluate_recon(model, traj_loader, device) -> Tuple:
    """RMSE of encoder → decoder round-trip on all test states and heat fields [K]."""
    model.eval()
    se_s, n_s = 0.0, 0
    se_h, n_h = 0.0, 0
    for traj_s, traj_h, _, _ in traj_loader:
        traj_s = traj_s.to(device)
        traj_h = traj_h.to(device)
        B, T1, D = traj_s.shape
        s_flat = traj_s.reshape(B * T1, D)
        recon_s = model.decode(model.encode(s_flat))
        se_s += (recon_s - s_flat).pow(2).mean(dim=-1).sum().item()
        n_s  += B * T1

        _, T, _ = traj_h.shape
        h_flat = traj_h.reshape(B * T, D)
        recon_h = model.decode(model.encode(h_flat))
        se_h += (recon_h - h_flat).pow(2).mean(dim=-1).sum().item()
        n_h  += B * T
    return float(np.sqrt(se_s / max(n_s, 1))), float(np.sqrt(se_h / max(n_h, 1)))


@torch.no_grad()
def compute_per_layer_mean_gt(traj_loader, device, traj_len: int = 12):
    """Mean ground-truth temperature (raw Kelvin) per layer, for heat and next-state fields."""
    heat_sums, heat_counts = np.zeros(traj_len), np.zeros(traj_len)
    next_sums, next_counts = np.zeros(traj_len), np.zeros(traj_len)

    for traj_s, traj_h, _, _ in traj_loader:
        traj_s, traj_h = traj_s.to(device), traj_h.to(device)
        B, T1, D = traj_s.shape
        T = min(T1 - 1, traj_len)
        for t in range(T):
            heat_sums[t]  += traj_h[:, t, :].mean().item() * B
            heat_counts[t] += B
            next_sums[t]  += traj_s[:, t + 1, :].mean().item() * B
            next_counts[t] += B

    return heat_sums / np.maximum(heat_counts, 1), next_sums / np.maximum(next_counts, 1)


# =============================================================================
# Plotting helpers  (matched to surrogate_model_latent_uncertainty/evaluate.py style)
# =============================================================================

def _plot_field_2d(ax, values, triang, title, vmin=300, vmax=5000):
    tpc = ax.tripcolor(triang, values, cmap="jet", vmin=vmin, vmax=vmax, shading="gouraud")
    ax.triplot(triang, color="k", linewidth=0.15, alpha=0.3)
    plt.colorbar(tpc, ax=ax, label="Temperature [K]")
    ax.set_aspect("auto"); ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(title, fontsize=9)


def _plot_field_1d(ax, values, title, color="steelblue"):
    ax.plot(values, linewidth=0.6, color=color)
    ax.set_xlabel("Node index"); ax.set_ylabel("Temperature [K]")
    ax.set_ylim(300, 5500); ax.set_title(title, fontsize=9); ax.grid(True, alpha=0.3)


def plot_action_sequence(actions, cooltimes, traj_idx, out_path):
    T = len(actions)
    layers = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 3))
    axes[0].bar(layers, actions, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Laser Power [W]")
    axes[0].set_title(f"Trajectory {traj_idx} — Laser Power")
    axes[0].set_xticks(layers); axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(layers, cooltimes, color="tab:cyan", alpha=0.8)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Cool Time [s]")
    axes[1].set_title(f"Trajectory {traj_idx} — Cool Time")
    axes[1].set_xticks(layers); axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


def plot_sigma_sequence(heat_epi, heat_ale, cool_epi, cool_ale, traj_idx, out_path):
    T = len(heat_epi)
    layers = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 3))
    axes[0].plot(layers, heat_epi, marker="o", color="tab:orange", linewidth=1.5, label="Epistemic")
    axes[0].plot(layers, heat_ale, marker="s", color="tab:purple", linewidth=1.5, label="Aleatoric")
    axes[0].set_title(f"Traj {traj_idx} — Heating σ"); axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean latent σ"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(layers)

    axes[1].plot(layers, cool_epi, marker="o", color="tab:orange", linewidth=1.5, label="Epistemic")
    axes[1].plot(layers, cool_ale, marker="s", color="tab:purple", linewidth=1.5, label="Aleatoric")
    axes[1].set_title(f"Traj {traj_idx} — Cooling σ"); axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean latent σ"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(layers)

    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


def plot_example_trajectory(
    heat_pred, heat_gt, next_pred, next_gt,
    actions, cooltimes, heat_epi, heat_ale, cool_epi, cool_ale,
    traj_idx, out_dir, triang, vmin=300, vmax=5000,
):
    T = heat_pred.shape[0]
    traj_dir = os.path.join(out_dir, f"traj_{traj_idx:03d}")
    os.makedirs(traj_dir, exist_ok=True)

    txt_path = os.path.join(traj_dir, "actions.txt")
    with open(txt_path, "w") as f:
        f.write(f"Trajectory {traj_idx} — Laser Power / Cool Time per Layer\n")
        f.write(f"{'Layer':>6}  {'LP [W]':>10}  {'CoolTime [s]':>14}\n")
        f.write("-" * 34 + "\n")
        for t in range(T):
            f.write(f"{t+1:>6}  {actions[t]:>10.1f}  {cooltimes[t]:>14.4f}\n")
    print(f"[evaluate] Saved → {txt_path}")

    plot_action_sequence(actions, cooltimes, traj_idx,
                         os.path.join(traj_dir, "action_sequence.png"))
    plot_sigma_sequence(heat_epi, heat_ale, cool_epi, cool_ale, traj_idx,
                        os.path.join(traj_dir, "sigma_per_layer.png"))

    for layer in range(T):
        hp, hg = heat_pred[layer], heat_gt[layer]
        np_, ng = next_pred[layer], next_gt[layer]
        h_mae = np.abs(hp - hg).mean()
        n_mae = np.abs(np_ - ng).mean()
        lp    = actions[layer]

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle(
            f"Traj {traj_idx} — Layer {layer+1}  |  LP = {lp:.0f} W  |  "
            f"Heat MAE = {h_mae:.2f} K  |  Cool MAE = {n_mae:.2f} K",
            fontsize=10,
        )

        if triang is not None:
            _plot_field_2d(axes[0, 0], hg, triang, "Heating GT [K]", vmin, vmax)
            _plot_field_2d(axes[0, 1], hp, triang, "Heating Pred [K]", vmin, vmax)
            herr = np.abs(hp - hg)
            tpc = axes[0, 2].tripcolor(triang, herr, cmap="hot", vmin=0,
                                       vmax=max(herr.max(), 1.0), shading="gouraud")
            axes[0, 2].triplot(triang, color="k", linewidth=0.15, alpha=0.3)
            plt.colorbar(tpc, ax=axes[0, 2], label="Abs Error [K]")
            axes[0, 2].set_aspect("auto")
            axes[0, 2].set_title(f"Heating Error  mean={h_mae:.2f}K", fontsize=9)

            _plot_field_2d(axes[1, 0], ng, triang, "Cooling GT [K]", vmin, vmax)
            _plot_field_2d(axes[1, 1], np_, triang, "Cooling Pred [K]", vmin, vmax)
            nerr = np.abs(np_ - ng)
            tpc2 = axes[1, 2].tripcolor(triang, nerr, cmap="hot", vmin=0,
                                        vmax=max(nerr.max(), 1.0), shading="gouraud")
            axes[1, 2].triplot(triang, color="k", linewidth=0.15, alpha=0.3)
            plt.colorbar(tpc2, ax=axes[1, 2], label="Abs Error [K]")
            axes[1, 2].set_aspect("auto")
            axes[1, 2].set_title(f"Cooling Error  mean={n_mae:.2f}K", fontsize=9)
        else:
            _plot_field_1d(axes[0, 0], hg, "Heating GT [K]", "steelblue")
            _plot_field_1d(axes[0, 1], hp, "Heating Pred [K]", "darkorange")
            axes[0, 2].plot(np.abs(hp - hg), linewidth=0.6, color="crimson")
            axes[0, 2].set_title(f"Heating Abs Error  mean={h_mae:.2f}K", fontsize=9)
            axes[0, 2].grid(True, alpha=0.3)

            _plot_field_1d(axes[1, 0], ng, "Cooling GT [K]", "steelblue")
            _plot_field_1d(axes[1, 1], np_, "Cooling Pred [K]", "darkorange")
            axes[1, 2].plot(np.abs(np_ - ng), linewidth=0.6, color="crimson")
            axes[1, 2].set_title(f"Cooling Abs Error  mean={n_mae:.2f}K", fontsize=9)
            axes[1, 2].grid(True, alpha=0.3)

        fig.tight_layout()
        fname = os.path.join(traj_dir, f"layer_{layer+1:02d}_LP{lp:.0f}W.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[evaluate] Saved → {fname}")


_SPLIT_STYLES = {
    "train": ("tab:blue",   "o", "-"),
    "val":   ("tab:orange", "s", "--"),
    "test":  ("tab:green",  "^", ":"),
}


def _plot_two_panel_metric(results, out_path, ylabel, title, pct_refs=None):
    """results: {split: (heat_ro, heat_ss, cool_ro, cool_ss)}"""
    traj_len = len(next(iter(results.values()))[0])
    layers = np.arange(1, traj_len + 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for split, (h_ro, h_ss, c_ro, c_ss) in results.items():
        color, marker, _ = _SPLIT_STYLES.get(split, ("grey", "o", "-"))
        axes[0].plot(layers, h_ro, marker=marker, color=color, linewidth=1.5, label=f"{split} rollout")
        axes[0].plot(layers, h_ss, marker=marker, color=color, linewidth=1.0,
                     linestyle="--", alpha=0.6, label=f"{split} single-step")
        axes[1].plot(layers, c_ro, marker=marker, color=color, linewidth=1.5, label=f"{split} rollout")
        axes[1].plot(layers, c_ss, marker=marker, color=color, linewidth=1.0,
                     linestyle="--", alpha=0.6, label=f"{split} single-step")

    if pct_refs is not None and "test" in results and "test" in pct_refs:
        h_ro_test = results["test"][0]
        c_ro_test = results["test"][2]
        heat_ref, next_ref = pct_refs["test"]
        for i, (l, v, r) in enumerate(zip(layers, h_ro_test, heat_ref)):
            axes[0].annotate(f"{v / r * 100:.1f}%", (l, v), xytext=(0, 6),
                             textcoords="offset points", fontsize=6.5, ha="center", va="bottom",
                             color=_SPLIT_STYLES["test"][0])
        for i, (l, v, r) in enumerate(zip(layers, c_ro_test, next_ref)):
            axes[1].annotate(f"{v / r * 100:.1f}%", (l, v), xytext=(0, 6),
                             textcoords="offset points", fontsize=6.5, ha="center", va="bottom",
                             color=_SPLIT_STYLES["test"][0])

    axes[0].set_title(f"Heating — {title}", fontsize=10)
    axes[1].set_title(f"Cooling — {title}", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Layer index"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xticks(layers)

    fig.suptitle(f"{title} per Layer — Train / Val / Test  (% = relative to mean GT, rollout)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


def plot_per_layer_uncertainty(results, out_path):
    """results: {split: (heat_epi, heat_ale, cool_epi, cool_ale, total_epi, total_ale, total_std)}.
    Combined row uses the model's own combine_stage_uncertainties output
    (from model.rollout), not a re-derivation here."""
    traj_len = len(next(iter(results.values()))[0])
    layers = np.arange(1, traj_len + 1)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for split, (h_epi, h_ale, c_epi, c_ale, t_epi, t_ale, t_std) in results.items():
        color, marker, ls = _SPLIT_STYLES.get(split, ("grey", "o", "-"))
        axes[0, 0].plot(layers, h_epi, marker=marker, color=color, linestyle=ls, linewidth=1.5, label=split)
        axes[0, 1].plot(layers, h_ale, marker=marker, color=color, linestyle=ls, linewidth=1.5, label=split)
        h_tot = np.sqrt(h_epi ** 2 + h_ale ** 2)
        axes[0, 2].plot(layers, h_tot, marker=marker, color=color, linestyle=ls, linewidth=1.5, label=split)

        axes[1, 0].plot(layers, c_epi, marker=marker, color=color, linestyle=ls, linewidth=1.5, label=split)
        axes[1, 1].plot(layers, c_ale, marker=marker, color=color, linestyle=ls, linewidth=1.5, label=split)
        c_tot = np.sqrt(c_epi ** 2 + c_ale ** 2)
        axes[1, 2].plot(layers, c_tot, marker=marker, color=color, linestyle=ls, linewidth=1.5, label=split)

        axes[2, 0].plot(layers, t_epi, marker=marker, color=color, linestyle=ls, linewidth=1.5, label=split)
        axes[2, 1].plot(layers, t_ale, marker=marker, color=color, linestyle=ls, linewidth=1.5, label=split)
        axes[2, 2].plot(layers, t_std, marker=marker, color=color, linestyle=ls, linewidth=1.5, label=split)

    titles = [["Heating Epistemic σ", "Heating Aleatoric σ", "Heating Total σ"],
              ["Cooling Epistemic σ", "Cooling Aleatoric σ", "Cooling Total σ"],
              ["Combined Epistemic σ  (√(heat²+cool²))", "Combined Aleatoric σ  (√(heat²+cool²))",
               "Combined Total σ — read by RL"]]
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            ax.set_title(titles[i][j], fontsize=9)
            ax.set_xlabel("Layer index"); ax.set_ylabel("Mean latent σ")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(layers)

    fig.suptitle("Latent-Space Uncertainty per Layer — Train / Val / Test", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the two-stage LPBF Latent-Space Gaussian Ensemble surrogate."
    )
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--data_path",     type=str, required=True)
    p.add_argument("--mesh_path",     type=str, default="surrogate_model/mesh.mat")
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--n_examples",    type=int,   default=3)
    p.add_argument("--vmin",          type=float, default=300.0)
    p.add_argument("--vmax",          type=float, default=5000.0)
    p.add_argument("--out_dir",       type=str,   default="")
    p.add_argument("--device",        type=str,   default="")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[evaluate] Output dir : {out_dir}")
    print(f"[evaluate] Device     : {device}")

    triang = load_mesh(args.mesh_path)

    (model, state_mean, state_std, lp_mean, lp_std,
     cool_mean, cool_std, roi_table) = load_two_stage_surrogate(args.checkpoint, device)
    print(f"[evaluate] {model}")

    all_trajs = load_trajectories(args.data_path)
    train_trajs, val_trajs, test_trajs = split_trajectories(
        all_trajs, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
    )
    traj_len = len(test_trajs[0])

    ds_kwargs = dict(
        state_mean=state_mean.cpu(), state_std=state_std.cpu(),
        lp_mean=lp_mean, lp_std=lp_std, cool_mean=cool_mean, cool_std=cool_std,
        initial_temp=args.initial_temp,
    )
    flat_kwargs = dict(ds_kwargs, n_ensemble=model.n_ensemble, bootstrap_seed=args.seed)
    loader_kw   = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    def _traj_loader_only(trajs):
        base = DataLoader(
            TwoStageLatentTrajectoryDataset(trajs, **ds_kwargs, n_ensemble=model.n_ensemble,
                                             bootstrap_seed=args.seed),
            **loader_kw,
        )
        return _DropBootstrapMask(base)

    split_loaders = {
        "train": (_traj_loader_only(train_trajs),
                  DataLoader(TwoStageLatentSurrogateDataset(train_trajs, **flat_kwargs), **loader_kw)),
        "val":   (_traj_loader_only(val_trajs),
                  DataLoader(TwoStageLatentSurrogateDataset(val_trajs,   **flat_kwargs), **loader_kw)),
        "test":  (_traj_loader_only(test_trajs),
                  DataLoader(TwoStageLatentSurrogateDataset(test_trajs,  **flat_kwargs), **loader_kw)),
    }

    metric_results = {}       # {split: (h_ro_rmse, h_ss_rmse, c_ro_rmse, c_ss_rmse)}
    mae_results    = {}       # {split: (h_ro_mae,  h_ss_mae,  c_ro_mae,  c_ss_mae)}
    unc_results    = {}       # {split: (h_epi, h_ale, c_epi, c_ale)}
    summary = {}

    test_rollout = None

    for split_name, (traj_loader, flat_loader) in split_loaders.items():
        print(f"\n[evaluate] {'═'*20} {split_name.upper()} SET {'═'*20}")

        comps = evaluate_loss_components(model, flat_loader, device, roi_table)
        for k, v in comps.items():
            print(f"  {k:<14}: {v:.6f}")

        ae_rmse_s, ae_rmse_h = evaluate_recon(model, traj_loader, device)
        print(f"  AE RMSE (s_t)      : {ae_rmse_s:.4f} K")
        print(f"  AE RMSE (u_heat_t) : {ae_rmse_h:.4f} K")

        h_ss_mae, h_ss_rmse, c_ss_mae, c_ss_rmse = evaluate_single_step(
            model, traj_loader, device, traj_len=traj_len
        )

        ro = evaluate_rollout(model, traj_loader, lp_mean, lp_std, cool_mean, cool_std,
                              device, traj_len=traj_len)

        print(f"\n  {'Layer':>6}  {'Heat SS RMSE':>13}  {'Heat RO RMSE':>13}  "
              f"{'Cool SS RMSE':>13}  {'Cool RO RMSE':>13}")
        for i in range(traj_len):
            print(f"  {i+1:>6}  {h_ss_rmse[i]:>13.3f}  {ro['heat_rmse'][i]:>13.3f}  "
                  f"{c_ss_rmse[i]:>13.3f}  {ro['cool_rmse'][i]:>13.3f}")

        metric_results[split_name] = (ro["heat_rmse"], h_ss_rmse, ro["cool_rmse"], c_ss_rmse)
        mae_results[split_name]    = (ro["heat_mae"],  h_ss_mae,  ro["cool_mae"],  c_ss_mae)
        unc_results[split_name]    = (ro["heat_epi"], ro["heat_ale"], ro["cool_epi"], ro["cool_ale"],
                                       ro["total_epi"], ro["total_ale"], ro["total_std"])

        summary[split_name] = {
            "ae_rmse_s [K]":    ae_rmse_s,
            "ae_rmse_h [K]":    ae_rmse_h,
            "heat_ss_rmse [K]": h_ss_rmse.mean(),
            "heat_ro_rmse [K]": ro["heat_rmse"].mean(),
            "cool_ss_rmse [K]": c_ss_rmse.mean(),
            "cool_ro_rmse [K]": ro["cool_rmse"].mean(),
            "heat_epist_std":   ro["heat_epi"].mean(),
            "cool_epist_std":   ro["cool_epi"].mean(),
            "total_std":        ro["total_std"].mean(),
        }

        if split_name == "test":
            test_rollout = ro

    metrics = list(next(iter(summary.values())).keys())
    col_w = 15
    header = f"{'Split':<8}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print(f"\n[evaluate] {'═'*len(header)}\n[evaluate] SUMMARY\n[evaluate] {'═'*len(header)}")
    print(f"  {header}")
    for split_name, vals in summary.items():
        print(f"  {split_name:<8}" + "".join(f"{vals[m]:>{col_w}.4f}" for m in metrics))
    print(f"[evaluate] {'═'*len(header)}\n")

    pct_refs = {}
    for split_name, (traj_loader, _) in split_loaders.items():
        pct_refs[split_name] = compute_per_layer_mean_gt(traj_loader, device, traj_len=traj_len)

    _plot_two_panel_metric(metric_results, os.path.join(out_dir, "per_layer_rmse.png"),
                           "RMSE [K]", "RMSE", pct_refs=pct_refs)
    _plot_two_panel_metric(mae_results, os.path.join(out_dir, "per_layer_mae.png"),
                           "MAE [K]", "MAE", pct_refs=pct_refs)
    plot_per_layer_uncertainty(unc_results, os.path.join(out_dir, "per_layer_uncertainty.png"))

    n_ex = min(args.n_examples, len(test_rollout["all_heat_pred"]))
    print(f"[evaluate] Saving field plots for {n_ex} test trajectories ...")
    for i in range(n_ex):
        plot_example_trajectory(
            test_rollout["all_heat_pred"][i], test_rollout["all_heat_gt"][i],
            test_rollout["all_next_pred"][i], test_rollout["all_next_gt"][i],
            actions=test_rollout["all_actions"][i], cooltimes=test_rollout["all_cooltimes"][i],
            heat_epi=test_rollout["all_heat_epi"][i], heat_ale=test_rollout["all_heat_ale"][i],
            cool_epi=test_rollout["all_cool_epi"][i], cool_ale=test_rollout["all_cool_ale"][i],
            traj_idx=i, out_dir=out_dir, triang=triang, vmin=args.vmin, vmax=args.vmax,
        )

    print(f"\n[evaluate] Complete. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
