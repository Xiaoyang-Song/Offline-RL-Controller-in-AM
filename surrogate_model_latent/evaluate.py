"""
surrogate_model_latent/evaluate.py
------------------------------------
Evaluation and visualisation for the trained LPBF Latent-Space surrogate.

Plot style matches surrogate_model/evaluate.py exactly:
  jet colormap, caxis [300, 5000], Gouraud shading, triplot mesh overlay,
  hot colormap for error field — all mimicking MATLAB pdeplot style.

Metrics
-------
  1. Single-step MAE / RMSE per layer  (teacher-forced)
  2. Auto-regressive rollout RMSE per layer
  3. Per-layer mean transition σ  (uncertainty from GaussianTransitionMLP)
  4. Autoencoder round-trip RMSE  (encoder → decoder)
  5. Per-component test-set loss  (recon_st, recon_st1, nll)

Plots saved to --out_dir (defaults to checkpoint directory)
------------------------------------------------------------
  per_layer_mae.png
  per_layer_rmse.png
  per_layer_uncertainty.png
  traj_<i>/actions.txt
  traj_<i>/action_sequence.png
  traj_<i>/sigma_per_layer.png
  traj_<i>/layer_<l>_LP<p>W.png   — GT | Pred | Error on PDE mesh

Usage
-----
    python -m surrogate_model_latent.evaluate \\
        --checkpoint surrogate_model_latent/runs/<ts>/latent_best.pt \\
        --data_path  Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl
"""

import argparse
import os
import sys

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

from surrogate_model_latent.dataset import (
    load_trajectories,
    split_trajectories,
    build_normalizers,
    LatentTrajectoryDataset,
    LatentSurrogateDataset,
)
from surrogate_model_latent.train import load_latent_surrogate, compute_single_step_losses, weighted_mse


# =============================================================================
# Mesh loading  (identical to surrogate_model/evaluate.py)
# =============================================================================

def load_mesh(mesh_path: str):
    if not os.path.exists(mesh_path):
        print(f"[evaluate] mesh.mat not found at {mesh_path}. "
              "Run surrogate_model/extract_mesh.m in MATLAB first for 2-D plots. "
              "Falling back to 1-D node-index plots.")
        return None
    data     = loadmat(mesh_path)
    nodes    = data["nodes"]            # (2, N_nodes)
    elements = data["elements"] - 1    # 1-indexed → 0-indexed
    x   = nodes[0, :].astype(np.float64)
    y   = nodes[1, :].astype(np.float64)
    tri = elements[:3, :].T.astype(np.int32)
    triang = mtri.Triangulation(x, y, tri)
    print(f"[evaluate] Mesh loaded: {len(x)} nodes, {len(tri)} triangles. "
          f"X∈[{x.min():.2f},{x.max():.2f}], Y∈[{y.min():.2f},{y.max():.2f}]")
    return triang


# =============================================================================
# Core evaluation
# =============================================================================

@torch.no_grad()
def evaluate_single_step(
    model,
    test_loader: DataLoader,
    state_mean:  torch.Tensor,
    state_std:   torch.Tensor,
    device:      str,
    traj_len:    int = 12,
):
    """
    Teacher-forced 1-step evaluation (no error accumulation).
    Returns per_layer_mae, per_layer_rmse — both in raw Kelvin.
    """
    model.eval()
    abs_err = [[] for _ in range(traj_len)]
    sq_err  = [[] for _ in range(traj_len)]

    sm = state_mean.to(device)
    ss = state_std.to(device)

    for traj_s, traj_a in test_loader:
        traj_s = traj_s.to(device)
        traj_a = traj_a.to(device)
        B, T1, D = traj_s.shape
        T = T1 - 1

        for t in range(min(T, traj_len)):
            s_in = traj_s[:, t,     :]
            a_in = traj_a[:, t,     :]
            s_gt = traj_s[:, t + 1, :]

            layer_idx      = torch.full((s_in.shape[0],), t, dtype=torch.long, device=device)
            z_t            = model.encode(s_in)
            mu_mean, _     = model.predict_ensemble(z_t, a_in, layer_idx)
            s_pred         = model.decode(z_t + mu_mean)
            diff_k         = (s_pred - s_gt) * ss    # Kelvin

            abs_err[t].extend(diff_k.abs().mean(dim=-1).cpu().numpy().tolist())
            sq_err[t].extend((diff_k ** 2).mean(dim=-1).cpu().numpy().tolist())

    per_layer_mae  = np.array([np.mean(abs_err[t])            for t in range(traj_len)])
    per_layer_rmse = np.array([np.sqrt(np.mean(sq_err[t]))    for t in range(traj_len)])
    return per_layer_mae, per_layer_rmse


@torch.no_grad()
def evaluate_rollout(
    model,
    test_loader:  DataLoader,
    state_mean:   torch.Tensor,
    state_std:    torch.Tensor,
    action_mean:  float,
    action_std:   float,
    device:       str,
    traj_len:     int = 12,
):
    """
    Auto-regressive rollout: latent model feeds its own predictions forward.

    Returns
    -------
    per_layer_rmse  : (traj_len,)   RMSE in raw Kelvin per layer
    per_layer_sigma : (traj_len,)   mean transition σ per layer
    all_preds       : list of (T, D) np.ndarray  — predicted fields [K]
    all_gts         : list of (T, D) np.ndarray  — ground-truth fields [K]
    all_actions     : list of (T,)   np.ndarray  — laser power [W]
    all_sigmas      : list of (T,)   np.ndarray  — mean σ per layer [latent space]
    """
    model.eval()
    sq_errors  = [[] for _ in range(traj_len)]
    abs_errors = [[] for _ in range(traj_len)]
    all_preds, all_gts, all_actions, all_sigmas = [], [], [], []

    sm = state_mean.to(device)
    ss = state_std.to(device)

    for traj_s, traj_a in test_loader:
        traj_s = traj_s.to(device)   # (B, T+1, D) normalised
        traj_a = traj_a.to(device)   # (B, T,   1) normalised
        B, T1, D = traj_s.shape
        T = min(T1 - 1, traj_len)

        # Auto-regressive rollout in latent space
        pred_states, epistemic_stds = model.rollout(
            traj_s[:, 0, :], traj_a[:, :T, :]
        )
        # pred_states    : (B, T, D)   epistemic_stds : (B, T)

        for t in range(T):
            s_pred_k = pred_states[:, t, :] * ss + sm
            s_gt_k   = traj_s[:, t + 1, :] * ss + sm
            diff_k   = s_pred_k - s_gt_k
            sq_errors[t].extend(diff_k.pow(2).mean(dim=-1).cpu().numpy().tolist())
            abs_errors[t].extend(diff_k.abs().mean(dim=-1).cpu().numpy().tolist())

        # Denormalise to raw Kelvin / Watts
        pred_np    = (pred_states * ss + sm).cpu().numpy()                      # (B, T, D)
        gt_np      = (traj_s[:, 1:T+1, :] * ss + sm).cpu().numpy()             # (B, T, D)
        actions_np = traj_a[:, :T, 0].cpu().numpy() * action_std + action_mean  # (B, T)
        sigma_np   = epistemic_stds.cpu().numpy()                               # (B, T)

        for b in range(B):
            all_preds.append(pred_np[b])
            all_gts.append(gt_np[b])
            all_actions.append(actions_np[b])
            all_sigmas.append(sigma_np[b])

    per_layer_rmse  = np.array([np.sqrt(np.mean(sq_errors[t]))           for t in range(traj_len)])
    per_layer_mae   = np.array([np.mean(abs_errors[t])                   for t in range(traj_len)])
    per_layer_sigma = np.array([np.mean([s[t] for s in all_sigmas])      for t in range(traj_len)])

    return per_layer_rmse, per_layer_mae, per_layer_sigma, all_preds, all_gts, all_actions, all_sigmas


@torch.no_grad()
def evaluate_loss_components(
    model,
    flat_loader: DataLoader,
    device:      str,
    roi_table,
) -> dict:
    """Mean per-component losses on the test set (flat transition dataset)."""
    model.eval()
    agg = defaultdict(float)
    n   = 0
    for s, a, s2, layer_idx in flat_loader:
        s         = s.to(device)
        a         = a.to(device)
        s2        = s2.to(device)
        layer_idx = layer_idx.to(device)
        L_rs, L_rs1 = compute_single_step_losses(
            model, s, a, s2, layer_idx, roi_table
        )
        agg["recon_st"]  += L_rs.item()
        agg["recon_st1"] += L_rs1.item()
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


@torch.no_grad()
def evaluate_recon(
    model,
    test_loader: DataLoader,
    state_mean:  torch.Tensor,
    state_std:   torch.Tensor,
    device:      str,
) -> float:
    """RMSE of encoder → decoder round-trip on all test states [K]."""
    model.eval()
    total_se = 0.0
    count    = 0
    ss = state_std.to(device)
    for traj_s, _ in test_loader:
        traj_s  = traj_s.to(device)
        B, T1, D = traj_s.shape
        s_flat  = traj_s.reshape(B * T1, D)
        recon   = model.decode(model.encode(s_flat))
        err_raw = (recon - s_flat) * ss
        total_se += err_raw.pow(2).mean(dim=-1).sum().item()
        count    += B * T1
    return float(np.sqrt(total_se / max(count, 1)))


# =============================================================================
# Per-layer mean ground-truth temperature  (percentage reference)
# =============================================================================

@torch.no_grad()
def compute_per_layer_mean_gt(
    traj_loader: DataLoader,
    state_mean:  torch.Tensor,
    state_std:   torch.Tensor,
    device:      str,
    traj_len:    int = 12,
) -> np.ndarray:
    """
    Mean ground-truth temperature (raw Kelvin) at each layer, averaged over
    all nodes and all trajectories.  Used as a per-layer reference for
    computing relative percentage errors on the metric plots.

    Returns (traj_len,) float64 array.
    """
    ss  = state_std.to(device)
    sm  = state_mean.to(device)
    sums   = np.zeros(traj_len)
    counts = np.zeros(traj_len)

    for traj_s, _ in traj_loader:
        traj_s = traj_s.to(device)
        B, T1, D = traj_s.shape
        T = min(T1 - 1, traj_len)
        for t in range(T):
            s_gt_k = traj_s[:, t + 1, :] * ss + sm   # (B, D) raw Kelvin
            sums[t]   += s_gt_k.mean().item() * B
            counts[t] += B

    return sums / np.maximum(counts, 1)


# =============================================================================
# Per-layer diagnostic decomposition
# =============================================================================

@torch.no_grad()
def compute_per_layer_diagnostics(
    model,
    traj_loader: DataLoader,
    state_mean:  torch.Tensor,
    state_std:   torch.Tensor,
    device:      str,
    traj_len:    int = 12,
) -> dict:
    """
    Decompose prediction error into three independent sources per layer.

    Returns a dict with arrays of length traj_len:

      delta_mag     : RMS state change  ||s_{t+1} - s_t|| [K]
                      — how much the physics changes each layer regardless of model
      ae_rmse       : encoder-decoder round-trip RMSE on s_t [K]
                      — how well the shared AE represents states at this layer
      trans_lat_err : RMS latent transition error
                      ||encode(s_{t+1}) - (z_t + mu_mean)|| in normalised latent space
                      — pure transition model error after isolating the AE
    """
    model.eval()
    ss = state_std.to(device)

    delta_sq  = [[] for _ in range(traj_len)]
    ae_sq     = [[] for _ in range(traj_len)]
    lat_sq    = [[] for _ in range(traj_len)]

    for traj_s, traj_a in traj_loader:
        traj_s = traj_s.to(device)
        traj_a = traj_a.to(device)
        B, T1, D = traj_s.shape
        T = min(T1 - 1, traj_len)

        for t in range(T):
            s_t  = traj_s[:, t,     :]
            a_t  = traj_a[:, t,     :]
            s_t1 = traj_s[:, t + 1, :]

            # State change magnitude (raw Kelvin)
            delta_k = ((s_t1 - s_t) * ss).pow(2).mean(dim=-1)   # (B,)
            delta_sq[t].extend(delta_k.cpu().numpy().tolist())

            # Autoencoder error on s_t (normalised → Kelvin)
            z_t     = model.encode(s_t)
            s_hat   = model.decode(z_t)
            ae_k    = ((s_hat - s_t) * ss).pow(2).mean(dim=-1)  # (B,)
            ae_sq[t].extend(ae_k.cpu().numpy().tolist())

            # Latent transition error: how well does z_t + mu_mean match encode(s_{t+1})?
            layer_idx  = torch.full((s_t.shape[0],), t, dtype=torch.long, device=device)
            mu_mean, _ = model.predict_ensemble(z_t, a_t, layer_idx)
            z_t1_pred  = z_t + mu_mean
            z_t1_enc   = model.encode(s_t1)
            lat_k      = (z_t1_pred - z_t1_enc).pow(2).mean(dim=-1)   # (B,) normalised
            lat_sq[t].extend(lat_k.cpu().numpy().tolist())

    return {
        "delta_mag":     np.array([np.sqrt(np.mean(delta_sq[t])) for t in range(traj_len)]),
        "ae_rmse":       np.array([np.sqrt(np.mean(ae_sq[t]))    for t in range(traj_len)]),
        "trans_lat_err": np.array([np.sqrt(np.mean(lat_sq[t]))   for t in range(traj_len)]),
    }


def plot_per_layer_diagnostics(
    diag_by_split: dict,   # {split: {metric: array}}
    out_path:      str,
) -> None:
    """
    3-panel diagnostic: state change magnitude, AE RMSE, latent transition error.
    Separates the three root causes of high per-layer prediction error.
    """
    traj_len = len(next(iter(diag_by_split.values()))["ae_rmse"])
    layers   = np.arange(1, traj_len + 1)

    panels = [
        ("delta_mag",     "||s_{t+1} - s_t|| [K]",        "State Change Magnitude"),
        ("ae_rmse",       "AE RMSE [K]",                   "Autoencoder Error on s_t"),
        ("trans_lat_err", "||z_{pred} - z_{enc}|| (norm.)", "Latent Transition Error"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (key, ylabel, title) in zip(axes, panels):
        for split, diag in diag_by_split.items():
            color, marker, ls = _SPLIT_STYLES.get(split, ("grey", "o", "-"))
            ax.plot(layers, diag[key], marker=marker, color=color,
                    linestyle=ls, linewidth=1.5, label=split)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)

    fig.suptitle("Per-Layer Error Source Diagnostics", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


# =============================================================================
# Plotting helpers  (matched to surrogate_model/evaluate.py style)
# =============================================================================

def _plot_field_2d(ax, values, triang, title, vmin=300, vmax=5000):
    """
    2-D PDE field plot matching MATLAB pdeplot style:
      jet colormap, Gouraud shading, mesh overlay, aspect='auto'.
    """
    tpc = ax.tripcolor(triang, values, cmap="jet", vmin=vmin, vmax=vmax,
                       shading="gouraud")
    ax.triplot(triang, color="k", linewidth=0.15, alpha=0.3)
    plt.colorbar(tpc, ax=ax, label="Temperature [K]")
    ax.set_aspect("auto")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title, fontsize=9)


def _plot_field_1d(ax, values, title, color="steelblue"):
    """Fallback: node-index line plot when mesh.mat is not available."""
    ax.plot(values, linewidth=0.6, color=color)
    ax.set_xlabel("Node index")
    ax.set_ylabel("Temperature [K]")
    ax.set_ylim(300, 5500)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_action_sequence(
    actions:  np.ndarray,   # (T,) raw laser power [W]
    traj_idx: int,
    out_path: str,
) -> None:
    """Bar chart of laser power per layer for one trajectory."""
    T      = len(actions)
    layers = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(layers, actions, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Laser Power [W]")
    ax.set_title(f"Trajectory {traj_idx} — Action Sequence")
    ax.set_xticks(layers)
    ax.set_ylim(0, max(actions.max() * 1.15, 500))
    for l, v in zip(layers, actions):
        ax.text(l, v + 5, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


def plot_sigma_sequence(
    sigmas:   np.ndarray,   # (T,) mean ensemble std per layer
    traj_idx: int,
    out_path: str,
) -> None:
    """Per-layer mean epistemic uncertainty (ensemble std) for one trajectory."""
    T      = len(sigmas)
    layers = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(layers, sigmas, marker="o", color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean ensemble std (epistemic uncertainty)")
    ax.set_title(f"Trajectory {traj_idx} — Per-layer Epistemic Uncertainty")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


def plot_example_trajectory(
    pred_traj:  np.ndarray,   # (T, D)  raw Kelvin
    gt_traj:    np.ndarray,   # (T, D)  raw Kelvin
    actions:    np.ndarray,   # (T,)    laser power [W]
    sigmas:     np.ndarray,   # (T,)    mean σ per layer
    traj_idx:   int,
    out_dir:    str,
    triang,
    vmin: float = 300,
    vmax: float = 5000,
) -> None:
    """
    Save every layer as a 3-panel PNG (GT | Pred | Error) into out_dir/traj_{i}/.
    Also saves action_sequence.png, sigma_per_layer.png, and actions.txt.
    Matches MATLAB: jet colormap, caxis [vmin, vmax], hot colormap for error.
    """
    T        = pred_traj.shape[0]
    traj_dir = os.path.join(out_dir, f"traj_{traj_idx:03d}")
    os.makedirs(traj_dir, exist_ok=True)

    # ── action list text ──────────────────────────────────────────────────────
    txt_path = os.path.join(traj_dir, "actions.txt")
    with open(txt_path, "w") as f:
        f.write(f"Trajectory {traj_idx} — Laser Power per Layer\n")
        f.write(f"{'Layer':>6}  {'LP [W]':>10}\n")
        f.write("-" * 20 + "\n")
        for t in range(T):
            f.write(f"{t+1:>6}  {actions[t]:>10.1f}\n")
    print(f"[evaluate] Saved → {txt_path}")

    # ── action bar chart ──────────────────────────────────────────────────────
    plot_action_sequence(actions, traj_idx,
                         os.path.join(traj_dir, "action_sequence.png"))

    # ── sigma curve ───────────────────────────────────────────────────────────
    plot_sigma_sequence(sigmas, traj_idx,
                        os.path.join(traj_dir, "sigma_per_layer.png"))

    # ── one PNG per layer ─────────────────────────────────────────────────────
    for layer in range(T):
        pred    = pred_traj[layer]
        gt      = gt_traj[layer]
        err_abs = np.abs(pred - gt)
        mae     = err_abs.mean()
        # Percentage relative error: |pred - gt| / |gt| × 100
        # gt is always ≥ 300 K so division is safe, but clamp to avoid tiny denominators
        pct_err = err_abs / np.maximum(np.abs(gt), 1.0) * 100.0
        mape    = pct_err.mean()
        lp      = actions[layer]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(
            f"Traj {traj_idx} — Layer {layer+1}  |  LP = {lp:.0f} W"
            f"  |  MAE = {mae:.2f} K  |  MAPE = {mape:.2f}%",
            fontsize=10,
        )

        if triang is not None:
            _plot_field_2d(axes[0], gt,   triang, "Ground Truth [K]", vmin, vmax)
            _plot_field_2d(axes[1], pred, triang, "Predicted [K]",    vmin, vmax)
            vmax_pct = max(pct_err.max(), 1.0)
            tpc = axes[2].tripcolor(triang, pct_err, cmap="hot",
                                    vmin=0, vmax=vmax_pct,
                                    shading="gouraud")
            axes[2].triplot(triang, color="k", linewidth=0.15, alpha=0.3)
            plt.colorbar(tpc, ax=axes[2], label="Relative Error [%]")
            axes[2].set_aspect("auto")
            axes[2].set_xlabel("X"); axes[2].set_ylabel("Y")
            axes[2].set_title(
                f"Relative Error [%]  mean={mape:.2f}%  max={pct_err.max():.1f}%",
                fontsize=9,
            )
        else:
            _plot_field_1d(axes[0], gt,   "Ground Truth [K]", color="steelblue")
            _plot_field_1d(axes[1], pred, "Predicted [K]",    color="darkorange")
            axes[2].plot(pct_err, linewidth=0.6, color="crimson")
            axes[2].set_xlabel("Node index")
            axes[2].set_ylabel("Relative Error [%]")
            axes[2].set_title(
                f"Relative Error [%]  mean={mape:.2f}%  max={pct_err.max():.1f}%",
                fontsize=9,
            )
            axes[2].grid(True, alpha=0.3)

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


def plot_per_layer_rmse(
    results:   dict,            # {split_name: (rollout_rmse, single_step_rmse)}
    out_path:  str,
    pct_refs:  dict = None,     # {split_name: per_layer_mean_gt_K} for % annotations
) -> None:
    """
    One line per split: solid = rollout, dashed = single-step.
    If pct_refs supplied, annotates the test rollout line with per-layer % values.
    """
    traj_len = len(next(iter(results.values()))[0])
    layers   = np.arange(1, traj_len + 1)
    fig, ax  = plt.subplots(figsize=(10, 5))
    for split, (ro_rmse, ss_rmse) in results.items():
        color, marker, _ = _SPLIT_STYLES.get(split, ("grey", "o", "-"))
        ax.plot(layers, ro_rmse, marker=marker, color=color,
                linewidth=1.5, label=f"{split} rollout")
        ax.plot(layers, ss_rmse, marker=marker, color=color,
                linewidth=1.0, linestyle="--", alpha=0.6, label=f"{split} single-step")

    # Annotate test rollout with per-layer relative %
    if pct_refs is not None and "test" in results and "test" in pct_refs:
        ro_rmse_test = results["test"][0]
        ref          = pct_refs["test"]
        for i, (l, v, r) in enumerate(zip(layers, ro_rmse_test, ref)):
            ha = "right" if i == len(layers) - 1 else "center"
            ax.annotate(f"{v / r * 100:.1f}%", (l, v),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=6.5, ha=ha, va="bottom",
                        color=_SPLIT_STYLES["test"][0])

    ax.set_xlabel("Layer index"); ax.set_ylabel("RMSE [K]")
    ax.set_title("Prediction RMSE per Layer — Train / Val / Test  (% = relative to mean GT, test rollout)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(layers)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


def plot_mae_per_layer(
    results:   dict,            # {split_name: (ro_mae, ss_mae)}
    out_path:  str,
    pct_refs:  dict = None,     # {split_name: per_layer_mean_gt_K} for % annotations
) -> None:
    """
    One line per split: solid = rollout MAE, dashed = single-step MAE.
    If pct_refs supplied, annotates the test rollout line with per-layer % values.
    """
    traj_len = len(next(iter(results.values()))[0])
    layers   = np.arange(1, traj_len + 1)
    fig, ax  = plt.subplots(figsize=(10, 4))
    for split, (ro_mae, ss_mae) in results.items():
        color, marker, _ = _SPLIT_STYLES.get(split, ("grey", "o", "-"))
        ax.plot(layers, ro_mae, marker=marker, color=color,
                linewidth=1.5, label=f"{split} rollout")
        ax.plot(layers, ss_mae, marker=marker, color=color,
                linewidth=1.0, linestyle="--", alpha=0.6, label=f"{split} single-step")

    # Annotate test rollout with per-layer relative %
    if pct_refs is not None and "test" in results and "test" in pct_refs:
        ro_mae_test = results["test"][0]
        ref         = pct_refs["test"]
        for i, (l, v, r) in enumerate(zip(layers, ro_mae_test, ref)):
            ha = "right" if i == len(layers) - 1 else "center"
            ax.annotate(f"{v / r * 100:.1f}%", (l, v),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=6.5, ha=ha, va="bottom",
                        color=_SPLIT_STYLES["test"][0])

    ax.set_xlabel("Layer index"); ax.set_ylabel("MAE [K]")
    ax.set_title("MAE per Layer — Rollout vs Single-Step | Train / Val / Test  (% = relative to mean GT, test rollout)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(layers)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


def plot_per_layer_uncertainty(
    results:  dict,   # {split_name: per_layer_sigma}
    out_path: str,
) -> None:
    """Epistemic uncertainty (ensemble std) per layer for each split."""
    traj_len = len(next(iter(results.values())))
    layers   = np.arange(1, traj_len + 1)
    fig, ax  = plt.subplots(figsize=(10, 5))
    for split, sigma in results.items():
        color, marker, ls = _SPLIT_STYLES.get(split, ("grey", "o", "-"))
        ax.plot(layers, sigma, marker=marker, color=color, linestyle=ls,
                linewidth=1.5, label=split)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Mean ensemble std")
    ax.set_title("Epistemic Uncertainty per Layer — Train / Val / Test")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(layers)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the LPBF Latent-Space surrogate on held-out test trajectories."
    )
    p.add_argument("--checkpoint",    type=str, required=True,
                   help="Path to latent_best.pt or latent_final.pt.")
    p.add_argument("--data_path",     type=str, required=True,
                   help="Path to the same pickled dataset used for training.")
    p.add_argument("--mesh_path",     type=str,
                   default="surrogate_model/mesh.mat",
                   help="Path to mesh.mat. Falls back to 1-D plots if absent.")
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--n_examples",    type=int,   default=3,
                   help="Number of example trajectories to visualise.")
    p.add_argument("--vmin",          type=float, default=300.0,
                   help="Color axis minimum [K] (matches MATLAB caxis lower bound).")
    p.add_argument("--vmax",          type=float, default=5000.0,
                   help="Color axis maximum [K] (matches MATLAB caxis upper bound).")
    p.add_argument("--out_dir",       type=str,   default="",
                   help="Output directory. Defaults to same folder as checkpoint.")
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

    # ── load mesh ─────────────────────────────────────────────────────────────
    triang = load_mesh(args.mesh_path)

    # ── load model ────────────────────────────────────────────────────────────
    model, state_mean, state_std, action_mean, action_std, roi_table = \
        load_latent_surrogate(args.checkpoint, device)
    state_mean_cpu = state_mean.cpu()
    state_std_cpu  = state_std.cpu()
    print(f"[evaluate] {model}")

    # ── load & split dataset (same seed → reproducible splits) ───────────────
    all_trajs = load_trajectories(args.data_path)
    train_trajs, val_trajs, test_trajs = split_trajectories(
        all_trajs,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    traj_len = len(test_trajs[0])

    ds_kwargs  = dict(
        state_mean=state_mean_cpu, state_std=state_std_cpu,
        action_mean=action_mean,   action_std=action_std,
        initial_temp=args.initial_temp,
    )
    loader_kw  = dict(batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers)

    def _make_loaders(trajs):
        tl = DataLoader(LatentTrajectoryDataset(trajs, **ds_kwargs), **loader_kw)
        fl = DataLoader(LatentSurrogateDataset(trajs,  **ds_kwargs), **loader_kw)
        return tl, fl

    split_loaders = {
        "train": _make_loaders(train_trajs),
        "val":   _make_loaders(val_trajs),
        "test":  _make_loaders(test_trajs),
    }

    # ── evaluate each split ───────────────────────────────────────────────────
    # Collected results for cross-split plots
    rmse_results  = {}   # {split: (ro_rmse, ss_rmse)}
    mae_results   = {}   # {split: ss_mae}
    sigma_results = {}   # {split: ro_sigma}

    # Summary scalars for the final table
    summary = {}   # {split: {metric: value}}

    test_preds, test_gts, test_actions, test_sigmas = None, None, None, None

    for split_name, (traj_loader, flat_loader) in split_loaders.items():
        print(f"\n[evaluate] {'═'*20} {split_name.upper()} SET {'═'*20}")

        # Loss components
        comps = evaluate_loss_components(model, flat_loader, device, roi_table)
        print(f"  recon_st  : {comps['recon_st']:.6f}")
        print(f"  recon_st1 : {comps['recon_st1']:.6f}")

        # Autoencoder round-trip RMSE
        recon_rmse = evaluate_recon(model, traj_loader, state_mean, state_std, device)
        print(f"  AE RMSE   : {recon_rmse:.4f} K")

        # Single-step (teacher-forced)
        ss_mae, ss_rmse = evaluate_single_step(
            model, traj_loader, state_mean_cpu, state_std_cpu, device, traj_len=traj_len
        )
        print(f"\n  {'Layer':>6}  {'SS MAE [K]':>12}  {'SS RMSE [K]':>12}")
        for i in range(traj_len):
            print(f"  {i+1:>6}  {ss_mae[i]:>12.3f}  {ss_rmse[i]:>12.3f}")
        print(f"  {'Mean':>6}  {ss_mae.mean():>12.3f}  {ss_rmse.mean():>12.3f}")

        # Auto-regressive rollout
        ro_rmse, ro_mae, ro_sigma, all_preds, all_gts, all_actions, all_sigmas = evaluate_rollout(
            model, traj_loader,
            state_mean_cpu, state_std_cpu,
            action_mean, action_std,
            device, traj_len=traj_len,
        )
        print(f"\n  {'Layer':>6}  {'RO RMSE [K]':>12}  {'RO MAE [K]':>12}  {'Epist σ':>10}")
        for i in range(traj_len):
            print(f"  {i+1:>6}  {ro_rmse[i]:>12.3f}  {ro_mae[i]:>12.3f}  {ro_sigma[i]:>10.4f}")
        print(f"  {'Mean':>6}  {ro_rmse.mean():>12.3f}  {ro_mae.mean():>12.3f}  {ro_sigma.mean():>10.4f}")

        # Stash for plots
        rmse_results[split_name]  = (ro_rmse, ss_rmse)
        mae_results[split_name]   = (ro_mae,  ss_mae)
        sigma_results[split_name] = ro_sigma

        summary[split_name] = {
            "recon_rmse [K]":  recon_rmse,
            "ss_mae [K]":      ss_mae.mean(),
            "ss_rmse [K]":     ss_rmse.mean(),
            "ro_mae [K]":      ro_mae.mean(),
            "ro_rmse [K]":     ro_rmse.mean(),
            "epist_std":       ro_sigma.mean(),
        }

        # Keep test trajectories for field plots
        if split_name == "test":
            test_preds, test_gts   = all_preds, all_gts
            test_actions, test_sigmas = all_actions, all_sigmas

    # ── consolidated summary table ────────────────────────────────────────────
    metrics = ["recon_rmse [K]", "ss_mae [K]", "ss_rmse [K]", "ro_mae [K]", "ro_rmse [K]", "epist_std"]
    col_w   = 14
    header  = f"{'Split':<8}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print(f"\n[evaluate] {'═'*len(header)}")
    print(f"[evaluate] SUMMARY")
    print(f"[evaluate] {'═'*len(header)}")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for split_name, vals in summary.items():
        row = f"  {split_name:<8}" + "".join(f"{vals[m]:>{col_w}.4f}" for m in metrics)
        print(row)
    print(f"[evaluate] {'═'*len(header)}\n")

    # ── per-layer mean GT temperature (for % annotations on plots) ───────────
    pct_refs = {}
    for split_name, (traj_loader, _) in split_loaders.items():
        pct_refs[split_name] = compute_per_layer_mean_gt(
            traj_loader, state_mean_cpu, state_std_cpu, device, traj_len=traj_len
        )

    # ── per-layer error source diagnostics ───────────────────────────────────
    print("[evaluate] Computing per-layer diagnostics ...")
    diag_by_split = {}
    for split_name, (traj_loader, _) in split_loaders.items():
        diag_by_split[split_name] = compute_per_layer_diagnostics(
            model, traj_loader, state_mean_cpu, state_std_cpu, device, traj_len=traj_len
        )

    plot_per_layer_diagnostics(diag_by_split,
                               os.path.join(out_dir, "per_layer_diagnostics.png"))

    # Print test-set diagnostic table
    test_diag = diag_by_split["test"]
    print(f"\n  {'Layer':>6}  {'ΔS mag [K]':>12}  {'AE RMSE [K]':>13}  {'Lat Trans Err':>14}")
    print(f"  {'-'*50}")
    for t in range(traj_len):
        print(f"  {t+1:>6}  {test_diag['delta_mag'][t]:>12.3f}"
              f"  {test_diag['ae_rmse'][t]:>13.3f}"
              f"  {test_diag['trans_lat_err'][t]:>14.4f}")

    # ── metric plots (all splits on same axes) ────────────────────────────────
    plot_per_layer_rmse(rmse_results,
                        os.path.join(out_dir, "per_layer_rmse.png"),
                        pct_refs=pct_refs)
    plot_mae_per_layer(mae_results,
                       os.path.join(out_dir, "per_layer_mae.png"),
                       pct_refs=pct_refs)
    plot_per_layer_uncertainty(sigma_results,
                               os.path.join(out_dir, "per_layer_uncertainty.png"))

    # ── per-trajectory field plots (test set only) ────────────────────────────
    n_ex = min(args.n_examples, len(test_preds))
    print(f"[evaluate] Saving field plots for {n_ex} test trajectories ...")
    for i in range(n_ex):
        plot_example_trajectory(
            test_preds[i], test_gts[i],
            actions=test_actions[i],
            sigmas=test_sigmas[i],
            traj_idx=i,
            out_dir=out_dir,
            triang=triang,
            vmin=args.vmin,
            vmax=args.vmax,
        )

    print(f"\n[evaluate] Complete. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
