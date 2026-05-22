"""
surrogate_model/evaluate.py
----------------------------
Evaluation and visualisation for a trained LPBF surrogate model.

Visualization matches test.py / MATLAB style:
  pdeplot(model, 'XYData', uFinal, 'Mesh','on', 'ColorMap','jet')
  colorbar; caxis([300 5000])

  → Python equivalent: matplotlib.tri.tripcolor with jet colormap,
    vmin=300, vmax=5000, drawn on the actual PDE triangular mesh.

Mesh file prerequisite
-----------------------
  Run surrogate_model/extract_mesh.m in MATLAB once to generate
  surrogate_model/mesh.mat  (contains 'nodes' and 'elements').
  If mesh.mat is absent the script falls back to a 1-D node-index plot.

Metrics reported
----------------
  1. Single-step MAE / RMSE  — per layer, teacher-forced
  2. Auto-regressive rollout RMSE — per layer (error accumulation test)

Plots produced (saved to --out_dir)
------------------------------------
  per_layer_rmse.png              — rollout vs single-step RMSE curves
  per_layer_mae.png               — single-step MAE bar chart
  example_traj{i}_layer{L}.png   — predicted vs GT 2-D temperature field

Usage
-----
    python -m surrogate_model.evaluate \\
        --checkpoint surrogate_model/runs/<ts>/surrogate_best.pt \\
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
from scipy.io import loadmat
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model.dataset import (
    load_trajectories,
    split_trajectories,
    build_normalizers,
    TrajectoryDataset,
)
from surrogate_model.train import load_surrogate


# =============================================================================
# Mesh loading
# =============================================================================

def load_mesh(mesh_path: str):
    """
    Load PDE mesh extracted by surrogate_model/extract_mesh.m.

    Returns
    -------
    triang : matplotlib.tri.Triangulation  (or None if file missing)
    """
    if not os.path.exists(mesh_path):
        print(f"[evaluate] mesh.mat not found at {mesh_path}. "
              "Run surrogate_model/extract_mesh.m in MATLAB first for 2-D plots. "
              "Falling back to 1-D node-index plots.")
        return None

    data     = loadmat(mesh_path)
    nodes    = data["nodes"]        # (2, N_nodes)  X,Y
    elements = data["elements"] - 1 # (3, N_tri) — convert MATLAB 1-index → 0-index

    x  = nodes[0, :].astype(np.float64)
    y  = nodes[1, :].astype(np.float64)
    tri = elements[:3, :].T.astype(np.int32)   # (N_tri, 3)

    triang = mtri.Triangulation(x, y, tri)
    print(f"[evaluate] Mesh loaded: {len(x)} nodes, {len(tri)} triangles. "
          f"X∈[{x.min():.2f},{x.max():.2f}], Y∈[{y.min():.2f},{y.max():.2f}]")
    return triang


# =============================================================================
# Core evaluation
# =============================================================================

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
    Auto-regressive rollout: feed model's own prediction as next input.

    Returns
    -------
    per_layer_rmse : (traj_len,)   RMSE in raw Kelvin per layer
    all_preds      : list of (T, D) np.ndarray  — predicted fields [K]
    all_gts        : list of (T, D) np.ndarray  — ground-truth fields [K]
    all_actions    : list of (T,)   np.ndarray  — laser power [W] per layer
    """
    model.eval()
    sq_errors = [[] for _ in range(traj_len)]
    all_preds, all_gts, all_actions = [], [], []

    sm = state_mean.to(device)
    ss = state_std.to(device)

    for traj_s, traj_a in test_loader:
        traj_s = traj_s.to(device)   # (B, T+1, D) normalised
        traj_a = traj_a.to(device)   # (B, T,   1) normalised
        B, T1, D = traj_s.shape
        T = T1 - 1

        s_pred_norm = traj_s[:, 0, :]   # start from ground-truth s_0
        pred_list, gt_list = [], []

        for t in range(min(T, traj_len)):
            s_pred_norm = model(s_pred_norm, traj_a[:, t, :])
            s_gt_norm   = traj_s[:, t + 1, :]

            # Denormalise → raw Kelvin
            s_pred_k = s_pred_norm * ss + sm
            s_gt_k   = s_gt_norm   * ss + sm

            se = ((s_pred_k - s_gt_k) ** 2).mean(dim=-1)   # (B,)
            sq_errors[t].extend(se.cpu().numpy().tolist())

            pred_list.append(s_pred_k.cpu().numpy())
            gt_list.append(s_gt_k.cpu().numpy())

        # Denormalise actions → raw laser power [W]
        actions_raw = (traj_a.squeeze(-1).cpu().numpy() * action_std
                       + action_mean)   # (B, T)

        for b in range(B):
            all_preds.append(  np.stack([p[b] for p in pred_list], axis=0))  # (T,D)
            all_gts.append(    np.stack([g[b] for g in gt_list],   axis=0))  # (T,D)
            all_actions.append(actions_raw[b, :min(T, traj_len)])             # (T,)

    per_layer_rmse = np.array([
        np.sqrt(np.mean(sq_errors[t])) for t in range(traj_len)
    ])
    return per_layer_rmse, all_preds, all_gts, all_actions


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
            s_in  = traj_s[:, t,     :]
            a_in  = traj_a[:, t,     :]
            s_gt  = traj_s[:, t + 1, :]

            s_pred  = model(s_in, a_in)
            diff_k  = (s_pred - s_gt) * ss     # in Kelvin

            abs_err[t].extend(diff_k.abs().mean(dim=-1).cpu().numpy().tolist())
            sq_err[t].extend((diff_k ** 2).mean(dim=-1).cpu().numpy().tolist())

    per_layer_mae  = np.array([np.mean(abs_err[t]) for t in range(traj_len)])
    per_layer_rmse = np.array([np.sqrt(np.mean(sq_err[t])) for t in range(traj_len)])
    return per_layer_mae, per_layer_rmse


# =============================================================================
# Plotting helpers
# =============================================================================

def _plot_field_2d(ax, values, triang, title, vmin=300, vmax=5000):
    """
    Draw a 2-D PDE temperature field on ax — matches MATLAB pdeplot style:
      pdeplot(model,'XYData',u,'Mesh','on','ColorMap','jet')
      colorbar; caxis([300 5000])

    Axes are stretched to fill the axes box (aspect='auto'), matching
    MATLAB's default behaviour — both axes span the same pixel width
    even though the physical domain is 12×3.
    """
    tpc = ax.tripcolor(triang, values, cmap="jet", vmin=vmin, vmax=vmax,
                       shading="gouraud")
    ax.triplot(triang, color="k", linewidth=0.15, alpha=0.3)  # mesh overlay
    plt.colorbar(tpc, ax=ax, label="Temperature [K]")
    ax.set_aspect("auto")   # stretch to fill axes box, like MATLAB default
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


def plot_example_trajectory(
    pred_traj:    np.ndarray,   # (T, D)  raw Kelvin
    gt_traj:      np.ndarray,   # (T, D)  raw Kelvin
    traj_idx:     int,
    out_dir:      str,
    triang,                     # mtri.Triangulation or None
    layers_to_show: list = None,
    vmin: float = 300,
    vmax: float = 5000,
) -> None:
    """
    For each selected layer, plot a 3-panel figure:
      left   — ground-truth temperature field  (2D or 1D)
      middle — predicted temperature field     (2D or 1D)
      right  — absolute error field            (2D or 1D)

    Matches MATLAB: jet colormap, caxis [300, 5000].
    """
    T = pred_traj.shape[0]
    if layers_to_show is None:
        layers_to_show = sorted({0, T // 2, T - 1})

    for layer in layers_to_show:
        if layer >= T:
            continue
        pred  = pred_traj[layer]     # (D,)
        gt    = gt_traj[layer]       # (D,)
        err   = np.abs(pred - gt)
        mae   = err.mean()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(
            f"Traj {traj_idx} — Layer {layer+1} | MAE = {mae:.2f} K",
            fontsize=10
        )

        if triang is not None:
            _plot_field_2d(axes[0], gt,   triang, "Ground truth [K]",  vmin, vmax)
            _plot_field_2d(axes[1], pred, triang, "Predicted [K]",     vmin, vmax)
            # Error: use full range so small errors are still visible
            tpc = axes[2].tripcolor(triang, err, cmap="hot",
                                    vmin=0, vmax=max(err.max(), 1.0),
                                    shading="gouraud")
            axes[2].triplot(triang, color="k", linewidth=0.15, alpha=0.3)
            plt.colorbar(tpc, ax=axes[2], label="|Error| [K]")
            axes[2].set_aspect("auto")
            axes[2].set_xlabel("X"); axes[2].set_ylabel("Y")
            axes[2].set_title(f"|Error| max={err.max():.1f} K", fontsize=9)
        else:
            _plot_field_1d(axes[0], gt,   "Ground truth [K]",  color="steelblue")
            _plot_field_1d(axes[1], pred, "Predicted [K]",     color="darkorange")
            axes[2].plot(err, linewidth=0.6, color="crimson")
            axes[2].set_xlabel("Node index")
            axes[2].set_ylabel("|Error| [K]")
            axes[2].set_title(f"|Error| max={err.max():.1f} K", fontsize=9)
            axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fname = os.path.join(out_dir, f"example_traj{traj_idx}_layer{layer+1}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[evaluate] Saved → {fname}")


def plot_per_layer_rmse(
    rollout_rmse:     np.ndarray,
    single_step_rmse: np.ndarray,
    out_path:         str,
) -> None:
    layers = np.arange(1, len(rollout_rmse) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layers, rollout_rmse,     marker="o", label="Auto-regressive rollout RMSE [K]")
    ax.plot(layers, single_step_rmse, marker="s", label="Single-step (teacher-forced) RMSE [K]",
            linestyle="--")
    best = int(np.argmin(rollout_rmse))
    ax.axvline(x=best + 1, color="grey", linestyle=":", linewidth=1,
               label=f"Best rollout layer {best+1}")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("RMSE [K]")
    ax.set_title("Surrogate Prediction Error per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


def plot_mae_per_layer(per_layer_mae: np.ndarray, out_path: str) -> None:
    layers = np.arange(1, len(per_layer_mae) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(layers, per_layer_mae, alpha=0.75)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("MAE [K]")
    ax.set_title("Single-Step MAE per Layer (teacher-forced)")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the LPBF surrogate model on held-out test trajectories."
    )
    p.add_argument("--checkpoint",    type=str, required=True,
                   help="Path to surrogate_best.pt or surrogate_final.pt.")
    p.add_argument("--data_path",     type=str, required=True,
                   help="Path to the same pickled dataset used for training.")
    p.add_argument("--mesh_path",     type=str,
                   default="surrogate_model/mesh.mat",
                   help="Path to mesh.mat produced by extract_mesh.m. "
                        "If absent, falls back to 1-D plots.")
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--n_examples",    type=int,   default=3,
                   help="Number of example trajectories to visualise.")
    p.add_argument("--vmin",          type=float, default=300.0,
                   help="Color axis minimum [K] (matches MATLAB caxis lower bound).")
    p.add_argument("--vmax",          type=float, default=5000.0,
                   help="Color axis maximum [K] (matches MATLAB caxis upper bound).")
    p.add_argument("--out_dir",       type=str,   default="",
                   help="Output directory. Defaults to same folder as checkpoint.")
    p.add_argument("--device",        type=str,   default="",
                   help="'cuda' or 'cpu'. Auto-detected if empty.")
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

    # ── load mesh (optional) ─────────────────────────────────────────────────
    triang = load_mesh(args.mesh_path)

    # ── load model ───────────────────────────────────────────────────────────
    model, state_mean, state_std, action_mean, action_std = load_surrogate(
        args.checkpoint, device=device
    )
    # Normalizers stay on CPU for dataset construction; evaluation functions
    # move them to device internally as needed.
    state_mean_cpu = state_mean.cpu()
    state_std_cpu  = state_std.cpu()
    print(f"[evaluate] {model}")

    # ── load & split dataset (same seed → identical test set) ────────────────
    all_trajs = load_trajectories(args.data_path)
    _, _, test_trajs = split_trajectories(
        all_trajs,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    test_ds = TrajectoryDataset(
        test_trajs, state_mean_cpu, state_std_cpu, action_mean, action_std,
        initial_temp=args.initial_temp,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)
    traj_len = len(test_trajs[0])

    # ── single-step evaluation ────────────────────────────────────────────────
    print("\n[evaluate] Single-step (teacher-forced) evaluation ...")
    ss_mae, ss_rmse = evaluate_single_step(
        model, test_loader, state_mean_cpu, state_std_cpu, device, traj_len=traj_len
    )
    print(f"\n{'Layer':>6}  {'MAE [K]':>10}  {'RMSE [K]':>10}")
    for i in range(traj_len):
        print(f"{i+1:>6}  {ss_mae[i]:>10.3f}  {ss_rmse[i]:>10.3f}")
    print(f"{'Mean':>6}  {ss_mae.mean():>10.3f}  {ss_rmse.mean():>10.3f}")

    # ── auto-regressive rollout ────────────────────────────────────────────────
    print("\n[evaluate] Auto-regressive rollout evaluation ...")
    ro_rmse, all_preds, all_gts = evaluate_rollout(
        model, test_loader, state_mean_cpu, state_std_cpu, device, traj_len=traj_len
    )
    print(f"\n{'Layer':>6}  {'Rollout RMSE [K]':>18}")
    for i in range(traj_len):
        print(f"{i+1:>6}  {ro_rmse[i]:>18.3f}")
    print(f"{'Mean':>6}  {ro_rmse.mean():>18.3f}")

    # ── metric plots ──────────────────────────────────────────────────────────
    plot_per_layer_rmse(ro_rmse, ss_rmse,
                        os.path.join(out_dir, "per_layer_rmse.png"))
    plot_mae_per_layer(ss_mae,
                       os.path.join(out_dir, "per_layer_mae.png"))

    # ── example field plots ───────────────────────────────────────────────────
    n_ex = min(args.n_examples, len(all_preds))
    for i in range(n_ex):
        plot_example_trajectory(
            all_preds[i], all_gts[i],
            traj_idx=i,
            out_dir=out_dir,
            triang=triang,
            vmin=args.vmin,
            vmax=args.vmax,
        )

    print(f"\n[evaluate] Complete. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
