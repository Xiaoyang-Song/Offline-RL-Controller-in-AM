"""
surrogate_model_latent/train.py
---------------------------------
Training script for the LPBF Latent-Space Ensemble Dynamics surrogate model.

Usage (single-step losses only):
    python -m surrogate_model_latent.train \\
        --data_path Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl

Usage (with multi-step rollout loss):
    python -m surrogate_model_latent.train \\
        --data_path Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl \\
        --rollout_steps 12 \\
        --rollout_weight 0.1

Loss terms
----------
  L_recon_st   : weighted MSE between decoder(encoder(s_t)) and s_t
  L_recon_st1  : mean over K members of weighted MSE between
                 decoder(z_t + μ_Δz_k) and s_{t+1}
  L_rollout    : k-step auto-regressive reconstruction loss (optional,
                 uses ensemble mean for stepping)

  total = recon_st_w·L_recon_st + recon_st1_w·L_recon_st1
        [+ rollout_w·L_rollout]

Physics-aware weights
---------------------
  Requires surrogate_model/mesh.mat (nodes + elements from the PDE mesh).
  If absent, uniform weights are used (a warning is printed).
  Weights are Gaussian-shaped per layer, centred on the layer's build height.

All outputs are written under --out_dir (default: surrogate_model_latent/runs/<timestamp>/).
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_latent.dataset import (
    load_trajectories,
    split_trajectories,
    build_normalizers,
    LatentSurrogateDataset,
    LatentTrajectoryDataset,
    compute_roi_weights_table,
)
from surrogate_model_latent.model import EnsembleLatentDynamicsModel


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the LPBF Latent-Space Ensemble Dynamics surrogate model."
    )

    # ── data ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_path",     type=str,  required=True)
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--seed",          type=int,   default=42)

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--latent_dim",    type=int,   default=64)
    p.add_argument("--n_ensemble",    type=int,   default=5,
                   help="Number of ensemble transition members K.")
    p.add_argument("--enc_hidden",    type=int,   default=256)
    p.add_argument("--enc_depth",     type=int,   default=3)
    p.add_argument("--trans_hidden",  type=int,   default=128)
    p.add_argument("--trans_depth",   type=int,   default=3)
    p.add_argument("--dec_hidden",    type=int,   default=256)
    p.add_argument("--dec_depth",     type=int,   default=3)
    p.add_argument("--dropout",       type=float, default=0.0)

    # ── loss weights ─────────────────────────────────────────────────────────
    p.add_argument("--recon_st_weight",  type=float, default=1.0,
                   help="Weight for autoencoder reconstruction loss L_recon_st.")
    p.add_argument("--recon_st1_weight", type=float, default=1.0,
                   help="Weight for per-member next-state prediction loss L_recon_st1.")

    # ── multi-step rollout ────────────────────────────────────────────────────
    p.add_argument("--rollout_steps",  type=int,   default=0,
                   help="Unroll k steps for rollout loss (0 = off; recommended: 12).")
    p.add_argument("--rollout_weight", type=float, default=0.1)

    # ── physics ROI weights ───────────────────────────────────────────────────
    p.add_argument("--mesh_path", type=str, default="")
    p.add_argument("--roi_boost",            type=float, default=5.0)
    p.add_argument("--roi_initial_fraction", type=float, default=0.4)
    p.add_argument("--roi_final_fraction",   type=float, default=0.5)
    p.add_argument("--roi_edge_sigma_frac",  type=float, default=0.05)

    # ── optimiser ─────────────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int,   default=30)
    p.add_argument("--num_workers",  type=int,   default=4)

    # ── output ───────────────────────────────────────────────────────────────
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--device",  type=str, default="")

    return p.parse_args()


# =============================================================================
# Loss functions
# =============================================================================

def weighted_mse(
    pred:    torch.Tensor,            # (..., D)
    target:  torch.Tensor,            # (..., D)
    weights: Optional[torch.Tensor],  # (..., D) or None
) -> torch.Tensor:
    sq = (pred - target).pow(2)
    if weights is not None:
        sq = sq * weights
    return sq.mean()


def compute_single_step_losses(
    model:         EnsembleLatentDynamicsModel,
    s:             torch.Tensor,              # (B, D)
    a:             torch.Tensor,              # (B, 1)
    s2:            torch.Tensor,              # (B, D)
    layer_indices: torch.Tensor,              # (B,)
    roi_table:     Optional[torch.Tensor],    # (n_layers, D) or None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (L_recon_st, L_recon_st1).

    L_recon_st   : autoencoder reconstruction of s_t  (shared enc/dec)
    L_recon_st1  : per-member next-state prediction, averaged over K members.
                   Uses a single batched decoder call for efficiency:
                   all K predicted z_{t+1} are stacked and decoded together.
    """
    s_t_recon, _, mu_deltas, z_t = model(s, a)
    # mu_deltas : (K, B, latent_dim)
    # z_t       : (B, latent_dim)

    w = roi_table[layer_indices] if roi_table is not None else None  # (B, D) or None

    L_recon_st = weighted_mse(s_t_recon, s, w)

    # Batched per-member reconstruction: decode all K predictions in one call
    K, B, Dz = mu_deltas.shape
    z_t1_preds = z_t.unsqueeze(0) + mu_deltas           # (K, B, latent_dim)
    s_preds    = model.decode(
        z_t1_preds.reshape(K * B, Dz)
    ).reshape(K, B, -1)                                  # (K, B, state_dim)

    s2_exp = s2.unsqueeze(0).expand(K, -1, -1)           # (K, B, D)
    w_exp  = w.unsqueeze(0).expand(K, -1, -1) if w is not None else None
    L_recon_st1 = weighted_mse(s_preds, s2_exp, w_exp)

    return L_recon_st, L_recon_st1


def compute_rollout_loss(
    model:     EnsembleLatentDynamicsModel,
    traj_s:    torch.Tensor,              # (B, T+1, D)
    traj_a:    torch.Tensor,              # (B, T,   1)
    k:         int,
    roi_table: Optional[torch.Tensor],    # (n_layers, D) or None
) -> torch.Tensor:
    """
    k-step auto-regressive reconstruction loss using ensemble mean for stepping.
    """
    B, T1, D = traj_s.shape
    z_t   = model.encode(traj_s[:, 0, :])
    total = torch.zeros(1, device=traj_s.device)

    for t in range(k):
        a_t            = traj_a[:, t, :]
        mu_mean, _     = model.predict_ensemble(z_t, a_t)
        z_t            = z_t + mu_mean
        s_pred         = model.decode(z_t)
        s_gt           = traj_s[:, t + 1, :]
        w              = roi_table[t].unsqueeze(0) if roi_table is not None else None
        total          = total + weighted_mse(s_pred, s_gt, w)

    return total / k


# =============================================================================
# Training / validation epoch
# =============================================================================

def run_epoch(
    model:       EnsembleLatentDynamicsModel,
    loader:      DataLoader,
    optimizer:   Optional[torch.optim.Optimizer],
    device:      str,
    roi_table:   Optional[torch.Tensor],
    recon_st_w:  float,
    recon_st1_w: float,
    rollout_k:   int,
    rollout_w:   float,
    is_traj:     bool,
) -> Tuple[float, Dict[str, float]]:
    """
    One full pass.  Returns (avg_total_loss, avg_components_dict).
    optimizer=None → evaluation mode (no grad).
    """
    training = optimizer is not None
    model.train(training)

    agg = defaultdict(float)
    n   = 0

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        if is_traj:
            for traj_s, traj_a in loader:
                traj_s = traj_s.to(device)   # (B, T+1, D)
                traj_a = traj_a.to(device)   # (B, T,   1)
                B, T1, D = traj_s.shape
                T = T1 - 1
                k = min(rollout_k, T)

                s_flat  = traj_s[:, :-1, :].reshape(B * T, D)
                a_flat  = traj_a.reshape(B * T, 1)
                s2_flat = traj_s[:, 1:,  :].reshape(B * T, D)
                layer_flat = (
                    torch.arange(T, device=device)
                    .unsqueeze(0).expand(B, -1).reshape(B * T)
                )

                L_rs, L_rs1 = compute_single_step_losses(
                    model, s_flat, a_flat, s2_flat, layer_flat, roi_table
                )
                L_rollout = (
                    compute_rollout_loss(model, traj_s, traj_a, k, roi_table)
                    if k > 1 else torch.zeros(1, device=device)
                )
                loss = recon_st_w * L_rs + recon_st1_w * L_rs1 + rollout_w * L_rollout

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                agg["total"]     += loss.item()
                agg["recon_st"]  += L_rs.item()
                agg["recon_st1"] += L_rs1.item()
                agg["rollout"]   += L_rollout.item()
                n += 1

        else:
            for s, a, s2, layer_idx in loader:
                s         = s.to(device)
                a         = a.to(device)
                s2        = s2.to(device)
                layer_idx = layer_idx.to(device)

                L_rs, L_rs1 = compute_single_step_losses(
                    model, s, a, s2, layer_idx, roi_table
                )
                loss = recon_st_w * L_rs + recon_st1_w * L_rs1

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                agg["total"]     += loss.item()
                agg["recon_st"]  += L_rs.item()
                agg["recon_st1"] += L_rs1.item()
                agg["rollout"]   += 0.0
                n += 1

    denom = max(n, 1)
    comps = {k: v / denom for k, v in agg.items()}
    return comps["total"], comps


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_loss_curves(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs  = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train total",  linewidth=1.5)
    ax.plot(epochs, val_losses,   label="Val total",    linewidth=1.5, linestyle="--")
    best_e = int(np.argmin(val_losses)) + 1
    ax.axvline(best_e, color="grey", linestyle=":", linewidth=1,
               label=f"Best val epoch {best_e}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted loss")
    ax.set_title("Latent Ensemble Surrogate — Total Loss Curves")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[train] Loss plot → {out_path}")


def plot_loss_components(train_comps, val_comps, out_path):
    keys   = ["recon_st", "recon_st1", "rollout"]
    titles = ["L_recon_st", "L_recon_st1", "L_rollout"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(train_comps["recon_st"]) + 1)
    for ax, k, title in zip(axes, keys, titles):
        ax.plot(epochs, train_comps[k], label="Train", linewidth=1.5)
        ax.plot(epochs, val_comps[k],   label="Val",   linewidth=1.5, linestyle="--")
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("Latent Ensemble Surrogate — Loss Components")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[train] Component loss plot → {out_path}")


# =============================================================================
# Checkpoint helpers
# =============================================================================

def _save_checkpoint(
    model:        EnsembleLatentDynamicsModel,
    state_mean:   torch.Tensor,
    state_std:    torch.Tensor,
    action_mean:  float,
    action_std:   float,
    roi_table_np: Optional[np.ndarray],
    args:         argparse.Namespace,
    epoch:        int,
    val_loss:     float,
    path:         str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_mean":       state_mean.cpu(),
            "state_std":        state_std.cpu(),
            "action_mean":      action_mean,
            "action_std":       action_std,
            "roi_table":        roi_table_np,
            "epoch":            epoch,
            "val_loss":         val_loss,
            "model_config": {
                "state_dim":    model.state_dim,
                "action_dim":   model.action_dim,
                "latent_dim":   model.latent_dim,
                "n_ensemble":   model.n_ensemble,
                "enc_hidden":   args.enc_hidden,
                "enc_depth":    args.enc_depth,
                "trans_hidden": args.trans_hidden,
                "trans_depth":  args.trans_depth,
                "dec_hidden":   args.dec_hidden,
                "dec_depth":    args.dec_depth,
                "dropout":      args.dropout,
            },
            "train_args": vars(args),
        },
        path,
    )


def load_latent_surrogate(checkpoint_path: str, device: str = "cpu"):
    """
    Load a saved ensemble latent surrogate checkpoint.

    Returns
    -------
    model       : EnsembleLatentDynamicsModel  (eval mode)
    state_mean  : torch.Tensor (state_dim,)
    state_std   : torch.Tensor (state_dim,)
    action_mean : float
    action_std  : float
    roi_table   : torch.Tensor (n_layers, state_dim) or None
    """
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg   = ckpt["model_config"]
    model = EnsembleLatentDynamicsModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    roi_table = None
    if ckpt.get("roi_table") is not None:
        roi_table = torch.tensor(ckpt["roi_table"], dtype=torch.float32, device=device)

    return (
        model,
        ckpt["state_mean"].to(device),
        ckpt["state_std"].to(device),
        ckpt["action_mean"],
        ckpt["action_std"],
        roi_table,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── output directory ─────────────────────────────────────────────────────
    if not args.out_dir:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("surrogate_model_latent", "runs", ts)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] Output dir : {out_dir}")
    print(f"[train] Device     : {device}")

    # ── load & split trajectories ────────────────────────────────────────────
    all_trajs = load_trajectories(args.data_path)
    train_trajs, val_trajs, _ = split_trajectories(
        all_trajs,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    state_mean, state_std, action_mean, action_std = build_normalizers(
        train_trajs, initial_temp=args.initial_temp
    )

    # ── physics ROI weights ───────────────────────────────────────────────────
    state_dim = state_mean.shape[0]
    n_layers  = len(train_trajs[0])

    mesh_path = args.mesh_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "surrogate_model", "mesh.mat"
    )

    roi_table_np: Optional[np.ndarray] = None
    if os.path.exists(mesh_path):
        data   = loadmat(mesh_path)
        nodes  = data["nodes"]
        roi_table_np = compute_roi_weights_table(
            nodes, n_layers=n_layers,
            initial_fraction=args.roi_initial_fraction,
            final_fraction=args.roi_final_fraction,
            roi_boost=args.roi_boost,
            edge_sigma_frac=args.roi_edge_sigma_frac,
        )
        print(f"[train] ROI weights loaded from mesh: {mesh_path}")
    else:
        print(f"[train] mesh.mat not found at {mesh_path} — using uniform weights.")

    roi_table_t: Optional[torch.Tensor] = None
    if roi_table_np is not None:
        roi_table_t = torch.tensor(roi_table_np, dtype=torch.float32, device=device)

    # ── datasets ─────────────────────────────────────────────────────────────
    use_rollout = args.rollout_steps > 1
    ds_kwargs   = dict(
        state_mean=state_mean, state_std=state_std,
        action_mean=action_mean, action_std=action_std,
        initial_temp=args.initial_temp,
    )
    if use_rollout:
        train_ds = LatentTrajectoryDataset(train_trajs, **ds_kwargs)
        val_ds   = LatentTrajectoryDataset(val_trajs,   **ds_kwargs)
    else:
        train_ds = LatentSurrogateDataset(train_trajs, **ds_kwargs)
        val_ds   = LatentSurrogateDataset(val_trajs,   **ds_kwargs)

    loader_kw    = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    # ── model ─────────────────────────────────────────────────────────────────
    model = EnsembleLatentDynamicsModel(
        state_dim=state_dim,
        action_dim=1,
        latent_dim=args.latent_dim,
        n_ensemble=args.n_ensemble,
        enc_hidden=args.enc_hidden,
        enc_depth=args.enc_depth,
        trans_hidden=args.trans_hidden,
        trans_depth=args.trans_depth,
        dec_hidden=args.dec_hidden,
        dec_depth=args.dec_depth,
        dropout=args.dropout,
    ).to(device)
    print(f"[train] {model}")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100
    )

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    epochs_no_impr = 0
    train_losses, val_losses = [], []
    train_comps_hist: Dict[str, list] = defaultdict(list)
    val_comps_hist:   Dict[str, list] = defaultdict(list)

    ckpt_best  = os.path.join(out_dir, "latent_best.pt")
    ckpt_final = os.path.join(out_dir, "latent_final.pt")

    epoch_kw = dict(
        roi_table=roi_table_t,
        recon_st_w=args.recon_st_weight,
        recon_st1_w=args.recon_st1_weight,
        rollout_k=args.rollout_steps,
        rollout_w=args.rollout_weight,
        is_traj=use_rollout,
    )

    mode_str = f"rollout k={args.rollout_steps}" if use_rollout else "single-step only"
    print(f"\n[train] Starting training for up to {args.epochs} epochs "
          f"(patience={args.patience})")
    print(f"[train] Loss mode : {mode_str}")
    print(f"[train] Weights   → recon_st={args.recon_st_weight}  "
          f"recon_st1={args.recon_st1_weight}  rollout={args.rollout_weight}")
    print(f"[train] Ensemble  → K={args.n_ensemble} members")
    print("-" * 80)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):

        tr_loss, tr_comps = run_epoch(model, train_loader, optimizer, device, **epoch_kw)
        va_loss, va_comps = run_epoch(model, val_loader,   None,      device, **epoch_kw)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        for k in tr_comps:
            train_comps_hist[k].append(tr_comps[k])
            val_comps_hist[k].append(va_comps[k])

        improved = va_loss < best_val_loss
        if improved:
            best_val_loss  = va_loss
            epochs_no_impr = 0
            _save_checkpoint(model, state_mean, state_std, action_mean, action_std,
                             roi_table_np, args, epoch, best_val_loss, ckpt_best)
            marker = " ✓ best"
        else:
            epochs_no_impr += 1
            marker = f" (no improvement {epochs_no_impr}/{args.patience})"

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train {tr_loss:.5f} "
            f"[rs={tr_comps['recon_st']:.4f} rs1={tr_comps['recon_st1']:.4f} "
            f"ro={tr_comps['rollout']:.4f}] | "
            f"val {va_loss:.5f} | lr {lr_now:.2e} | {elapsed:6.1f}s{marker}"
        )

        if epochs_no_impr >= args.patience:
            print(f"[train] Early stopping at epoch {epoch}.")
            break

    # ── final checkpoint + plots ──────────────────────────────────────────────
    _save_checkpoint(model, state_mean, state_std, action_mean, action_std,
                     roi_table_np, args, epoch, va_loss, ckpt_final)
    plot_loss_curves(train_losses, val_losses,
                     os.path.join(out_dir, "loss_curves.png"))
    plot_loss_components(train_comps_hist, val_comps_hist,
                         os.path.join(out_dir, "loss_components.png"))

    print(f"\n[train] Done.  Best val loss : {best_val_loss:.6f}")
    print(f"[train] Best checkpoint  : {ckpt_best}")
    print(f"[train] Final checkpoint : {ckpt_final}")


if __name__ == "__main__":
    main()
