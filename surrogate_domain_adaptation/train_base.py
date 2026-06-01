"""
surrogate_domain_adaptation/train_base.py
------------------------------------------
Train the base EnsembleLatentDynamicsModel on pooled source-domain data
(cooling times ct_050 to ct_100).  The resulting checkpoint is the
starting point for few-shot domain adaptation to ct_150.

Architecture and loss are identical to surrogate_model_latent/train.py;
the only difference is that data is loaded from per-domain .mat files
rather than a single .pkl.

Usage
-----
    python -m surrogate_domain_adaptation.train_base

    # Custom source cooling times and trajectories per domain:
    python -m surrogate_domain_adaptation.train_base \\
        --ct_list 50 60 70 80 90 100 \\
        --n_trajs_each 200

All outputs → surrogate_domain_adaptation/runs/base/<timestamp>/
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_domain_adaptation.dataset import (
    load_source_domains,
    split_trajectories,
    build_normalizers,
    MultiDomainFlatDataset,
    MultiDomainTrajectoryDataset,
    DATA_ROOT,
    SOURCE_COOLING_TIMES,
)
from surrogate_model_latent.model import EnsembleLatentDynamicsModel
from surrogate_model_latent.train import (
    compute_single_step_losses,
    compute_rollout_loss,
    weighted_mse,
)
from surrogate_model_latent.dataset import compute_roi_weights_table


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train base surrogate on pooled source cooling-time domains."
    )

    # ── data ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_root",    type=str,  default=DATA_ROOT)
    p.add_argument("--ct_list",      type=int,  nargs="+",
                   default=SOURCE_COOLING_TIMES,
                   help="Source cooling times (×10, e.g. 50 60 70 80 90 100).")
    p.add_argument("--n_trajs_each", type=int,  default=None,
                   help="Max trajectories per domain (None = all 300).")
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--seed",          type=int,   default=42)

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--latent_dim",      type=int,   default=64)
    p.add_argument("--n_ensemble",      type=int,   default=5)
    p.add_argument("--n_layers",        type=int,   default=12)
    p.add_argument("--layer_embed_dim", type=int,   default=8)
    p.add_argument("--enc_hidden",      type=int,   default=256)
    p.add_argument("--enc_depth",       type=int,   default=3)
    p.add_argument("--trans_hidden",    type=int,   default=128)
    p.add_argument("--trans_depth",     type=int,   default=3)
    p.add_argument("--dec_hidden",      type=int,   default=256)
    p.add_argument("--dec_depth",       type=int,   default=3)
    p.add_argument("--dropout",         type=float, default=0.0)

    # ── loss ─────────────────────────────────────────────────────────────────
    p.add_argument("--recon_st_weight",  type=float, default=1.0)
    p.add_argument("--recon_st1_weight", type=float, default=1.0)
    p.add_argument("--rollout_steps",    type=int,   default=12)
    p.add_argument("--rollout_weight",   type=float, default=0.1)

    # ── physics ROI ───────────────────────────────────────────────────────────
    p.add_argument("--mesh_path",            type=str,   default="")
    p.add_argument("--roi_boost",            type=float, default=5.0)
    p.add_argument("--roi_initial_fraction", type=float, default=0.4)
    p.add_argument("--roi_final_fraction",   type=float, default=0.5)
    p.add_argument("--roi_edge_sigma_frac",  type=float, default=0.05)

    # ── optimiser ─────────────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int,   default=30)
    p.add_argument("--num_workers",  type=int,   default=4)

    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--device",  type=str, default="")

    return p.parse_args()


# =============================================================================
# Training / validation epoch  (mirrors surrogate_model_latent/train.py)
# =============================================================================

def run_epoch(
    model,
    loader,
    optimizer,
    device,
    roi_table,
    recon_st_w, recon_st1_w, rollout_k, rollout_w, is_traj,
):
    training = optimizer is not None
    model.train(training)
    agg = defaultdict(float)
    n   = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        if is_traj:
            for traj_s, traj_a in loader:
                traj_s = traj_s.to(device)
                traj_a = traj_a.to(device)
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
                L_ro = (
                    compute_rollout_loss(model, traj_s, traj_a, k, roi_table)
                    if k > 1 else torch.zeros(1, device=device)
                )
                loss = recon_st_w * L_rs + recon_st1_w * L_rs1 + rollout_w * L_ro

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                agg["total"]     += loss.item()
                agg["recon_st"]  += L_rs.item()
                agg["recon_st1"] += L_rs1.item()
                agg["rollout"]   += L_ro.item()
                n += 1
        else:
            for s, a, s2, layer_idx in loader:
                s, a, s2, layer_idx = (
                    s.to(device), a.to(device), s2.to(device), layer_idx.to(device)
                )
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
# Checkpoint save / load
# =============================================================================

def save_base_checkpoint(
    model, state_mean, state_std, action_mean, action_std,
    roi_table_np, args, epoch, val_loss, path,
):
    torch.save({
        "model_state_dict": model.state_dict(),
        "state_mean":       state_mean.cpu(),
        "state_std":        state_std.cpu(),
        "action_mean":      action_mean,
        "action_std":       action_std,
        "roi_table":        roi_table_np,
        "epoch":            epoch,
        "val_loss":         val_loss,
        "model_config": {
            "state_dim":       model.state_dim,
            "action_dim":      model.action_dim,
            "latent_dim":      model.latent_dim,
            "n_ensemble":      model.n_ensemble,
            "n_layers":        model.n_layers,
            "layer_embed_dim": model.layer_embed_dim,
            "enc_hidden":      args.enc_hidden,
            "enc_depth":       args.enc_depth,
            "trans_hidden":    args.trans_hidden,
            "trans_depth":     args.trans_depth,
            "dec_hidden":      args.dec_hidden,
            "dec_depth":       args.dec_depth,
            "dropout":         args.dropout,
        },
        "train_args": vars(args),
        "source_ct":  args.ct_list,
    }, path)


def load_base_checkpoint(path: str, device: str = "cpu"):
    """Return (model, state_mean, state_std, action_mean, action_std, roi_table)."""
    ckpt  = torch.load(path, map_location=device, weights_only=False)
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
# Plots
# =============================================================================

def _plot_curves(train_vals, val_vals, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ep = range(1, len(train_vals) + 1)
    ax.plot(ep, train_vals, label="Train", linewidth=1.5)
    ax.plot(ep, val_vals,   label="Val",   linewidth=1.5, linestyle="--")
    best = int(np.argmin(val_vals)) + 1
    ax.axvline(best, color="grey", linestyle=":", linewidth=1, label=f"Best ep {best}")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[train] → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if not args.out_dir:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("surrogate_domain_adaptation", "runs", "base", ts)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] Output dir : {out_dir}")
    print(f"[train] Device     : {device}")
    print(f"[train] Source CTs : ct_{'_'.join(str(c) for c in args.ct_list)}")

    # ── load source data ─────────────────────────────────────────────────────
    all_trajs = load_source_domains(
        data_root=args.data_root,
        ct_list=args.ct_list,
        n_trajs_each=args.n_trajs_each,
    )
    train_trajs, val_trajs, _ = split_trajectories(
        all_trajs, val_fraction=args.val_fraction,
        test_fraction=args.test_fraction, seed=args.seed,
    )
    state_mean, state_std, action_mean, action_std = build_normalizers(
        train_trajs, initial_temp=args.initial_temp
    )

    # ── ROI weights ───────────────────────────────────────────────────────────
    state_dim  = state_mean.shape[0]
    n_layers   = args.n_layers
    mesh_path  = args.mesh_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "surrogate_model", "mesh.mat"
    )
    roi_table_np = None
    if os.path.exists(mesh_path):
        data = loadmat(mesh_path)
        roi_table_np = compute_roi_weights_table(
            data["nodes"], n_layers=n_layers,
            initial_fraction=args.roi_initial_fraction,
            final_fraction=args.roi_final_fraction,
            roi_boost=args.roi_boost,
            edge_sigma_frac=args.roi_edge_sigma_frac,
        )
        print(f"[train] ROI weights from {mesh_path}")
    roi_table_t = (
        torch.tensor(roi_table_np, dtype=torch.float32, device=device)
        if roi_table_np is not None else None
    )

    # ── datasets ─────────────────────────────────────────────────────────────
    use_rollout = args.rollout_steps > 1
    ds_kw = dict(state_mean=state_mean, state_std=state_std,
                 action_mean=action_mean, action_std=action_std,
                 initial_temp=args.initial_temp)
    DS    = MultiDomainTrajectoryDataset if use_rollout else MultiDomainFlatDataset
    ldr_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=(device == "cuda"))
    train_loader = DataLoader(DS(train_trajs, **ds_kw), shuffle=True,  **ldr_kw)
    val_loader   = DataLoader(DS(val_trajs,   **ds_kw), shuffle=False, **ldr_kw)

    # ── model ─────────────────────────────────────────────────────────────────
    model = EnsembleLatentDynamicsModel(
        state_dim=state_dim, action_dim=1,
        latent_dim=args.latent_dim, n_ensemble=args.n_ensemble,
        n_layers=args.n_layers, layer_embed_dim=args.layer_embed_dim,
        enc_hidden=args.enc_hidden, enc_depth=args.enc_depth,
        trans_hidden=args.trans_hidden, trans_depth=args.trans_depth,
        dec_hidden=args.dec_hidden, dec_depth=args.dec_depth,
        dropout=args.dropout,
    ).to(device)
    print(f"[train] {model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/100)

    # ── training loop ─────────────────────────────────────────────────────────
    best_val, patience_cnt = float("inf"), 0
    train_losses, val_losses = [], []
    ckpt_best  = os.path.join(out_dir, "base_best.pt")
    ckpt_final = os.path.join(out_dir, "base_final.pt")
    epoch_kw = dict(roi_table=roi_table_t,
                    recon_st_w=args.recon_st_weight, recon_st1_w=args.recon_st1_weight,
                    rollout_k=args.rollout_steps, rollout_w=args.rollout_weight,
                    is_traj=use_rollout)
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_c = run_epoch(model, train_loader, optimizer, device, **epoch_kw)
        va_loss, va_c = run_epoch(model, val_loader,   None,      device, **epoch_kw)
        scheduler.step()
        train_losses.append(tr_loss); val_losses.append(va_loss)

        if va_loss < best_val:
            best_val, patience_cnt = va_loss, 0
            save_base_checkpoint(model, state_mean, state_std, action_mean, action_std,
                                  roi_table_np, args, epoch, best_val, ckpt_best)
            marker = " ✓"
        else:
            patience_cnt += 1
            marker = f" ({patience_cnt}/{args.patience})"

        if epoch % 10 == 0 or epoch == 1:
            print(f"Ep {epoch:4d}/{args.epochs} | "
                  f"train {tr_loss:.5f} [rs={tr_c['recon_st']:.4f} "
                  f"rs1={tr_c['recon_st1']:.4f} ro={tr_c['rollout']:.4f}] | "
                  f"val {va_loss:.5f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"{time.time()-t0:.0f}s{marker}")

        if patience_cnt >= args.patience:
            print(f"[train] Early stop at epoch {epoch}.")
            break

    save_base_checkpoint(model, state_mean, state_std, action_mean, action_std,
                          roi_table_np, args, epoch, va_loss, ckpt_final)
    _plot_curves(train_losses, val_losses, "Loss", "Base surrogate — source domains",
                 os.path.join(out_dir, "loss_curves.png"))
    print(f"\n[train] Best val loss  : {best_val:.6f}")
    print(f"[train] Best checkpoint: {ckpt_best}")


if __name__ == "__main__":
    main()
