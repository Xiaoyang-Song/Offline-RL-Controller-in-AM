"""
surrogate_model/train.py
------------------------
Training script for the LPBF Residual Dynamics surrogate model.

Usage (single-step loss only):
    python -m surrogate_model.train \\
        --data_path Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl

Usage (with multi-step rollout loss):
    python -m surrogate_model.train \\
        --data_path Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl \\
        --rollout_steps 6 \\
        --rollout_weight 0.5

All outputs are written under --out_dir (default: surrogate_model/runs/<timestamp>/).
"""

import argparse
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── make sure repo root is on the path when called as a script ───────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model.dataset import (
    load_trajectories,
    split_trajectories,
    build_normalizers,
    SurrogateDataset,
    TrajectoryDataset,
)
from surrogate_model.model import ResidualDynamicsModel


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the LPBF Residual Dynamics surrogate model."
    )

    # ── data ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to the pickled trajectory dataset (.pkl).")
    p.add_argument("--val_fraction",  type=float, default=0.10,
                   help="Fraction of trajectories held out for validation.")
    p.add_argument("--test_fraction", type=float, default=0.10,
                   help="Fraction of trajectories held out for test.")
    p.add_argument("--initial_temp",  type=float, default=300.0,
                   help="Initial uniform temperature [K] for the first state.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--hidden",  type=int,   default=512,
                   help="Hidden layer width.")
    p.add_argument("--depth",   type=int,   default=4,
                   help="Number of ResidualBlocks in the trunk.")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout probability inside each block (0 = off).")

    # ── training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--patience",    type=int,   default=20,
                   help="Early-stopping patience (epochs without val improvement).")

    # ── multi-step rollout loss ───────────────────────────────────────────────
    p.add_argument("--rollout_steps",  type=int,   default=0,
                   help="Unroll this many steps for rollout loss (0 = off). "
                        "E.g. 6 means the model sees 1-step AND 6-step loss.")
    p.add_argument("--rollout_weight", type=float, default=0.5,
                   help="Weight of multi-step rollout loss vs 1-step MSE. "
                        "total_loss = (1 - w)*L1 + w*L_rollout")

    # ── output ───────────────────────────────────────────────────────────────
    p.add_argument("--out_dir", type=str, default="",
                   help="Directory for checkpoints + plots. "
                        "Defaults to surrogate_model/runs/<timestamp>/.")
    p.add_argument("--device",  type=str, default="",
                   help="'cuda' or 'cpu'. Auto-detected if empty.")

    return p.parse_args()


# =============================================================================
# Loss functions
# =============================================================================

def single_step_loss(
    model:   ResidualDynamicsModel,
    states:  torch.Tensor,   # (B, D) normalised
    actions: torch.Tensor,   # (B, 1) normalised
    targets: torch.Tensor,   # (B, D) normalised
) -> torch.Tensor:
    preds = model(states, actions)
    return nn.functional.mse_loss(preds, targets)


def rollout_loss(
    model:          ResidualDynamicsModel,
    traj_states:    torch.Tensor,   # (B, T+1, D) normalised
    traj_actions:   torch.Tensor,   # (B, T,   1) normalised
    k:              int,            # number of steps to unroll
) -> torch.Tensor:
    """
    Auto-regressive rollout loss: unroll the model for k steps from s_0,
    feeding its own predictions as input at each step (no teacher forcing).

    loss = (1/k) * sum_{t=1}^{k} MSE(ŝ_t, s_t)
    """
    B, T1, D = traj_states.shape
    assert k <= T1 - 1, f"rollout_steps ({k}) must be ≤ trajectory length ({T1-1})"

    s_pred = traj_states[:, 0, :]   # (B, D)  — start from ground-truth s_0
    total  = torch.tensor(0.0, device=traj_states.device)

    for t in range(k):
        a_t    = traj_actions[:, t, :]          # (B, 1)
        s_pred = model(s_pred, a_t)             # (B, D) — model's own prediction
        s_gt   = traj_states[:, t + 1, :]       # (B, D)
        total  = total + nn.functional.mse_loss(s_pred, s_gt)

    return total / k


# =============================================================================
# Training and validation loops
# =============================================================================

def run_epoch(
    model:       ResidualDynamicsModel,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer | None,
    device:      str,
    rollout_k:   int   = 0,
    rollout_w:   float = 0.5,
    is_traj:     bool  = False,
) -> float:
    """
    One full pass over `loader`.
    If optimizer is None → evaluation mode (no grad).
    is_traj : True when loader yields (traj_states, traj_actions) from TrajectoryDataset.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        if is_traj:
            # ── multi-step rollout mode ────────────────────────────────────
            for traj_s, traj_a in loader:
                traj_s = traj_s.to(device)   # (B, T+1, D)
                traj_a = traj_a.to(device)   # (B, T,   1)

                T = traj_a.shape[1]
                k = min(rollout_k, T)

                # 1-step loss on all transitions in this batch
                s_flat  = traj_s[:, :-1, :].reshape(-1, traj_s.shape[-1])   # (B*T, D)
                a_flat  = traj_a.reshape(-1, 1)                              # (B*T, 1)
                s2_flat = traj_s[:,  1:, :].reshape(-1, traj_s.shape[-1])   # (B*T, D)
                L1 = single_step_loss(model, s_flat, a_flat, s2_flat)

                # k-step rollout loss
                Lk = rollout_loss(model, traj_s, traj_a, k) if k > 1 else L1

                loss = (1.0 - rollout_w) * L1 + rollout_w * Lk

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

        else:
            # ── single-step mode ──────────────────────────────────────────
            for s, a, s2 in loader:
                s  = s.to(device)
                a  = a.to(device)
                s2 = s2.to(device)

                loss = single_step_loss(model, s, a, s2)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

    return total_loss / max(n_batches, 1)


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_losses(
    train_losses: list,
    val_losses:   list,
    out_path:     str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train MSE",      linewidth=1.5)
    ax.plot(epochs, val_losses,   label="Val MSE",        linewidth=1.5, linestyle="--")
    ax.axvline(
        x=int(np.argmin(val_losses)) + 1,
        color="grey", linestyle=":", linewidth=1,
        label=f"Best val epoch {int(np.argmin(val_losses))+1}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalised space)")
    ax.set_title("Surrogate Training — Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[train] Loss plot saved → {out_path}")


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
        out_dir = os.path.join("surrogate_model", "runs", ts)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] Output dir: {out_dir}")
    print(f"[train] Device    : {device}")

    # ── load & split trajectories ────────────────────────────────────────────
    all_trajs = load_trajectories(args.data_path)
    train_trajs, val_trajs, test_trajs = split_trajectories(
        all_trajs,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    # ── normalisation stats (training set only) ───────────────────────────────
    state_mean, state_std, action_mean, action_std = build_normalizers(
        train_trajs, initial_temp=args.initial_temp
    )

    # ── datasets ─────────────────────────────────────────────────────────────
    use_rollout = args.rollout_steps > 1

    if use_rollout:
        train_ds = TrajectoryDataset(
            train_trajs, state_mean, state_std, action_mean, action_std,
            initial_temp=args.initial_temp,
        )
        val_ds = TrajectoryDataset(
            val_trajs, state_mean, state_std, action_mean, action_std,
            initial_temp=args.initial_temp,
        )
    else:
        train_ds = SurrogateDataset(
            train_trajs, state_mean, state_std, action_mean, action_std,
            initial_temp=args.initial_temp,
        )
        val_ds = SurrogateDataset(
            val_trajs, state_mean, state_std, action_mean, action_std,
            initial_temp=args.initial_temp,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    # ── model ─────────────────────────────────────────────────────────────────
    state_dim = state_mean.shape[0]
    model = ResidualDynamicsModel(
        state_dim=state_dim,
        action_dim=1,
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    print(f"[train] {model}")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine annealing: smoothly decays LR to lr/100 by the end of training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100
    )

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    epochs_no_impr = 0
    train_losses   = []
    val_losses     = []

    ckpt_best  = os.path.join(out_dir, "surrogate_best.pt")
    ckpt_final = os.path.join(out_dir, "surrogate_final.pt")

    print(f"\n[train] Starting training for up to {args.epochs} epochs "
          f"(patience={args.patience})")
    print(f"[train] Loss mode: {'rollout k=' + str(args.rollout_steps) if use_rollout else '1-step only'}")
    print("-" * 65)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):

        tr_loss = run_epoch(
            model, train_loader, optimizer, device,
            rollout_k=args.rollout_steps,
            rollout_w=args.rollout_weight,
            is_traj=use_rollout,
        )
        va_loss = run_epoch(
            model, val_loader, None, device,
            rollout_k=args.rollout_steps,
            rollout_w=args.rollout_weight,
            is_traj=use_rollout,
        )
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        improved = va_loss < best_val_loss
        if improved:
            best_val_loss  = va_loss
            epochs_no_impr = 0
            _save_checkpoint(model, state_mean, state_std, action_mean, action_std,
                             args, epoch, best_val_loss, ckpt_best)
            marker = " ✓ best"
        else:
            epochs_no_impr += 1
            marker = f" (no improvement {epochs_no_impr}/{args.patience})"

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train {tr_loss:.6f} | val {va_loss:.6f} | "
            f"lr {lr_now:.2e} | {elapsed:6.1f}s{marker}"
        )

        if epochs_no_impr >= args.patience:
            print(f"[train] Early stopping at epoch {epoch}.")
            break

    # ── save final checkpoint + plots ─────────────────────────────────────────
    _save_checkpoint(model, state_mean, state_std, action_mean, action_std,
                     args, epoch, va_loss, ckpt_final)
    plot_losses(train_losses, val_losses,
                os.path.join(out_dir, "loss_curves.png"))

    print(f"\n[train] Done.  Best val MSE: {best_val_loss:.6f}")
    print(f"[train] Best checkpoint : {ckpt_best}")
    print(f"[train] Final checkpoint: {ckpt_final}")


# =============================================================================
# Checkpoint helpers
# =============================================================================

def _save_checkpoint(
    model:        ResidualDynamicsModel,
    state_mean:   torch.Tensor,
    state_std:    torch.Tensor,
    action_mean:  float,
    action_std:   float,
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
            "epoch":            epoch,
            "val_loss":         val_loss,
            "model_config": {
                "state_dim":  model.state_dim,
                "action_dim": model.action_dim,
                "hidden":     args.hidden,
                "depth":      args.depth,
                "dropout":    args.dropout,
            },
            "train_args": vars(args),
        },
        path,
    )


def load_surrogate(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple:
    """
    Load a saved surrogate checkpoint.

    Returns
    -------
    model       : ResidualDynamicsModel  (eval mode)
    state_mean  : torch.Tensor (1053,)
    state_std   : torch.Tensor (1053,)
    action_mean : float
    action_std  : float
    """
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg    = ckpt["model_config"]
    model  = ResidualDynamicsModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return (
        model,
        ckpt["state_mean"].to(device),
        ckpt["state_std"].to(device),
        ckpt["action_mean"],
        ckpt["action_std"],
    )


# =============================================================================
if __name__ == "__main__":
    main()
