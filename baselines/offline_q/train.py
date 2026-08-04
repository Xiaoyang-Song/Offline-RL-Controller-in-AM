"""
baselines/offline_q/train.py
--------------------------------
Baseline 1: offline (batch/fitted) Q-learning, discrete action space, RAW
temperature field — no surrogate model, no environment interaction, no
online rollouts. Standard offline DQN: Bellman backups computed entirely
over a FIXED buffer built once from the pickled dataset
(baselines/common/data_utils.build_offline_transitions), with a target
network and periodic hard sync. This is as close to "just run Q-learning on
the raw dataset" as it gets.

MDP used here (single-step, no two-stage decomposition — Q-learning doesn't
care how a transition happened, only that (s, a, r, s') occurred):
    s      = temperature field before this layer's laser pass
    a      = laser power [W], snapped to the training data's discrete grid (100:10:400)
    r      = reward (already -meanDeviation of the end-of-heating field)
    s'     = temperature field before the NEXT layer (this layer's post-cooling field)

Usage
-----
    python -m baselines.offline_q.train \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --epochs 50
"""

import argparse
import copy
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.dataset_v2 import load_trajectories, split_trajectories
from baselines.common.data_utils import build_offline_transitions
from baselines.offline_q.model import RawQNet, ACTION_GRID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline 1: offline (fitted) Q-learning on the raw dataset.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--n_layers",      type=int,   default=12)

    p.add_argument("--hidden",          type=int, default=256)
    p.add_argument("--depth",           type=int, default=3)
    p.add_argument("--layer_embed_dim", type=int, default=8)

    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--target_sync_freq", type=int, default=2,
                   help="Hard-sync the target network every this many epochs.")
    p.add_argument("--weight_decay", type=float, default=1e-5)

    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--device",  type=str, default="")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def _snap_to_grid(actions: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Nearest-grid index for each action (should be an exact match — the
    dataset's laser powers ARE drawn from this grid)."""
    return np.abs(actions[:, None] - grid[None, :]).argmin(axis=1).astype(np.int64)


def _to_tensors(tr: dict, action_idx: np.ndarray, device: str):
    return (
        torch.tensor(tr["s"],         dtype=torch.float32, device=device),
        torch.tensor(action_idx,      dtype=torch.long,    device=device),
        torch.tensor(tr["r"],         dtype=torch.float32, device=device),
        torch.tensor(tr["s2"],        dtype=torch.float32, device=device),
        torch.tensor(tr["layer"],     dtype=torch.long,    device=device),
        torch.tensor(tr["done"],      dtype=torch.bool,    device=device),
        torch.tensor(tr["cool_time"], dtype=torch.float32, device=device),
    )


@torch.no_grad()
def _eval_loss(qnet, target, s, a, r, s2, layer, done, cool, gamma, n_layers, batch_size=4096):
    total, n = 0.0, 0
    for i in range(0, s.shape[0], batch_size):
        sb, ab, rb, s2b, lb, db, cb = (t[i:i + batch_size] for t in (s, a, r, s2, layer, done, cool))
        next_layer = (lb + 1).clamp(max=n_layers - 1)
        q_next = target(s2b, next_layer, cb).max(dim=-1).values
        y = rb + gamma * (~db).float() * q_next
        q_pred = qnet(sb, lb, cb).gather(-1, ab.unsqueeze(-1)).squeeze(-1)
        total += nn.functional.mse_loss(q_pred, y, reduction="sum").item()
        n += sb.shape[0]
    return total / max(n, 1)


def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.out_dir or os.path.join("baselines", "offline_q", "runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 65)
    print("[offline_q] Baseline 1 — offline (fitted) Q-learning, discrete actions, raw field")
    print(f"[offline_q] Output dir : {out_dir}")
    print(f"[offline_q] Device     : {device}")
    print("=" * 65)

    all_trajs = load_trajectories(args.data_path)
    train_trajs, val_trajs, _test_trajs = split_trajectories(
        all_trajs, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
    )

    tr_data = build_offline_transitions(train_trajs, initial_temp=args.initial_temp)
    va_data = build_offline_transitions(val_trajs,   initial_temp=args.initial_temp)
    print(f"[offline_q] Train transitions: {tr_data['s'].shape[0]}  |  Val transitions: {va_data['s'].shape[0]}")

    # Standardisation fit purely on the TRAIN split's raw fields — this
    # baseline's own normaliser, independent of the surrogate's.
    state_mean = tr_data["s"].mean(axis=0)
    state_std  = tr_data["s"].std(axis=0)
    state_std[state_std < 1e-6] = 1e-6
    cool_mean = float(tr_data["cool_time"].mean())
    cool_std  = float(tr_data["cool_time"].std())
    if cool_std < 1e-6:
        cool_std = 1.0

    state_dim = tr_data["s"].shape[1]
    n_actions = len(ACTION_GRID)

    def normalise(tr):
        tr = dict(tr)
        tr["s"]         = (tr["s"]         - state_mean) / state_std
        tr["s2"]        = (tr["s2"]        - state_mean) / state_std
        tr["cool_time"] = (tr["cool_time"] - cool_mean)  / cool_std
        return tr

    tr_idx = _snap_to_grid(tr_data["a"], ACTION_GRID)
    va_idx = _snap_to_grid(va_data["a"], ACTION_GRID)
    s, a, r, s2, layer, done, cool = _to_tensors(normalise(tr_data), tr_idx, device)
    vs, va, vr, vs2, vlayer, vdone, vcool = _to_tensors(normalise(va_data), va_idx, device)

    qnet   = RawQNet(state_dim, n_actions, args.hidden, args.depth, args.n_layers, args.layer_embed_dim).to(device)
    target = copy.deepcopy(qnet).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[offline_q] {qnet}")

    n = s.shape[0]
    train_losses, val_losses = [], []
    best_val = float("inf")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            sb, ab, rb, s2b, lb, db, cb = s[idx], a[idx], r[idx], s2[idx], layer[idx], done[idx], cool[idx]

            with torch.no_grad():
                next_layer = (lb + 1).clamp(max=args.n_layers - 1)
                q_next = target(s2b, next_layer, cb).max(dim=-1).values
                y = rb + args.gamma * (~db).float() * q_next

            q_pred = qnet(sb, lb, cb).gather(-1, ab.unsqueeze(-1)).squeeze(-1)
            loss = nn.functional.mse_loss(q_pred, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item(); n_batches += 1

        if epoch % args.target_sync_freq == 0:
            target.load_state_dict(qnet.state_dict())

        val_loss = _eval_loss(qnet, target, vs, va, vr, vs2, vlayer, vdone, vcool, args.gamma, args.n_layers)
        train_losses.append(epoch_loss / max(n_batches, 1))
        val_losses.append(val_loss)

        print(f"Epoch {epoch:4d}/{args.epochs} | train {train_losses[-1]:.5f} | "
              f"val {val_loss:.5f} | {time.time()-t0:.0f}s")

        ckpt = {
            "qnet_state_dict": qnet.state_dict(),
            "model_config": dict(state_dim=state_dim, n_actions=n_actions, hidden=args.hidden,
                                 depth=args.depth, n_layers=args.n_layers, layer_embed_dim=args.layer_embed_dim),
            "state_mean": state_mean, "state_std": state_std,
            "cool_mean": cool_mean, "cool_std": cool_std, "action_grid": ACTION_GRID,
            "gamma": args.gamma, "epoch": epoch, "val_loss": val_loss, "train_args": vars(args),
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(out_dir, "offline_q_best.pt"))
    torch.save(ckpt, os.path.join(out_dir, "offline_q_final.pt"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ep = np.arange(1, len(train_losses) + 1)
    ax.plot(ep, train_losses, label="Train"); ax.plot(ep, val_losses, label="Val", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Bellman MSE loss"); ax.set_title("Offline Q-learning — Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss.png"), dpi=150); plt.close(fig)

    print(f"\n[offline_q] Done. Best val loss: {best_val:.5f}")
    print(f"[offline_q] Best checkpoint: {os.path.join(out_dir, 'offline_q_best.pt')}")


if __name__ == "__main__":
    main()
