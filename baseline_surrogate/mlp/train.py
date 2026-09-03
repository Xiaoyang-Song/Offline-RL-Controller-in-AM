"""
baseline_surrogate/mlp/train.py
-----------------------------------
Train the plain MLP baseline (baseline_surrogate/mlp/model.py).

Usage (matches the "train on 200-300W" convention used throughout this
repo's other narrow/gapped surrogate experiments):

    python -m baseline_surrogate.mlp.train \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --lp_filter_min 200 --lp_filter_max 300 \\
        --out_dir baseline_surrogate/mlp/runs/narrow_200_300W
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.dataset_v2 import load_trajectories, split_trajectories
from baseline_surrogate.common.data import build_single_stage_normalizers, SingleStageFlatDataset
from baseline_surrogate.common.train_loop import run_training
from baseline_surrogate.mlp.model import PlainMLPSurrogate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the plain-MLP surrogate baseline.")
    p.add_argument("--data_path",     type=str, required=True)
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--lp_filter_min", type=float, default=None)
    p.add_argument("--lp_filter_max", type=float, default=None)

    p.add_argument("--hidden",          type=int, default=512)
    p.add_argument("--depth",           type=int, default=4)
    p.add_argument("--layer_embed_dim", type=int, default=8)
    p.add_argument("--dropout",         type=float, default=0.0)

    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--num_workers",  type=int,   default=4)

    p.add_argument("--out_dir", type=str, default="baseline_surrogate/mlp/runs/default")
    p.add_argument("--device",  type=str, default="")
    return p.parse_args()


def mse_loss_fn(model, batch, device):
    s, a, c, s2, layer_idx, _bmask = (t.to(device) for t in batch)
    pred = model(s, a, c, layer_idx)
    return (pred - s2).pow(2).mean()


def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[mlp.train] Output dir : {args.out_dir}")
    print(f"[mlp.train] Device     : {device}")

    all_trajs = load_trajectories(args.data_path)
    train_trajs, val_trajs, _ = split_trajectories(
        all_trajs, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
    )
    state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std = build_single_stage_normalizers(
        train_trajs, initial_temp=args.initial_temp
    )
    state_dim = state_mean.shape[0]
    n_layers  = len(train_trajs[0])

    lp_filter = None
    if args.lp_filter_min is not None or args.lp_filter_max is not None:
        assert args.lp_filter_min is not None and args.lp_filter_max is not None, \
            "--lp_filter_min and --lp_filter_max must be given together."
        lp_filter = (args.lp_filter_min, args.lp_filter_max)
        print(f"[mlp.train] LP filter active: [{args.lp_filter_min}, {args.lp_filter_max}] W")

    ds_kwargs = dict(
        state_mean=state_mean, state_std=state_std, lp_mean=lp_mean, lp_std=lp_std,
        cool_mean=cool_mean, cool_std=cool_std, initial_temp=args.initial_temp, lp_filter=lp_filter,
    )
    train_ds = SingleStageFlatDataset(train_trajs, **ds_kwargs)
    val_ds   = SingleStageFlatDataset(val_trajs,   **ds_kwargs)

    loader_kw    = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    model = PlainMLPSurrogate(
        state_dim=state_dim, hidden=args.hidden, depth=args.depth,
        n_layers=n_layers, layer_embed_dim=args.layer_embed_dim, dropout=args.dropout,
    ).to(device)
    print(f"[mlp.train] PlainMLPSurrogate params={model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    ckpt_best = os.path.join(args.out_dir, "mlp_best.pt")

    def save_best(epoch, val_loss):
        torch.save({
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean, "state_std": state_std,
            "lp_mean": lp_mean, "lp_std": lp_std, "cool_mean": cool_mean, "cool_std": cool_std,
            "epoch": epoch, "val_loss": val_loss,
            "model_config": dict(state_dim=state_dim, hidden=args.hidden, depth=args.depth,
                                 n_layers=n_layers, layer_embed_dim=args.layer_embed_dim,
                                 dropout=args.dropout),
            "train_args": vars(args),
        }, ckpt_best)

    run_training(model, train_loader, val_loader, mse_loss_fn, optimizer, device,
                 epochs=args.epochs, patience=args.patience, scheduler=scheduler,
                 save_best_fn=save_best, log_prefix="[mlp.train] ")

    print(f"[mlp.train] Done. Best checkpoint: {ckpt_best}")


if __name__ == "__main__":
    main()
