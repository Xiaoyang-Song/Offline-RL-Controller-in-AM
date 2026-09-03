"""
baseline_surrogate/ablation_no_latent/train.py
--------------------------------------------------
Train the "no latent space" ablation (baseline_surrogate/ablation_no_latent/model.py).
Reuses TwoStageLatentSurrogateDataset / build_normalizers / weighted_mse /
gaussian_nll directly from surrogate_model_latent_uncertainty_v2 — this
ablation needs the true two-stage target (u_heat_t), so it shares that
package's data pipeline rather than baseline_surrogate/common/data.py's
single-stage one (see model.py's module docstring).

Usage:
    python -m baseline_surrogate.ablation_no_latent.train \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --lp_filter_min 200 --lp_filter_max 300 \\
        --out_dir baseline_surrogate/ablation_no_latent/runs/narrow_200_300W
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.dataset_v2 import (
    load_trajectories, split_trajectories, build_normalizers, TwoStageLatentSurrogateDataset,
)
from surrogate_model_latent_uncertainty_v2.train import weighted_mse, gaussian_nll
from baseline_surrogate.common.train_loop import run_training
from baseline_surrogate.ablation_no_latent.model import NoLatentTwoStageSurrogate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the no-latent-space ablation of the main surrogate.")
    p.add_argument("--data_path",     type=str, required=True)
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--lp_filter_min", type=float, default=None)
    p.add_argument("--lp_filter_max", type=float, default=None)

    p.add_argument("--n_ensemble",      type=int, default=5)
    p.add_argument("--layer_embed_dim", type=int, default=8)
    p.add_argument("--trans_hidden",    type=int, default=128)
    p.add_argument("--trans_depth",     type=int, default=3)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--mu_init_scale",   type=float, default=1e-3)
    p.add_argument("--member_init_seed", type=int, default=None)
    p.add_argument("--bootstrap_seed",  type=int, default=-1)

    p.add_argument("--recon_heat_weight", type=float, default=1.0)
    p.add_argument("--nll_heat_weight",   type=float, default=0.1)
    p.add_argument("--recon_cool_weight", type=float, default=1.0)
    p.add_argument("--nll_cool_weight",   type=float, default=0.1)

    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--num_workers",  type=int,   default=4)

    p.add_argument("--out_dir", type=str, default="baseline_surrogate/ablation_no_latent/runs/default")
    p.add_argument("--device",  type=str, default="")
    return p.parse_args()


def make_loss_fn(recon_heat_w, nll_heat_w, recon_cool_w, nll_cool_w):
    def loss_fn(model, batch, device):
        s, a, c, h, s2, layer_idx, bmask = (t.to(device) for t in batch)
        out = model(s, a, c, h, layer_idx)
        z_t, mu_heat, log_sigma_heat = out["z_t"], out["mu_heat"], out["log_sigma_heat"]
        z_heat_enc, mu_cool, log_sigma_cool = out["z_heat_enc"], out["mu_cool"], out["log_sigma_cool"]

        K, B, D = mu_heat.shape
        bw = bmask.t()   # (K, B)

        z_heat_preds = z_t.unsqueeze(0) + mu_heat
        h_exp = h.unsqueeze(0).expand(K, -1, -1)
        L_recon_heat = weighted_mse(z_heat_preds, h_exp, None, sample_weights=bw)
        delta_heat_target = (h - z_t).unsqueeze(0).expand(K, -1, -1)
        L_nll_heat = gaussian_nll(mu_heat, log_sigma_heat, delta_heat_target, sample_weights=bw)

        z_next_preds = z_heat_enc.unsqueeze(0) + mu_cool
        s2_exp = s2.unsqueeze(0).expand(K, -1, -1)
        L_recon_cool = weighted_mse(z_next_preds, s2_exp, None, sample_weights=bw)
        delta_cool_target = (s2 - z_heat_enc).unsqueeze(0).expand(K, -1, -1)
        L_nll_cool = gaussian_nll(mu_cool, log_sigma_cool, delta_cool_target, sample_weights=bw)

        return (recon_heat_w * L_recon_heat + nll_heat_w * L_nll_heat
                + recon_cool_w * L_recon_cool + nll_cool_w * L_nll_cool)
    return loss_fn


def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bootstrap_seed = args.bootstrap_seed if args.bootstrap_seed >= 0 else args.seed
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[ablation_no_latent.train] Output dir : {args.out_dir}")
    print(f"[ablation_no_latent.train] Device     : {device}")

    all_trajs = load_trajectories(args.data_path)
    train_trajs, val_trajs, _ = split_trajectories(
        all_trajs, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
    )
    state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std = build_normalizers(
        train_trajs, initial_temp=args.initial_temp
    )
    state_dim = state_mean.shape[0]
    n_layers  = len(train_trajs[0])

    lp_filter = None
    if args.lp_filter_min is not None or args.lp_filter_max is not None:
        assert args.lp_filter_min is not None and args.lp_filter_max is not None, \
            "--lp_filter_min and --lp_filter_max must be given together."
        lp_filter = (args.lp_filter_min, args.lp_filter_max)
        print(f"[ablation_no_latent.train] LP filter active: [{args.lp_filter_min}, {args.lp_filter_max}] W")

    ds_kwargs = dict(
        state_mean=state_mean, state_std=state_std, lp_mean=lp_mean, lp_std=lp_std,
        cool_mean=cool_mean, cool_std=cool_std, initial_temp=args.initial_temp, lp_filter=lp_filter,
        n_ensemble=args.n_ensemble, bootstrap_seed=bootstrap_seed,
    )
    train_ds = TwoStageLatentSurrogateDataset(train_trajs, **ds_kwargs)
    val_ds   = TwoStageLatentSurrogateDataset(val_trajs,   **ds_kwargs)

    loader_kw    = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    model_kwargs = dict(
        state_dim=state_dim, lp_dim=1, cool_dim=1, n_ensemble=args.n_ensemble, n_layers=n_layers,
        layer_embed_dim=args.layer_embed_dim, trans_hidden=args.trans_hidden, trans_depth=args.trans_depth,
        dropout=args.dropout, mu_init_scale=args.mu_init_scale, member_init_seed=args.member_init_seed,
    )
    model = NoLatentTwoStageSurrogate(**model_kwargs).to(device)
    print(f"[ablation_no_latent.train] NoLatentTwoStageSurrogate params={model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)
    loss_fn = make_loss_fn(args.recon_heat_weight, args.nll_heat_weight,
                           args.recon_cool_weight, args.nll_cool_weight)

    ckpt_best = os.path.join(args.out_dir, "ablation_no_latent_best.pt")

    def save_best(epoch, val_loss):
        torch.save({
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean, "state_std": state_std,
            "lp_mean": lp_mean, "lp_std": lp_std, "cool_mean": cool_mean, "cool_std": cool_std,
            "epoch": epoch, "val_loss": val_loss,
            "model_config": model_kwargs,
            "train_args": vars(args),
        }, ckpt_best)

    run_training(model, train_loader, val_loader, loss_fn, optimizer, device,
                 epochs=args.epochs, patience=args.patience, scheduler=scheduler,
                 save_best_fn=save_best, log_prefix="[ablation_no_latent.train] ")

    print(f"[ablation_no_latent.train] Done. Best checkpoint: {ckpt_best}")


if __name__ == "__main__":
    main()
