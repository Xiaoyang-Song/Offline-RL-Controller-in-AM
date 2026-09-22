"""
surrogate_model_v3/train.py
------------------------------------
Training script for the two-stage (heating / cooling) LPBF surrogate —
bootstrap-resampled Gaussian ensemble, NO learned latent bottleneck.

Usage:
    python -m surrogate_model_v3.train \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl

Narrow-laser-power-range experiment:
    python -m surrogate_model_v3.train \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --lp_filter_min 200 --lp_filter_max 300 \\
        --out_dir surrogate_model_v3/runs/narrow_200_300W

Gapped/patchy + perturbation + decorrelated-bootstrap experiment (for an
epistemic-uncertainty stress test via evaluate_ood.py/evaluate_ood_ratio.py):
    python -m surrogate_model_v3.train \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --lp_filter_ranges "100-150,200-250,300-350" \\
        --perturb_frac 0.1 --bootstrap_frac 0.5 \\
        --out_dir surrogate_model_v3/runs/patchy_100-150_200-250_300-350_perturb0.1

Loss terms
----------
  L_recon_heat : bootstrap-weighted mean_k MSE( s_t + μ_Δ_heat_k, u_heat_t )
  L_nll_heat   : bootstrap-weighted Gaussian NLL, target = u_heat_t - s_t
  L_recon_cool : bootstrap-weighted mean_k MSE( u_heat_t + μ_Δ_cool_k, s_{t+1} )
  L_nll_cool   : bootstrap-weighted Gaussian NLL, target = s_{t+1} - u_heat_t

  total = recon_heat_w·L_recon_heat + nll_heat_w·L_nll_heat
        + recon_cool_w·L_recon_cool + nll_cool_w·L_nll_cool

The cooling stage is teacher-forced on the ground-truth u_heat_t during
training (see model.forward), so the two stages' losses are computed
independently of each other's prediction error — matching the physical
fact that "no matter what laser power you applied previously, the cooling
mechanism is the same."

All outputs are written under --out_dir (default:
surrogate_model_v3/runs/<timestamp>/).
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.dataset import (
    load_trajectories, split_trajectories, build_normalizers, TwoStageSurrogateDataset,
)
from surrogate_model_v3.model import TwoStageSurrogate

LOG_2PI = float(np.log(2.0 * np.pi))


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the two-stage (heating/cooling) LPBF Gaussian ensemble surrogate (no latent space)."
    )

    # ── data ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_path",     type=str,  required=True)
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--lp_filter_min", type=float, default=None,
                   help="Narrow-surrogate experiment: restrict training transitions to "
                        "laser power in [min, max]. Requires --lp_filter_max too; "
                        "mutually exclusive with --lp_filter_ranges.")
    p.add_argument("--lp_filter_max", type=float, default=None)
    p.add_argument("--lp_filter_ranges", type=str, default=None,
                   help="Gapped/patchy-surrogate experiment: comma-separated 'lo-hi' laser-power "
                        "ranges, e.g. '100-150,200-250,300-350' -- a transition is kept if its "
                        "power falls in the UNION of these ranges, leaving interior gaps bracketed "
                        "by training data on both sides. Mutually exclusive with "
                        "--lp_filter_min/--lp_filter_max (pick one). See README.md's 'Harder / "
                        "gapped-surrogate experiments' section for the full rationale.")
    p.add_argument("--perturb_frac", type=float, default=0.0,
                   help="Additive Gaussian noise on the u_heat_t/s_{t+1} TARGET fields only, scaled "
                        "per-node as perturb_frac * that node's own state_std (e.g. 0.1 = 10%% of each "
                        "node's natural variation). 0.0 (default) is a no-op. See "
                        "TwoStageSurrogateDataset's perturb_frac docstring for the full rationale.")
    p.add_argument("--perturb_seed", type=int, default=0,
                   help="RNG seed for --perturb_frac noise (independent of --bootstrap_seed).")

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--n_ensemble",      type=int, default=5,
                   help="Number of ensemble transition members K (shared by both stages).")
    p.add_argument("--n_layers",        type=int, default=12,
                   help="Number of build layers (sets layer embedding table size).")
    p.add_argument("--layer_embed_dim", type=int, default=8)
    p.add_argument("--trans_hidden",    type=int, default=128)
    p.add_argument("--trans_depth",     type=int, default=3)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--mu_init_scale",   type=float, default=1e-3,
                   help="Uniform init range for each ensemble member's mu_head weights. See "
                        "GaussianTransitionMLP's docstring for why raising this (with "
                        "--member_init_seed) sharpens epistemic disagreement in training gaps.")
    p.add_argument("--member_init_seed", type=int, default=None,
                   help="Optional: seed each of the K ensemble members' weight init independently "
                        "(base seed + per-member offset). Default None = every member draws "
                        "sequentially from one global RNG stream.")

    # ── bootstrap ensemble ────────────────────────────────────────────────────
    p.add_argument("--bootstrap_seed", type=int, default=-1,
                   help="RNG seed for the per-member bootstrap resamples (default: same as --seed).")
    p.add_argument("--bootstrap_frac", type=float, default=1.0,
                   help="Fraction of N samples each ensemble member draws (with replacement) for "
                        "its bootstrap resample (default 1.0 = standard N-out-of-N bootstrap). "
                        "Lowering this (e.g. 0.5) decorrelates the K members further and makes "
                        "epistemic (ensemble-disagreement) uncertainty larger and more informative.")

    # ── loss weights ─────────────────────────────────────────────────────────
    p.add_argument("--recon_heat_weight", type=float, default=1.0,
                   help="Weight for the stage-1 (heating) next-field prediction loss.")
    p.add_argument("--nll_heat_weight",   type=float, default=0.1,
                   help="Weight for the stage-1 (heating) Gaussian NLL loss.")
    p.add_argument("--recon_cool_weight", type=float, default=1.0,
                   help="Weight for the stage-2 (cooling) next-state prediction loss.")
    p.add_argument("--nll_cool_weight",   type=float, default=0.1,
                   help="Weight for the stage-2 (cooling) Gaussian NLL loss.")

    # ── optimiser ─────────────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--num_workers",  type=int,   default=4)

    # ── output ───────────────────────────────────────────────────────────────
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--device",  type=str, default="")

    return p.parse_args()


def _parse_lp_filter_ranges(spec: str):
    """'150-200,300-350' -> [(150.0, 200.0), (300.0, 350.0)]"""
    ranges = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo_str, hi_str = part.split("-")
        ranges.append((float(lo_str), float(hi_str)))
    if not ranges:
        raise ValueError(f"--lp_filter_ranges='{spec}' parsed to an empty range list.")
    return ranges


# =============================================================================
# Loss functions
# =============================================================================

def weighted_mse(
    pred:           torch.Tensor,
    target:         torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sq = (pred - target).pow(2)
    per_sample = sq.mean(dim=-1)
    if sample_weights is not None:
        denom = sample_weights.sum().clamp_min(1e-8)
        return (per_sample * sample_weights).sum() / denom
    return per_sample.mean()


def gaussian_nll(
    mu:             torch.Tensor,   # (K, B, D)
    log_sigma:      torch.Tensor,   # (K, B, D)
    target:         torch.Tensor,   # (K, B, D)
    sample_weights: Optional[torch.Tensor] = None,  # (K, B) or None
) -> torch.Tensor:
    var = (2.0 * log_sigma).exp()
    nll = 0.5 * (LOG_2PI + 2.0 * log_sigma + (target - mu).pow(2) / var)
    per_sample = nll.mean(dim=-1)
    if sample_weights is not None:
        denom = sample_weights.sum().clamp_min(1e-8)
        return (per_sample * sample_weights).sum() / denom
    return per_sample.mean()


def compute_losses(
    model:            TwoStageSurrogate,
    s:                torch.Tensor,              # (B, D)
    a:                torch.Tensor,              # (B, 1)  laser power
    c:                torch.Tensor,              # (B, 1)  cool time
    h:                torch.Tensor,              # (B, D)  ground-truth u_heat
    s2:               torch.Tensor,              # (B, D)  ground-truth next state
    layer_indices:    torch.Tensor,              # (B,)
    bootstrap_masks:  torch.Tensor,              # (B, K)
) -> Dict[str, torch.Tensor]:
    """Returns dict with L_recon_heat, L_nll_heat, L_recon_cool, L_nll_cool."""
    out = model(s, a, c, h, layer_indices)
    mu_heat, log_sigma_heat = out["mu_heat"], out["log_sigma_heat"]
    mu_cool, log_sigma_cool = out["mu_cool"], out["log_sigma_cool"]

    K, B, D = mu_heat.shape
    bw = bootstrap_masks.t()  # (K, B)

    # ── heating stage ───────────────────────────────────────────────────────
    heat_preds   = s.unsqueeze(0) + mu_heat                            # (K, B, D)
    h_exp        = h.unsqueeze(0).expand(K, -1, -1)
    L_recon_heat = weighted_mse(heat_preds, h_exp, sample_weights=bw)

    delta_heat_target = (h - s).unsqueeze(0).expand(K, -1, -1)
    L_nll_heat = gaussian_nll(mu_heat, log_sigma_heat, delta_heat_target, sample_weights=bw)

    # ── cooling stage (teacher-forced on ground-truth u_heat) ────────────────
    next_preds   = h.unsqueeze(0) + mu_cool                            # (K, B, D)
    s2_exp       = s2.unsqueeze(0).expand(K, -1, -1)
    L_recon_cool = weighted_mse(next_preds, s2_exp, sample_weights=bw)

    delta_cool_target = (s2 - h).unsqueeze(0).expand(K, -1, -1)
    L_nll_cool = gaussian_nll(mu_cool, log_sigma_cool, delta_cool_target, sample_weights=bw)

    return dict(
        recon_heat=L_recon_heat, nll_heat=L_nll_heat,
        recon_cool=L_recon_cool, nll_cool=L_nll_cool,
    )


# =============================================================================
# Training / validation epoch
# =============================================================================

def run_epoch(
    model:        TwoStageSurrogate,
    loader:       DataLoader,
    optimizer:    Optional[torch.optim.Optimizer],
    device:       str,
    recon_heat_w: float,
    nll_heat_w:   float,
    recon_cool_w: float,
    nll_cool_w:   float,
) -> Tuple[float, Dict[str, float]]:
    training = optimizer is not None
    model.train(training)

    agg = defaultdict(float)
    n   = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for s, a, c, h, s2, layer_idx, bmask in loader:
            s, a, c, h, s2 = (t.to(device) for t in (s, a, c, h, s2))
            layer_idx = layer_idx.to(device)
            bmask     = bmask.to(device)

            L = compute_losses(model, s, a, c, h, s2, layer_idx, bmask)
            loss = (
                recon_heat_w * L["recon_heat"] + nll_heat_w * L["nll_heat"]
                + recon_cool_w * L["recon_cool"] + nll_cool_w * L["nll_cool"]
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            agg["total"]      += loss.item()
            agg["recon_heat"] += L["recon_heat"].item()
            agg["nll_heat"]   += L["nll_heat"].item()
            agg["recon_cool"] += L["recon_cool"].item()
            agg["nll_cool"]   += L["nll_cool"].item()
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
    ax.plot(epochs, train_losses, label="Train total", linewidth=1.5)
    ax.plot(epochs, val_losses,   label="Val total",   linewidth=1.5, linestyle="--")
    best_e = int(np.argmin(val_losses)) + 1
    ax.axvline(best_e, color="grey", linestyle=":", linewidth=1,
               label=f"Best val epoch {best_e}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted loss")
    ax.set_title("Two-Stage Gaussian Ensemble Surrogate (no latent) — Total Loss Curves")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[train] Loss plot → {out_path}")


def plot_loss_components(train_comps, val_comps, out_path):
    keys   = ["recon_heat", "nll_heat", "recon_cool", "nll_cool"]
    titles = ["L_recon_heat", "L_nll_heat", "L_recon_cool", "L_nll_cool"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    epochs = range(1, len(train_comps["recon_heat"]) + 1)
    for ax, k, title in zip(axes, keys, titles):
        ax.plot(epochs, train_comps[k], label="Train", linewidth=1.5)
        ax.plot(epochs, val_comps[k],   label="Val",   linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=9); ax.set_xlabel("Epoch")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Two-Stage Gaussian Ensemble Surrogate (no latent) — Loss Components")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[train] Component loss plot → {out_path}")


# =============================================================================
# Checkpoint helpers
# =============================================================================

def _save_checkpoint(
    model:        TwoStageSurrogate,
    state_mean:   torch.Tensor,
    state_std:    torch.Tensor,
    lp_mean:      float,
    lp_std:       float,
    cool_mean:    float,
    cool_std:     float,
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
            "lp_mean":          lp_mean,
            "lp_std":           lp_std,
            "cool_mean":        cool_mean,
            "cool_std":         cool_std,
            "epoch":            epoch,
            "val_loss":         val_loss,
            "model_config": {
                "state_dim":       model.state_dim,
                "lp_dim":          model.lp_dim,
                "cool_dim":        model.cool_dim,
                "n_ensemble":      model.n_ensemble,
                "n_layers":        model.n_layers,
                "layer_embed_dim": model.layer_embed_dim,
                "trans_hidden":    args.trans_hidden,
                "trans_depth":     args.trans_depth,
                "dropout":         args.dropout,
                "mu_init_scale":   args.mu_init_scale,
                "member_init_seed": args.member_init_seed,
            },
            "train_args": vars(args),
        },
        path,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bootstrap_seed = args.bootstrap_seed if args.bootstrap_seed >= 0 else args.seed

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.out_dir:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("surrogate_model_v3", "runs", ts)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] Output dir : {out_dir}")
    print(f"[train] Device     : {device}")

    all_trajs = load_trajectories(args.data_path)
    train_trajs, val_trajs, _ = split_trajectories(
        all_trajs, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
    )
    state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std = build_normalizers(
        train_trajs, initial_temp=args.initial_temp
    )
    state_dim = state_mean.shape[0]
    n_layers  = len(train_trajs[0])

    have_lp_minmax = args.lp_filter_min is not None or args.lp_filter_max is not None
    have_lp_ranges = args.lp_filter_ranges is not None
    if have_lp_minmax and have_lp_ranges:
        raise ValueError("--lp_filter_min/--lp_filter_max and --lp_filter_ranges are mutually "
                          "exclusive -- use --lp_filter_ranges 'lo-hi' for a single contiguous "
                          "range too if needed.")

    lp_filter = None
    if have_lp_ranges:
        lp_filter = _parse_lp_filter_ranges(args.lp_filter_ranges)
        range_str = ", ".join(f"[{lo}, {hi}]" for lo, hi in lp_filter)
        print(f"[train] LP filter (gapped) active: {range_str} W")
    elif have_lp_minmax:
        if args.lp_filter_min is None or args.lp_filter_max is None:
            raise ValueError("--lp_filter_min and --lp_filter_max must be given together.")
        lp_filter = (args.lp_filter_min, args.lp_filter_max)
        print(f"[train] LP filter active: [{args.lp_filter_min}, {args.lp_filter_max}] W")

    ds_kwargs = dict(
        state_mean=state_mean, state_std=state_std, lp_mean=lp_mean, lp_std=lp_std,
        cool_mean=cool_mean, cool_std=cool_std, initial_temp=args.initial_temp, lp_filter=lp_filter,
        n_ensemble=args.n_ensemble, bootstrap_seed=bootstrap_seed,
        bootstrap_resample_frac=args.bootstrap_frac,
        perturb_frac=args.perturb_frac, perturb_seed=args.perturb_seed,
    )
    train_ds = TwoStageSurrogateDataset(train_trajs, **ds_kwargs)
    val_ds   = TwoStageSurrogateDataset(val_trajs,   **ds_kwargs)

    loader_kw    = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    model_kwargs = dict(
        state_dim=state_dim, lp_dim=1, cool_dim=1, n_ensemble=args.n_ensemble, n_layers=n_layers,
        layer_embed_dim=args.layer_embed_dim, trans_hidden=args.trans_hidden, trans_depth=args.trans_depth,
        dropout=args.dropout, mu_init_scale=args.mu_init_scale, member_init_seed=args.member_init_seed,
    )
    model = TwoStageSurrogate(**model_kwargs).to(device)
    print(f"[train] {model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    epoch_kw = dict(
        recon_heat_w=args.recon_heat_weight, nll_heat_w=args.nll_heat_weight,
        recon_cool_w=args.recon_cool_weight, nll_cool_w=args.nll_cool_weight,
    )

    ckpt_best  = os.path.join(out_dir, "surrogate_best.pt")
    ckpt_final = os.path.join(out_dir, "surrogate_final.pt")

    best_val_loss  = float("inf")
    epochs_no_impr = 0
    train_losses, val_losses = [], []
    train_comps_hist: Dict[str, list] = defaultdict(list)
    val_comps_hist:   Dict[str, list] = defaultdict(list)

    print(f"\n[train] Starting training for up to {args.epochs} epochs (patience={args.patience})")
    print(f"[train] Weights   → recon_heat={args.recon_heat_weight}  nll_heat={args.nll_heat_weight}  "
          f"recon_cool={args.recon_cool_weight}  nll_cool={args.nll_cool_weight}")
    print(f"[train] Ensemble  → K={args.n_ensemble} members, bootstrap_seed={bootstrap_seed}, "
          f"bootstrap_frac={args.bootstrap_frac}")
    print("-" * 80)

    t0 = time.time()
    epoch = 0
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
            _save_checkpoint(model, state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std,
                             args, epoch, best_val_loss, ckpt_best)
            marker = " ✓ best"
        else:
            epochs_no_impr += 1
            marker = f" (no improvement {epochs_no_impr}/{args.patience})"

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train {tr_loss:.5f} [rh={tr_comps['recon_heat']:.4f} nh={tr_comps['nll_heat']:.4f} "
            f"rc={tr_comps['recon_cool']:.4f} nc={tr_comps['nll_cool']:.4f}] | "
            f"val {va_loss:.5f} | lr {lr_now:.2e} | {elapsed:6.1f}s{marker}"
        )

        if epochs_no_impr >= args.patience:
            print(f"[train] Early stopping at epoch {epoch}.")
            break

    _save_checkpoint(model, state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std,
                     args, epoch, va_loss, ckpt_final)
    plot_loss_curves(train_losses, val_losses, os.path.join(out_dir, "loss_curves.png"))
    plot_loss_components(train_comps_hist, val_comps_hist, os.path.join(out_dir, "loss_components.png"))

    print(f"\n[train] Done.  Best val loss : {best_val_loss:.6f}")
    print(f"[train] Best checkpoint  : {ckpt_best}")
    print(f"[train] Final checkpoint : {ckpt_final}")


if __name__ == "__main__":
    main()
