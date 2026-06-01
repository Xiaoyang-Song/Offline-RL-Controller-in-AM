"""
surrogate_domain_adaptation/adapt.py
--------------------------------------
Few-shot domain adaptation of the pretrained base surrogate to ct_150.

For each K ∈ K_LIST:
  1. Sample K trajectories from the target domain (ct_150).
  2. Freeze the base model; create an AdaptedEnsembleLatentDynamicsModel.
  3. Train ONLY the three adapter modules on K trajectories.
  4. Evaluate on the held-out ct_150 test set.
  5. Save adapted checkpoint + per-K metrics.

This generates:
    <out_dir>/
        K005/  adapted.pt  metrics.json
        K010/  adapted.pt  metrics.json
        ...
        metrics_summary.json   ← all K results side-by-side

Usage
-----
    python -m surrogate_domain_adaptation.adapt \\
        --base_checkpoint surrogate_domain_adaptation/runs/base/<ts>/base_best.pt

    # Specify target cooling time (default 150) and K values:
    python -m surrogate_domain_adaptation.adapt \\
        --base_checkpoint <path> \\
        --target_ct 150 \\
        --k_values 5 10 20 50 100
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_domain_adaptation.dataset import (
    load_target_domain,
    split_trajectories,
    sample_few_shot,
    MultiDomainFlatDataset,
    MultiDomainTrajectoryDataset,
    DATA_ROOT,
    TARGET_COOLING_TIME,
)
from surrogate_domain_adaptation.model import (
    AdaptedEnsembleLatentDynamicsModel,
    save_adapted,
)
from surrogate_domain_adaptation.train_base import load_base_checkpoint
from surrogate_model_latent.train import (
    compute_single_step_losses,
    compute_rollout_loss,
)


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Few-shot domain adaptation to target cooling time."
    )
    p.add_argument("--base_checkpoint", type=str, required=True,
                   help="Path to base_best.pt from train_base.py.")

    # ── target domain ─────────────────────────────────────────────────────────
    p.add_argument("--data_root",  type=str, default=DATA_ROOT)
    p.add_argument("--target_ct",  type=int, default=TARGET_COOLING_TIME,
                   help="Target cooling time (×10, default 150 → 0.15 s).")
    p.add_argument("--n_target_trajs", type=int, default=None,
                   help="Total target trajectories to load (None = all).")
    p.add_argument("--test_fraction",  type=float, default=0.30,
                   help="Fraction of target held out for evaluation.")
    p.add_argument("--initial_temp",   type=float, default=300.0)
    p.add_argument("--seed",           type=int,   default=42)

    # ── K values ──────────────────────────────────────────────────────────────
    p.add_argument("--k_values", type=int, nargs="+",
                   default=[5, 10, 20, 50, 100],
                   help="Number of adaptation shots to evaluate.")

    # ── adapter architecture ──────────────────────────────────────────────────
    p.add_argument("--bottleneck",        type=int,  default=32)
    p.add_argument("--no_enc_adapter",    action="store_true")
    p.add_argument("--no_trans_adapter",  action="store_true")
    p.add_argument("--no_dec_adapter",    action="store_true")

    # ── adaptation optimiser ──────────────────────────────────────────────────
    p.add_argument("--adapt_lr",           type=float, default=1e-3)
    p.add_argument("--adapt_weight_decay", type=float, default=1e-5)
    p.add_argument("--adapt_epochs",       type=int,   default=200,
                   help="Training epochs on K-shot data.")
    p.add_argument("--adapt_batch_size",   type=int,   default=32)
    p.add_argument("--adapt_patience",     type=int,   default=30,
                   help="Early stopping patience (val from remaining adapt data).")

    # ── loss ─────────────────────────────────────────────────────────────────
    p.add_argument("--recon_st_weight",  type=float, default=1.0)
    p.add_argument("--recon_st1_weight", type=float, default=1.0)
    # Rollout off for few-shot (can overfit with K=5)
    p.add_argument("--rollout_steps",    type=int,   default=0)
    p.add_argument("--rollout_weight",   type=float, default=0.0)

    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument("--out_dir", type=str, default="",
                   help="Output dir.  Defaults to same folder as base_checkpoint.")
    p.add_argument("--device",  type=str, default="")
    p.add_argument("--num_workers", type=int, default=2)

    return p.parse_args()


# =============================================================================
# Single-K adaptation
# =============================================================================

@torch.no_grad()
def evaluate_on_loader(model, loader, device, state_std):
    """Compute mean MAE and RMSE in raw Kelvin (teacher-forced)."""
    model.eval()
    ss     = state_std.to(device)
    abs_e  = [[] for _ in range(12)]
    sq_e   = [[] for _ in range(12)]

    for traj_s, traj_a in loader:
        traj_s = traj_s.to(device)
        traj_a = traj_a.to(device)
        B, T1, D = traj_s.shape
        T = T1 - 1
        for t in range(min(T, 12)):
            s_in  = traj_s[:, t,     :]
            a_in  = traj_a[:, t,     :]
            s_gt  = traj_s[:, t + 1, :]
            lidx  = torch.full((B,), t, dtype=torch.long, device=device)
            z_t   = model.encode(s_in)
            mu, _ = model.predict_ensemble(z_t, a_in, lidx)
            diff  = (model.decode(z_t + mu) - s_gt) * ss
            abs_e[t].extend(diff.abs().mean(-1).cpu().numpy().tolist())
            sq_e[t].extend(diff.pow(2).mean(-1).cpu().numpy().tolist())

    per_layer_mae  = np.array([np.mean(abs_e[t]) for t in range(12)])
    per_layer_rmse = np.array([np.sqrt(np.mean(sq_e[t])) for t in range(12)])
    return per_layer_mae, per_layer_rmse


def adapt_k_shot(
    base_model,
    adapt_trajs,
    eval_trajs,
    state_mean, state_std, action_mean, action_std,
    args, device, K,
):
    """
    Run adaptation for a single K and return (adapted_model, metrics_dict).
    """
    ds_kw = dict(state_mean=state_mean, state_std=state_std,
                 action_mean=action_mean, action_std=action_std,
                 initial_temp=args.initial_temp)

    # Split adapt_trajs 80/20 for training / early-stop val
    n_val_a  = max(1, int(len(adapt_trajs) * 0.2))
    adapt_tr = adapt_trajs[n_val_a:]
    adapt_va = adapt_trajs[:n_val_a]

    # Use flat dataset for adaptation (rollout can overfit at K=5)
    use_rollout = args.rollout_steps > 1 and K >= 20
    DS   = MultiDomainTrajectoryDataset if use_rollout else MultiDomainFlatDataset
    lkw  = dict(batch_size=min(args.adapt_batch_size, max(1, len(adapt_tr) * 12)),
                num_workers=args.num_workers, pin_memory=(device == "cuda"))
    tr_loader = DataLoader(DS(adapt_tr, **ds_kw), shuffle=True,  **lkw)
    va_loader = DataLoader(DS(adapt_va, **ds_kw), shuffle=False, **lkw)
    ev_loader = DataLoader(
        MultiDomainTrajectoryDataset(eval_trajs, **ds_kw),
        batch_size=64, shuffle=False, num_workers=args.num_workers,
    )

    # Build adapted model (fresh adapters each K)
    adapted = AdaptedEnsembleLatentDynamicsModel(
        base_model,
        bottleneck        = args.bottleneck,
        use_enc_adapter   = not args.no_enc_adapter,
        use_trans_adapter = not args.no_trans_adapter,
        use_dec_adapter   = not args.no_dec_adapter,
    ).to(device)
    adapted.freeze_base()
    adapted.unfreeze_adapters()

    optimizer = torch.optim.AdamW(
        adapted.adapter_parameters(),
        lr=args.adapt_lr, weight_decay=args.adapt_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.adapt_epochs, eta_min=args.adapt_lr / 100
    )

    best_va, patience_cnt = float("inf"), 0
    best_adapter_state    = {
        "enc":   adapted.enc_adapter.state_dict()   if adapted.enc_adapter   else None,
        "trans": adapted.trans_adapter.state_dict() if adapted.trans_adapter else None,
        "dec":   adapted.dec_adapter.state_dict()   if adapted.dec_adapter   else None,
    }

    for epoch in range(1, args.adapt_epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        adapted.train()
        for batch in tr_loader:
            if use_rollout:
                traj_s, traj_a = batch
                traj_s, traj_a = traj_s.to(device), traj_a.to(device)
                B, T1, D = traj_s.shape; T = T1 - 1
                s_f  = traj_s[:, :-1, :].reshape(B*T, D)
                a_f  = traj_a.reshape(B*T, 1)
                s2_f = traj_s[:, 1:,  :].reshape(B*T, D)
                li_f = torch.arange(T, device=device).unsqueeze(0).expand(B,-1).reshape(B*T)
                L_rs, L_rs1 = compute_single_step_losses(adapted, s_f, a_f, s2_f, li_f, None)
                L_ro = compute_rollout_loss(adapted, traj_s, traj_a, T, None)
                loss = args.recon_st_weight*L_rs + args.recon_st1_weight*L_rs1 + args.rollout_weight*L_ro
            else:
                s, a, s2, li = [x.to(device) for x in batch]
                L_rs, L_rs1  = compute_single_step_losses(adapted, s, a, s2, li, None)
                loss = args.recon_st_weight*L_rs + args.recon_st1_weight*L_rs1

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(adapted.adapter_parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        # ── val ───────────────────────────────────────────────────────────
        adapted.eval()
        with torch.no_grad():
            va_losses = []
            for batch in va_loader:
                if use_rollout:
                    traj_s, traj_a = batch
                    traj_s, traj_a = traj_s.to(device), traj_a.to(device)
                    B, T1, D = traj_s.shape; T = T1 - 1
                    s_f  = traj_s[:, :-1, :].reshape(B*T, D)
                    a_f  = traj_a.reshape(B*T, 1)
                    s2_f = traj_s[:, 1:,  :].reshape(B*T, D)
                    li_f = torch.arange(T, device=device).unsqueeze(0).expand(B,-1).reshape(B*T)
                    L_rs, L_rs1 = compute_single_step_losses(adapted, s_f, a_f, s2_f, li_f, None)
                    va_losses.append((L_rs + L_rs1).item())
                else:
                    s, a, s2, li = [x.to(device) for x in batch]
                    L_rs, L_rs1  = compute_single_step_losses(adapted, s, a, s2, li, None)
                    va_losses.append((L_rs + L_rs1).item())

        va_loss = float(np.mean(va_losses))
        if va_loss < best_va:
            best_va, patience_cnt = va_loss, 0
            best_adapter_state = {
                "enc":   adapted.enc_adapter.state_dict()   if adapted.enc_adapter   else None,
                "trans": adapted.trans_adapter.state_dict() if adapted.trans_adapter else None,
                "dec":   adapted.dec_adapter.state_dict()   if adapted.dec_adapter   else None,
            }
        else:
            patience_cnt += 1
            if patience_cnt >= args.adapt_patience:
                break

    # Restore best adapter weights
    if adapted.enc_adapter   and best_adapter_state["enc"]:
        adapted.enc_adapter.load_state_dict(best_adapter_state["enc"])
    if adapted.trans_adapter and best_adapter_state["trans"]:
        adapted.trans_adapter.load_state_dict(best_adapter_state["trans"])
    if adapted.dec_adapter   and best_adapter_state["dec"]:
        adapted.dec_adapter.load_state_dict(best_adapter_state["dec"])

    # ── evaluate on held-out test set ─────────────────────────────────────
    per_layer_mae, per_layer_rmse = evaluate_on_loader(adapted, ev_loader, device, state_std)

    metrics = {
        "K":            K,
        "mean_mae_K":   float(per_layer_mae.mean()),
        "mean_rmse_K":  float(per_layer_rmse.mean()),
        "per_layer_mae":  per_layer_mae.tolist(),
        "per_layer_rmse": per_layer_rmse.tolist(),
    }
    return adapted, metrics


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.base_checkpoint)),
        f"adapt_ct{args.target_ct:03d}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[adapt] Output dir  : {out_dir}")
    print(f"[adapt] Target CT   : ct_{args.target_ct:03d}  (0.{args.target_ct//10:02d} s)")

    # ── load base model + normalizers (from source training) ─────────────────
    print(f"\n[adapt] Loading base checkpoint: {args.base_checkpoint}")
    base_model, state_mean, state_std, action_mean, action_std, _ = \
        load_base_checkpoint(args.base_checkpoint, device=device)
    print(f"[adapt] {base_model}")

    # ── load target domain data ───────────────────────────────────────────────
    target_trajs = load_target_domain(
        data_root=args.data_root,
        ct=args.target_ct,
        n_trajs=args.n_target_trajs,
    )

    # Hold out a fixed test set (same split regardless of K)
    n_test = max(10, int(len(target_trajs) * args.test_fraction))
    rng    = np.random.default_rng(args.seed)
    perm   = rng.permutation(len(target_trajs))
    test_trajs  = [target_trajs[i] for i in perm[:n_test]]
    avail_trajs = [target_trajs[i] for i in perm[n_test:]]  # pool for K-shot sampling

    print(f"[adapt] Target: {len(target_trajs)} trajs | "
          f"test={len(test_trajs)} | adapt-pool={len(avail_trajs)}")

    # ── evaluate base model (K=0) on test set ────────────────────────────────
    ds_kw = dict(state_mean=state_mean, state_std=state_std,
                 action_mean=action_mean, action_std=action_std,
                 initial_temp=args.initial_temp)
    ev_loader = DataLoader(
        MultiDomainTrajectoryDataset(test_trajs, **ds_kw),
        batch_size=64, shuffle=False, num_workers=args.num_workers,
    )
    base_mae, base_rmse = evaluate_on_loader(base_model, ev_loader, device, state_std)
    print(f"\n[adapt] Base model (K=0) on target domain:")
    print(f"        mean MAE  = {base_mae.mean():.3f} K  "
          f"  mean RMSE = {base_rmse.mean():.3f} K")

    all_metrics = {
        "K0_base": {
            "K": 0,
            "mean_mae_K":  float(base_mae.mean()),
            "mean_rmse_K": float(base_rmse.mean()),
            "per_layer_mae":  base_mae.tolist(),
            "per_layer_rmse": base_rmse.tolist(),
        }
    }

    # ── few-shot adaptation for each K ────────────────────────────────────────
    for K in args.k_values:
        if K > len(avail_trajs):
            print(f"[adapt] K={K} exceeds available trajectories "
                  f"({len(avail_trajs)}), skipping.")
            continue

        print(f"\n[adapt] ── K = {K} ────────────────────────────")
        t0 = time.time()

        adapt_trajs, _ = sample_few_shot(avail_trajs, K, seed=args.seed)

        adapted_model, metrics = adapt_k_shot(
            base_model, adapt_trajs, test_trajs,
            state_mean, state_std, action_mean, action_std,
            args, device, K,
        )

        elapsed = time.time() - t0
        print(f"[adapt] K={K:3d} | "
              f"MAE={metrics['mean_mae_K']:.3f} K | "
              f"RMSE={metrics['mean_rmse_K']:.3f} K | "
              f"{elapsed:.0f}s")

        # Save checkpoint
        k_dir = os.path.join(out_dir, f"K{K:03d}")
        os.makedirs(k_dir, exist_ok=True)
        save_adapted(
            adapted_model, state_mean, state_std, action_mean, action_std,
            K=K, path=os.path.join(k_dir, "adapted.pt"),
            extra={"metrics": metrics, "base_checkpoint": args.base_checkpoint},
        )

        # Save metrics json
        with open(os.path.join(k_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        all_metrics[f"K{K:03d}"] = metrics

    # ── summary ──────────────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[adapt] Summary → {os.path.join(out_dir, 'metrics_summary.json')}")

    print("\n[adapt] Results:")
    print(f"  {'K':>6}  {'Mean MAE [K]':>14}  {'Mean RMSE [K]':>14}")
    print("  " + "-" * 40)
    for key in sorted(all_metrics.keys()):
        m = all_metrics[key]
        print(f"  {m['K']:>6}  {m['mean_mae_K']:>14.3f}  {m['mean_rmse_K']:>14.3f}")


if __name__ == "__main__":
    main()
