"""
baselines/naive_pg/train.py
-------------------------------
Baseline 0: naive policy gradient — plain Monte Carlo REINFORCE maximising
reward alone:

    theta <- theta + alpha_theta * (1/N) sum_i sum_t
                 nabla_theta log pi_theta(a_t^(i)|s_t^(i)) * G_r,t^(i)

No uncertainty term, no Lagrange multiplier, no dual ascent — the "before"
picture for online_RL_ucpg_v2's uncertainty constraint. Everything else
(environment, two-stage surrogate transition, continuous Gaussian policy
architecture, on-policy Monte Carlo data collection) is reused UNMODIFIED
from online_RL_ucpg_v2, imported here read-only, so the only difference
between this baseline and the real UCPG v2 run is whether the uncertainty
term is present in the advantage at all.

Usage
-----
    python -m baselines.naive_pg.train \\
        --surrogate surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt \\
        --action_min 100 --action_max 400 \\
        --n_iterations 2000
"""

import argparse
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.train import load_two_stage_surrogate
from online_RL_ucpg_v2.env   import TwoStageLatentLPBFEnv
from online_RL_ucpg_v2.agent import UCPGAgentV2
from online_RL_ucpg_v2.train import collect_batch, discounted_returns, _plot_series


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline 0: naive (unconstrained) continuous policy gradient.")
    p.add_argument("--surrogate", type=str, required=True)

    p.add_argument("--T_l",          type=float, default=2000.0)
    p.add_argument("--T_h",          type=float, default=2800.0)
    p.add_argument("--n_layers",     type=int,   default=12)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--mesh_path",    type=str,   default="surrogate_model/mesh.mat")
    p.add_argument("--width",        type=float, default=12.0)
    p.add_argument("--height",       type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end",   type=float, default=0.5)
    p.add_argument("--cool_time_min", type=float, default=0.05)
    p.add_argument("--cool_time_max", type=float, default=0.15)
    p.add_argument("--action_min",  type=float, default=100.0)
    p.add_argument("--action_max",  type=float, default=400.0)

    p.add_argument("--hidden",          type=int,   default=128)
    p.add_argument("--depth",           type=int,   default=3)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--layer_embed_dim", type=int,   default=8)
    p.add_argument("--log_sigma_init",  type=float, default=-1.0)
    p.add_argument("--min_log_sigma",   type=float, default=-3.0)
    p.add_argument("--max_log_sigma",   type=float, default=0.0)

    p.add_argument("--n_traj",       type=int,   default=32)
    p.add_argument("--n_iterations", type=int,   default=2000)
    p.add_argument("--gamma_r",      type=float, default=0.99)
    p.add_argument("--lr_theta",     type=float, default=3e-4)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--use_baseline", action="store_true",
                   help="Subtract the batch's per-timestep empirical mean return "
                        "(variance reduction only, see online_RL_ucpg_v2/README.md).")

    p.add_argument("--log_freq",  type=int, default=20)
    p.add_argument("--save_freq", type=int, default=200)
    p.add_argument("--out_dir",   type=str, default="")
    p.add_argument("--device",    type=str, default="")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.out_dir or os.path.join("baselines", "naive_pg", "runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 65)
    print("[naive_pg] Baseline 0 — naive (unconstrained) policy gradient")
    print(f"[naive_pg] Output dir : {out_dir}")
    print(f"[naive_pg] Device     : {device}")
    print("=" * 65)

    (surrogate, state_mean, state_std, lp_mean, lp_std,
     cool_mean, cool_std, _roi) = load_two_stage_surrogate(args.surrogate, device=device)
    surrogate.eval()

    env = TwoStageLatentLPBFEnv(
        surrogate=surrogate, state_mean=state_mean, state_std=state_std,
        lp_mean=lp_mean, lp_std=lp_std, cool_mean=cool_mean, cool_std=cool_std,
        temp_range=(args.T_l, args.T_h), n_layers=args.n_layers, initial_temp=args.initial_temp,
        device=device, mesh_path=args.mesh_path, width=args.width, height=args.height,
        sq_frac_start=args.sq_frac_start, sq_frac_end=args.sq_frac_end,
        action_min=args.action_min, action_max=args.action_max,
        cool_time_min=args.cool_time_min, cool_time_max=args.cool_time_max,
    )

    # lambda_init/lr_lambda are irrelevant here: we simply never call
    # update_lambda, and never subtract lambda*G_u from the advantage below,
    # so agent.lam stays at 0 for the agent's entire lifetime — the loss is
    # purely the reward-return policy gradient.
    agent = UCPGAgentV2(
        latent_dim=env.latent_dim, action_min=args.action_min, action_max=args.action_max,
        hidden=args.hidden, depth=args.depth, dropout=args.dropout,
        n_layers=args.n_layers, layer_embed_dim=args.layer_embed_dim,
        log_sigma_init=args.log_sigma_init, min_log_sigma=args.min_log_sigma, max_log_sigma=args.max_log_sigma,
        lr_theta=args.lr_theta, lambda_init=0.0, max_grad_norm=args.max_grad_norm, device=device,
    )
    print(f"[naive_pg] {agent}")

    hist_return, hist_loss, hist_entropy = [], [], []
    best_return = -float("inf")

    t0 = time.time()
    for k in range(1, args.n_iterations + 1):
        obs, actions, rewards, _u = collect_batch(env, agent, args.n_traj, args.n_layers)

        G_r = discounted_returns(rewards, args.gamma_r)
        advantage = G_r - G_r.mean(axis=0, keepdims=True) if args.use_baseline else G_r

        N, T, obs_dim = obs.shape
        obs_flat       = torch.tensor(obs.reshape(N * T, obs_dim), dtype=torch.float32, device=device)
        actions_flat   = torch.tensor(actions.reshape(N * T),      dtype=torch.float32, device=device)
        advantage_flat = torch.tensor(advantage.reshape(N * T),    dtype=torch.float32, device=device)

        loss, entropy = agent.policy_step(obs_flat, actions_flat, advantage_flat, N)
        raw_return = float(rewards.sum(axis=1).mean())

        hist_return.append(raw_return); hist_loss.append(loss); hist_entropy.append(entropy)

        if k % args.log_freq == 0 or k == 1:
            n_recent = min(args.log_freq, len(hist_return))
            print(f"Iter {k:5d}/{args.n_iterations} | return {raw_return:+.4f} "
                  f"(avg{n_recent} {np.mean(hist_return[-n_recent:]):+.4f}) | "
                  f"loss {loss:+.4f} | entropy {entropy:.3f} | {time.time()-t0:.0f}s")

        if raw_return > best_return:
            best_return = raw_return
            agent.save(os.path.join(out_dir, "naive_pg_best.pt"),
                      extra={"iteration": k, "best_return": best_return, "train_args": vars(args)})

        if k % args.save_freq == 0:
            agent.save(os.path.join(out_dir, f"naive_pg_iter{k:05d}.pt"),
                      extra={"iteration": k, "hist_return": hist_return, "train_args": vars(args)})
            _plot_series(hist_return, "Undiscounted episode return", "Naive PG — Reward Return",
                        os.path.join(out_dir, "return.png"))
            _plot_series(hist_loss, "Policy loss", "Naive PG — Policy Gradient Loss",
                        os.path.join(out_dir, "loss.png"), color="tab:orange")
            _plot_series(hist_entropy, "Mean policy entropy", "Naive PG — Policy Entropy",
                        os.path.join(out_dir, "entropy.png"), color="tab:green")

    agent.save(os.path.join(out_dir, "naive_pg_final.pt"),
              extra={"iteration": args.n_iterations, "hist_return": hist_return, "train_args": vars(args)})
    _plot_series(hist_return, "Undiscounted episode return", "Naive PG — Reward Return",
                os.path.join(out_dir, "return.png"))
    print(f"\n[naive_pg] Done. Best return: {best_return:.4f}")
    print(f"[naive_pg] Best checkpoint: {os.path.join(out_dir, 'naive_pg_best.pt')}")


if __name__ == "__main__":
    main()
