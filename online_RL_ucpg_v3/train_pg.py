"""
online_RL_ucpg_v3/train_pg.py
---------------------------------
Monte Carlo policy gradient (REINFORCE) on the v3 environment:

    theta <- theta + alpha (1/N) sum_i sum_t grad log pi(a_t|s_t) * G_t,   G_t = sum_k gamma^(k-t) r_k

with r_t the uncertainty-SHAPED reward (--reward_mode shaped, "UCPG v3") or the
plain reward (--reward_mode plain, the matched naive-PG baseline on the same
reward field). No uncertainty constraint, no Lagrange multiplier. Policy =
online_RL_ucpg_v2's UCPGAgentV2 (imported read-only, lambda fixed at 0), so
checkpoints load in every existing evaluator.

Logs both the trained reward and the PLAIN reward (true deviation of the
member-mean field), so shaped and plain runs are compared on the same scale.

Usage
-----
    python -m online_RL_ucpg_v3.train_pg \\
        --surrogate surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt \\
        --ood_min 200 --ood_max 300 --reward_mode shaped --use_baseline \\
        --out_dir online_RL_ucpg_v3/runs/narrow_pg_shaped
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_RL_ucpg_v2.agent import UCPGAgentV2
from online_RL_ucpg_v2.train import discounted_returns, _plot_series
from online_RL_ucpg_v3.common import add_env_args, make_env
from online_RL_ucpg_v3.env import in_ranges


def parse_args():
    p = argparse.ArgumentParser(description="v3: policy gradient with uncertainty-shaped (or plain) reward.")
    add_env_args(p)
    p.add_argument("--hidden",          type=int,   default=128)
    p.add_argument("--depth",           type=int,   default=3)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--layer_embed_dim", type=int,   default=8)
    p.add_argument("--log_sigma_init",  type=float, default=-1.0)
    p.add_argument("--min_log_sigma",   type=float, default=-3.0)
    p.add_argument("--max_log_sigma",   type=float, default=0.0)
    p.add_argument("--n_traj",       type=int,   default=32)
    p.add_argument("--n_iterations", type=int,   default=2000)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--lr_theta",     type=float, default=3e-4)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--use_baseline", action="store_true",
                   help="Subtract the batch's per-timestep mean return (variance reduction).")
    p.add_argument("--log_freq",  type=int, default=20)
    p.add_argument("--save_freq", type=int, default=200)
    p.add_argument("--out_dir",   type=str, required=True)
    p.add_argument("--device",    type=str, default="")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def collect(env, agent, n_traj, n_layers):
    obs = np.zeros((n_traj, n_layers, env.obs_dim), np.float32)
    act = np.zeros((n_traj, n_layers), np.float32)
    rew = np.zeros((n_traj, n_layers), np.float32)
    plain = np.zeros((n_traj, n_layers), np.float32)
    sig = np.zeros((n_traj, n_layers), np.float32)
    trust = np.zeros((n_traj, n_layers), np.float32)
    for i in range(n_traj):
        o = env.reset()
        for t in range(n_layers):
            a = agent.select_action(o, explore=True)
            o2, r, done, info = env.step(a)
            obs[i, t], act[i, t], rew[i, t] = o, a, r
            plain[i, t], sig[i, t], trust[i, t] = info["reward_plain"], info["sigma_epi_roi_K"], info["trust_roi"]
            o = o2
    return obs, act, rew, plain, sig, trust


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"v3-PG ({args.reward_mode}, {args.reward_field})"
    print("=" * 70); print(f"[{tag}] out_dir={args.out_dir} device={device}"); print("=" * 70)

    env, cal = make_env(args, device)
    with open(os.path.join(args.out_dir, "env_config.json"), "w") as f:
        json.dump(dict(vars(args), calibration=cal), f, indent=2, default=str)

    agent = UCPGAgentV2(
        latent_dim=env.latent_dim, action_min=args.action_min, action_max=args.action_max,
        hidden=args.hidden, depth=args.depth, dropout=args.dropout, n_layers=args.n_layers,
        layer_embed_dim=args.layer_embed_dim, log_sigma_init=args.log_sigma_init,
        min_log_sigma=args.min_log_sigma, max_log_sigma=args.max_log_sigma,
        lr_theta=args.lr_theta, lambda_init=0.0, max_grad_norm=args.max_grad_norm, device=device,
    )
    have_ood = args.id_range_list is not None
    H = {k: [] for k in ("ret", "plain", "sigma", "trust", "act_mean", "ood", "loss", "entropy")}
    best = -float("inf")
    t0 = time.time()
    for it in range(1, args.n_iterations + 1):
        obs, act, rew, plain, sig, trust = collect(env, agent, args.n_traj, args.n_layers)
        G = discounted_returns(rew, args.gamma)
        adv = G - G.mean(axis=0, keepdims=True) if args.use_baseline else G
        N, T, D = obs.shape
        loss, ent = agent.policy_step(
            torch.tensor(obs.reshape(N * T, D), device=device),
            torch.tensor(act.reshape(N * T), device=device),
            torch.tensor(adv.reshape(N * T), device=device), N)

        ret, ret_plain = float(rew.sum(1).mean()), float(plain.sum(1).mean())
        H["ret"].append(ret); H["plain"].append(ret_plain); H["sigma"].append(float(sig.mean()))
        H["trust"].append(float(np.nanmean(trust))); H["act_mean"].append(float(act.mean()))
        H["loss"].append(loss); H["entropy"].append(ent)
        if have_ood:
            H["ood"].append(float((~in_ranges(act, args.id_range_list)).mean()))

        if it % args.log_freq == 0 or it == 1:
            ood = f" | OOD {H['ood'][-1] * 100:5.1f}%" if have_ood else ""
            print(f"Iter {it:5d}/{args.n_iterations} | return {ret:+.4f} | plain {ret_plain:+.4f} "
                  f"| sigma {H['sigma'][-1]:.2f}K | trust {H['trust'][-1]:.3f} | action {act.mean():.1f}"
                  f"±{act.std():.1f}W{ood} | entropy {ent:.3f} | {time.time() - t0:.0f}s")

        if ret > best:                                     # select on the TRAINED objective
            best = ret
            agent.save(os.path.join(args.out_dir, "pg_best.pt"),
                       extra=dict(iteration=it, best_return=best, best_plain_return=ret_plain,
                                  train_args=vars(args), v3_calibration=cal))
        if it % args.save_freq == 0 or it == args.n_iterations:
            agent.save(os.path.join(args.out_dir, "pg_final.pt" if it == args.n_iterations else f"pg_iter{it:05d}.pt"),
                       extra=dict(iteration=it, history=H, train_args=vars(args), v3_calibration=cal))
            _plot_series(H["ret"], "Return (trained reward)", f"{tag} — return", os.path.join(args.out_dir, "return.png"))
            _plot_series(H["plain"], "Return (plain deviation)", f"{tag} — plain return",
                         os.path.join(args.out_dir, "return_plain.png"), color="tab:green")
            _plot_series(H["sigma"], "ROI epistemic sigma [K]", f"{tag} — uncertainty",
                         os.path.join(args.out_dir, "sigma.png"), color="tab:red")
            if have_ood:
                _plot_series(H["ood"], "OOD-action fraction", f"{tag} — OOD actions",
                             os.path.join(args.out_dir, "ood_action_fraction.png"), color="tab:brown")
    print(f"[{tag}] done. best trained return {best:+.4f}  checkpoint: {os.path.join(args.out_dir, 'pg_best.pt')}")


if __name__ == "__main__":
    main()
