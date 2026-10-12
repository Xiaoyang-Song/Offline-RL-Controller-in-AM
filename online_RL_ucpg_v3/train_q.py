"""
online_RL_ucpg_v3/train_q.py
--------------------------------
Online DQN (epsilon-greedy, replay buffer, target network) on the v3
environment, with the uncertainty-SHAPED reward (--reward_mode shaped) or the
plain reward (--reward_mode plain, the matched naive-Q baseline).

Same algorithm and hyperparameter defaults as baselines/online_q/train.py; the
Q-network (LatentQNet), action grid and replay buffer are imported read-only
from there, and checkpoints use the same format, so
baselines.online_q.model.load_online_q_controller loads them for evaluation.

Usage
-----
    python -m online_RL_ucpg_v3.train_q \\
        --surrogate surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt \\
        --ood_min 200 --ood_max 300 --reward_mode shaped --n_episodes 5000 \\
        --out_dir online_RL_ucpg_v3/runs/narrow_q_shaped
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_RL_ucpg_v2.train import _plot_series
from baselines.online_q.model import ACTION_GRID, LatentQNet
from baselines.online_q.train import ReplayBuffer
from online_RL_ucpg_v3.common import add_env_args, make_env


def parse_args():
    p = argparse.ArgumentParser(description="v3: online Q-learning with uncertainty-shaped (or plain) reward.")
    add_env_args(p)
    p.add_argument("--hidden",          type=int, default=128)
    p.add_argument("--depth",           type=int, default=3)
    p.add_argument("--layer_embed_dim", type=int, default=8)
    p.add_argument("--n_episodes",      type=int,   default=5000)
    p.add_argument("--warmup_episodes", type=int,   default=50)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight_decay",    type=float, default=1e-5)
    p.add_argument("--buffer_size",     type=int,   default=50_000)
    p.add_argument("--batch_size",      type=int,   default=256)
    p.add_argument("--updates_per_episode", type=int, default=12)
    p.add_argument("--target_sync_freq", type=int, default=10)
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_end",   type=float, default=0.05)
    p.add_argument("--epsilon_decay_episodes", type=int, default=None)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--eval_every",    type=int, default=100,
                   help="Every this many episodes, run greedy episodes to pick the best checkpoint "
                        "(selection on the greedy policy, not on noisy epsilon-greedy returns).")
    p.add_argument("--eval_episodes", type=int, default=16)
    p.add_argument("--log_freq",  type=int, default=50)
    p.add_argument("--save_freq", type=int, default=500)
    p.add_argument("--out_dir",   type=str, required=True)
    p.add_argument("--device",    type=str, default="")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def greedy_action(qnet, obs, t, device):
    with torch.no_grad():
        q = qnet(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                 torch.tensor([t], dtype=torch.long, device=device)).squeeze(0)
    return int(q.argmax().item())


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"v3-Q ({args.reward_mode}, {args.reward_field})"
    print("=" * 70); print(f"[{tag}] out_dir={args.out_dir} device={device}"); print("=" * 70)

    env, cal = make_env(args, device)
    with open(os.path.join(args.out_dir, "env_config.json"), "w") as f:
        json.dump(dict(vars(args), calibration=cal), f, indent=2, default=str)

    n_actions = len(ACTION_GRID)
    cfg = dict(obs_dim=env.obs_dim, latent_dim=env.latent_dim, n_actions=n_actions, hidden=args.hidden,
               depth=args.depth, n_layers=args.n_layers, layer_embed_dim=args.layer_embed_dim)
    qnet, target = LatentQNet(**cfg).to(device), LatentQNet(**cfg).to(device)
    target.load_state_dict(qnet.state_dict()); target.eval()
    for p_ in target.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.Adam(qnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buf = ReplayBuffer(args.buffer_size, env.obs_dim)
    decay = args.epsilon_decay_episodes or max(args.n_episodes // 2, 1)
    have_ood = args.ood_min is not None and args.ood_max is not None

    def save(path, **extra):
        torch.save(dict(qnet_state_dict=qnet.state_dict(), model_config=cfg, action_grid=ACTION_GRID,
                        gamma=args.gamma, train_args=vars(args), v3_calibration=cal, **extra), path)

    def greedy_eval():
        rs, ps = [], []
        for _ in range(args.eval_episodes):
            o = env.reset(); R = P = 0.0
            for t in range(args.n_layers):
                o, r, _d, info = env.step(float(ACTION_GRID[greedy_action(qnet, o, t, device)]))
                R += r; P += info["reward_plain"]
            rs.append(R); ps.append(P)
        return float(np.mean(rs)), float(np.mean(ps))

    H = {k: [] for k in ("ret", "plain", "sigma", "eps", "ood", "greedy_ret", "greedy_plain")}
    best, loss_val, t0 = -float("inf"), float("nan"), time.time()
    for ep in range(1, args.n_episodes + 1):
        eps = (args.epsilon_start if ep <= args.warmup_episodes else
               np.interp(ep - args.warmup_episodes, [0, decay], [args.epsilon_start, args.epsilon_end]))
        o = env.reset(); R = P = 0.0; acts, sig = [], []
        for t in range(args.n_layers):
            ai = np.random.randint(n_actions) if np.random.rand() < eps else greedy_action(qnet, o, t, device)
            o2, r, done, info = env.step(float(ACTION_GRID[ai]))
            buf.push(o, t, ai, r, o2, min(t + 1, args.n_layers - 1), done)
            o = o2; R += r; P += info["reward_plain"]
            acts.append(float(ACTION_GRID[ai])); sig.append(info["sigma_epi_roi_K"])

        if len(buf) >= args.batch_size and ep > args.warmup_episodes:
            ls = []
            for _ in range(args.updates_per_episode):
                ob, ly, ai_, rw, nob, nly, dn = buf.sample(args.batch_size, device)
                with torch.no_grad():
                    y = rw + args.gamma * (~dn).float() * target(nob, nly).max(-1).values
                q = qnet(ob, ly).gather(-1, ai_.unsqueeze(-1)).squeeze(-1)
                loss = nn.functional.mse_loss(q, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), args.max_grad_norm); opt.step()
                ls.append(loss.item())
            loss_val = float(np.mean(ls))
        if ep % args.target_sync_freq == 0:
            target.load_state_dict(qnet.state_dict())

        acts = np.array(acts)
        H["ret"].append(R); H["plain"].append(P); H["sigma"].append(float(np.mean(sig))); H["eps"].append(float(eps))
        if have_ood:
            H["ood"].append(float(((acts < args.ood_min) | (acts > args.ood_max)).mean()))

        if ep % args.eval_every == 0 and ep > args.warmup_episodes:
            g_ret, g_plain = greedy_eval()
            H["greedy_ret"].append(g_ret); H["greedy_plain"].append(g_plain)
            if g_ret > best:                               # select on the TRAINED objective, greedy policy
                best = g_ret
                save(os.path.join(args.out_dir, "q_best.pt"), episode=ep, best_return=best, best_plain_return=g_plain)
            print(f"  [greedy eval @ ep {ep}] return {g_ret:+.4f} | plain {g_plain:+.4f} | best {best:+.4f}")

        if ep % args.log_freq == 0 or ep == 1:
            ood = f" | OOD {H['ood'][-1] * 100:5.1f}%" if have_ood else ""
            print(f"Episode {ep:6d}/{args.n_episodes} | return {R:+.4f} | plain {P:+.4f} | sigma "
                  f"{H['sigma'][-1]:.2f}K | loss {loss_val:.5f} | eps {eps:.3f} | action "
                  f"{acts.mean():.1f}±{acts.std():.1f}W{ood} | {time.time() - t0:.0f}s")
        if ep % args.save_freq == 0:
            _plot_series(H["ret"], "Return (trained reward)", f"{tag} — return", os.path.join(args.out_dir, "return.png"))
            _plot_series(H["plain"], "Return (plain deviation)", f"{tag} — plain return",
                         os.path.join(args.out_dir, "return_plain.png"), color="tab:green")
            if have_ood:
                _plot_series(H["ood"], "OOD-action fraction", f"{tag} — OOD actions",
                             os.path.join(args.out_dir, "ood_action_fraction.png"), color="tab:brown")

    save(os.path.join(args.out_dir, "q_final.pt"), episode=args.n_episodes, history=H)
    print(f"[{tag}] done. best greedy trained return {best:+.4f}  checkpoint: {os.path.join(args.out_dir, 'q_best.pt')}")


if __name__ == "__main__":
    main()
