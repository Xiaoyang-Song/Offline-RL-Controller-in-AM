"""
baselines/online_q/train.py
--------------------------------
Baseline 5: online (interactive) Q-learning, discrete action space. Standard
epsilon-greedy DQN with a replay buffer and target network, trained by
stepping through the SAME surrogate environment (online_RL_ucpg_v2.env.
TwoStageLatentLPBFEnv) that naive_pg / UCPG v2 use — no fixed offline buffer,
no --data_path, no pickled dataset anywhere in this file. This isolates one
further variable against baselines/offline_q (which trains purely from the
static dataset, no environment interaction at all): does letting a
value-based method interact with the (surrogate) environment change
anything, independent of the reward-vs-uncertainty question naive_pg /
UCPG v2 already isolate.

Everything about the environment/surrogate wiring (surrogate loading,
TwoStageLatentLPBFEnv construction, --ood_min/--ood_max diagnostic) is
copied from baselines/naive_pg/train.py's conventions so the three on-policy
baselines (naive_pg, UCPG v2, online_q) share an identical CLI for every flag
they have in common — only the algorithm (policy gradient vs. Q-learning)
and its own hyperparameters differ.

MDP (same convention as everywhere else in this project):
    obs    = [z_t (raw normalised field) || layer_token || cool_time_token]
    action = laser power [W], snapped to the discrete ACTION_GRID (100:10:400W)
    reward = -meanDeviation of the END-OF-HEATING field (env.step())

Usage
-----
    python -m baselines.online_q.train \\
        --surrogate surrogate_model_v3/runs/<ts>/surrogate_best.pt \\
        --n_episodes 5000
"""

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_v3.model import load_surrogate
from online_RL_ucpg_v2.env   import TwoStageLatentLPBFEnv
from online_RL_ucpg_v2.train import _plot_series
from baselines.online_q.model import ACTION_GRID, LatentQNet


# =============================================================================
# Replay buffer
# =============================================================================

class ReplayBuffer:
    """Fixed-size ring buffer of (obs, layer, action_idx, reward, next_obs,
    next_layer, done) transitions, collected online from the environment."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs       = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.layer      = np.zeros(capacity, dtype=np.int64)
        self.next_layer = np.zeros(capacity, dtype=np.int64)
        self.action_idx = np.zeros(capacity, dtype=np.int64)
        self.reward     = np.zeros(capacity, dtype=np.float32)
        self.done       = np.zeros(capacity, dtype=np.bool_)
        self._ptr = 0
        self._size = 0

    def push(self, obs, layer, action_idx, reward, next_obs, next_layer, done):
        i = self._ptr
        self.obs[i] = obs; self.next_obs[i] = next_obs
        self.layer[i] = layer; self.next_layer[i] = next_layer
        self.action_idx[i] = action_idx; self.reward[i] = reward; self.done[i] = done
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int, device: str):
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.tensor(self.obs[idx],        dtype=torch.float32, device=device),
            torch.tensor(self.layer[idx],      dtype=torch.long,    device=device),
            torch.tensor(self.action_idx[idx], dtype=torch.long,    device=device),
            torch.tensor(self.reward[idx],     dtype=torch.float32, device=device),
            torch.tensor(self.next_obs[idx],   dtype=torch.float32, device=device),
            torch.tensor(self.next_layer[idx], dtype=torch.long,    device=device),
            torch.tensor(self.done[idx],       dtype=torch.bool,    device=device),
        )


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline 5: online (interactive) Q-learning, discrete actions.")
    p.add_argument("--surrogate", type=str, required=True)

    # ── environment (identical flags/defaults to baselines/naive_pg/train.py) ──
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
    p.add_argument("--action_min",  type=float, default=100.0,
                   help="Passed through to the env constructor only (informational — this "
                        "agent always chooses from the full ACTION_GRID, not a clipped range).")
    p.add_argument("--action_max",  type=float, default=400.0)
    p.add_argument("--ood_min", type=float, default=None,
                   help="Optional diagnostic: log the fraction of chosen actions outside "
                        "[ood_min, ood_max] every episode — same semantics as "
                        "baselines/naive_pg/train.py's --ood_min/--ood_max.")
    p.add_argument("--ood_max", type=float, default=None, help="See --ood_min.")

    # ── Q-network ────────────────────────────────────────────────────────────
    p.add_argument("--hidden",          type=int, default=128)
    p.add_argument("--depth",           type=int, default=3)
    p.add_argument("--layer_embed_dim", type=int, default=8)

    # ── DQN hyperparameters ──────────────────────────────────────────────────
    p.add_argument("--n_episodes",    type=int,   default=2000,
                   help="Total number of 12-layer episodes collected online from the environment.")
    p.add_argument("--warmup_episodes", type=int, default=50,
                   help="Episodes of pure random exploration (epsilon=1) before any gradient "
                        "update, so the replay buffer has something to sample from.")
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--buffer_size",   type=int,   default=50_000)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--updates_per_episode", type=int, default=12,
                   help="Gradient steps taken after each episode (default: one per layer "
                        "collected that episode, a common online-DQN update ratio).")
    p.add_argument("--target_sync_freq", type=int, default=10,
                   help="Hard-sync the target network every this many episodes.")
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_end",   type=float, default=0.05)
    p.add_argument("--epsilon_decay_episodes", type=int, default=None,
                   help="Episodes over which epsilon decays linearly from --epsilon_start to "
                        "--epsilon_end. Defaults to n_episodes // 2.")
    p.add_argument("--max_grad_norm", type=float, default=10.0)

    p.add_argument("--log_freq",  type=int, default=50)
    p.add_argument("--save_freq", type=int, default=500)
    p.add_argument("--out_dir",   type=str, default="")
    p.add_argument("--device",    type=str, default="")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    have_ood = args.ood_min is not None or args.ood_max is not None
    if have_ood and (args.ood_min is None or args.ood_max is None):
        raise ValueError("--ood_min and --ood_max must be given together.")

    epsilon_decay_episodes = args.epsilon_decay_episodes or max(args.n_episodes // 2, 1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.out_dir or os.path.join("baselines", "online_q", "runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 65)
    print("[online_q] Baseline 5 — online (interactive) Q-learning, discrete actions")
    print(f"[online_q] Output dir : {out_dir}")
    print(f"[online_q] Device     : {device}")
    print("=" * 65)

    (surrogate, state_mean, state_std, lp_mean, lp_std,
     cool_mean, cool_std) = load_surrogate(args.surrogate, device=device)
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
    obs_dim    = env.obs_dim
    latent_dim = env.latent_dim
    n_actions  = len(ACTION_GRID)
    action_grid_t = torch.tensor(ACTION_GRID, dtype=torch.float32, device=device)

    qnet = LatentQNet(obs_dim, latent_dim, n_actions, args.hidden, args.depth,
                      args.n_layers, args.layer_embed_dim).to(device)
    target = LatentQNet(obs_dim, latent_dim, n_actions, args.hidden, args.depth,
                        args.n_layers, args.layer_embed_dim).to(device)
    target.load_state_dict(qnet.state_dict())
    target.eval()
    for pparam in target.parameters():
        pparam.requires_grad_(False)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[online_q] {qnet}")

    buffer = ReplayBuffer(args.buffer_size, obs_dim)

    hist_return, hist_loss, hist_epsilon, hist_frac_ood = [], [], [], []
    hist_action_mean, hist_action_std = [], []
    best_return = -float("inf")
    loss_val = float("nan")

    t0 = time.time()
    for ep in range(1, args.n_episodes + 1):
        epsilon = (args.epsilon_start if ep <= args.warmup_episodes else
                  np.interp(ep - args.warmup_episodes, [0, epsilon_decay_episodes],
                            [args.epsilon_start, args.epsilon_end]))

        obs = env.reset()
        ep_reward, ep_actions = 0.0, []
        for t in range(args.n_layers):
            if np.random.rand() < epsilon:
                a_idx = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    obs_t   = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    layer_t = torch.tensor([t], dtype=torch.long, device=device)
                    a_idx = int(qnet(obs_t, layer_t).squeeze(0).argmax().item())
            action_W = float(ACTION_GRID[a_idx])
            next_obs, reward, done, _info = env.step(action_W)
            next_layer = min(t + 1, args.n_layers - 1)

            buffer.push(obs, t, a_idx, reward, next_obs, next_layer, done)
            obs = next_obs
            ep_reward += reward
            ep_actions.append(action_W)

        if len(buffer) >= args.batch_size and ep > args.warmup_episodes:
            ep_losses = []
            for _ in range(args.updates_per_episode):
                ob, ly, ai, rw, nob, nly, dn = buffer.sample(args.batch_size, device)
                with torch.no_grad():
                    q_next = target(nob, nly).max(dim=-1).values
                    y = rw + args.gamma * (~dn).float() * q_next
                q_pred = qnet(ob, ly).gather(-1, ai.unsqueeze(-1)).squeeze(-1)
                loss = nn.functional.mse_loss(q_pred, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), args.max_grad_norm)
                optimizer.step()
                ep_losses.append(loss.item())
            loss_val = float(np.mean(ep_losses))

        if ep % args.target_sync_freq == 0:
            target.load_state_dict(qnet.state_dict())

        ep_actions = np.array(ep_actions)
        hist_return.append(ep_reward)
        hist_loss.append(loss_val)
        hist_epsilon.append(float(epsilon))
        hist_action_mean.append(float(ep_actions.mean()))
        hist_action_std.append(float(ep_actions.std()))
        if have_ood:
            hist_frac_ood.append(float(((ep_actions < args.ood_min) | (ep_actions > args.ood_max)).mean()))

        if ep % args.log_freq == 0 or ep == 1:
            n_recent = min(args.log_freq, len(hist_return))
            ood_str  = (f" | OOD-action {hist_frac_ood[-1]*100:5.1f}%" if have_ood else "")
            print(f"Episode {ep:6d}/{args.n_episodes} | return {ep_reward:+.4f} "
                  f"(avg{n_recent} {np.mean(hist_return[-n_recent:]):+.4f}) | "
                  f"loss {loss_val:.5f} | epsilon {epsilon:.3f} | "
                  f"action {ep_actions.mean():.1f}±{ep_actions.std():.1f}W"
                  f"{ood_str} | {time.time()-t0:.0f}s")

        if ep_reward > best_return:
            best_return = ep_reward
            torch.save({
                "qnet_state_dict": qnet.state_dict(),
                "model_config": dict(obs_dim=obs_dim, latent_dim=latent_dim, n_actions=n_actions,
                                     hidden=args.hidden, depth=args.depth, n_layers=args.n_layers,
                                     layer_embed_dim=args.layer_embed_dim),
                "action_grid": ACTION_GRID, "gamma": args.gamma,
                "episode": ep, "best_return": best_return, "train_args": vars(args),
            }, os.path.join(out_dir, "online_q_best.pt"))

        if ep % args.save_freq == 0:
            _plot_series(hist_return, "Undiscounted episode return", "Online Q-learning — Reward Return",
                        os.path.join(out_dir, "return.png"))
            _plot_series(hist_loss, "Bellman MSE loss", "Online Q-learning — Loss",
                        os.path.join(out_dir, "loss.png"), color="tab:orange")
            _plot_series(hist_epsilon, "Epsilon", "Online Q-learning — Exploration Schedule",
                        os.path.join(out_dir, "epsilon.png"), color="tab:purple")
            if have_ood:
                _plot_series(hist_frac_ood, "Fraction of actions outside [ood_min, ood_max]",
                            f"Online Q-learning — Fraction of Chosen Actions Outside "
                            f"[{args.ood_min:.0f}, {args.ood_max:.0f}] W",
                            os.path.join(out_dir, "ood_action_fraction.png"), color="tab:brown")

    torch.save({
        "qnet_state_dict": qnet.state_dict(),
        "model_config": dict(obs_dim=obs_dim, latent_dim=latent_dim, n_actions=n_actions,
                             hidden=args.hidden, depth=args.depth, n_layers=args.n_layers,
                             layer_embed_dim=args.layer_embed_dim),
        "action_grid": ACTION_GRID, "gamma": args.gamma,
        "episode": args.n_episodes, "hist_return": hist_return, "hist_frac_ood": hist_frac_ood,
        "train_args": vars(args),
    }, os.path.join(out_dir, "online_q_final.pt"))

    _plot_series(hist_return, "Undiscounted episode return", "Online Q-learning — Reward Return",
                os.path.join(out_dir, "return.png"))
    if have_ood:
        _plot_series(hist_frac_ood, "Fraction of actions outside [ood_min, ood_max]",
                    f"Online Q-learning — Fraction of Chosen Actions Outside "
                    f"[{args.ood_min:.0f}, {args.ood_max:.0f}] W",
                    os.path.join(out_dir, "ood_action_fraction.png"), color="tab:brown")

    print(f"\n[online_q] Done. Best return: {best_return:.4f}")
    print(f"[online_q] Best checkpoint: {os.path.join(out_dir, 'online_q_best.pt')}")


if __name__ == "__main__":
    main()
