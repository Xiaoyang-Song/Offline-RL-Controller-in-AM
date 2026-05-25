"""
online_RL/train.py
------------------
Online Double-DQN training loop for the LPBF laser-power controller.

The agent interacts with the LPBF surrogate environment episode by episode:
  state  : 1053-D normalised temperature field
  action : discrete laser power index  (ACTION_LIST: 150→400 W, step 10)
  reward : −meanDeviation from nominal temperature window [T_l, T_h]

Training protocol
-----------------
  1. Warm-up  : `warmup_episodes` episodes of pure random exploration to
                pre-fill the replay buffer before any learning.
  2. Learning : ε-greedy exploration with linear ε decay; after each
                environment step the agent performs `updates_per_step`
                gradient updates from the replay buffer.
  3. Target   : hard target-network copy every `target_update_freq` episodes.
  4. Checkpts : best-return checkpoint + periodic episode checkpoints.
  5. Plots    : returns curve and Q-loss curve (updated at save_freq).

Usage
-----
    python -m online_RL.train \\
        --surrogate surrogate_model/runs/20260521_210923/surrogate_best.pt

All outputs go under --out_dir (default: online_RL/runs/<timestamp>/).
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

# ── ensure repo root is on sys.path when invoked as a script ─────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model.train import load_surrogate
from online_RL.env         import LPBFEnv, ACTION_LIST, NUM_ACTIONS
from online_RL.replay_buffer import ReplayBuffer
from online_RL.agent       import DQNAgent


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online Double-DQN training for the LPBF laser-power controller."
    )

    # ── surrogate model ───────────────────────────────────────────────────────
    p.add_argument(
        "--surrogate", type=str,
        default="surrogate_model/runs/20260521_210923/surrogate_best.pt",
        help="Path to the trained surrogate checkpoint (surrogate_best.pt).",
    )

    # ── environment ───────────────────────────────────────────────────────────
    p.add_argument("--T_l",          type=float, default=2000.0,
                   help="Lower nominal temperature bound [K]  (from test.py tempRange).")
    p.add_argument("--T_h",          type=float, default=2800.0,
                   help="Upper nominal temperature bound [K]  (from test.py tempRange).")
    p.add_argument("--n_layers",     type=int,   default=12,
                   help="Number of LPBF layers per episode.")
    p.add_argument("--initial_temp", type=float, default=300.0,
                   help="Initial uniform temperature field [K].")
    p.add_argument("--mesh_path",    type=str,   default="surrogate_model/mesh.mat",
                   help="Path to mesh.mat for exact scan-region node masks. "
                        "Falls back to all-nodes if file is missing.")
    p.add_argument("--width",         type=float, default=12.0,
                   help="Domain width  (must match MATLAB params.width).")
    p.add_argument("--height",        type=float, default=3.0,
                   help="Domain height (must match MATLAB params.height).")
    p.add_argument("--sq_frac_start", type=float, default=0.4,
                   help="squareSideFraction at layer 1 (matches test.py initialFraction).")
    p.add_argument("--sq_frac_end",   type=float, default=0.5,
                   help="squareSideFraction at last layer (matches test.py finalFraction).")

    # ── Q-network architecture ────────────────────────────────────────────────
    p.add_argument("--hidden",  type=int,   default=256,
                   help="Q-network hidden width.")
    p.add_argument("--depth",   type=int,   default=4,
                   help="Number of residual blocks in Q-network.")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout inside residual blocks (0 = off).")

    # ── RL hyperparameters ────────────────────────────────────────────────────
    p.add_argument("--n_episodes",          type=int,   default=3000,
                   help="Total number of training episodes.")
    p.add_argument("--gamma",               type=float, default=0.99,
                   help="Discount factor γ.")
    p.add_argument("--lr",                  type=float, default=3e-4,
                   help="Adam learning rate.")
    p.add_argument("--batch_size",          type=int,   default=256,
                   help="Mini-batch size for Q-network updates.")
    p.add_argument("--buffer_capacity",     type=int,   default=100_000,
                   help="Replay buffer capacity.")
    p.add_argument("--warmup_episodes",     type=int,   default=50,
                   help="Episodes of pure random exploration before learning starts.")
    p.add_argument("--target_update_freq",  type=int,   default=10,
                   help="Hard target-network update every N episodes.")
    p.add_argument("--updates_per_step",    type=int,   default=1,
                   help="Q-network gradient updates per environment step.")

    p.add_argument("--epsilon_start",       type=float, default=1.0,
                   help="Initial exploration probability ε.")
    p.add_argument("--epsilon_end",         type=float, default=0.05,
                   help="Minimum exploration probability ε.")
    p.add_argument("--epsilon_decay_steps", type=int,   default=50_000,
                   help="Number of Q-updates over which ε decays linearly.")

    p.add_argument("--double_dqn",    action="store_true",  default=True,
                   help="Use Double-DQN (default: True).")
    p.add_argument("--no_double_dqn", dest="double_dqn", action="store_false",
                   help="Disable Double-DQN (use vanilla DQN instead).")

    # ── logging / checkpointing ────────────────────────────────────────────────
    p.add_argument("--log_freq",  type=int, default=50,
                   help="Print metrics every N episodes.")
    p.add_argument("--save_freq", type=int, default=500,
                   help="Save checkpoint + plots every N episodes.")
    p.add_argument("--out_dir",   type=str, default="",
                   help="Output directory. Defaults to online_RL/runs/<timestamp>/.")
    p.add_argument("--device",    type=str, default="",
                   help="'cuda' or 'cpu'. Auto-detected if empty.")
    p.add_argument("--seed",      type=int, default=42,
                   help="Random seed for reproducibility.")

    # ── optional warm-start from an existing online-RL checkpoint ────────────
    p.add_argument("--resume",    type=str, default="",
                   help="Path to a previous DQN checkpoint to resume training from.")

    return p.parse_args()


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_returns(returns: list, out_path: str, window: int = 50) -> None:
    """Episode-return curve with rolling average."""
    fig, ax = plt.subplots(figsize=(10, 5))
    eps = np.arange(1, len(returns) + 1)
    ax.plot(eps, returns, alpha=0.35, color="steelblue",
            linewidth=0.7, label="Episode return")
    if len(returns) >= window:
        smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(eps[window - 1:], smooth, color="steelblue",
                linewidth=2.0, label=f"Rolling avg ({window})")
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Ideal (0 = zero deviation)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (sum of rewards per episode)")
    ax.set_title("Online DQN — Episode Returns")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[train] Returns plot → {out_path}")


def plot_q_loss(losses: list, out_path: str) -> None:
    """Q-network Huber-loss curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, linewidth=0.4, alpha=0.6, color="darkorange")
    ax.set_xlabel("Q-network update step")
    ax.set_ylabel("Huber loss")
    ax.set_title("Online DQN — Q-loss Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[train] Q-loss plot  → {out_path}")


def plot_epsilon(epsilons: list, out_path: str) -> None:
    """ε vs episode."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(1, len(epsilons) + 1), epsilons, color="seagreen", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon (exploration probability)")
    ax.set_title("ε-greedy Decay Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[train] Epsilon plot → {out_path}")


# =============================================================================
# Main training loop
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── output directory ─────────────────────────────────────────────────────
    if not args.out_dir:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("online_RL", "runs", ts)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print(f"[train] Online DQN — LPBF Laser-Power Controller")
    print(f"[train] Output dir : {out_dir}")
    print(f"[train] Device     : {device}")
    print("=" * 65)

    # ── load surrogate model ─────────────────────────────────────────────────
    print(f"\n[train] Loading surrogate: {args.surrogate}")
    surrogate, state_mean, state_std, action_mean, action_std = load_surrogate(
        args.surrogate, device=device
    )
    surrogate.eval()
    state_dim = int(state_mean.shape[0])   # raw temperature field dim (1053)
    print(f"[train] Surrogate   : state_dim={state_dim}, "
          f"action_mean={action_mean:.1f} W, action_std={action_std:.1f} W")

    # ── create LPBF environment ───────────────────────────────────────────────
    env = LPBFEnv(
        surrogate      = surrogate,
        state_mean     = state_mean,
        state_std      = state_std,
        action_mean    = action_mean,
        action_std     = action_std,
        temp_range     = (args.T_l, args.T_h),
        n_layers       = args.n_layers,
        initial_temp   = args.initial_temp,
        device         = device,
        mesh_path      = args.mesh_path,
        width          = args.width,
        height         = args.height,
        sq_frac_start  = args.sq_frac_start,
        sq_frac_end    = args.sq_frac_end,
    )
    # obs_dim = surrogate state_dim + 1 layer token  (e.g. 1053 + 1 = 1054)
    obs_dim = env.obs_dim
    print(f"[train] Action space : {NUM_ACTIONS} actions  "
          f"({ACTION_LIST[0]:.0f}–{ACTION_LIST[-1]:.0f} W, step 10 W)")
    print(f"[train] Temp window  : [{args.T_l:.0f}, {args.T_h:.0f}] K")
    print(f"[train] Layers/ep    : {args.n_layers}")
    print(f"[train] obs_dim      : {obs_dim}  "
          f"({state_dim} temp nodes + 1 layer token)")

    # ── create agent ──────────────────────────────────────────────────────────
    if args.resume:
        print(f"\n[train] Resuming agent from: {args.resume}")
        agent = DQNAgent.load(args.resume, device=device)
    else:
        agent = DQNAgent(
            state_dim            = obs_dim,   # Q-net input = temp field + layer index
            n_actions            = NUM_ACTIONS,
            hidden               = args.hidden,
            depth                = args.depth,
            dropout              = args.dropout,
            lr                   = args.lr,
            gamma                = args.gamma,
            epsilon_start        = args.epsilon_start,
            epsilon_end          = args.epsilon_end,
            epsilon_decay_steps  = args.epsilon_decay_steps,
            double_dqn           = args.double_dqn,
            device               = device,
        )
    print(f"[train] {agent.q_net}")

    # ── replay buffer ─────────────────────────────────────────────────────────
    buffer = ReplayBuffer(capacity=args.buffer_capacity, state_dim=obs_dim)
    print(f"[train] {buffer}")

    # ── training ──────────────────────────────────────────────────────────────
    all_returns:  list = []
    all_losses:   list = []
    all_epsilons: list = []
    best_return       = -float("inf")

    print(f"\n[train] Starting {args.n_episodes} episodes  "
          f"(warmup={args.warmup_episodes}, "
          f"Double-DQN={args.double_dqn}, "
          f"target_update_freq={args.target_update_freq})")
    print("-" * 65)

    t0 = time.time()

    for episode in range(1, args.n_episodes + 1):

        # ── episode rollout ───────────────────────────────────────────────
        state      = env.reset()
        ep_return  = 0.0
        ep_losses  = []
        in_warmup  = episode <= args.warmup_episodes

        for _step in range(args.n_layers):
            # Always pass explore=True.
            # • During warmup   : ε = 1.0 (no updates → ε never decays)
            #                     → random.random() < 1.0 always → purely random action
            # • After warmup    : ε decays with each update → ε-greedy as intended
            # (Using explore=False during warmup was a bug: it would use the
            # randomly-initialised Q-net greedily, which is not uniformly random
            # and reduces buffer diversity.)
            action = agent.select_action(state, explore=True)
            next_state, reward, done, _info = env.step(action)

            buffer.push(state, action, reward, next_state, done)
            ep_return += reward
            state      = next_state

            # ── Q-network updates ─────────────────────────────────────────
            if not in_warmup:
                for _ in range(args.updates_per_step):
                    loss = agent.update(buffer, args.batch_size)
                    if loss is not None:
                        ep_losses.append(loss)

            if done:
                break

        # ── target network hard update ────────────────────────────────────
        if episode % args.target_update_freq == 0:
            agent.update_target()

        # ── book-keeping ──────────────────────────────────────────────────
        all_returns.append(ep_return)
        all_epsilons.append(agent.epsilon)
        if ep_losses:
            all_losses.extend(ep_losses)

        # ── console logging ───────────────────────────────────────────────
        if episode % args.log_freq == 0 or episode == 1:
            n_recent = min(args.log_freq, len(all_returns))
            avg_ret  = np.mean(all_returns[-n_recent:])
            avg_loss = np.mean(ep_losses) if ep_losses else float("nan")
            elapsed  = time.time() - t0
            tag      = " [warmup]" if in_warmup else ""
            print(
                f"Ep {episode:5d}/{args.n_episodes} | "
                f"ret {ep_return:+.4f} | "
                f"avg({n_recent}) {avg_ret:+.4f} | "
                f"ε {agent.epsilon:.3f} | "
                f"loss {avg_loss:.5f} | "
                f"buf {len(buffer):,} | "
                f"{elapsed:.0f}s{tag}"
            )

        # ── best checkpoint ────────────────────────────────────────────────
        if not in_warmup and ep_return > best_return:
            best_return = ep_return
            agent.save(
                os.path.join(out_dir, "dqn_best.pt"),
                extra={
                    "episode":      episode,
                    "best_return":  best_return,
                    "action_list":  ACTION_LIST,
                    "train_args":   vars(args),
                },
            )

        # ── periodic checkpoint + plots ────────────────────────────────────
        if episode % args.save_freq == 0:
            agent.save(
                os.path.join(out_dir, f"dqn_ep{episode:05d}.pt"),
                extra={
                    "episode":     episode,
                    "all_returns": all_returns,
                    "action_list": ACTION_LIST,
                    "train_args":  vars(args),
                },
            )
            plot_returns(all_returns,
                         os.path.join(out_dir, "returns.png"))
            if all_losses:
                plot_q_loss(all_losses,
                            os.path.join(out_dir, "q_loss.png"))
            plot_epsilon(all_epsilons,
                         os.path.join(out_dir, "epsilon.png"))

    # ── final checkpoint + plots ───────────────────────────────────────────────
    agent.save(
        os.path.join(out_dir, "dqn_final.pt"),
        extra={
            "episode":     args.n_episodes,
            "all_returns": all_returns,
            "action_list": ACTION_LIST,
            "train_args":  vars(args),
        },
    )
    plot_returns(all_returns,  os.path.join(out_dir, "returns.png"))
    if all_losses:
        plot_q_loss(all_losses, os.path.join(out_dir, "q_loss.png"))
    plot_epsilon(all_epsilons,  os.path.join(out_dir, "epsilon.png"))

    total_time = time.time() - t0
    print("\n" + "=" * 65)
    print(f"[train] Training complete in {total_time/60:.1f} min")
    print(f"[train] Best episode return : {best_return:.4f}")
    print(f"[train] Best checkpoint     : {os.path.join(out_dir, 'dqn_best.pt')}")
    print(f"[train] Final checkpoint    : {os.path.join(out_dir, 'dqn_final.pt')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
