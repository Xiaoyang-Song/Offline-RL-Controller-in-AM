"""
online_RL/train.py
------------------
Online Double-DQN training loop for the LPBF laser-power controller,
backed by the latent-space ensemble surrogate.

Two training modes
------------------
  standard  (--uncertainty_mode standard)
      Reward = −meanDeviation   (identical to previous behaviour)
      The ensemble surrogate is used purely as a dynamics model; uncertainty
      is recorded in the log but does not affect learning.

  penalty   (--uncertainty_mode penalty)
      Reward = −meanDeviation − λ · σ_epist
      where σ_epist is the mean ensemble std (latent space) at the current
      (state, action, layer).  The penalty λ is set by
      --uncertainty_penalty_weight (default 0.5).

      Effect: the Q-net learns to avoid (state, action) pairs where the
      ensemble disagrees strongly — i.e. OOD regions where the surrogate's
      predictions are unreliable.  Encourages conservative, in-distribution
      policies during surrogate-based training.

Layer embedding in the Q-network
---------------------------------
  The obs vector is (D+1,): normalised temperature field (D=1053) + float
  layer token in [0,1].  The Q-net internally converts the float token to
  an integer index and looks up a learned layer embedding, giving the
  network the same layer-specific conditioning as the surrogate transition
  MLP.  --layer_embed_dim controls the embedding size (default 8).

Usage
-----
    # Mode 1 — standard (no uncertainty penalty)
    python -m online_RL.train \\
        --surrogate surrogate_model_latent/runs/<ts>/latent_best.pt \\
        --uncertainty_mode standard

    # Mode 2 — uncertainty penalty
    python -m online_RL.train \\
        --surrogate surrogate_model_latent/runs/<ts>/latent_best.pt \\
        --uncertainty_mode penalty \\
        --uncertainty_penalty_weight 0.5
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_latent.train import load_latent_surrogate
from online_RL.env          import LPBFEnv, ACTION_LIST, NUM_ACTIONS
from online_RL.replay_buffer import ReplayBuffer
from online_RL.agent        import DQNAgent


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
        default="surrogate_model_latent/runs/latent_best.pt",
        help="Path to the trained latent ensemble surrogate checkpoint.",
    )

    # ── training mode ─────────────────────────────────────────────────────────
    p.add_argument(
        "--uncertainty_mode", type=str, default="standard",
        choices=["standard", "penalty"],
        help=(
            "standard : reward = −meanDeviation  (no uncertainty term)\n"
            "penalty  : reward = −meanDeviation − λ·σ_epist  "
            "(penalises OOD actions)"
        ),
    )
    p.add_argument(
        "--uncertainty_penalty_weight", type=float, default=0.5,
        help="λ for uncertainty penalty mode (default 0.5). "
             "Ignored when --uncertainty_mode standard.",
    )

    # ── environment ───────────────────────────────────────────────────────────
    p.add_argument("--T_l",          type=float, default=2000.0)
    p.add_argument("--T_h",          type=float, default=2800.0)
    p.add_argument("--n_layers",     type=int,   default=12)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--mesh_path",    type=str,   default="surrogate_model/mesh.mat")
    p.add_argument("--width",        type=float, default=12.0)
    p.add_argument("--height",       type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end",   type=float, default=0.5)

    # ── Q-network architecture ────────────────────────────────────────────────
    p.add_argument("--hidden",          type=int,   default=256)
    p.add_argument("--depth",           type=int,   default=4)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--layer_embed_dim", type=int,   default=8,
                   help="Dimension of the learned layer embedding in the Q-net.")

    # ── RL hyperparameters ────────────────────────────────────────────────────
    p.add_argument("--n_episodes",          type=int,   default=3000)
    p.add_argument("--gamma",               type=float, default=0.99)
    p.add_argument("--lr",                  type=float, default=3e-4)
    p.add_argument("--batch_size",          type=int,   default=256)
    p.add_argument("--buffer_capacity",     type=int,   default=100_000)
    p.add_argument("--warmup_episodes",     type=int,   default=50)
    p.add_argument("--target_update_freq",  type=int,   default=10)
    p.add_argument("--updates_per_step",    type=int,   default=1)
    p.add_argument("--epsilon_start",       type=float, default=1.0)
    p.add_argument("--epsilon_end",         type=float, default=0.05)
    p.add_argument("--epsilon_decay_steps", type=int,   default=50_000)
    p.add_argument("--double_dqn",    action="store_true",  default=True)
    p.add_argument("--no_double_dqn", dest="double_dqn", action="store_false")

    # ── logging / checkpointing ────────────────────────────────────────────────
    p.add_argument("--log_freq",  type=int, default=50)
    p.add_argument("--save_freq", type=int, default=500)
    p.add_argument("--out_dir",   type=str, default="")
    p.add_argument("--device",    type=str, default="")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--resume",    type=str, default="")

    return p.parse_args()


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_returns(returns: list, out_path: str, window: int = 50) -> None:
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
    ax.set_xlabel("Episode"); ax.set_ylabel("Return")
    ax.set_title("Online DQN — Episode Returns")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[train] Returns plot → {out_path}")


def plot_q_loss(losses: list, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, linewidth=0.4, alpha=0.6, color="darkorange")
    ax.set_xlabel("Q-network update step"); ax.set_ylabel("Huber loss")
    ax.set_title("Online DQN — Q-loss Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[train] Q-loss plot  → {out_path}")


def plot_epsilon(epsilons: list, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(1, len(epsilons) + 1), epsilons,
            color="seagreen", linewidth=1.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
    ax.set_title("ε-greedy Decay Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[train] Epsilon plot → {out_path}")


def plot_epistemic_std(ep_stds: list, out_path: str, window: int = 50) -> None:
    """Mean per-episode ensemble uncertainty — should decrease as the policy
    learns to stay in well-covered (low-uncertainty) regions of state space."""
    fig, ax = plt.subplots(figsize=(10, 4))
    eps = np.arange(1, len(ep_stds) + 1)
    ax.plot(eps, ep_stds, alpha=0.35, color="tab:purple",
            linewidth=0.7, label="Mean epistemic std per episode")
    if len(ep_stds) >= window:
        smooth = np.convolve(ep_stds, np.ones(window) / window, mode="valid")
        ax.plot(eps[window - 1:], smooth, color="tab:purple",
                linewidth=2.0, label=f"Rolling avg ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean ensemble std (latent space)")
    ax.set_title("Epistemic Uncertainty per Episode\n"
                 "(penalty mode: decreasing = policy avoids OOD regions)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[train] Epistemic-std plot → {out_path}")


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

    penalty_w = args.uncertainty_penalty_weight if args.uncertainty_mode == "penalty" else 0.0

    print("=" * 65)
    print(f"[train] Online DQN — LPBF Laser-Power Controller")
    print(f"[train] Output dir  : {out_dir}")
    print(f"[train] Device      : {device}")
    print(f"[train] Mode        : {args.uncertainty_mode}"
          + (f"  (λ={penalty_w})" if args.uncertainty_mode == "penalty" else ""))
    print("=" * 65)

    # ── load latent ensemble surrogate ────────────────────────────────────────
    print(f"\n[train] Loading surrogate: {args.surrogate}")
    surrogate, state_mean, state_std, action_mean, action_std, _ = \
        load_latent_surrogate(args.surrogate, device=device)
    surrogate.eval()
    state_dim = int(state_mean.shape[0])
    print(f"[train] Surrogate   : {surrogate}")

    # ── create LPBF environment ───────────────────────────────────────────────
    env = LPBFEnv(
        surrogate                   = surrogate,
        state_mean                  = state_mean,
        state_std                   = state_std,
        action_mean                 = action_mean,
        action_std                  = action_std,
        temp_range                  = (args.T_l, args.T_h),
        n_layers                    = args.n_layers,
        initial_temp                = args.initial_temp,
        device                      = device,
        mesh_path                   = args.mesh_path,
        width                       = args.width,
        height                      = args.height,
        sq_frac_start               = args.sq_frac_start,
        sq_frac_end                 = args.sq_frac_end,
        uncertainty_penalty_weight  = penalty_w,
    )
    obs_dim = env.obs_dim   # D + 1

    print(f"[train] Action space : {NUM_ACTIONS} actions  "
          f"({ACTION_LIST[0]:.0f}–{ACTION_LIST[-1]:.0f} W)")
    print(f"[train] Temp window  : [{args.T_l:.0f}, {args.T_h:.0f}] K")
    print(f"[train] obs_dim      : {obs_dim}  ({state_dim} nodes + 1 layer token)")

    # ── create agent ──────────────────────────────────────────────────────────
    if args.resume:
        print(f"\n[train] Resuming agent from: {args.resume}")
        agent = DQNAgent.load(args.resume, device=device)
    else:
        agent = DQNAgent(
            state_dim            = obs_dim,
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
            n_layers             = args.n_layers,
            layer_embed_dim      = args.layer_embed_dim,
        )
    print(f"[train] {agent.q_net}")

    # ── replay buffer ─────────────────────────────────────────────────────────
    buffer = ReplayBuffer(capacity=args.buffer_capacity, state_dim=obs_dim)
    print(f"[train] {buffer}")

    # ── training ──────────────────────────────────────────────────────────────
    all_returns:   list = []
    all_losses:    list = []
    all_epsilons:  list = []
    all_ep_stds:   list = []   # mean epistemic std per episode
    best_return        = -float("inf")

    print(f"\n[train] Starting {args.n_episodes} episodes  "
          f"(warmup={args.warmup_episodes}, "
          f"Double-DQN={args.double_dqn}, "
          f"target_update_freq={args.target_update_freq})")
    print("-" * 65)

    t0 = time.time()

    for episode in range(1, args.n_episodes + 1):

        state     = env.reset()
        ep_return = 0.0
        ep_losses = []
        ep_stds   = []
        in_warmup = episode <= args.warmup_episodes

        for _step in range(args.n_layers):
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)

            ep_stds.append(info["epistemic_std"])
            buffer.push(state, action, reward, next_state, done)
            ep_return += reward
            state      = next_state

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
        all_ep_stds.append(float(np.mean(ep_stds)) if ep_stds else 0.0)
        if ep_losses:
            all_losses.extend(ep_losses)

        # ── console logging ───────────────────────────────────────────────
        if episode % args.log_freq == 0 or episode == 1:
            n_recent  = min(args.log_freq, len(all_returns))
            avg_ret   = np.mean(all_returns[-n_recent:])
            avg_loss  = np.mean(ep_losses) if ep_losses else float("nan")
            avg_std   = np.mean(all_ep_stds[-n_recent:])
            elapsed   = time.time() - t0
            tag       = " [warmup]" if in_warmup else ""
            print(
                f"Ep {episode:5d}/{args.n_episodes} | "
                f"ret {ep_return:+.4f} | "
                f"avg({n_recent}) {avg_ret:+.4f} | "
                f"ε {agent.epsilon:.3f} | "
                f"loss {avg_loss:.5f} | "
                f"σ_epist {avg_std:.4f} | "
                f"buf {len(buffer):,} | "
                f"{elapsed:.0f}s{tag}"
            )

        # ── best checkpoint ────────────────────────────────────────────────
        if not in_warmup and ep_return > best_return:
            best_return = ep_return
            agent.save(
                os.path.join(out_dir, "dqn_best.pt"),
                extra={
                    "episode":          episode,
                    "best_return":      best_return,
                    "uncertainty_mode": args.uncertainty_mode,
                    "penalty_weight":   penalty_w,
                    "action_list":      ACTION_LIST,
                    "train_args":       vars(args),
                },
            )

        # ── periodic checkpoint + plots ────────────────────────────────────
        if episode % args.save_freq == 0:
            agent.save(
                os.path.join(out_dir, f"dqn_ep{episode:05d}.pt"),
                extra={
                    "episode":          episode,
                    "all_returns":      all_returns,
                    "uncertainty_mode": args.uncertainty_mode,
                    "action_list":      ACTION_LIST,
                    "train_args":       vars(args),
                },
            )
            plot_returns(all_returns,
                         os.path.join(out_dir, "returns.png"))
            if all_losses:
                plot_q_loss(all_losses,
                            os.path.join(out_dir, "q_loss.png"))
            plot_epsilon(all_epsilons,
                         os.path.join(out_dir, "epsilon.png"))
            plot_epistemic_std(all_ep_stds,
                               os.path.join(out_dir, "epistemic_std.png"))

    # ── final checkpoint + plots ───────────────────────────────────────────────
    agent.save(
        os.path.join(out_dir, "dqn_final.pt"),
        extra={
            "episode":          args.n_episodes,
            "all_returns":      all_returns,
            "uncertainty_mode": args.uncertainty_mode,
            "action_list":      ACTION_LIST,
            "train_args":       vars(args),
        },
    )
    plot_returns(all_returns,    os.path.join(out_dir, "returns.png"))
    if all_losses:
        plot_q_loss(all_losses,  os.path.join(out_dir, "q_loss.png"))
    plot_epsilon(all_epsilons,   os.path.join(out_dir, "epsilon.png"))
    plot_epistemic_std(all_ep_stds, os.path.join(out_dir, "epistemic_std.png"))

    total_time = time.time() - t0
    print("\n" + "=" * 65)
    print(f"[train] Training complete in {total_time/60:.1f} min")
    print(f"[train] Mode               : {args.uncertainty_mode}")
    print(f"[train] Best episode return: {best_return:.4f}")
    print(f"[train] Best checkpoint    : {os.path.join(out_dir, 'dqn_best.pt')}")
    print(f"[train] Final checkpoint   : {os.path.join(out_dir, 'dqn_final.pt')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
