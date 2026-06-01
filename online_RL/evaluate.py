"""
online_RL/evaluate.py
---------------------
Surrogate-based evaluation of a trained online DQN policy.

Mirrors the per-layer output of test.py but replaces the MATLAB external call
with a forward pass through the trained surrogate model.  This lets you:

  • Quickly assess per-layer reward without waiting for MATLAB
  • Isolate the surrogate's sim-to-real gap  (compare this vs test.py rewards)
  • Run many evaluation episodes for statistics

The greedy policy (ε = 0) is used throughout — no exploration.

Usage
-----
    python -m online_RL.evaluate \\
        --checkpoint online_RL/runs/<timestamp>/dqn_best.pt \\
        --surrogate  surrogate_model_latent/runs/<timestamp>/latent_best.pt

    # Run 10 episodes and save a plot
    python -m online_RL.evaluate \\
        --checkpoint online_RL/runs/<ts>/dqn_best.pt \\
        --n_episodes 10 --plot

Output
------
  Per-layer table (action chosen, surrogate reward, mean/max temperature)
  Summary statistics across all episodes
  Optional plot: per-layer reward heatmap + mean curve  (--plot)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_latent.train import load_latent_surrogate
from online_RL.env                import LPBFEnv, ACTION_LIST, NUM_ACTIONS
from online_RL.agent       import DQNAgent


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained online DQN policy using the surrogate model."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a trained DQN checkpoint (dqn_best.pt).")
    p.add_argument("--surrogate",  type=str,
                   default="surrogate_model_latent/runs/20260601_161335/latent_best.pt",
                   help="Path to the latent surrogate checkpoint.")

    # ── environment (must match training) ────────────────────────────────────
    p.add_argument("--T_l",          type=float, default=2000.0)
    p.add_argument("--T_h",          type=float, default=2800.0)
    p.add_argument("--n_layers",     type=int,   default=12)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--mesh_path",    type=str,
                   default="surrogate_model/mesh.mat")
    p.add_argument("--width",         type=float, default=12.0)
    p.add_argument("--height",        type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end",   type=float, default=0.5)

    # ── evaluation settings ───────────────────────────────────────────────────
    p.add_argument("--n_episodes", type=int, default=1,
                   help="Number of evaluation episodes to run.")
    p.add_argument("--plot",  action="store_true",
                   help="Save a per-layer reward plot alongside the checkpoint.")
    p.add_argument("--device", type=str, default="",
                   help="'cuda' or 'cpu'. Auto-detected if empty.")
    return p.parse_args()


# =============================================================================
# Single episode rollout (greedy)
# =============================================================================

def run_episode(env: LPBFEnv, agent: DQNAgent):
    """
    Run one greedy episode.

    Returns
    -------
    actions  : (n_layers,)  laser powers [W]
    rewards  : (n_layers,)  per-layer reward
    mean_Ts  : (n_layers,)  mean temperature [K] after each layer
    max_Ts   : (n_layers,)  max  temperature [K] after each layer
    """
    state = env.reset()
    actions, rewards, mean_Ts, max_Ts = [], [], [], []

    for _ in range(env.n_layers):
        action_idx = agent.select_action_greedy(state)
        state, reward, done, info = env.step(action_idx)

        actions.append(info["action_W"])
        rewards.append(reward)
        mean_Ts.append(info["mean_temp_K"])
        max_Ts.append(info["max_temp_K"])

        if done:
            break

    return (
        np.array(actions),
        np.array(rewards),
        np.array(mean_Ts),
        np.array(max_Ts),
    )


# =============================================================================
# Printing helpers
# =============================================================================

def print_episode(
    ep_idx:  int,
    actions: np.ndarray,
    rewards: np.ndarray,
    mean_Ts: np.ndarray,
    max_Ts:  np.ndarray,
    T_l:     float,
    T_h:     float,
) -> None:
    """Print a per-layer table identical in style to test.py output."""
    total = rewards.sum()
    print(f"\n{'='*62}")
    print(f"  Episode {ep_idx+1}   |   Total return: {total:+.4f}")
    print(f"{'='*62}")
    print(f"{'Layer':>6}  {'LP [W]':>8}  {'Reward':>10}  "
          f"{'Mean T [K]':>12}  {'Max T [K]':>11}  {'Status':>8}")
    print(f"{'-'*62}")
    for i, (a, r, mt, xt) in enumerate(zip(actions, rewards, mean_Ts, max_Ts)):
        # rough flag: check if mean is in range (approximate, uses all nodes)
        in_range = T_l <= mt <= T_h
        status   = "in range" if in_range else "OUT"
        print(f"{i+1:>6}  {a:>8.0f}  {r:>+10.5f}  "
              f"{mt:>12.1f}  {xt:>11.1f}  {status:>8}")
    print(f"{'-'*62}")
    print(f"{'Total':>6}  {'':>8}  {total:>+10.5f}")


# =============================================================================
# Plotting
# =============================================================================

def plot_results(
    all_rewards: list,    # list of (n_layers,) arrays
    all_actions: list,    # list of (n_layers,) arrays
    out_path:    str,
    T_l: float,
    T_h: float,
) -> None:
    """
    Two-panel figure:
      Top   : per-layer reward for every episode (thin lines) + mean (thick)
      Bottom: per-layer laser power (mean ± std across episodes)
    """
    n_ep    = len(all_rewards)
    n_lay   = len(all_rewards[0])
    layers  = np.arange(1, n_lay + 1)

    rewards_mat = np.stack(all_rewards)   # (n_ep, n_lay)
    actions_mat = np.stack(all_actions)   # (n_ep, n_lay)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ── top: rewards ──────────────────────────────────────────────────────────
    for i in range(n_ep):
        ax1.plot(layers, rewards_mat[i], alpha=0.3, linewidth=0.8,
                 color="steelblue")
    mean_r = rewards_mat.mean(axis=0)
    ax1.plot(layers, mean_r, color="steelblue", linewidth=2.5,
             label=f"Mean reward (n={n_ep})")
    ax1.axhline(0, color="green", linestyle="--", linewidth=1,
                label="Perfect (0 deviation)")
    ax1.set_ylabel("Reward (−meanDeviation)")
    ax1.set_title("Surrogate-based Policy Evaluation — Per-Layer Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── bottom: actions ───────────────────────────────────────────────────────
    mean_a = actions_mat.mean(axis=0)
    std_a  = actions_mat.std(axis=0)
    ax2.bar(layers, mean_a, alpha=0.7, color="darkorange", label="Mean LP [W]")
    ax2.errorbar(layers, mean_a, yerr=std_a, fmt="none",
                 color="black", capsize=3, linewidth=1)
    ax2.set_ylabel("Laser Power [W]")
    ax2.set_xlabel("Layer")
    ax2.set_title("Policy Action Sequence (mean ± std)")
    ax2.set_xticks(layers)
    ax2.set_ylim(
        max(0, mean_a.min() - 3 * std_a.max() - 30),
        min(400, mean_a.max() + 3 * std_a.max() + 30),
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[evaluate] Plot saved → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 62)
    print("  Online DQN — Surrogate-based Evaluation")
    print("=" * 62)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Surrogate  : {args.surrogate}")
    print(f"  Device     : {device}")
    print(f"  Episodes   : {args.n_episodes}")
    print(f"  Temp range : [{args.T_l:.0f}, {args.T_h:.0f}] K")
    print("=" * 62)

    # ── load latent surrogate ─────────────────────────────────────────────────
    surrogate, state_mean, state_std, action_mean, action_std, _ = load_latent_surrogate(
        args.surrogate, device=device
    )
    surrogate.eval()

    # ── build environment ─────────────────────────────────────────────────────
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

    # ── load agent ────────────────────────────────────────────────────────────
    agent = DQNAgent.load(args.checkpoint, device=device)
    agent.q_net.eval()
    print(f"\n  {agent.q_net}")

    # ── run evaluation episodes ───────────────────────────────────────────────
    all_rewards, all_actions = [], []

    for ep in range(args.n_episodes):
        actions, rewards, mean_Ts, max_Ts = run_episode(env, agent)
        all_rewards.append(rewards)
        all_actions.append(actions)
        print_episode(ep, actions, rewards, mean_Ts, max_Ts, args.T_l, args.T_h)

    # ── summary across episodes ───────────────────────────────────────────────
    rewards_mat = np.stack(all_rewards)   # (n_ep, n_layers)

    print(f"\n{'='*62}")
    print(f"  Summary over {args.n_episodes} episode(s)")
    print(f"{'='*62}")
    print(f"  {'Layer':>6}  {'Mean reward':>13}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*54}")
    for i in range(args.n_layers):
        col = rewards_mat[:, i]
        print(f"  {i+1:>6}  {col.mean():>+13.5f}  "
              f"{col.std():>8.5f}  {col.min():>+8.5f}  {col.max():>+8.5f}")
    print(f"  {'-'*54}")
    ep_returns = rewards_mat.sum(axis=1)
    print(f"  {'Return':>6}  {ep_returns.mean():>+13.4f}  "
          f"{ep_returns.std():>8.4f}  {ep_returns.min():>+8.4f}  "
          f"{ep_returns.max():>+8.4f}")
    print(f"{'='*62}")

    # ── optional plot ─────────────────────────────────────────────────────────
    if args.plot:
        out_dir  = os.path.dirname(os.path.abspath(args.checkpoint))
        out_path = os.path.join(out_dir, "surrogate_eval.png")
        plot_results(all_rewards, all_actions, out_path, args.T_l, args.T_h)


if __name__ == "__main__":
    main()
