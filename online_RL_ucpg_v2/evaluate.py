"""
online_RL_ucpg_v2/evaluate.py
---------------------------------
Surrogate-based evaluation of a trained UCPG v2 (continuous-action) policy.

Reports, for both the greedy (deterministic mean) and stochastic (sampled)
policy: mean return, mean uncertainty, and the chosen-action distribution
(mean/std/min/max, plus an optional OOD fraction if --ood_threshold is
given). With --plot, also saves:
  eval_action_histogram.png     — continuous action density (greedy vs. stochastic)
  eval_reward_and_action.png    — per-layer reward/action trace (mean ± std)
  eval_policy_mean_std.png      — the POLICY's own mu_theta(z_t) ± sigma_theta
                                   per layer along one fresh rollout — something
                                   a categorical policy (online_RL_ucpg) can't
                                   show directly, since there's no single
                                   scalar "mean ± std" for a categorical
                                   distribution.

Usage
-----
    python -m online_RL_ucpg_v2.evaluate \\
        --checkpoint online_RL_ucpg_v2/runs/<ts>/ucpg_best.pt \\
        --surrogate  surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt \\
        --n_episodes 50 --plot
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

from surrogate_model_latent_uncertainty_v2.train import load_two_stage_surrogate
from online_RL_ucpg_v2.env   import TwoStageLatentLPBFEnv
from online_RL_ucpg_v2.agent import UCPGAgentV2


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained UCPG v2 (continuous-action) policy against "
                    "the two-stage surrogate model."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a trained UCPG v2 checkpoint (ucpg_best.pt or ucpg_final.pt).")
    p.add_argument("--surrogate",  type=str, required=True,
                   help="Path to the surrogate_model_latent_uncertainty_v2 checkpoint "
                        "used to back the environment (should match training).")

    # ── environment (should match training) ───────────────────────────────────
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

    # ── evaluation settings ───────────────────────────────────────────────────
    p.add_argument("--ood_threshold", type=float, default=None,
                   help="Optional: actions above this [W] are counted as OOD in the summary.")
    p.add_argument("--n_episodes", type=int, default=50,
                   help="Number of evaluation episodes for each of (greedy, stochastic).")
    p.add_argument("--plot",  action="store_true",
                   help="Save action-histogram, per-layer reward, and policy mean±std plots.")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed",   type=int, default=123)
    return p.parse_args()


# =============================================================================
# Rollout
# =============================================================================

def run_episode(env: TwoStageLatentLPBFEnv, agent: UCPGAgentV2, greedy: bool):
    """
    Returns
    -------
    actions   : (T,) laser powers [W]
    rewards   : (T,)
    u         : (T,) combined (heating+cooling) epistemic_std + aleatoric_std
    mean_Ts   : (T,) mean end-of-heating temperature [K]
    """
    state = env.reset()
    actions, rewards, u, mean_Ts = [], [], [], []

    for _ in range(env.n_layers):
        a = agent.select_action(state, explore=not greedy)
        state, reward, done, info = env.step(a)

        actions.append(info["action_W"])
        rewards.append(reward)
        u.append(info["uncertainty"])
        mean_Ts.append(info["mean_heat_temp_K"])

        if done:
            break

    return np.array(actions), np.array(rewards), np.array(u), np.array(mean_Ts)


def run_many(env, agent, n_episodes: int, greedy: bool):
    all_actions, all_rewards, all_u = [], [], []
    for _ in range(n_episodes):
        actions, rewards, u, _mean_Ts = run_episode(env, agent, greedy=greedy)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_u.append(u)
    return np.stack(all_actions), np.stack(all_rewards), np.stack(all_u)


@torch.no_grad()
def policy_mean_std_per_layer(env: TwoStageLatentLPBFEnv, agent: UCPGAgentV2):
    """
    Roll out ONE greedy episode while recording the policy's own
    (mu_theta(z_t), sigma_theta) at each layer — the policy's internal
    belief, not empirical action statistics from sampling.
    """
    state = env.reset()
    mus, sigmas = [], []
    for _ in range(env.n_layers):
        obs_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        mu, sigma = agent.policy(obs_t)
        mus.append(float(mu.item())); sigmas.append(float(sigma.item()))
        a = float(mu.item())   # step with the greedy (mean) action
        state, _reward, _done, _info = env.step(a)
    return np.array(mus), np.array(sigmas)


# =============================================================================
# Reporting
# =============================================================================

def summarize(label: str, actions: np.ndarray, rewards: np.ndarray, u: np.ndarray,
             ood_threshold) -> None:
    n_total = actions.size
    ep_return = rewards.sum(axis=1)
    print(f"\n  [{label}]  {actions.shape[0]} episodes × {actions.shape[1]} layers "
          f"= {n_total} actions")
    print(f"    Return           : mean={ep_return.mean():+.4f}  std={ep_return.std():.4f}  "
          f"min={ep_return.min():+.4f}  max={ep_return.max():+.4f}")
    print(f"    Uncertainty u_t  : mean={u.mean():.5f}  max={u.max():.5f}")
    print(f"    Action power     : mean={actions.mean():.1f}W  std={actions.std():.1f}W  "
          f"median={np.median(actions):.1f}W  min={actions.min():.1f}W  max={actions.max():.1f}W")
    if ood_threshold is not None:
        n_ood = int((actions > ood_threshold).sum())
        print(f"    Actions > {ood_threshold:.0f}W : {n_ood}/{n_total} "
              f"({n_ood/n_total*100:.1f}%)")


def plot_action_histogram(
    actions_greedy: np.ndarray,
    actions_stoch:  np.ndarray,
    action_min:     float,
    action_max:     float,
    ood_threshold,
    out_path:       str,
) -> None:
    lo = min(actions_greedy.min(), actions_stoch.min(), action_min)
    hi = max(actions_greedy.max(), actions_stoch.max(), action_max)
    bins = np.linspace(lo, hi, 41)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(actions_greedy.flatten(), bins=bins, alpha=0.6, color="steelblue",
            density=True, label="Greedy policy (mean action)")
    ax.hist(actions_stoch.flatten(), bins=bins, alpha=0.5, color="darkorange",
            density=True, label="Stochastic policy (sampled)")
    ax.axvspan(action_min, action_max, color="green", alpha=0.05,
               label=f"Policy aim range [{action_min:.0f}, {action_max:.0f}]W")
    if ood_threshold is not None:
        ax.axvline(ood_threshold, color="red", linestyle="--", linewidth=1.5,
                   label=f"OOD threshold ({ood_threshold:.0f} W)")
    ax.set_xlabel("Laser power [W]")
    ax.set_ylabel("Density")
    ax.set_title("UCPG v2 — Chosen Action Distribution (continuous)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


def plot_reward_and_action_per_layer(
    rewards: np.ndarray,   # (n_ep, T)
    actions: np.ndarray,   # (n_ep, T)
    out_path: str,
) -> None:
    n_ep, T = rewards.shape
    layers  = np.arange(1, T + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for i in range(n_ep):
        ax1.plot(layers, rewards[i], alpha=0.25, linewidth=0.7, color="steelblue")
    ax1.plot(layers, rewards.mean(axis=0), color="steelblue", linewidth=2.5,
             label=f"Mean reward (n={n_ep})")
    ax1.axhline(0, color="green", linestyle="--", linewidth=1, label="Perfect (0 deviation)")
    ax1.set_ylabel("Reward (−meanDeviation, end-of-heating)")
    ax1.set_title("UCPG v2 Policy Evaluation (greedy) — Per-Layer Reward")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    mean_a = actions.mean(axis=0)
    std_a  = actions.std(axis=0)
    ax2.bar(layers, mean_a, alpha=0.7, color="darkorange", label="Mean LP [W]")
    ax2.errorbar(layers, mean_a, yerr=std_a, fmt="none", color="black",
                capsize=3, linewidth=1)
    ax2.set_ylabel("Laser Power [W]")
    ax2.set_xlabel("Layer")
    ax2.set_title("Policy Action Sequence (mean ± std)")
    ax2.set_xticks(layers)
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


def plot_policy_mean_std(mus: np.ndarray, sigmas: np.ndarray, out_path: str) -> None:
    T = len(mus)
    layers = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(layers, mus, color="tab:blue", linewidth=2.0, marker="o", label="mu_theta(z_t)")
    ax.fill_between(layers, mus - sigmas, mus + sigmas, color="tab:blue", alpha=0.2,
                    label="± sigma_theta")
    ax.set_xlabel("Layer"); ax.set_ylabel("Laser power [W]")
    ax.set_xticks(layers)
    ax.set_title("UCPG v2 — Policy's Own mu ± sigma per Layer (one greedy rollout)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 65)
    print("  UCPG v2 — Surrogate-based Policy Evaluation (continuous action)")
    print("=" * 65)
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Surrogate      : {args.surrogate}")
    print(f"  Device         : {device}")
    print(f"  Episodes       : {args.n_episodes} (× greedy, × stochastic)")
    print("=" * 65)

    (surrogate, state_mean, state_std, lp_mean, lp_std,
     cool_mean, cool_std, _roi_table) = load_two_stage_surrogate(args.surrogate, device=device)
    surrogate.eval()

    env = TwoStageLatentLPBFEnv(
        surrogate     = surrogate,
        state_mean    = state_mean,
        state_std     = state_std,
        lp_mean       = lp_mean,
        lp_std        = lp_std,
        cool_mean     = cool_mean,
        cool_std      = cool_std,
        temp_range    = (args.T_l, args.T_h),
        n_layers      = args.n_layers,
        initial_temp  = args.initial_temp,
        device        = device,
        mesh_path     = args.mesh_path,
        width         = args.width,
        height        = args.height,
        sq_frac_start = args.sq_frac_start,
        sq_frac_end   = args.sq_frac_end,
        action_min    = args.action_min,
        action_max    = args.action_max,
        cool_time_min = args.cool_time_min,
        cool_time_max = args.cool_time_max,
    )

    agent = UCPGAgentV2.load(args.checkpoint, device=device)
    print(f"\n  {agent}")

    # ── greedy evaluation ─────────────────────────────────────────────────────
    actions_g, rewards_g, u_g = run_many(env, agent, args.n_episodes, greedy=True)
    summarize("GREEDY", actions_g, rewards_g, u_g, args.ood_threshold)

    # ── stochastic evaluation (matches training-time behaviour) ──────────────
    actions_s, rewards_s, u_s = run_many(env, agent, args.n_episodes, greedy=False)
    summarize("STOCHASTIC", actions_s, rewards_s, u_s, args.ood_threshold)

    print(f"\n{'='*65}")
    if args.ood_threshold is not None:
        n_ood_g = int((actions_g > args.ood_threshold).sum())
        n_ood_s = int((actions_s > args.ood_threshold).sum())
        print(f"  Greedy policy chose an action above {args.ood_threshold:.0f} W in "
              f"{n_ood_g}/{actions_g.size} steps ({n_ood_g/actions_g.size*100:.1f}%)")
        print(f"  Stochastic policy: {n_ood_s}/{actions_s.size} steps "
              f"({n_ood_s/actions_s.size*100:.1f}%)")
    print(f"{'='*65}")

    if args.plot:
        out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        plot_action_histogram(
            actions_g, actions_s, args.action_min, args.action_max, args.ood_threshold,
            os.path.join(out_dir, "eval_action_histogram.png"),
        )
        plot_reward_and_action_per_layer(
            rewards_g, actions_g,
            os.path.join(out_dir, "eval_reward_and_action.png"),
        )
        mus, sigmas = policy_mean_std_per_layer(env, agent)
        plot_policy_mean_std(mus, sigmas, os.path.join(out_dir, "eval_policy_mean_std.png"))


if __name__ == "__main__":
    main()
