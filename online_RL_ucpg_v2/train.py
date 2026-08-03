"""
online_RL_ucpg_v2/train.py
------------------------------
Monte Carlo Uncertainty-Constrained Policy Gradient (UCPG) training loop —
continuous laser-power action, two-stage (heating/cooling) surrogate.

Implements Algorithm 1 exactly, unchanged from online_RL_ucpg/train.py:

  for k = 1 .. K:
    collect N trajectories using a_t ~ pi_theta(.|s_t)   (s_t = latent obs)
    for each transition: u_t = sigma_epis,t + sigma_alea,t   (combined heat+cool, from the surrogate env)
    G_r,t = sum_{k=t}^{T-1} gamma_r^{k-t} r_k
    G_u,t = sum_{k=t}^{T-1} gamma_u^{k-t} u_k
    G_lambda,t = G_r,t - lambda * G_u,t
    theta <- theta + alpha_theta * (1/N) sum_i sum_t nabla_theta log pi_theta(a_t|s_t) * G_lambda,t
    J_u_hat = (1/N) sum_i sum_t gamma_u^t u_t^(i)          [ = (1/N) sum_i G_u,0^(i) ]
    lambda <- [lambda + alpha_lambda * (J_u_hat - delta)]_+

No replay buffer, no entropy bonus. Each batch of N trajectories is generated
fresh by the CURRENT policy and used for exactly one gradient step (on-policy
Monte Carlo), matching the algorithm box.

What changed vs. online_RL_ucpg/train.py
--------------------------------------------
  - Action is continuous (a_t ~ Normal(mu_theta(s_t), sigma_theta), see
    model.ContinuousLatentPolicyNet) instead of categorical over a discrete
    grid — actions/actions_flat are float32 laser powers [W], not int64 grid
    indices. No --action_step.
  - The environment (env.TwoStageLatentLPBFEnv) chains a two-stage
    heating->cooling transition per layer and computes reward from the
    END-OF-HEATING field (see env.py's docstring); it also draws one
    cool_time per EPISODE (not per layer, not an action) — see
    --cool_time_min/--cool_time_max.
  - --ood_threshold is now optional (default: none logged) since this
    package isn't tied to the narrow-vs-wide surrogate ablation online_RL_ucpg
    was built for; pass it if you want that diagnostic.
  - New per-iteration diagnostic: mean/std of the policy's chosen action
    (action_stats.png) — the continuous analogue of watching a categorical
    distribution sharpen, directly visualising exploration collapsing (or
    not) over training.

Optional per-timestep baseline (--use_baseline)
--------------------------------------------------
Identical mechanics to online_RL_ucpg/train.py — see that file's docstring.

Usage
-----
    python -m online_RL_ucpg_v2.train \\
        --surrogate surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt \\
        --action_min 100 --action_max 400 \\
        --delta 0.05
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

from surrogate_model_latent_uncertainty_v2.train import load_two_stage_surrogate
from online_RL_ucpg_v2.env   import TwoStageLatentLPBFEnv
from online_RL_ucpg_v2.agent import UCPGAgentV2


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo Uncertainty-Constrained Policy Gradient (UCPG) "
                    "training for the LPBF laser-power controller (continuous action, "
                    "two-stage surrogate)."
    )

    # ── surrogate model ───────────────────────────────────────────────────────
    p.add_argument("--surrogate", type=str, required=True,
                   help="Path to a trained surrogate_model_latent_uncertainty_v2 checkpoint "
                        "(TwoStageEnsembleGaussianLatentDynamicsModel).")

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
    p.add_argument("--cool_time_min", type=float, default=0.05,
                   help="cool_time is drawn Uniform(cool_time_min, cool_time_max) once "
                        "per EPISODE (not an action) — matches multilayer_random_v2.m.")
    p.add_argument("--cool_time_max", type=float, default=0.15)

    # ── policy action range (soft "aim range" for the Gaussian mean, NOT a hard bound) ──
    p.add_argument("--action_min",  type=float, default=100.0)
    p.add_argument("--action_max",  type=float, default=400.0)
    p.add_argument("--ood_threshold", type=float, default=None,
                   help="Optional diagnostic: actions above this [W] are logged as "
                        "'OOD fraction' every iteration (does not affect training). "
                        "Omit to skip this diagnostic.")

    # ── policy network architecture ───────────────────────────────────────────
    p.add_argument("--hidden",          type=int,   default=128)
    p.add_argument("--depth",           type=int,   default=3)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--layer_embed_dim", type=int,   default=8)
    p.add_argument("--log_sigma_init",  type=float, default=-1.0,
                   help="Initial Gaussian std, as log(sigma / action_half_range).")
    p.add_argument("--min_log_sigma",   type=float, default=-3.0)
    p.add_argument("--max_log_sigma",   type=float, default=0.0)

    # ── UCPG hyperparameters ──────────────────────────────────────────────────
    p.add_argument("--n_traj",       type=int,   default=32,
                   help="N — number of Monte Carlo trajectories collected per iteration.")
    p.add_argument("--n_iterations", type=int,   default=2000,
                   help="K — number of outer (collect, update theta, update lambda) iterations.")
    p.add_argument("--gamma_r",      type=float, default=0.99, help="Reward discount factor.")
    p.add_argument("--gamma_u",      type=float, default=0.99, help="Uncertainty discount factor.")
    p.add_argument("--delta",        type=float, required=True,
                   help="Uncertainty budget: constrains E[sum_t gamma_u^t u_t] <= delta. "
                        "Problem-specific — inspect a few --n_traj rollouts' J_u_hat "
                        "under lambda=0 first (e.g. via a short warm-start run) to pick "
                        "a sensible value for your surrogate's uncertainty scale.")
    p.add_argument("--lambda_init",  type=float, default=0.0)
    p.add_argument("--lr_theta",     type=float, default=3e-4, help="alpha_theta.")
    p.add_argument("--lr_lambda",    type=float, default=1e-2, help="alpha_lambda.")
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--use_baseline", action="store_true",
                   help="Subtract the batch's per-timestep empirical mean return "
                        "from G_r,t and G_u,t before forming the advantage (variance "
                        "reduction; fixes early-layer credit assignment). Off by "
                        "default to match the algorithm literally. Never affects "
                        "J_u_hat / the lambda update, which always use raw G_u,0.")

    # ── logging / checkpointing ────────────────────────────────────────────────
    p.add_argument("--log_freq",  type=int, default=20)
    p.add_argument("--save_freq", type=int, default=200)
    p.add_argument("--out_dir",   type=str, default="")
    p.add_argument("--device",    type=str, default="")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--resume",    type=str, default="")

    return p.parse_args()


# =============================================================================
# Return computation
# =============================================================================

def discounted_returns(values: np.ndarray, gamma: float) -> np.ndarray:
    """
    values : (N, T) — per-trajectory, per-step scalars (reward or uncertainty)
    returns: (N, T) float32 — G_t = sum_{k=t}^{T-1} gamma^{k-t} values_k,
             computed per-trajectory via the standard reverse recursion
             G_t = values_t + gamma * G_{t+1}.
    """
    N, T = values.shape
    returns = np.zeros_like(values, dtype=np.float64)
    running = np.zeros(N, dtype=np.float64)
    for t in reversed(range(T)):
        running = values[:, t] + gamma * running
        returns[:, t] = running
    return returns.astype(np.float32)


# =============================================================================
# Trajectory collection
# =============================================================================

def collect_batch(env: TwoStageLatentLPBFEnv, agent: UCPGAgentV2, n_traj: int, n_layers: int):
    """
    Roll out n_traj fresh trajectories with the current stochastic policy.

    Returns
    -------
    obs      : (N, T, obs_dim) float32
    actions  : (N, T) float32   — continuous laser power [W] chosen
    rewards  : (N, T) float32
    u        : (N, T) float32   — combined (heating+cooling) epistemic_std + aleatoric_std
    """
    obs_dim = env.obs_dim
    obs     = np.zeros((n_traj, n_layers, obs_dim), dtype=np.float32)
    actions = np.zeros((n_traj, n_layers),           dtype=np.float32)
    rewards = np.zeros((n_traj, n_layers),           dtype=np.float32)
    u       = np.zeros((n_traj, n_layers),           dtype=np.float32)

    for i in range(n_traj):
        state = env.reset()
        for t in range(n_layers):
            a = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(a)

            obs[i, t]     = state
            actions[i, t] = a
            rewards[i, t] = reward
            u[i, t]       = info["uncertainty"]

            state = next_state
            if done:
                break

    return obs, actions, rewards, u


# =============================================================================
# Plotting helpers
# =============================================================================

def _plot_series(values, ylabel, title, out_path, color="steelblue",
                 hline=None, hline_label=None, window=20):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    it = np.arange(1, len(values) + 1)
    ax.plot(it, values, alpha=0.35, color=color, linewidth=0.8, label=ylabel)
    if len(values) >= window:
        smooth = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(it[window - 1:], smooth, color=color, linewidth=2.0,
                label=f"Rolling avg ({window})")
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle="--", linewidth=1, label=hline_label)
    ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[train] Plot saved → {out_path}")


def _plot_action_stats(hist_mean, hist_std, action_min, action_max, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    it = np.arange(1, len(hist_mean) + 1)
    mean_arr = np.array(hist_mean)
    std_arr  = np.array(hist_std)
    ax.plot(it, mean_arr, color="darkorange", linewidth=1.5, label="Mean chosen action")
    ax.fill_between(it, mean_arr - std_arr, mean_arr + std_arr,
                    color="darkorange", alpha=0.2, label="± std")
    ax.axhline(action_min, color="grey", linestyle=":", linewidth=1, label="action_min/max")
    ax.axhline(action_max, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Laser power [W]")
    ax.set_title("UCPG v2 — Chosen Action Mean ± Std per Iteration")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[train] Plot saved → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.out_dir:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("online_RL_ucpg_v2", "runs", ts)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("[train] Uncertainty-Constrained Policy Gradient v2 — LPBF Controller")
    print("[train] (continuous action, two-stage heating/cooling surrogate)")
    print(f"[train] Output dir : {out_dir}")
    print(f"[train] Device     : {device}")
    print(f"[train] delta (uncertainty budget) : {args.delta}")
    print("=" * 65)

    # ── load surrogate ────────────────────────────────────────────────────────
    print(f"\n[train] Loading surrogate: {args.surrogate}")
    (surrogate, state_mean, state_std, lp_mean, lp_std,
     cool_mean, cool_std, _roi_table) = load_two_stage_surrogate(args.surrogate, device=device)
    surrogate.eval()
    print(f"[train] Surrogate   : {surrogate}")

    # ── environment ────────────────────────────────────────────────────────────
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
    print(f"[train] Action aim range : [{args.action_min:.0f}, {args.action_max:.0f}] W  (continuous, unclipped)")
    print(f"[train] cool_time range  : [{args.cool_time_min:.3f}, {args.cool_time_max:.3f}] s  (drawn once/episode)")
    print(f"[train] obs_dim     : {env.obs_dim}  (latent_dim={env.latent_dim} + layer token + cool_time token)")

    # ── agent ──────────────────────────────────────────────────────────────────
    if args.resume:
        print(f"\n[train] Resuming agent from: {args.resume}")
        agent = UCPGAgentV2.load(args.resume, device=device)
    else:
        agent = UCPGAgentV2(
            latent_dim      = env.latent_dim,
            action_min      = args.action_min,
            action_max      = args.action_max,
            hidden          = args.hidden,
            depth           = args.depth,
            dropout         = args.dropout,
            n_layers        = args.n_layers,
            layer_embed_dim = args.layer_embed_dim,
            log_sigma_init  = args.log_sigma_init,
            min_log_sigma   = args.min_log_sigma,
            max_log_sigma   = args.max_log_sigma,
            lr_theta        = args.lr_theta,
            lambda_init     = args.lambda_init,
            max_grad_norm   = args.max_grad_norm,
            device          = device,
        )
    print(f"[train] {agent}")

    # ── training loop ─────────────────────────────────────────────────────────
    hist_return, hist_ju, hist_lambda, hist_loss, hist_entropy = [], [], [], [], []
    hist_action_mean, hist_action_std, hist_frac_ood = [], [], []
    best_return = -float("inf")

    print(f"\n[train] Starting {args.n_iterations} iterations "
          f"(N={args.n_traj} trajectories/iter, T={args.n_layers} steps/traj)")
    print(f"[train] gamma_r={args.gamma_r}  gamma_u={args.gamma_u}  "
          f"lr_theta={args.lr_theta}  lr_lambda={args.lr_lambda}")
    print("-" * 65)

    t0 = time.time()
    for k in range(1, args.n_iterations + 1):

        obs, actions, rewards, u = collect_batch(env, agent, args.n_traj, args.n_layers)

        G_r = discounted_returns(rewards, args.gamma_r)   # (N, T)
        G_u = discounted_returns(u,       args.gamma_u)   # (N, T)

        if args.use_baseline:
            # Per-timestep empirical baseline (batch mean at each t) — pure Monte
            # Carlo, no bootstrapping. Reduces variance and repairs early-layer
            # credit assignment without changing the gradient estimator's expectation.
            G_r_adv = G_r - G_r.mean(axis=0, keepdims=True)
            G_u_adv = G_u - G_u.mean(axis=0, keepdims=True)
        else:
            G_r_adv = G_r
            G_u_adv = G_u

        advantage = G_r_adv - agent.lam * G_u_adv           # uses lambda from PREVIOUS iteration

        N, T, obs_dim = obs.shape
        obs_flat       = torch.tensor(obs.reshape(N * T, obs_dim), dtype=torch.float32, device=device)
        actions_flat   = torch.tensor(actions.reshape(N * T),      dtype=torch.float32, device=device)
        advantage_flat = torch.tensor(advantage.reshape(N * T),    dtype=torch.float32, device=device)

        loss, entropy = agent.policy_step(obs_flat, actions_flat, advantage_flat, N)

        j_u_hat = float(G_u[:, 0].mean())
        lam     = agent.update_lambda(j_u_hat, args.delta, args.lr_lambda)

        raw_return  = float(rewards.sum(axis=1).mean())   # undiscounted, interpretable in reward units
        act_mean    = float(actions.mean())
        act_std     = float(actions.std())

        hist_return.append(raw_return)
        hist_ju.append(j_u_hat)
        hist_lambda.append(lam)
        hist_loss.append(loss)
        hist_entropy.append(entropy)
        hist_action_mean.append(act_mean)
        hist_action_std.append(act_std)
        if args.ood_threshold is not None:
            hist_frac_ood.append(float((actions > args.ood_threshold).mean()))

        if k % args.log_freq == 0 or k == 1:
            n_recent = min(args.log_freq, len(hist_return))
            elapsed  = time.time() - t0
            ood_str  = (f" | OOD-action {hist_frac_ood[-1]*100:5.1f}%"
                       if args.ood_threshold is not None else "")
            print(
                f"Iter {k:5d}/{args.n_iterations} | "
                f"return {raw_return:+.4f} (avg{n_recent} {np.mean(hist_return[-n_recent:]):+.4f}) | "
                f"J_u {j_u_hat:.5f} (δ={args.delta}) | "
                f"λ {lam:.5f} | "
                f"loss {loss:+.4f} | "
                f"entropy {entropy:.3f} | "
                f"action {act_mean:.1f}±{act_std:.1f}W"
                f"{ood_str} | "
                f"{elapsed:.0f}s"
            )

        if raw_return > best_return:
            best_return = raw_return
            agent.save(
                os.path.join(out_dir, "ucpg_best.pt"),
                extra={
                    "iteration":    k,
                    "best_return":  best_return,
                    "j_u_hat":      j_u_hat,
                    "delta":        args.delta,
                    "train_args":   vars(args),
                },
            )

        if k % args.save_freq == 0:
            agent.save(
                os.path.join(out_dir, f"ucpg_iter{k:05d}.pt"),
                extra={
                    "iteration":        k,
                    "hist_return":      hist_return,
                    "hist_ju":          hist_ju,
                    "hist_lambda":      hist_lambda,
                    "hist_action_mean": hist_action_mean,
                    "hist_action_std":  hist_action_std,
                    "hist_frac_ood":    hist_frac_ood,
                    "delta":            args.delta,
                    "train_args":       vars(args),
                },
            )
            _plot_series(hist_return, "Undiscounted episode return", "UCPG v2 — Reward Return",
                        os.path.join(out_dir, "return.png"))
            _plot_series(hist_ju, "J_u_hat", "UCPG v2 — Estimated Uncertainty Return vs. Budget",
                        os.path.join(out_dir, "j_u.png"), color="tab:red",
                        hline=args.delta, hline_label="δ (budget)")
            _plot_series(hist_lambda, "λ", "UCPG v2 — Lagrange Multiplier",
                        os.path.join(out_dir, "lambda.png"), color="tab:purple")
            _plot_series(hist_loss, "Policy loss", "UCPG v2 — Policy Gradient Loss",
                        os.path.join(out_dir, "loss.png"), color="tab:orange")
            _plot_series(hist_entropy, "Mean policy entropy", "UCPG v2 — Policy Entropy",
                        os.path.join(out_dir, "entropy.png"), color="tab:green")
            _plot_action_stats(hist_action_mean, hist_action_std,
                               args.action_min, args.action_max,
                               os.path.join(out_dir, "action_stats.png"))
            if args.ood_threshold is not None:
                _plot_series(hist_frac_ood, "Fraction of actions > threshold",
                            f"UCPG v2 — Fraction of Chosen Actions Above {args.ood_threshold:.0f} W",
                            os.path.join(out_dir, "ood_action_fraction.png"), color="tab:brown")

    # ── final checkpoint + plots ───────────────────────────────────────────────
    agent.save(
        os.path.join(out_dir, "ucpg_final.pt"),
        extra={
            "iteration":        args.n_iterations,
            "hist_return":      hist_return,
            "hist_ju":          hist_ju,
            "hist_lambda":      hist_lambda,
            "hist_action_mean": hist_action_mean,
            "hist_action_std":  hist_action_std,
            "hist_frac_ood":    hist_frac_ood,
            "delta":            args.delta,
            "train_args":       vars(args),
        },
    )
    _plot_series(hist_return, "Undiscounted episode return", "UCPG v2 — Reward Return",
                os.path.join(out_dir, "return.png"))
    _plot_series(hist_ju, "J_u_hat", "UCPG v2 — Estimated Uncertainty Return vs. Budget",
                os.path.join(out_dir, "j_u.png"), color="tab:red",
                hline=args.delta, hline_label="δ (budget)")
    _plot_series(hist_lambda, "λ", "UCPG v2 — Lagrange Multiplier",
                os.path.join(out_dir, "lambda.png"), color="tab:purple")
    _plot_series(hist_loss, "Policy loss", "UCPG v2 — Policy Gradient Loss",
                os.path.join(out_dir, "loss.png"), color="tab:orange")
    _plot_series(hist_entropy, "Mean policy entropy", "UCPG v2 — Policy Entropy",
                os.path.join(out_dir, "entropy.png"), color="tab:green")
    _plot_action_stats(hist_action_mean, hist_action_std,
                       args.action_min, args.action_max,
                       os.path.join(out_dir, "action_stats.png"))
    if args.ood_threshold is not None:
        _plot_series(hist_frac_ood, "Fraction of actions > threshold",
                    f"UCPG v2 — Fraction of Chosen Actions Above {args.ood_threshold:.0f} W",
                    os.path.join(out_dir, "ood_action_fraction.png"), color="tab:brown")

    total_time = time.time() - t0
    print("\n" + "=" * 65)
    print(f"[train] Training complete in {total_time/60:.1f} min")
    print(f"[train] Best undiscounted return : {best_return:.4f}")
    print(f"[train] Final J_u_hat / delta     : {hist_ju[-1]:.5f} / {args.delta}")
    print(f"[train] Final lambda              : {hist_lambda[-1]:.5f}")
    print(f"[train] Final action              : {hist_action_mean[-1]:.1f} ± {hist_action_std[-1]:.1f} W")
    print(f"[train] Best checkpoint  : {os.path.join(out_dir, 'ucpg_best.pt')}")
    print(f"[train] Final checkpoint : {os.path.join(out_dir, 'ucpg_final.pt')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
