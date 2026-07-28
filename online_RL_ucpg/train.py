"""
online_RL_ucpg/train.py
--------------------------
Monte Carlo Uncertainty-Constrained Policy Gradient (UCPG) training loop.

Implements Algorithm 1 exactly:

  for k = 1 .. K:
    collect N trajectories using a_t ~ pi_theta(.|s_t)   (s_t = latent obs)
    for each transition: u_t = sigma_epis,t + sigma_alea,t   (from the surrogate env)
    G_r,t = sum_{k=t}^{T-1} gamma_r^{k-t} r_k
    G_u,t = sum_{k=t}^{T-1} gamma_u^{k-t} u_k
    G_lambda,t = G_r,t - lambda * G_u,t
    theta <- theta + alpha_theta * (1/N) sum_i sum_t nabla_theta log pi_theta(a_t|s_t) * G_lambda,t
    J_u_hat = (1/N) sum_i sum_t gamma_u^t u_t^(i)          [ = (1/N) sum_i G_u,0^(i) ]
    lambda <- [lambda + alpha_lambda * (J_u_hat - delta)]_+

No replay buffer, no baseline subtraction, no entropy bonus — this is a
faithful, literal implementation of the algorithm as specified, not a
variance-reduced variant. Each batch of N trajectories is generated fresh by
the CURRENT policy and used for exactly one gradient step (on-policy Monte
Carlo), matching the algorithm box.

Purpose of this experiment
----------------------------
Train the policy against a surrogate that only ever saw a NARROW laser-power
range (e.g. 150-300 W) while allowing the policy to choose from a WIDER
action grid (e.g. 150-400 W, via --action_max). Because the surrogate is
uncertain (high epistemic sigma) outside its training range, a working
uncertainty constraint should push the policy away from the unseen
300-400 W actions to keep J_u <= delta, even though the reward signal alone
has no idea that region is untrustworthy. --ood_threshold (default 300, the
narrow surrogate's training ceiling) is used only for the diagnostic
"fraction of chosen actions above this threshold" logged every iteration —
it plays no role in the constraint/loss itself.

Usage
-----
    python -m online_RL_ucpg.train \\
        --surrogate surrogate_model_latent_uncertainty/runs/narrow_150_300W/latent_best.pt \\
        --action_min 150 --action_max 400 --action_step 10 \\
        --delta 0.05 --ood_threshold 300
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

from surrogate_model_latent_uncertainty.train import load_latent_surrogate
from online_RL_ucpg.env   import LatentLPBFEnv
from online_RL_ucpg.agent import UCPGAgent


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo Uncertainty-Constrained Policy Gradient (UCPG) "
                    "training for the LPBF laser-power controller."
    )

    # ── surrogate model ───────────────────────────────────────────────────────
    p.add_argument("--surrogate", type=str, required=True,
                   help="Path to a trained surrogate_model_latent_uncertainty checkpoint "
                        "(EnsembleGaussianLatentDynamicsModel).")

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

    # ── action grid — intentionally may extend past the surrogate's training range ──
    p.add_argument("--action_min",  type=float, default=150.0)
    p.add_argument("--action_max",  type=float, default=400.0)
    p.add_argument("--action_step", type=float, default=10.0)
    p.add_argument("--ood_threshold", type=float, default=300.0,
                   help="Diagnostic only: actions above this [W] are logged as "
                        "'OOD fraction' every iteration (does not affect training).")

    # ── policy network architecture ───────────────────────────────────────────
    p.add_argument("--hidden",          type=int,   default=128)
    p.add_argument("--depth",           type=int,   default=3)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--layer_embed_dim", type=int,   default=8)

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

def collect_batch(env: LatentLPBFEnv, agent: UCPGAgent, n_traj: int, n_layers: int):
    """
    Roll out n_traj fresh trajectories with the current stochastic policy.

    Returns
    -------
    obs      : (N, T, obs_dim) float32
    actions  : (N, T) int64
    rewards  : (N, T) float32
    u        : (N, T) float32   — epistemic_std + aleatoric_std per step
    action_W : (N, T) float32   — raw laser power chosen, for OOD diagnostics
    """
    obs_dim = env.obs_dim
    obs      = np.zeros((n_traj, n_layers, obs_dim), dtype=np.float32)
    actions  = np.zeros((n_traj, n_layers),           dtype=np.int64)
    rewards  = np.zeros((n_traj, n_layers),           dtype=np.float32)
    u        = np.zeros((n_traj, n_layers),           dtype=np.float32)
    action_W = np.zeros((n_traj, n_layers),           dtype=np.float32)

    for i in range(n_traj):
        state = env.reset()
        for t in range(n_layers):
            a = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(a)

            obs[i, t]      = state
            actions[i, t]  = a
            rewards[i, t]  = reward
            u[i, t]        = info["uncertainty"]
            action_W[i, t] = info["action_W"]

            state = next_state
            if done:
                break

    return obs, actions, rewards, u, action_W


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
        out_dir = os.path.join("online_RL_ucpg", "runs", ts)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("[train] Uncertainty-Constrained Policy Gradient — LPBF Controller")
    print(f"[train] Output dir : {out_dir}")
    print(f"[train] Device     : {device}")
    print(f"[train] delta (uncertainty budget) : {args.delta}")
    print("=" * 65)

    # ── load surrogate ────────────────────────────────────────────────────────
    print(f"\n[train] Loading surrogate: {args.surrogate}")
    surrogate, state_mean, state_std, action_mean, action_std, _ = \
        load_latent_surrogate(args.surrogate, device=device)
    surrogate.eval()
    print(f"[train] Surrogate   : {surrogate}")

    # ── environment ────────────────────────────────────────────────────────────
    env = LatentLPBFEnv(
        surrogate     = surrogate,
        state_mean    = state_mean,
        state_std     = state_std,
        action_mean   = action_mean,
        action_std    = action_std,
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
        action_step   = args.action_step,
    )
    print(f"[train] Action grid : {env.n_actions} actions "
          f"({env.action_list[0]:.0f}-{env.action_list[-1]:.0f} W)  "
          f"| OOD diagnostic threshold: {args.ood_threshold:.0f} W")
    print(f"[train] obs_dim     : {env.obs_dim}  (latent_dim={env.latent_dim} + 1 layer token)")

    # ── agent ──────────────────────────────────────────────────────────────────
    if args.resume:
        print(f"\n[train] Resuming agent from: {args.resume}")
        agent = UCPGAgent.load(args.resume, device=device)
    else:
        agent = UCPGAgent(
            latent_dim      = env.latent_dim,
            n_actions       = env.n_actions,
            hidden          = args.hidden,
            depth           = args.depth,
            dropout         = args.dropout,
            n_layers        = args.n_layers,
            layer_embed_dim = args.layer_embed_dim,
            lr_theta        = args.lr_theta,
            lambda_init     = args.lambda_init,
            max_grad_norm   = args.max_grad_norm,
            device          = device,
        )
    print(f"[train] {agent}")

    # ── training loop ─────────────────────────────────────────────────────────
    hist_return, hist_ju, hist_lambda, hist_loss, hist_entropy, hist_frac_ood = \
        [], [], [], [], [], []
    best_return = -float("inf")

    print(f"\n[train] Starting {args.n_iterations} iterations "
          f"(N={args.n_traj} trajectories/iter, T={args.n_layers} steps/traj)")
    print(f"[train] gamma_r={args.gamma_r}  gamma_u={args.gamma_u}  "
          f"lr_theta={args.lr_theta}  lr_lambda={args.lr_lambda}")
    print("-" * 65)

    t0 = time.time()
    for k in range(1, args.n_iterations + 1):

        obs, actions, rewards, u, action_W = collect_batch(
            env, agent, args.n_traj, args.n_layers
        )

        G_r = discounted_returns(rewards, args.gamma_r)   # (N, T)
        G_u = discounted_returns(u,       args.gamma_u)   # (N, T)
        advantage = G_r - agent.lam * G_u                  # uses lambda from PREVIOUS iteration

        N, T, obs_dim = obs.shape
        obs_flat       = torch.tensor(obs.reshape(N * T, obs_dim), dtype=torch.float32, device=device)
        actions_flat   = torch.tensor(actions.reshape(N * T),      dtype=torch.long,    device=device)
        advantage_flat = torch.tensor(advantage.reshape(N * T),    dtype=torch.float32, device=device)

        loss, entropy = agent.policy_step(obs_flat, actions_flat, advantage_flat, N)

        j_u_hat = float(G_u[:, 0].mean())
        lam     = agent.update_lambda(j_u_hat, args.delta, args.lr_lambda)

        raw_return  = float(rewards.sum(axis=1).mean())   # undiscounted, interpretable in reward units
        frac_ood    = float((action_W > args.ood_threshold).mean())

        hist_return.append(raw_return)
        hist_ju.append(j_u_hat)
        hist_lambda.append(lam)
        hist_loss.append(loss)
        hist_entropy.append(entropy)
        hist_frac_ood.append(frac_ood)

        if k % args.log_freq == 0 or k == 1:
            n_recent = min(args.log_freq, len(hist_return))
            elapsed  = time.time() - t0
            print(
                f"Iter {k:5d}/{args.n_iterations} | "
                f"return {raw_return:+.4f} (avg{n_recent} {np.mean(hist_return[-n_recent:]):+.4f}) | "
                f"J_u {j_u_hat:.5f} (δ={args.delta}) | "
                f"λ {lam:.5f} | "
                f"loss {loss:+.4f} | "
                f"entropy {entropy:.3f} | "
                f"OOD-action {frac_ood*100:5.1f}% | "
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
                    "action_list":  env.action_list,
                    "train_args":   vars(args),
                },
            )

        if k % args.save_freq == 0:
            agent.save(
                os.path.join(out_dir, f"ucpg_iter{k:05d}.pt"),
                extra={
                    "iteration":     k,
                    "hist_return":   hist_return,
                    "hist_ju":       hist_ju,
                    "hist_lambda":   hist_lambda,
                    "hist_frac_ood": hist_frac_ood,
                    "delta":         args.delta,
                    "action_list":   env.action_list,
                    "train_args":    vars(args),
                },
            )
            _plot_series(hist_return, "Undiscounted episode return", "UCPG — Reward Return",
                        os.path.join(out_dir, "return.png"))
            _plot_series(hist_ju, "J_u_hat", "UCPG — Estimated Uncertainty Return vs. Budget",
                        os.path.join(out_dir, "j_u.png"), color="tab:red",
                        hline=args.delta, hline_label="δ (budget)")
            _plot_series(hist_lambda, "λ", "UCPG — Lagrange Multiplier",
                        os.path.join(out_dir, "lambda.png"), color="tab:purple")
            _plot_series(hist_loss, "Policy loss", "UCPG — Policy Gradient Loss",
                        os.path.join(out_dir, "loss.png"), color="tab:orange")
            _plot_series(hist_entropy, "Mean policy entropy", "UCPG — Policy Entropy",
                        os.path.join(out_dir, "entropy.png"), color="tab:green")
            _plot_series(hist_frac_ood, "Fraction of actions > threshold",
                        f"UCPG — Fraction of Chosen Actions Above {args.ood_threshold:.0f} W\n"
                        "(the region the surrogate was never trained on)",
                        os.path.join(out_dir, "ood_action_fraction.png"), color="tab:brown")

    # ── final checkpoint + plots ───────────────────────────────────────────────
    agent.save(
        os.path.join(out_dir, "ucpg_final.pt"),
        extra={
            "iteration":     args.n_iterations,
            "hist_return":   hist_return,
            "hist_ju":       hist_ju,
            "hist_lambda":   hist_lambda,
            "hist_frac_ood": hist_frac_ood,
            "delta":         args.delta,
            "action_list":   env.action_list,
            "train_args":    vars(args),
        },
    )
    _plot_series(hist_return, "Undiscounted episode return", "UCPG — Reward Return",
                os.path.join(out_dir, "return.png"))
    _plot_series(hist_ju, "J_u_hat", "UCPG — Estimated Uncertainty Return vs. Budget",
                os.path.join(out_dir, "j_u.png"), color="tab:red",
                hline=args.delta, hline_label="δ (budget)")
    _plot_series(hist_lambda, "λ", "UCPG — Lagrange Multiplier",
                os.path.join(out_dir, "lambda.png"), color="tab:purple")
    _plot_series(hist_loss, "Policy loss", "UCPG — Policy Gradient Loss",
                os.path.join(out_dir, "loss.png"), color="tab:orange")
    _plot_series(hist_entropy, "Mean policy entropy", "UCPG — Policy Entropy",
                os.path.join(out_dir, "entropy.png"), color="tab:green")
    _plot_series(hist_frac_ood, "Fraction of actions > threshold",
                f"UCPG — Fraction of Chosen Actions Above {args.ood_threshold:.0f} W\n"
                "(the region the surrogate was never trained on)",
                os.path.join(out_dir, "ood_action_fraction.png"), color="tab:brown")

    total_time = time.time() - t0
    print("\n" + "=" * 65)
    print(f"[train] Training complete in {total_time/60:.1f} min")
    print(f"[train] Best undiscounted return : {best_return:.4f}")
    print(f"[train] Final J_u_hat / delta     : {hist_ju[-1]:.5f} / {args.delta}")
    print(f"[train] Final lambda              : {hist_lambda[-1]:.5f}")
    print(f"[train] Final OOD-action fraction : {hist_frac_ood[-1]*100:.1f}%")
    print(f"[train] Best checkpoint  : {os.path.join(out_dir, 'ucpg_best.pt')}")
    print(f"[train] Final checkpoint : {os.path.join(out_dir, 'ucpg_final.pt')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
