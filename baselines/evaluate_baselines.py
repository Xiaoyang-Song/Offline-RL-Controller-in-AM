"""
baselines/evaluate_baselines.py
-----------------------------------
Aggregation CLI: rolls out every requested baseline (and, optionally, the
real online_RL_ucpg_v2 policy) through the SAME two-stage surrogate
environment and prints/saves one comparison leaderboard.

Every baseline is optional and independently gated by whether you pass its
checkpoint/fitted-params path — omit a flag to skip that method. The
constant-power sweep has no checkpoint and always runs unless
--skip_constant is given.

Usage
-----
    python -m baselines.evaluate_baselines \\
        --surrogate surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt \\
        --naive_pg_checkpoint     baselines/naive_pg/runs/<ts>/naive_pg_best.pt \\
        --offline_q_checkpoint    baselines/offline_q/runs/<ts>/offline_q_best.pt \\
        --proportional_fitted     baselines/proportional/fitted.pt \\
        --kalman_particle_fitted  baselines/kalman_particle/fitted.pt \\
        --ucpg_v2_checkpoint      online_RL_ucpg_v2/runs/<ts>/ucpg_best.pt \\
        --n_episodes 50
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.common.eval_harness import (
    Harness, LatentAgentController, summarize, print_leaderboard,
    plot_reward_and_action_per_layer, slugify,
)
from baselines.constant.controller import sweep_constant_controllers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and aggregate all baselines against the shared surrogate environment.")
    p.add_argument("--surrogate", type=str, required=True,
                   help="Two-stage surrogate checkpoint — the common evaluation environment for every method.")

    # environment (should match how the surrogate/baselines were fit)
    p.add_argument("--T_l",          type=float, default=2000.0)
    p.add_argument("--T_h",          type=float, default=2800.0)
    p.add_argument("--n_layers",     type=int,   default=12)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--mesh_path",    type=str,   default="surrogate_model/mesh.mat")
    p.add_argument("--width",        type=float, default=12.0)
    p.add_argument("--height",       type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end",   type=float, default=0.5)
    p.add_argument("--action_min",  type=float, default=100.0)
    p.add_argument("--action_max",  type=float, default=400.0)
    p.add_argument("--cool_time_min", type=float, default=0.05)
    p.add_argument("--cool_time_max", type=float, default=0.15)

    # baselines (each optional — omit to skip)
    p.add_argument("--naive_pg_checkpoint",    type=str, default=None)
    p.add_argument("--ucpg_v2_checkpoint",     type=str, default=None,
                   help="Optional: also include the real online_RL_ucpg_v2 policy for a full leaderboard.")
    p.add_argument("--offline_q_checkpoint",   type=str, default=None)
    p.add_argument("--proportional_fitted",    type=str, default=None)
    p.add_argument("--kalman_particle_fitted", type=str, default=None)
    p.add_argument("--kalman_R",    type=float, default=2500.0, help="Synthetic sensor-noise variance [K^2] for the Kalman controller.")
    p.add_argument("--particle_R",  type=float, default=2500.0, help="Synthetic sensor-noise variance [K^2] for the particle filter.")
    p.add_argument("--particle_n",  type=int,   default=200,    help="Number of particles.")
    p.add_argument("--skip_constant", action="store_true")
    p.add_argument("--n_episodes_constant", type=int, default=10,
                   help="Episodes per constant value in the sweep (kept small — there are 31 values).")

    p.add_argument("--n_episodes", type=int, default=50, help="Episodes per (non-constant) method.")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed",   type=int, default=123)
    p.add_argument("--out_dir", type=str, default="baselines/results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("  Baselines — Aggregated Evaluation")
    print("=" * 70)
    print(f"  Surrogate (shared environment) : {args.surrogate}")
    print(f"  Episodes/method                : {args.n_episodes}")
    print("=" * 70)

    harness = Harness(
        surrogate_path=args.surrogate, device=device,
        T_l=args.T_l, T_h=args.T_h, n_layers=args.n_layers, initial_temp=args.initial_temp,
        mesh_path=args.mesh_path, width=args.width, height=args.height,
        sq_frac_start=args.sq_frac_start, sq_frac_end=args.sq_frac_end,
        action_min=args.action_min, action_max=args.action_max,
        cool_time_min=args.cool_time_min, cool_time_max=args.cool_time_max,
    )

    rows = []

    def _evaluate(name: str, ctrl, n_episodes: int) -> None:
        """Roll out `ctrl`, add its row to the leaderboard, AND save its
        per-layer reward/action plot — every method gets both from one rollout."""
        result = harness.run_many(ctrl, n_episodes)
        rows.append(summarize(name, result))
        plot_reward_and_action_per_layer(
            result["rewards"], result["actions"], name,
            os.path.join(args.out_dir, f"per_layer_{slugify(name)}.png"),
        )

    # ── 0. Naive policy gradient ────────────────────────────────────────────
    if args.naive_pg_checkpoint:
        ctrl = LatentAgentController(args.naive_pg_checkpoint, device=device, greedy=True)
        _evaluate("0. Naive PG", ctrl, args.n_episodes)

    # ── 1. Offline Q-learning ────────────────────────────────────────────────
    if args.offline_q_checkpoint:
        from baselines.offline_q.model import load_offline_q_controller
        ctrl = load_offline_q_controller(args.offline_q_checkpoint, device=device)
        _evaluate("1. Offline Q-learning", ctrl, args.n_episodes)

    # ── 2. Proportional controller ──────────────────────────────────────────
    if args.proportional_fitted:
        from baselines.proportional.controller import load_proportional_controller
        ctrl = load_proportional_controller(args.proportional_fitted)
        _evaluate("2. Proportional", ctrl, args.n_episodes)

    # ── 3. Constant policy sweep ─────────────────────────────────────────────
    if not args.skip_constant:
        sweep_rows, sweep_results = [], []
        for w, ctrl in sweep_constant_controllers():
            result = harness.run_many(ctrl, args.n_episodes_constant)
            sweep_rows.append(summarize(f"const_{w:.0f}W", result))
            sweep_results.append(result)
        means = np.array([r["return_mean"]  for r in sweep_rows])
        watts = np.array([float(r["name"].split("_")[1][:-1]) for r in sweep_rows])
        best_idx = int(np.argmax(means))
        best = sweep_rows[best_idx]

        rows.append(dict(name="3. Constant (mean of sweep)", return_mean=float(means.mean()),
                         return_std=float(means.std()),
                         uncertainty_mean=float(np.mean([r["uncertainty_mean"] for r in sweep_rows])),
                         action_mean=float(watts.mean()), action_std=float(watts.std())))
        best_name = f"3. Constant (best={best['action_mean']:.0f}W, reference only)"
        rows.append(dict(name=best_name,
                         return_mean=best["return_mean"], return_std=best["return_std"],
                         uncertainty_mean=best["uncertainty_mean"],
                         action_mean=best["action_mean"], action_std=0.0))
        best_result = sweep_results[best_idx]
        plot_reward_and_action_per_layer(
            best_result["rewards"], best_result["actions"], best_name,
            os.path.join(args.out_dir, f"per_layer_{slugify(best_name)}.png"),
        )

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(watts, means, marker="o", color="darkorange")
        ax.axhline(means.mean(), color="grey", linestyle="--", label=f"Mean over sweep ({means.mean():+.4f})")
        ax.set_xlabel("Constant laser power [W]"); ax.set_ylabel("Mean undiscounted return")
        ax.set_title("Baseline 3 — Constant-Power Sweep")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "constant_sweep.png"), dpi=150)
        plt.close(fig)
        print(f"[evaluate_baselines] Saved → {os.path.join(args.out_dir, 'constant_sweep.png')}")

    # ── 4. Kalman / particle filter ─────────────────────────────────────────
    if args.kalman_particle_fitted:
        from baselines.kalman_particle.filters import load_kalman_controller, load_particle_controller
        kctrl = load_kalman_controller(args.kalman_particle_fitted, R=args.kalman_R, seed=args.seed)
        pctrl = load_particle_controller(args.kalman_particle_fitted, R=args.particle_R,
                                         n_particles=args.particle_n, seed=args.seed)
        _evaluate("4. Kalman filter", kctrl, args.n_episodes)
        _evaluate("4. Particle filter", pctrl, args.n_episodes)

    # ── (optional) real UCPG v2 policy, for a full leaderboard ───────────────
    if args.ucpg_v2_checkpoint:
        ctrl = LatentAgentController(args.ucpg_v2_checkpoint, device=device, greedy=True)
        _evaluate("UCPG v2 (ours)", ctrl, args.n_episodes)

    if not rows:
        print("[evaluate_baselines] No methods selected — pass at least one checkpoint/fitted-params "
              "flag, or drop --skip_constant.")
        return

    print()
    print_leaderboard(rows)

    csv_path = os.path.join(args.out_dir, "leaderboard.csv")
    with open(csv_path, "w") as f:
        f.write("name,return_mean,return_std,uncertainty_mean,action_mean,action_std\n")
        for r in sorted(rows, key=lambda r: r["return_mean"], reverse=True):
            f.write(f"{r['name']},{r['return_mean']:.6f},{r['return_std']:.6f},"
                    f"{r['uncertainty_mean']:.6f},{r['action_mean']:.4f},{r['action_std']:.4f}\n")
    print(f"\n[evaluate_baselines] Saved → {csv_path}")

    rows_sorted = sorted(rows, key=lambda r: r["return_mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["name"] for r in rows_sorted]
    means = [r["return_mean"] for r in rows_sorted]
    stds  = [r["return_std"]  for r in rows_sorted]
    ax.barh(names, means, xerr=stds, color="steelblue", alpha=0.8)
    ax.set_xlabel("Undiscounted episode return (mean ± std)")
    ax.set_title("Baseline Comparison — Undiscounted Return")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "leaderboard.png"), dpi=150)
    plt.close(fig)
    print(f"[evaluate_baselines] Saved → {os.path.join(args.out_dir, 'leaderboard.png')}")


if __name__ == "__main__":
    main()
