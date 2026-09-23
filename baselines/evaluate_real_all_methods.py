"""
baselines/evaluate_real_all_methods.py
------------------------------------------
Real-PHYSICS ground-truth evaluation for ALL controllers (naive_pg,
offline_q, proportional, kalman_particle, UCPG v2), not just the two
UCPGAgentV2-checkpoint-based ones online_RL_ucpg_v2/evaluate_real.py already
covers. Every layer's transition is a REAL simulateHeatingCooling_v2.m PDE
solve (via a fresh `matlab -batch` subprocess), not the neural surrogate —
this is the ground-truth check for "did each policy/controller actually
solve the process, or just learn to exploit the surrogate."

This module deliberately does NOT reimplement the MATLAB-interfacing
machinery: `_build_params_dict`, `run_one_layer`, `_check_mesh_consistency`,
`_to_matlab_numeric`, `FIXED_PARAMS`, `SIMULATION_V2_DIR` are imported
directly from online_RL_ucpg_v2.evaluate_real, which is the proven,
already-working implementation of one real PDE solve per layer with
IC-chaining across layers. Only the CONTROLLER side is generalized here:
instead of being hard-wired to a single UCPGAgentV2.select_action call,
each layer builds a baselines.common.eval_harness.StepContext (the same
interface every baseline already implements against the surrogate
environment in evaluate_baselines.py) from the real MATLAB field, so any
controller — RL policy or classical baseline alike — can be dropped in
unchanged.

Real PDE solves are slow (~30-45s/layer observed, ~6-9 min per 12-layer
episode — see jobs/evaluate_real_ucpg_v2.sh's comment) and scale linearly
with (n_episodes x n_layers x n_methods), so --n_episodes defaults small
(3) here, deliberately much lower than evaluate_baselines.py's surrogate-
driven default of 50 — this script is a ground-truth spot check on top of
that larger surrogate-driven comparison, not a replacement for it.

Usage
-----
    module load matlab   # must be on $PATH before running
    python -m baselines.evaluate_real_all_methods \\
        --surrogate surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt \\
        --naive_pg_checkpoint    baselines/naive_pg/runs/narrow_200_300W/naive_pg_best.pt \\
        --offline_q_checkpoint   baselines/offline_q/runs/narrow_200_300W/offline_q_best.pt \\
        --ucpg_v2_checkpoint     online_RL_ucpg_v2/runs/narrow_200_300W/ucpg_best.pt \\
        --proportional_fitted    baselines/proportional/fitted_narrow_200_300W.pt \\
        --kalman_particle_fitted baselines/kalman_particle/fitted_narrow_200_300W.pt \\
        --n_episodes 3 --out_dir baselines/results_narrow_200_300W_real
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.model import load_surrogate
from online_RL_ucpg_v2.evaluate_real import (
    _build_params_dict, run_one_layer, _check_mesh_consistency, FIXED_PARAMS,
)
from baselines.common.eval_harness import (
    StepContext, LatentAgentController, summarize, print_leaderboard,
    plot_reward_and_action_per_layer, slugify, write_leaderboard,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ALL baseline/RL controllers against the REAL PDE simulator."
    )
    p.add_argument("--surrogate", type=str, required=True,
                   help="surrogate_model_v3 checkpoint — normalisation stats only "
                        "(builds RL-policy observations from the real MATLAB field); "
                        "its transition model is never called.")

    # ── environment / reward window (must match training) ────────────────────
    p.add_argument("--T_l",      type=float, default=2000.0)
    p.add_argument("--T_h",      type=float, default=2800.0)
    p.add_argument("--n_layers", type=int,   default=12)
    p.add_argument("--width",    type=float, default=12.0)
    p.add_argument("--height",   type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end",   type=float, default=0.5)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--cool_time_min", type=float, default=0.05)
    p.add_argument("--cool_time_max", type=float, default=0.15)
    p.add_argument("--action_min",  type=float, default=100.0)
    p.add_argument("--action_max",  type=float, default=400.0)

    # ── this evaluation's cool_time (real, physically applied — fixed per episode) ──
    p.add_argument("--cool_time", type=float, default=0.10,
                   help="Fixed cool_time [s] applied every episode, for every method — "
                        "keeps the cross-method comparison fair. Ignored if --random_cool_time.")
    p.add_argument("--random_cool_time", action="store_true")

    # ── controllers (each optional — omit to skip; same flags as evaluate_baselines.py) ──
    p.add_argument("--naive_pg_checkpoint",    type=str, default=None)
    p.add_argument("--ucpg_v2_checkpoint",     type=str, default=None)
    p.add_argument("--offline_q_checkpoint",   type=str, default=None)
    p.add_argument("--proportional_fitted",    type=str, default=None)
    p.add_argument("--kalman_particle_fitted", type=str, default=None)
    p.add_argument("--kalman_R",   type=float, default=2500.0)
    p.add_argument("--particle_R", type=float, default=2500.0)
    p.add_argument("--particle_n", type=int,   default=200)

    # ── real-simulator evaluation settings ────────────────────────────────────
    p.add_argument("--n_episodes", type=int, default=3,
                   help="Episodes per method. Real PDE solves are slow — keep this small.")
    p.add_argument("--matlab_timeout", type=float, default=300.0)
    p.add_argument("--work_dir",  type=str, default="")
    p.add_argument("--mesh_path", type=str, default="surrogate_model/mesh.mat")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed",   type=int, default=123)
    p.add_argument("--out_dir", type=str, default="baselines/results_real")
    return p.parse_args()


def run_episode_real(
    controller, state_mean, state_std, cool_time_s: float, args, work_dir: str, check_mesh: bool,
):
    """Generalized version of online_RL_ucpg_v2.evaluate_real.run_episode:
    same MATLAB-subprocess-per-layer mechanism, but calls `controller.act(ctx)`
    (the baselines.common.eval_harness.StepContext interface every baseline
    already implements) instead of assuming a UCPGAgentV2 checkpoint."""
    device = state_mean.device
    state_dim = state_mean.shape[0]
    n_layers = args.n_layers
    fracs = np.linspace(args.sq_frac_start, args.sq_frac_end, n_layers)
    cool_token = (cool_time_s - args.cool_time_min) / max(args.cool_time_max - args.cool_time_min, 1e-8)

    raw_state = np.full(state_dim, args.initial_temp, dtype=np.float32)
    actions, rewards = [], []

    if hasattr(controller, "reset"):   # stateful controllers (Kalman/particle filter)
        controller.reset()

    for t in range(n_layers):
        with torch.no_grad():
            s_t = torch.tensor(raw_state, dtype=torch.float32, device=device).unsqueeze(0)
            s_n = (s_t - state_mean) / state_std
            z_t = s_n.squeeze(0).cpu().numpy()   # no encoder — raw normalised field IS the observation

        layer_token = t / max(n_layers - 1, 1)
        obs = np.concatenate([z_t, [layer_token, cool_token]]).astype(np.float32)
        ctx = StepContext(obs=obs, z=z_t, raw_state=raw_state, layer=t, cool_time=cool_time_s)
        a = float(np.clip(controller.act(ctx), args.action_min, args.action_max))

        params = _build_params_dict(
            lp_W=a, cool_time_s=cool_time_s, sq_frac=fracs[t],
            T_l=args.T_l, T_h=args.T_h, width=args.width, height=args.height,
        )
        t0 = time.time()
        out = run_one_layer(params, t, work_dir, args.matlab_timeout, save_mesh=(check_mesh and t == 0))
        elapsed = time.time() - t0

        if check_mesh and t == 0:
            _check_mesh_consistency(out["nodes"], os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", args.mesh_path))

        actions.append(a)
        rewards.append(out["reward"])
        raw_state = out["u_final"]

        print(f"      layer {t+1:2d}/{n_layers}  LP={a:7.2f}W  reward={out['reward']:+.4f}  ({elapsed:.0f}s)")

    return np.array(actions), np.array(rewards)


def _run_controller(name, controller, state_mean, state_std, args, work_dir):
    """Runs --n_episodes real episodes for one controller, prints per-episode
    progress, and returns a run_many()-shaped result dict (with a zero `u`
    array — no surrogate uncertainty is available from a real-sim rollout)."""
    all_a, all_r = [], []
    for ep in range(args.n_episodes):
        cool_time_s = (
            float(np.random.uniform(args.cool_time_min, args.cool_time_max))
            if args.random_cool_time else args.cool_time
        )
        print(f"    [{name}] episode {ep+1}/{args.n_episodes}  (cool_time={cool_time_s:.3f}s)")
        a, r = run_episode_real(controller, state_mean, state_std, cool_time_s, args,
                                work_dir, check_mesh=(ep == 0))
        print(f"      → episode return: {r.sum():+.4f}")
        all_a.append(a); all_r.append(r)
    actions = np.stack(all_a); rewards = np.stack(all_r)
    return dict(actions=actions, rewards=rewards, u=np.zeros_like(rewards))


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if shutil.which("matlab") is None:
        raise RuntimeError(
            "matlab not found on $PATH. Run `module load matlab` before this script."
        )

    print("=" * 70)
    print("  All-Methods REAL PHYSICS Evaluation (simulateHeatingCooling_v2.m)")
    print("=" * 70)
    print(f"  Surrogate (norm stats only) : {args.surrogate}")
    print(f"  Episodes/method              : {args.n_episodes}")
    print("=" * 70)

    (_surrogate, state_mean, state_std, *_rest) = load_surrogate(args.surrogate, device=device)
    state_mean = state_mean.to(device)
    state_std  = state_std.to(device)

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="real_eval_all_")
    os.makedirs(work_dir, exist_ok=True)
    print(f"  Scratch dir : {work_dir}\n")

    rows = []

    def _evaluate(name: str, ctrl) -> None:
        t0 = time.time()
        result = _run_controller(name, ctrl, state_mean, state_std, args, work_dir)
        rows.append(summarize(name, result))
        plot_reward_and_action_per_layer(
            result["rewards"], result["actions"], f"{name} (REAL physics)",
            os.path.join(args.out_dir, f"per_layer_real_{slugify(name)}.png"),
            temp_range=(args.T_l, args.T_h),
        )
        print(f"  [{name}] done in {(time.time()-t0)/60:.1f} min\n")

    if args.naive_pg_checkpoint:
        print("=== Naive PG ===")
        _evaluate("0. Naive PG", LatentAgentController(args.naive_pg_checkpoint, device=device, greedy=True))

    if args.offline_q_checkpoint:
        print("=== Offline Q ===")
        from baselines.offline_q.model import load_offline_q_controller
        _evaluate("1. Offline Q-learning", load_offline_q_controller(args.offline_q_checkpoint, device=device))

    if args.proportional_fitted:
        print("=== Proportional ===")
        from baselines.proportional.controller import load_proportional_controller
        _evaluate("2. Proportional", load_proportional_controller(args.proportional_fitted))

    if args.kalman_particle_fitted:
        print("=== Kalman / Particle filter ===")
        from baselines.kalman_particle.filters import load_kalman_controller, load_particle_controller
        _evaluate("4. Kalman filter", load_kalman_controller(args.kalman_particle_fitted, R=args.kalman_R, seed=args.seed))
        _evaluate("4. Particle filter", load_particle_controller(
            args.kalman_particle_fitted, R=args.particle_R, n_particles=args.particle_n, seed=args.seed))

    if args.ucpg_v2_checkpoint:
        print("=== UCPG v2 (ours) ===")
        _evaluate("UCPG v2 (ours)", LatentAgentController(args.ucpg_v2_checkpoint, device=device, greedy=True))

    if not rows:
        print("[evaluate_real_all_methods] No methods selected — pass at least one checkpoint/fitted-params flag.")
        return

    print()
    print_leaderboard(rows)
    write_leaderboard(
        rows, args.out_dir, n_layers=args.n_layers, T_l=args.T_l, T_h=args.T_h,
        title="Baseline Comparison (REAL PHYSICS)",
        csv_name="leaderboard_real.csv", png_name="leaderboard_real.png",
    )

    if not args.work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
