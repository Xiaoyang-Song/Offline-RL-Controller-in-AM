"""
baselines/sweep_diagnostic.py
------------------------------------
Cheap "where is the surrogate optimistic?" diagnostic: rolls a CONSTANT-power
policy through both (a) the given surrogate checkpoint's environment and
(b) the real simulateHeatingCooling_v2.m PDE simulator, at each power level
in --powers, and reports the SIGNED gap (surrogate_return - real_return).

A positive gap means the surrogate over-estimates that constant power's
return relative to reality — exactly the kind of exploitable "optimism" a
reward-greedy policy (naive PG) would chase, and that an uncertainty-
constrained policy (UCPG) should be able to avoid, PROVIDED that power is
also outside the surrogate's training coverage (an uncertainty penalty has
no signal to work with over an IN-distribution power, even if a genuine
model-fit error exists there).

Use this BEFORE committing to a training-coverage design (e.g. which powers
to hold out as a gap) — a gap placed where the sign never flips buys
nothing, which is exactly what the single-sided 200-300W band showed
(naive_pg and UCPG were statistically tied, both in the surrogate and in
real physics — see baselines/README.md and the narrow_200_300W results).

Also reports a PER-LAYER breakdown, not just the summed episode return,
since the optimal power is layer-dependent (observed ~250W early, ~100W
late in this project) — the aggregate return can hide a layer-local
optimism pocket that only shows up early or late in the build.

This module only ever reuses existing, already-tested pieces —
baselines.common.eval_harness.Harness (surrogate rollout),
baselines.constant.controller.ConstantController, and
baselines.evaluate_real_all_methods.run_episode_real (real PDE rollout) —
nothing here reimplements the MATLAB-interfacing machinery.

Usage (run from the repo root; needs `module load matlab` first; real PDE
solves are slow, ~30-45s/layer — run as a batch job, not on the login node):
    python -m baselines.sweep_diagnostic \\
        --surrogate surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt \\
        --powers 100,130,160,190,220,250,280,310,340,370,400 \\
        --n_episodes_surrogate 20 --n_episodes_real 1 \\
        --out_dir baselines/results_sweep_diagnostic
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.common.eval_harness import Harness
from baselines.constant.controller import ConstantController
from baselines.evaluate_real_all_methods import run_episode_real
from surrogate_model_v3.model import load_surrogate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Constant-power surrogate-vs-real optimism sweep.")
    p.add_argument("--surrogate", type=str, required=True)
    p.add_argument("--powers", type=str, default="100,130,160,190,220,250,280,310,340,370,400",
                   help="Comma-separated constant laser powers [W] to sweep.")
    p.add_argument("--n_episodes_surrogate", type=int, default=20)
    p.add_argument("--n_episodes_real", type=int, default=1,
                   help="Real PDE solves are slow (~30-45s/layer) — keep this small.")
    p.add_argument("--cool_time", type=float, default=0.10,
                   help="Fixed cooling duration for BOTH surrogate and real rollouts "
                        "(keeps the two directly comparable; also used as both "
                        "cool_time_min/max for the surrogate env, so every surrogate "
                        "episode uses this exact value too, no randomisation).")
    p.add_argument("--T_l", type=float, default=2000.0)
    p.add_argument("--T_h", type=float, default=2800.0)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--mesh_path", type=str, default="surrogate_model/mesh.mat")
    p.add_argument("--width", type=float, default=12.0)
    p.add_argument("--height", type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end", type=float, default=0.5)
    p.add_argument("--action_min", type=float, default=100.0)
    p.add_argument("--action_max", type=float, default=400.0)
    p.add_argument("--matlab_timeout", type=float, default=300.0)
    p.add_argument("--work_dir", type=str, default="")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out_dir", type=str, default="baselines/results_sweep_diagnostic")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("matlab") is None:
        raise RuntimeError("matlab not found on $PATH. Run `module load matlab` before this script.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    powers = [float(x) for x in args.powers.split(",")]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # run_episode_real (baselines.evaluate_real_all_methods) reads
    # args.cool_time_min/args.cool_time_max off this same Namespace to build
    # the (normalised) cool_time observation token — this script only exposes
    # a single fixed --cool_time, so pin both bounds to it (zero-width range,
    # same trick already used for the surrogate Harness below).
    args.cool_time_min = args.cool_time
    args.cool_time_max = args.cool_time

    print(f"[sweep_diagnostic] Surrogate : {args.surrogate}")
    print(f"[sweep_diagnostic] Powers    : {powers}")
    print(f"[sweep_diagnostic] Episodes  : surrogate={args.n_episodes_surrogate}  real={args.n_episodes_real}")
    print(f"[sweep_diagnostic] Cool time : {args.cool_time}s (fixed for both)")

    harness = Harness(
        args.surrogate, device=device, T_l=args.T_l, T_h=args.T_h, n_layers=args.n_layers,
        initial_temp=args.initial_temp, mesh_path=args.mesh_path, width=args.width, height=args.height,
        sq_frac_start=args.sq_frac_start, sq_frac_end=args.sq_frac_end,
        action_min=args.action_min, action_max=args.action_max,
        cool_time_min=args.cool_time, cool_time_max=args.cool_time,
    )
    _model, state_mean, state_std, *_rest = load_surrogate(args.surrogate, device=device)

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="sweep_diag_")
    os.makedirs(work_dir, exist_ok=True)
    print(f"[sweep_diagnostic] Scratch dir: {work_dir}\n")

    rows = []
    per_layer_surr, per_layer_real = {}, {}
    for i, pw in enumerate(powers):
        ctrl = ConstantController(pw)
        t0 = time.time()

        surr = harness.run_many(ctrl, args.n_episodes_surrogate)
        surr_ret = surr["rewards"].sum(axis=1)
        per_layer_surr[pw] = surr["rewards"].mean(axis=0)

        real_rewards = []
        for ep in range(args.n_episodes_real):
            _actions, r = run_episode_real(
                ctrl, state_mean, state_std, args.cool_time, args, work_dir,
                check_mesh=(i == 0 and ep == 0),
            )
            real_rewards.append(r)
        real_rewards = np.stack(real_rewards)
        real_ret = real_rewards.sum(axis=1)
        per_layer_real[pw] = real_rewards.mean(axis=0)

        gap = float(surr_ret.mean() - real_ret.mean())
        rows.append(dict(power=pw, surr_mean=float(surr_ret.mean()), surr_std=float(surr_ret.std()),
                         real_mean=float(real_ret.mean()), real_std=float(real_ret.std()), gap=gap))
        print(f"  [{pw:6.1f}W] surrogate={surr_ret.mean():+.3f}±{surr_ret.std():.3f}  "
              f"real={real_ret.mean():+.3f}±{real_ret.std():.3f}  "
              f"gap(surr-real)={gap:+.3f}  ({time.time()-t0:.0f}s)")

    csv_path = os.path.join(args.out_dir, "sweep_diagnostic.csv")
    with open(csv_path, "w") as f:
        f.write("power_W,surrogate_return_mean,surrogate_return_std,real_return_mean,"
                "real_return_std,gap_surr_minus_real\n")
        for r in rows:
            f.write(f"{r['power']},{r['surr_mean']:.4f},{r['surr_std']:.4f},"
                    f"{r['real_mean']:.4f},{r['real_std']:.4f},{r['gap']:.4f}\n")
    print(f"\n[sweep_diagnostic] Saved → {csv_path}")

    ps = [r["power"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.errorbar(ps, [r["surr_mean"] for r in rows], yerr=[r["surr_std"] for r in rows],
                marker="o", label="Surrogate", color="tab:blue", capsize=3)
    ax1.errorbar(ps, [r["real_mean"] for r in rows], yerr=[r["real_std"] for r in rows],
                marker="s", label="Real physics", color="tab:red", capsize=3)
    ax1.set_xlabel("Constant laser power [W]"); ax1.set_ylabel("Episode return")
    ax1.set_title("Surrogate vs. real return"); ax1.legend(); ax1.grid(True, alpha=0.3)

    bar_w = (ps[1] - ps[0]) * 0.6 if len(ps) > 1 else 10.0
    colors = ["tab:red" if r["gap"] > 0 else "tab:blue" for r in rows]
    ax2.bar(ps, [r["gap"] for r in rows], width=bar_w, color=colors)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("Constant laser power [W]"); ax2.set_ylabel("Gap  (surrogate − real)")
    ax2.set_title("Surrogate optimism (>0 = surrogate over-estimates return)")
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sweep_diagnostic.png"), dpi=150)
    plt.close(fig)
    print(f"[sweep_diagnostic] Saved → {os.path.join(args.out_dir, 'sweep_diagnostic.png')}")

    layers = np.arange(1, args.n_layers + 1)
    gap_grid = np.array([per_layer_surr[pw] - per_layer_real[pw] for pw in powers])   # (n_powers, n_layers)
    fig, ax = plt.subplots(figsize=(9, 5))
    vmax = max(np.abs(gap_grid).max(), 1e-6)
    im = ax.imshow(gap_grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[0.5, args.n_layers + 0.5, powers[-1], powers[0]])
    ax.set_xlabel("Layer"); ax.set_ylabel("Constant laser power [W]")
    ax.set_xticks(layers)
    ax.set_title("Per-layer reward gap (surrogate − real): red = surrogate optimistic")
    fig.colorbar(im, ax=ax, label="Reward gap")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sweep_diagnostic_per_layer.png"), dpi=150)
    plt.close(fig)
    print(f"[sweep_diagnostic] Saved → {os.path.join(args.out_dir, 'sweep_diagnostic_per_layer.png')}")

    print("\n[sweep_diagnostic] Most surrogate-optimistic powers (best gap-placement candidates):")
    for r in sorted(rows, key=lambda r: r["gap"], reverse=True)[:5]:
        print(f"    {r['power']:6.1f}W  gap={r['gap']:+.3f}")

    if not args.work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
