"""
online_RL_ucpg_v2/calibrate_delta.py
----------------------------------------
Pick UCPG's uncertainty budget delta from two reference points measured on
the SAME surrogate environment train.py uses, instead of a fixed fraction of
a near-random policy's J_u:

    J_min  = min over a constant-laser-power sweep of J_u  (an achievable floor:
             every policy that always plays that power reaches it, so a
             state-dependent policy can do at least as well)
    J_rand = J_u of i.i.d. uniform-random actions over [action_min, action_max]

    delta  = J_min + alpha * (J_rand - J_min)          alpha in (0, 1)

alpha -> 0 : "only play the least-uncertain actions";  alpha -> 1 : no constraint.

Why not delta = frac * J_rand: that silently assumes J_min ~ 0. With
u_t = epistemic + aleatoric, the aleatoric part is roughly flat across laser
power and irreducible, so J_min is a large fraction of J_rand and
frac * J_rand can sit BELOW J_min — an infeasible budget, under which lambda
grows without bound. This script reports whether that is the case.

J_u uses exactly train.py's definition: J_u = mean_i sum_t gamma_u^t u_t^(i),
with u_t chosen by --uncertainty (total | epistemic), so --uncertainty and
--gamma_u MUST match the train.py run that consumes the delta.

Every sweep point (and the random policy) reuses the same per-episode
cool_time draws (common random numbers), so J_u differences across powers are
not cool_time noise.

Output
------
  <out_dir>/delta_calibration.json   — sweep table, J_min/J_rand, delta
  <out_dir>/delta_calibration.png    — J_u vs. constant power, with J_rand/delta lines
  stdout last line: "DELTA=<value>"  — machine-readable for job scripts

Usage
-----
    python -m online_RL_ucpg_v2.calibrate_delta \\
        --surrogate surrogate_model_v3/runs/<run>/surrogate_best.pt \\
        --action_min 100 --action_max 400 \\
        --uncertainty epistemic --alpha 0.4 \\
        --out_dir online_RL_ucpg_v2/runs/<run>_delta_calib
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.model import load_surrogate
from online_RL_ucpg_v2.env   import TwoStageLatentLPBFEnv
from online_RL_ucpg_v2.train import UNCERTAINTY_KEYS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate UCPG's delta from J_min / J_rand")

    p.add_argument("--surrogate", type=str, required=True)

    # ── environment (defaults mirror train.py — keep them in sync with the run) ──
    p.add_argument("--T_l",           type=float, default=2000.0)
    p.add_argument("--T_h",           type=float, default=2800.0)
    p.add_argument("--n_layers",      type=int,   default=12)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--mesh_path",     type=str,   default="surrogate_model/mesh.mat")
    p.add_argument("--width",         type=float, default=12.0)
    p.add_argument("--height",        type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end",   type=float, default=0.5)
    p.add_argument("--cool_time_min", type=float, default=0.05)
    p.add_argument("--cool_time_max", type=float, default=0.15)
    p.add_argument("--action_min",    type=float, default=100.0)
    p.add_argument("--action_max",    type=float, default=400.0)

    # ── calibration ────────────────────────────────────────────────────────────
    p.add_argument("--uncertainty", type=str, default="total", choices=list(UNCERTAINTY_KEYS),
                   help="Must match train.py's --uncertainty.")
    p.add_argument("--gamma_u",     type=float, default=0.99,
                   help="Must match train.py's --gamma_u.")
    p.add_argument("--alpha",       type=float, default=0.4,
                   help="delta = J_min + alpha * (J_rand - J_min).")
    p.add_argument("--sweep_step",  type=float, default=10.0,
                   help="Constant-power sweep spacing [W] over [action_min, action_max].")
    p.add_argument("--n_episodes",  type=int,   default=16,
                   help="Episodes per sweep point (and for the random policy).")
    p.add_argument("--n_episodes_rand", type=int, default=64,
                   help="Episodes for the uniform-random reference (noisier than a constant).")

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device",  type=str, default="")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def run_episodes(env, policy_fn, cool_times, n_layers, u_key, gamma_u):
    """Roll out one episode per entry of cool_times; return (J_u per ep, return per ep)."""
    disc = gamma_u ** np.arange(n_layers)
    j_u, ret = [], []
    for c in cool_times:
        env.reset()
        env._cool_time = float(c)          # common random numbers across sweep points
        u, r = np.zeros(n_layers), np.zeros(n_layers)
        for t in range(n_layers):
            _, reward, done, info = env.step(policy_fn(t))
            u[t], r[t] = info[u_key], reward
            if done:
                break
        j_u.append(float((disc * u).sum()))
        ret.append(float(r.sum()))
    return np.array(j_u), np.array(ret)


def main() -> None:
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in (0, 1).")
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    (surrogate, state_mean, state_std, lp_mean, lp_std,
     cool_mean, cool_std) = load_surrogate(args.surrogate, device=device)
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
    u_key = UNCERTAINTY_KEYS[args.uncertainty]

    print("=" * 65)
    print("[calibrate_delta] UCPG v2 — delta calibration from J_min / J_rand")
    print(f"[calibrate_delta] Surrogate   : {args.surrogate}")
    print(f"[calibrate_delta] Uncertainty : {args.uncertainty} (info['{u_key}'])  gamma_u={args.gamma_u}")
    print("=" * 65)

    cool_times = rng.uniform(args.cool_time_min, args.cool_time_max, size=args.n_episodes)

    # ── constant-power sweep → J_min ──────────────────────────────────────────
    powers = np.arange(args.action_min, args.action_max + 1e-6, args.sweep_step)
    sweep  = []
    print(f"{'Power [W]':>10} {'J_u mean':>12} {'J_u std':>10} {'return':>10}")
    for P in powers:
        j_u, ret = run_episodes(env, lambda t, P=P: float(P), cool_times,
                                args.n_layers, u_key, args.gamma_u)
        sweep.append({"power_W": float(P), "j_u_mean": float(j_u.mean()),
                      "j_u_std": float(j_u.std()), "return_mean": float(ret.mean())})
        print(f"{P:10.1f} {j_u.mean():12.6f} {j_u.std():10.6f} {ret.mean():10.4f}")

    j_sweep = np.array([s["j_u_mean"] for s in sweep])
    i_min   = int(j_sweep.argmin())
    J_min   = float(j_sweep[i_min])
    P_min   = float(powers[i_min])

    # ── uniform-random policy → J_rand ────────────────────────────────────────
    cool_rand = rng.uniform(args.cool_time_min, args.cool_time_max, size=args.n_episodes_rand)
    j_rand, _ = run_episodes(env, lambda t: float(rng.uniform(args.action_min, args.action_max)),
                             cool_rand, args.n_layers, u_key, args.gamma_u)
    J_rand = float(j_rand.mean())

    delta = J_min + args.alpha * (J_rand - J_min)
    spread = (J_rand - J_min) / J_rand if J_rand > 0 else 0.0

    print("-" * 65)
    print(f"[calibrate_delta] J_min  = {J_min:.6f}  (constant {P_min:.0f} W)")
    print(f"[calibrate_delta] J_rand = {J_rand:.6f}  ± {j_rand.std() / np.sqrt(len(j_rand)):.6f} (s.e.)")
    print(f"[calibrate_delta] (J_rand - J_min) / J_rand = {spread * 100:.1f}%  "
          f"— the share of J_u a policy can actually influence")
    if J_rand <= J_min:
        print("[calibrate_delta] WARNING: J_rand <= J_min — uncertainty does not discriminate "
              "between actions on this surrogate; the constraint cannot do anything useful.")
    elif spread < 0.1:
        print("[calibrate_delta] WARNING: < 10% of J_u is action-dependent — the constraint "
              "signal is weak (consider --uncertainty epistemic).")
    print(f"[calibrate_delta] delta  = J_min + {args.alpha} * (J_rand - J_min) = {delta:.6f}")

    out = {
        "surrogate": args.surrogate, "uncertainty": args.uncertainty,
        "gamma_u": args.gamma_u, "alpha": args.alpha,
        "J_min": J_min, "J_min_power_W": P_min, "J_rand": J_rand,
        "J_rand_se": float(j_rand.std() / np.sqrt(len(j_rand))),
        "delta": delta, "sweep": sweep,
    }
    json_path = os.path.join(args.out_dir, "delta_calibration.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calibrate_delta] Saved → {json_path}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(powers, j_sweep, yerr=[s["j_u_std"] for s in sweep], color="tab:red",
                marker="o", markersize=3, capsize=2, label="J_u, constant power")
    ax.axhline(J_rand, color="grey",       linestyle="--", label=f"J_rand = {J_rand:.4g}")
    ax.axhline(delta,  color="tab:purple", linestyle="-",  label=f"δ = {delta:.4g} (α={args.alpha})")
    ax.axhline(J_min,  color="tab:green",  linestyle=":",  label=f"J_min = {J_min:.4g} @ {P_min:.0f} W")
    ax.set_xlabel("Constant laser power [W]"); ax.set_ylabel(f"J_u ({args.uncertainty})")
    ax.set_title("UCPG v2 — δ calibration: uncertainty return vs. constant action")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, "delta_calibration.png")
    fig.savefig(png_path, dpi=150); plt.close(fig)
    print(f"[calibrate_delta] Saved → {png_path}")

    print(f"DELTA={delta:.6g}")


if __name__ == "__main__":
    main()
