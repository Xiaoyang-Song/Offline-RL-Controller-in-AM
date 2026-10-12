"""
online_RL_ucpg_v3/common.py
-------------------------------
Environment CLI + construction shared by train_pg.py and train_q.py, so both
algorithms see exactly the same environment, reward and trust scale.
"""

import argparse

from surrogate_model_v3.model import load_surrogate
from online_RL_ucpg_v3.env import UncertaintyShapedLPBFEnv, calibrate_tau


def add_env_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--surrogate", type=str, required=True)
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
    p.add_argument("--ood_min", type=float, default=None,
                   help="Surrogate's trained laser-power range (diagnostic OOD-action fraction, and the "
                        "in-distribution range used to calibrate tau). Give with --ood_max.")
    p.add_argument("--ood_max", type=float, default=None)

    # v3 reward
    p.add_argument("--reward_field", choices=["cool", "heat"], default="cool",
                   help="'cool' = post-cooling field s_{t+1} (default); 'heat' = end-of-heating field (v2).")
    p.add_argument("--reward_mode", choices=["shaped", "plain"], default="shaped",
                   help="'shaped' = uncertainty-shaped per-node reward; 'plain' = no uncertainty "
                        "(the matched naive baseline).")
    p.add_argument("--tau", type=float, default=None,
                   help="Trust scale [K]. Default: tau_k x median in-distribution ROI epistemic sigma "
                        "of multi-step member rollouts (needs --ood_min/--ood_max).")
    p.add_argument("--tau_k", type=float, default=2.0)
    p.add_argument("--pess_dev", type=float, default=None,
                   help="Penalty c [K] for a fully untrusted node. Default: T_h - T_l.")
    p.add_argument("--shaping", choices=["additive", "shrink"], default="additive",
                   help="'additive': d + (1-w) c (never rewards uncertainty, default); "
                        "'shrink': w d + (1-w) c (exploitable when d > c; for comparison only).")


def make_env(args, device: str):
    (surrogate, sm, ss, lm, ls, cm, cs) = load_surrogate(args.surrogate, device=device)
    surrogate.eval()
    env = UncertaintyShapedLPBFEnv(
        surrogate=surrogate, state_mean=sm, state_std=ss, lp_mean=lm, lp_std=ls,
        cool_mean=cm, cool_std=cs, temp_range=(args.T_l, args.T_h), n_layers=args.n_layers,
        initial_temp=args.initial_temp, device=device, mesh_path=args.mesh_path,
        width=args.width, height=args.height, sq_frac_start=args.sq_frac_start, sq_frac_end=args.sq_frac_end,
        action_min=args.action_min, action_max=args.action_max,
        cool_time_min=args.cool_time_min, cool_time_max=args.cool_time_max,
        reward_field=args.reward_field, reward_mode="plain", tau=None, pess_dev=args.pess_dev,
        shaping=args.shaping,
    )
    # trust scale — computed for BOTH modes so a plain run logs the same shaped diagnostic
    if args.tau is not None:
        tau, med = args.tau, float("nan")
    elif args.ood_min is not None and args.ood_max is not None:
        tau, med = calibrate_tau(env, args.ood_min, args.ood_max, k=args.tau_k)
    else:
        if args.reward_mode == "shaped":
            raise ValueError("--reward_mode shaped needs --tau, or --ood_min/--ood_max to calibrate it.")
        tau, med = None, float("nan")
    env.tau = tau
    env.reward_mode = args.reward_mode
    print(f"[v3] reward_field={args.reward_field}  reward_mode={args.reward_mode}  "
          f"tau={tau if tau is None else f'{tau:.3f} K'}  (median ID sigma {med:.3f} K, k={args.tau_k})  "
          f"pess_dev={env.pess_dev:.1f} K  shaping={env.shaping}  members={env.K}")
    return env, dict(tau=tau, median_id_sigma=med)
