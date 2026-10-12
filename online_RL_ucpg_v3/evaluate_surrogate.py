"""
online_RL_ucpg_v3/evaluate_surrogate.py
-------------------------------------------
SURROGATE evaluation of PG / Q checkpoints in the v3 environment (member
rollouts), scored with the PLAIN (unpenalised) reward — the true ROI deviation
of the member-mean field on --field — never with the shaped training reward.

Every method sees exactly the same episodes (same cool_time sequence, fixed
seed), acts greedily, and actions are clipped to [action_min, action_max] as in
the real-physics evaluator. Also reports the mean ROI epistemic sigma and the
fraction of out-of-distribution actions.

Writes <out_dir>/per_episode_surrogate.json and leaderboard_surrogate_v3.csv;
plot_comparison.py picks these up next to the real-physics results.

Usage
-----
    python -m online_RL_ucpg_v3.evaluate_surrogate \\
        --surrogate surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt --field cool \\
        --ood_min 200 --ood_max 300 \\
        --pg "UCPG v3 shaped (cool)=online_RL_ucpg_v3/runs/narrow_pg_shaped_cool/pg_best.pt" ... \\
        --out_dir online_RL_ucpg_v3/results_real_narrow_cool
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.model import load_surrogate
from baselines.common.eval_harness import StepContext, LatentAgentController
from online_RL_ucpg_v3.env import UncertaintyShapedLPBFEnv


def parse_args():
    p = argparse.ArgumentParser(description="v3 surrogate evaluation with the plain (unpenalised) reward.")
    p.add_argument("--surrogate", required=True)
    p.add_argument("--field", choices=["cool", "heat"], required=True)
    p.add_argument("--pg", action="append", default=[], metavar="NAME=CKPT")
    p.add_argument("--q", action="append", default=[], metavar="NAME=CKPT")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--ood_min", type=float, default=None)
    p.add_argument("--ood_max", type=float, default=None)
    p.add_argument("--T_l", type=float, default=2000.0)
    p.add_argument("--T_h", type=float, default=2800.0)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--mesh_path", type=str, default="surrogate_model/mesh.mat")
    p.add_argument("--cool_time_min", type=float, default=0.05)
    p.add_argument("--cool_time_max", type=float, default=0.15)
    p.add_argument("--action_min", type=float, default=100.0)
    p.add_argument("--action_max", type=float, default=400.0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    s, sm, ss, lm, ls, cm, cs = load_surrogate(args.surrogate, device=device)
    s.eval()
    env = UncertaintyShapedLPBFEnv(s, sm, ss, lm, ls, cm, cs, temp_range=(args.T_l, args.T_h),
                                   n_layers=args.n_layers, device=device, mesh_path=args.mesh_path,
                                   cool_time_min=args.cool_time_min, cool_time_max=args.cool_time_max,
                                   reward_field=args.field, reward_mode="plain")
    cool_times = np.random.default_rng(args.seed).uniform(args.cool_time_min, args.cool_time_max, args.n_episodes)
    W = args.T_h - args.T_l

    from baselines.online_q.model import load_online_q_controller
    split = lambda spec: [x.strip() for x in spec.split("=", 1)]
    methods = ([(n, lambda p=p: LatentAgentController(p, device=device, greedy=True)) for n, p in map(split, args.pg)] +
               [(n, lambda p=p: load_online_q_controller(p, device=device)) for n, p in map(split, args.q)])
    if not methods:
        raise SystemExit("Pass at least one --pg or --q.")
    have_ood = args.ood_min is not None and args.ood_max is not None

    episodes, summary = [], []
    for name, make in methods:
        ctrl = make()
        devs, sigs, oods, acts = [], [], [], []
        for ct in cool_times:
            obs = env.reset(cool_time=float(ct))
            d_ep, s_ep, a_ep = [], [], []
            for t in range(args.n_layers):
                ctx = StepContext(obs=obs, z=obs[:-2], raw_state=None, layer=t, cool_time=float(ct))
                a = float(np.clip(ctrl.act(ctx), args.action_min, args.action_max))
                obs, _r, _d, info = env.step(a)
                d_ep.append(-info["reward_plain"] * W)            # plain deviation, K — never the shaped reward
                s_ep.append(info["sigma_epi_roi_K"]); a_ep.append(a)
            a_ep = np.array(a_ep)
            devs.append(float(np.mean(d_ep))); sigs.append(float(np.mean(s_ep))); acts.append(float(a_ep.mean()))
            oods.append(float(((a_ep < args.ood_min) | (a_ep > args.ood_max)).mean()) if have_ood else float("nan"))
            episodes.append(dict(method=name, cool_time=float(ct), dev_K=devs[-1], sigma_K=sigs[-1],
                                 action_W=acts[-1], ood_frac=oods[-1], actions=a_ep.tolist()))
        summary.append(dict(method=name, dev_K_mean=float(np.mean(devs)), dev_K_std=float(np.std(devs)),
                            sigma_K_mean=float(np.mean(sigs)), ood_frac_mean=float(np.mean(oods)),
                            action_W_mean=float(np.mean(acts)), n_episodes=len(cool_times)))
        print(f"  {name:<34} deviation {summary[-1]['dev_K_mean']:7.1f} ± {summary[-1]['dev_K_std']:5.1f} K | "
              f"sigma {summary[-1]['sigma_K_mean']:6.2f} K | OOD {100 * summary[-1]['ood_frac_mean']:5.1f}% | "
              f"LP {summary[-1]['action_W_mean']:.1f} W")

    with open(os.path.join(args.out_dir, "per_episode_surrogate.json"), "w") as f:
        json.dump(dict(field=args.field, episodes=episodes), f, indent=1)
    with open(os.path.join(args.out_dir, "leaderboard_surrogate_v3.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for r in sorted(summary, key=lambda r: r["dev_K_mean"]):
            w.writerow(r)
    print(f"Saved → {args.out_dir}/leaderboard_surrogate_v3.csv, per_episode_surrogate.json")


if __name__ == "__main__":
    main()
