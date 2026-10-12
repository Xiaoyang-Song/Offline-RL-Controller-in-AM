"""
online_RL_ucpg_v3/evaluate_real.py
--------------------------------------
REAL-physics evaluation (simulateHeatingCooling_v2.m via MATLAB, one PDE solve
per layer) of any mix of policy-gradient and Q-learning checkpoints, scoring
every method on BOTH reward fields:

  heat : ROI mean deviation of the END-OF-HEATING field — MATLAB's own
         meanDeviation (what every earlier real leaderboard reports)
  cool : ROI mean deviation of the POST-COOLING field u_final — what v3 is
         trained on — computed here with the same ROI mask/formula as the env

A per-layer consistency check recomputes the heating deviation in Python from
u_heat_final with that same mask and compares it with MATLAB's value, so the
post-cooling number is known to use the identical ROI definition.

Every method sees exactly the same episodes: one deterministic (greedy) episode
per value in --cool_times. The MATLAB runner is imported read-only from
online_RL_ucpg_v2.evaluate_real.

Usage
-----
    module load matlab
    python -m online_RL_ucpg_v3.evaluate_real \\
        --surrogate surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt \\
        --pg "PG (plain, cool)=online_RL_ucpg_v3/runs/narrow_pg_plain/pg_best.pt" \\
        --pg "UCPG v3 (shaped, cool)=online_RL_ucpg_v3/runs/narrow_pg_shaped/pg_best.pt" \\
        --q  "Q (plain, cool)=online_RL_ucpg_v3/runs/narrow_q_plain/q_best.pt" \\
        --q  "Q v3 (shaped, cool)=online_RL_ucpg_v3/runs/narrow_q_shaped/q_best.pt" \\
        --out_dir online_RL_ucpg_v3/results_real_narrow
"""

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.model import load_surrogate
from online_RL_ucpg_v2.evaluate_real import _build_params_dict, run_one_layer, _check_mesh_consistency
from baselines.common.eval_harness import StepContext, LatentAgentController
from online_RL_ucpg_v3.env import build_roi_masks, hinge_deviation


def parse_args():
    p = argparse.ArgumentParser(description="v3 real-physics evaluation on heat AND post-cooling deviation.")
    p.add_argument("--surrogate", required=True, help="Normalisation stats only (observation).")
    p.add_argument("--pg", action="append", default=[], metavar="NAME=CKPT",
                   help="Policy-gradient checkpoint (UCPGAgentV2 format: v3 PG, naive_pg, UCPG v2). Repeatable.")
    p.add_argument("--q", action="append", default=[], metavar="NAME=CKPT",
                   help="Online-Q checkpoint (LatentQNet format: v3 Q or baselines/online_q). Repeatable.")
    p.add_argument("--field", choices=["cool", "heat"], required=True,
                   help="Field the methods were trained on; the leaderboard is ranked on this deviation "
                        "(both are still saved per layer).")
    p.add_argument("--cool_times", type=float, nargs="+", default=[0.06, 0.10, 0.14],
                   help="One greedy episode per value; identical for every method.")
    p.add_argument("--T_l", type=float, default=2000.0)
    p.add_argument("--T_h", type=float, default=2800.0)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--width", type=float, default=12.0)
    p.add_argument("--height", type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end", type=float, default=0.5)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--cool_time_min", type=float, default=0.05)
    p.add_argument("--cool_time_max", type=float, default=0.15)
    p.add_argument("--action_min", type=float, default=100.0)
    p.add_argument("--action_max", type=float, default=400.0)
    p.add_argument("--matlab_timeout", type=float, default=300.0)
    p.add_argument("--mesh_path", type=str, default="surrogate_model/mesh.mat")
    p.add_argument("--work_dir", type=str, default="")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def _split(spec):
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def run_episode(ctrl, sm, ss, cool_time, masks, args, work_dir, check_mesh):
    device = sm.device
    fracs = np.linspace(args.sq_frac_start, args.sq_frac_end, args.n_layers)
    cool_tok = (cool_time - args.cool_time_min) / max(args.cool_time_max - args.cool_time_min, 1e-8)
    raw = np.full(sm.shape[0], args.initial_temp, dtype=np.float32)
    W = args.T_h - args.T_l
    rows = []
    for t in range(args.n_layers):
        with torch.no_grad():
            z = ((torch.tensor(raw, device=device) - sm) / ss).cpu().numpy()
        obs = np.concatenate([z, [t / max(args.n_layers - 1, 1), cool_tok]]).astype(np.float32)
        a = float(np.clip(ctrl.act(StepContext(obs=obs, z=z, raw_state=raw, layer=t, cool_time=cool_time)),
                          args.action_min, args.action_max))
        t0 = time.time()
        out = run_one_layer(_build_params_dict(a, cool_time, fracs[t], args.T_l, args.T_h, args.width, args.height),
                            t, work_dir, args.matlab_timeout, save_mesh=(check_mesh and t == 0))
        if check_mesh and t == 0:
            _check_mesh_consistency(out["nodes"], args.mesh_path)
        m = masks[t]
        sel = (lambda f: f[m]) if m is not None else (lambda f: f)
        dev_heat_py = float(hinge_deviation(sel(out["u_heat_final"]), args.T_l, args.T_h).mean())
        dev_cool = float(hinge_deviation(sel(out["u_final"]), args.T_l, args.T_h).mean())
        dev_heat = -out["reward"] * W                              # MATLAB's meanDeviation, in K
        rows.append(dict(layer=t + 1, action_W=a, dev_heat_K=dev_heat, dev_heat_python_K=dev_heat_py,
                         dev_cool_K=dev_cool))
        flag = "" if abs(dev_heat_py - dev_heat) <= 1e-3 * max(1.0, dev_heat) else "  <-- mask mismatch!"
        print(f"      layer {t + 1:2d}  LP={a:6.1f}W  dev_heat={dev_heat:7.1f}K (py {dev_heat_py:7.1f})  "
              f"dev_cool={dev_cool:7.1f}K  ({time.time() - t0:.0f}s){flag}")
        raw = out["u_final"]
    return rows


def main():
    args = parse_args()
    if shutil.which("matlab") is None:
        raise RuntimeError("matlab not on $PATH — run `module load matlab` first.")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    _s, sm, ss, *_ = load_surrogate(args.surrogate, device=device)
    sm, ss = sm.to(device), ss.to(device)
    masks = build_roi_masks(args.mesh_path, args.n_layers, args.width, args.height,
                            args.sq_frac_start, args.sq_frac_end)
    work_dir = args.work_dir or tempfile.mkdtemp(prefix="v3_real_eval_")
    os.makedirs(work_dir, exist_ok=True)

    from baselines.online_q.model import load_online_q_controller
    methods = ([(n, lambda p=p: LatentAgentController(p, device=device, greedy=True)) for n, p in map(_split, args.pg)] +
               [(n, lambda p=p: load_online_q_controller(p, device=device)) for n, p in map(_split, args.q)])
    if not methods:
        raise SystemExit("Pass at least one --pg or --q.")

    all_rows, summary = [], []
    first = True
    for name, make in methods:
        print(f"=== {name} ===")
        ctrl = make()
        ep_heat, ep_cool, ep_act = [], [], []
        for ct in args.cool_times:
            print(f"    episode cool_time={ct:.3f}s")
            rows = run_episode(ctrl, sm, ss, ct, masks, args, work_dir, check_mesh=first)
            first = False
            for r in rows:
                all_rows.append(dict(method=name, cool_time=ct, **r))
            ep_heat.append(np.mean([r["dev_heat_K"] for r in rows]))
            ep_cool.append(np.mean([r["dev_cool_K"] for r in rows]))
            ep_act.append(np.mean([r["action_W"] for r in rows]))
        summary.append(dict(method=name, dev_heat_K_mean=float(np.mean(ep_heat)), dev_heat_K_std=float(np.std(ep_heat)),
                            dev_cool_K_mean=float(np.mean(ep_cool)), dev_cool_K_std=float(np.std(ep_cool)),
                            action_W_mean=float(np.mean(ep_act)), n_episodes=len(args.cool_times)))
        print(f"    → {name}: {args.field} deviation {summary[-1][f'dev_{args.field}_K_mean']:.1f} K")
        # write after every method (real solves are slow)
        with open(os.path.join(args.out_dir, "per_layer_real.json"), "w") as f:
            json.dump(all_rows, f, indent=1)
        with open(os.path.join(args.out_dir, "leaderboard_real_v3.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            for s in sorted(summary, key=lambda s: s[f"dev_{args.field}_K_mean"]):
                w.writerow(s)

    fname = "post-cooling" if args.field == "cool" else "end-of-heating"
    print(f"\nREAL-physics leaderboard — {fname} ROI deviation per layer [K], lower is better")
    print(f"{'method':<34}{'deviation':>18}{'mean LP [W]':>14}")
    for s in sorted(summary, key=lambda s: s[f"dev_{args.field}_K_mean"]):
        print(f"{s['method']:<34}{s[f'dev_{args.field}_K_mean']:>10.1f} ± {s[f'dev_{args.field}_K_std']:<5.1f}"
              f"{s['action_W_mean']:>10.1f}")
    print(f"\nSaved → {args.out_dir}/leaderboard_real_v3.csv, per_layer_real.json")
    if not args.work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
