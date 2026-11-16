"""
rl_narrow/evaluate_real.py
--------------------------
Real-physics evaluation (simulateHeatingCooling_v2.m via MATLAB, one PDE solve per layer, heatTime
0.025 s as in the data) of ONE controller, on the same episodes for every method: one deterministic
episode per cooling time in --cool_times. Writes <out_dir>/<slug>.json, so methods can run as a SLURM
array and be combined by rl_narrow/aggregate.py. The real EoH / EoC fields of every
layer are saved to <slug>_fields.npz (for the deviation-map figure, rl_narrow/plot_maps.py).

Per layer it records the action and, on the END-OF-HEATING field (what the policies are trained on):
  dev_heat_K   : MATLAB's ROI mean hinge deviation [K]  (dev_heat_py_K: same, recomputed in Python)
  score_heat   : ROI mean exp(-d / W_T)  (method.tex reward with w = 1; the shared quality score)
and, for reference, the same two quantities on the post-cooling field.

--method TYPE:NAME=PATH, TYPE in
  pg      : policy-gradient checkpoint (online_RL_ucpg_v3 / naive_pg format), greedy mean action
  q       : online-Q checkpoint (online_RL_ucpg_v3 / baselines.online_q format), greedy
  offline : rl_narrow.offline_rl checkpoint (FQI / CQL / IQL), greedy
  prop    : baselines.proportional fitted .pt
  kalman  : baselines.kalman_particle fitted .pt (sensor noise R = --kalman_R, seed = --seed)
  const   : PATH is the constant laser power [W]

    module load matlab
    python -m rl_narrow.evaluate_real --method "pg:PG shaped seed 1=rl_narrow/runs/pg_shaped_s1/pg_best.pt" \
        --out_dir rl_narrow/results_real
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.model import load_surrogate
from online_RL_ucpg_v2.evaluate_real import _build_params_dict, run_one_layer, _check_mesh_consistency, FIXED_PARAMS
from baselines.common.eval_harness import StepContext
from online_RL_ucpg_v3.env import build_roi_masks, hinge_deviation


def parse_args():
    p = argparse.ArgumentParser(description="Real-physics evaluation of one controller (narrow experiment).")
    p.add_argument("--method", required=True, help="TYPE:NAME=PATH (see module docstring).")
    p.add_argument("--surrogate", default="surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt",
                   help="Normalisation stats for the observation of pg / q policies.")
    p.add_argument("--cool_times", type=float, nargs="+", default=[0.05, 0.07, 0.09, 0.11, 0.13, 0.15])
    p.add_argument("--kalman_R", type=float, default=2500.0)
    p.add_argument("--seed", type=int, default=0)
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
    p.add_argument("--mesh_path", default="surrogate_model/mesh.mat")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="")
    return p.parse_args()


def parse_method(spec):
    typ, rest = spec.split(":", 1)
    name, path = rest.rsplit("=", 1)
    return typ.strip(), name.strip(), path.strip()


def make_controller(typ, path, args, device):
    if typ == "pg":
        from baselines.common.eval_harness import LatentAgentController
        return LatentAgentController(path, device=device, greedy=True)
    if typ == "q":
        from baselines.online_q.model import load_online_q_controller
        return load_online_q_controller(path, device=device)
    if typ == "offline":
        from rl_narrow.offline_rl import OfflineController
        return OfflineController(path, device=device)
    if typ == "prop":
        from baselines.proportional.controller import load_proportional_controller
        return load_proportional_controller(path)
    if typ == "kalman":
        from baselines.kalman_particle.filters import load_kalman_controller
        return load_kalman_controller(path, R=args.kalman_R, seed=args.seed)
    if typ == "const":
        from baselines.constant.controller import ConstantController
        return ConstantController(float(path))
    raise ValueError(f"unknown method type {typ!r}")


def main():
    args = parse_args()
    if shutil.which("matlab") is None:
        raise RuntimeError("matlab not on $PATH — run `module load matlab` first.")
    assert abs(FIXED_PARAMS["heatTime"] - 0.025) < 1e-12, f"heatTime {FIXED_PARAMS['heatTime']} != 0.025 s (data)"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    typ, name, path = parse_method(args.method)
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    _s, sm, ss, *_ = load_surrogate(args.surrogate, device=device)
    sm, ss = sm.to(device), ss.to(device)
    masks = build_roi_masks(args.mesh_path, args.n_layers, args.width, args.height,
                            args.sq_frac_start, args.sq_frac_end)
    fracs = np.linspace(args.sq_frac_start, args.sq_frac_end, args.n_layers)
    W = args.T_h - args.T_l
    ctrl = make_controller(typ, path, args, device)
    work_dir = tempfile.mkdtemp(prefix=f"rlnarrow_{slug}_")
    print(f"=== {name} ({typ}: {path}) | heatTime {FIXED_PARAMS['heatTime']} s | cool times {args.cool_times}")

    rows, t_start = [], time.time()
    U_heat = np.full((len(args.cool_times), args.n_layers, sm.shape[0]), np.nan, np.float32)   # real EoH fields
    U_cool = np.full_like(U_heat, np.nan)                                                   # real EoC fields
    for e, ct in enumerate(args.cool_times):
        if hasattr(ctrl, "reset"):
            ctrl.reset()
        cool_tok = (ct - args.cool_time_min) / max(args.cool_time_max - args.cool_time_min, 1e-8)
        raw = np.full(sm.shape[0], args.initial_temp, dtype=np.float32)
        for t in range(args.n_layers):
            with torch.no_grad():
                z = ((torch.tensor(raw, device=device) - sm) / ss).cpu().numpy()
            obs = np.concatenate([z, [t / max(args.n_layers - 1, 1), cool_tok]]).astype(np.float32)
            a_raw = float(ctrl.act(StepContext(obs=obs, z=z, raw_state=raw, layer=t, cool_time=ct)))
            a = float(np.clip(a_raw, args.action_min, args.action_max))
            out = run_one_layer(_build_params_dict(a, ct, fracs[t], args.T_l, args.T_h, args.width, args.height),
                                t, work_dir, args.matlab_timeout, save_mesh=(e == 0 and t == 0))
            if e == 0 and t == 0:
                _check_mesh_consistency(out["nodes"], args.mesh_path)
            m = masks[t]
            dh = hinge_deviation(out["u_heat_final"][m], args.T_l, args.T_h)
            dc = hinge_deviation(out["u_final"][m], args.T_l, args.T_h)
            rows.append(dict(cool_time=ct, layer=t + 1, action_W=a, action_raw_W=a_raw,
                             dev_heat_K=-out["reward"] * W, dev_heat_py_K=float(dh.mean()),
                             score_heat=float(np.exp(-dh / W).mean()),
                             dev_cool_K=float(dc.mean()), score_cool=float(np.exp(-dc / W).mean()),
                             roi_mean_heat_K=float(out["u_heat_final"][m].mean())))
            r = rows[-1]
            print(f"  c={ct:.2f}s layer {t + 1:2d}  LP {a:6.1f} W  dev_heat {r['dev_heat_K']:7.1f} K "
                  f"(py {r['dev_heat_py_K']:7.1f})  score {r['score_heat']:.3f}  [{time.time() - t_start:.0f}s]")
            U_heat[e, t], U_cool[e, t] = out["u_heat_final"], out["u_final"]
            raw = out["u_final"]
        np.savez_compressed(os.path.join(args.out_dir, f"{slug}_fields.npz"), u_heat=U_heat, u_final=U_cool,
                            cool_times=np.asarray(args.cool_times),
                            actions=np.array([[r["action_W"] for r in rows if r["cool_time"] == c]
                                              + [np.nan] * (args.n_layers - sum(r["cool_time"] == c for r in rows))
                                              for c in args.cool_times]))
        with open(os.path.join(args.out_dir, f"{slug}.json"), "w") as f:     # after every episode
            json.dump(dict(method=name, type=typ, path=path, cool_times=args.cool_times, rows=rows,
                           complete=(e == len(args.cool_times) - 1)), f, indent=1)
    ep = [np.mean([r["score_heat"] for r in rows if r["cool_time"] == ct]) for ct in args.cool_times]
    print(f"→ {name}: EoH score {np.mean(ep):.4f} ± {np.std(ep):.4f}, "
          f"EoH deviation {np.mean([r['dev_heat_K'] for r in rows]):.1f} K")
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
