"""
online_RL_ucpg_v2/evaluate_real.py
--------------------------------------
Evaluate a trained UCPG v2 policy checkpoint against the REAL PDE simulator
(simulateHeatingCooling_v2.m), not the neural surrogate — the ground-truth
check for "did this policy actually solve the process, or just learn to
exploit the surrogate it was trained against."

Works identically for a naive_pg checkpoint or a real UCPG v2 checkpoint —
both save/load via UCPGAgentV2, so this script never needs to know which
one it was handed; only --checkpoint changes between runs.

Mechanism (see ../test.py for the v1 precedent this follows)
------------------------------------------------------------
Each layer is one SEPARATE `matlab -batch` subprocess call (not the Python
matlab.engine bridge — simpler, and already proven to work in this repo):
  1. Write this layer's paramsStruct (physics constants + this layer's LP +
     this trajectory's fixed cool_time + this layer's squareSideFraction)
     to a .mat file.
  2. Write a small wrapper .m script that loads it, sets the initial
     condition (300 K scalar for layer 0; the PREVIOUS layer's cooling
     PDE-solution OBJECT for later layers — this can only be reconstructed
     inside MATLAB itself, which is why the IC-chaining logic lives in the
     generated .m script rather than in the .mat file Python writes), calls
     simulateHeatingCooling_v2, and saves uFinal/uHeatFinal/meanDeviation/
     the next IC object back out.
  3. Run it via subprocess, read the results back.

The POLICY only ever needs the surrogate's ENCODER (+ its normalisation
stats) here — to turn the real MATLAB field into z_t for the latent policy
input. The surrogate's TRANSITION model is never called; every actual state
transition is the real PDE solve. Reward is read directly from MATLAB's own
meanDeviation output, which simulateHeatingCooling_v2.m already computes
from uHeatFinal internally — no reward recomputation needed here.

A one-time sanity check compares the live simulation's mesh node
coordinates against surrogate_model/mesh.mat's stored ones on the first
layer, since the surrogate's encoder assumes a specific node ORDER — if
MATLAB's mesh generation isn't deterministic across runs, the encoding
would be silently wrong without this check.

Usage
-----
    module load matlab   # must be on $PATH before running
    python -m online_RL_ucpg_v2.evaluate_real \\
        --checkpoint online_RL_ucpg_v2/runs/<ts>/ucpg_best.pt \\
        --surrogate  surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt \\
        --n_episodes 1 --cool_time 0.10 --plot
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy.io
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_latent_uncertainty_v2.train import load_two_stage_surrogate
from online_RL_ucpg_v2.agent import UCPGAgentV2

LPBF_SIM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "LPBF-Simulation")
)
SIMULATION_V2_DIR = os.path.join(LPBF_SIM_ROOT, "simulation_v2")

# Physics constants — fixed, matching multilayer_random_v2.m exactly (the
# script the surrogate's own training data was generated with). These are
# NOT exposed as CLI flags: they aren't experiment variables, they're the
# process's material/simulation parameters, and using anything else would
# silently evaluate a different (non-matching) physical setup.
FIXED_PARAMS = dict(
    k=0.0067, rho=0.00433, specificHeat=0.526,
    thick=4.0,                      # NOTE: v2 uses 4.0, not v1's 2.0 — see multilayer_random_v2.m
    hmax=0.4, style="simultaneous",
    heatTime=0.05, nTimeStepsHeat=50, nTimeStepsCool=50,
    doPlot=False,
    ss_fixed=600.0,                  # scan speed — fixed, not policy-controlled (unused by the surrogate too)
    eeta=0.3, r_b=0.06, H=0.1,
)


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained UCPG v2 (or naive_pg) checkpoint against the REAL "
                    "PDE simulator, not the neural surrogate."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="UCPGAgentV2-format checkpoint — works for naive_pg or real UCPG v2 alike.")
    p.add_argument("--surrogate",  type=str, required=True,
                   help="surrogate_model_latent_uncertainty_v2 checkpoint — used ONLY for its "
                        "encoder + normalisation stats (to build the policy's latent observation "
                        "from the real MATLAB field). Its transition model is never called.")

    # ── environment / reward window (must match training) ────────────────────
    p.add_argument("--T_l",      type=float, default=2000.0)
    p.add_argument("--T_h",      type=float, default=2800.0)
    p.add_argument("--n_layers", type=int,   default=12)
    p.add_argument("--width",    type=float, default=12.0)
    p.add_argument("--height",   type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end",   type=float, default=0.5)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--cool_time_min", type=float, default=0.05,
                   help="Only used to build the policy's cool_time observation token — "
                        "must match what the policy was trained with.")
    p.add_argument("--cool_time_max", type=float, default=0.15)

    # ── this evaluation's cool_time (real, physically applied — fixed per episode) ──
    p.add_argument("--cool_time", type=float, default=0.10,
                   help="Fixed cool_time [s] applied for every episode (same value used for "
                        "both naive_pg and UCPG checkpoints keeps the comparison fair). "
                        "Ignored if --random_cool_time is given.")
    p.add_argument("--random_cool_time", action="store_true",
                   help="Draw a fresh cool_time ~ Uniform(cool_time_min, cool_time_max) for "
                        "each episode instead of using the fixed --cool_time value.")

    # ── evaluation settings ────────────────────────────────────────────────────
    p.add_argument("--n_episodes", type=int, default=1,
                   help="Number of GREEDY episodes to run. Real PDE solves are slow "
                        "(~12 MATLAB calls/episode) — keep this small.")
    p.add_argument("--also_stochastic", action="store_true",
                   help="Also run --n_episodes STOCHASTIC (sampled) episodes. Off by default "
                        "given the cost; greedy is what matters for a deployment check.")
    p.add_argument("--matlab_timeout", type=float, default=300.0,
                   help="Per-layer MATLAB subprocess timeout [s].")
    p.add_argument("--work_dir", type=str, default="",
                   help="Scratch directory for per-layer .mat/.m files. Default: a fresh "
                        "temp directory (auto-cleaned on exit).")
    p.add_argument("--mesh_path", type=str, default="surrogate_model/mesh.mat",
                   help="Used only for the one-time mesh-consistency sanity check.")
    p.add_argument("--results_out", type=str, default="",
                   help="Optional path to save per-layer rewards/actions as JSON.")
    p.add_argument("--plot", action="store_true",
                   help="Save a per-layer reward/action plot (reuses baselines' plotting helper).")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed",   type=int, default=123)
    return p.parse_args()


# =============================================================================
# MATLAB call — one layer
# =============================================================================

def _build_params_dict(lp_W: float, cool_time_s: float, sq_frac: float,
                       T_l: float, T_h: float, width: float, height: float) -> dict:
    return {
        "k": FIXED_PARAMS["k"], "rho": FIXED_PARAMS["rho"], "specificHeat": FIXED_PARAMS["specificHeat"],
        "ic": 300.0,   # placeholder — overridden inside the .m script for layer > 0, see module docstring
        "thick": FIXED_PARAMS["thick"], "width": width, "height": height, "hmax": FIXED_PARAMS["hmax"],
        "squareSideFraction": sq_frac,
        "scan_pattern": np.linspace(0.0, width, 48),
        "style": FIXED_PARAMS["style"],
        "params": {
            "SS": FIXED_PARAMS["ss_fixed"], "LP": lp_W,
            "eeta": FIXED_PARAMS["eeta"], "r_b": FIXED_PARAMS["r_b"], "H": FIXED_PARAMS["H"],
        },
        "heatTime": FIXED_PARAMS["heatTime"], "coolTime": cool_time_s,
        "nTimeStepsHeat": FIXED_PARAMS["nTimeStepsHeat"], "nTimeStepsCool": FIXED_PARAMS["nTimeStepsCool"],
        "doPlot": FIXED_PARAMS["doPlot"],
        "tempRange": np.array([T_l, T_h], dtype=np.float64),
    }


def _to_matlab_numeric(obj):
    if isinstance(obj, dict):
        return {k: _to_matlab_numeric(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj if obj.dtype == np.bool_ else obj.astype(np.float64)
    if isinstance(obj, (np.integer, int, np.floating, float)):
        return np.float64(obj)
    return obj


def run_one_layer(
    params_dict: dict, layer_idx: int, work_dir: str, matlab_timeout: float,
    save_mesh: bool = False,
) -> dict:
    """
    Runs exactly one heating+cooling PDE solve via a fresh `matlab -batch`
    subprocess. Returns dict with u_final, u_heat_final (raw Kelvin numpy
    arrays, flattened), reward, and (if save_mesh) nodes for the one-time
    mesh sanity check.
    """
    params_path  = os.path.join(work_dir, f"params_{layer_idx:02d}.mat")
    results_path = os.path.join(work_dir, f"results_{layer_idx:02d}.mat")
    prev_results_path = os.path.join(work_dir, f"results_{layer_idx-1:02d}.mat")
    script_path  = os.path.join(work_dir, f"run_layer_{layer_idx:02d}.m")

    scipy.io.savemat(params_path, {"paramsStruct": _to_matlab_numeric(params_dict)})

    ic_lines = (
        "paramsStruct.ic = 300.0;"
        if layer_idx == 0 else
        f"prevData = load('{prev_results_path}', 'resultCool'); "
        f"paramsStruct.ic = prevData.resultCool;"
    )
    mesh_save_line = ", 'nodes'" if save_mesh else ""
    mesh_capture_line = "nodes = model.Mesh.Nodes;" if save_mesh else ""

    script = f"""
try
    cd('{SIMULATION_V2_DIR}');
    addpath('{LPBF_SIM_ROOT}');
    paramsStruct = load('{params_path}').paramsStruct;
    {ic_lines}
    [uFinal, tAll, uAll, resultAll, model, meanDeviation, uHeatFinal] = simulateHeatingCooling_v2(paramsStruct);
    resultCool = resultAll(2);
    {mesh_capture_line}
    save('{results_path}', 'uFinal', 'uHeatFinal', 'meanDeviation', 'resultCool'{mesh_save_line});
    exit(0);
catch e
    disp(getReport(e));
    exit(1);
end
"""
    with open(script_path, "w") as f:
        f.write(script)

    result = subprocess.run(
        ["matlab", "-nodisplay", "-nosplash", "-batch", f"run('{script_path}')"],
        timeout=matlab_timeout, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MATLAB failed on layer {layer_idx} (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    res = scipy.io.loadmat(results_path)
    out = dict(
        u_final=np.asarray(res["uFinal"], dtype=np.float32).reshape(-1),
        u_heat_final=np.asarray(res["uHeatFinal"], dtype=np.float32).reshape(-1),
        reward=-float(np.asarray(res["meanDeviation"]).reshape(-1)[0]),
    )
    if save_mesh:
        out["nodes"] = np.asarray(res["nodes"], dtype=np.float64)
    return out


def _check_mesh_consistency(live_nodes: np.ndarray, mesh_path: str) -> None:
    if not os.path.exists(mesh_path):
        print(f"[evaluate_real] WARNING: mesh_path '{mesh_path}' not found — skipping "
              f"mesh-consistency check. If the encoder's node order doesn't match the live "
              f"simulation's, results will be silently wrong.")
        return
    stored = scipy.io.loadmat(mesh_path)["nodes"]
    stored = stored.T if stored.shape[0] == 2 else stored
    live   = live_nodes.T if live_nodes.shape[0] == 2 else live_nodes
    if stored.shape != live.shape:
        print(f"[evaluate_real] WARNING: mesh node count mismatch — stored {stored.shape} "
              f"vs live {live.shape}. Encoder input will be malformed. Check mesh generation "
              f"determinism (hmax, geometry) between the surrogate's mesh.mat and this run.")
        return
    if not np.allclose(stored, live, atol=1e-4):
        max_diff = np.abs(stored - live).max()
        print(f"[evaluate_real] WARNING: live mesh node coordinates differ from "
              f"surrogate_model/mesh.mat by up to {max_diff:.6f} — the encoder assumes a "
              f"specific node ORDER, so any real mismatch here silently corrupts the policy's "
              f"observation. Investigate before trusting these results.")
    else:
        print(f"[evaluate_real] Mesh consistency check passed (matches surrogate_model/mesh.mat).")


# =============================================================================
# Episode rollout
# =============================================================================

def run_episode(
    agent: UCPGAgentV2, surrogate, state_mean, state_std,
    cool_time_s: float, greedy: bool, args, work_dir: str, check_mesh: bool,
):
    """Returns (actions[T], rewards[T])."""
    device = state_mean.device
    state_dim = state_mean.shape[0]
    n_layers = args.n_layers
    fracs = np.linspace(args.sq_frac_start, args.sq_frac_end, n_layers)
    cool_token = (cool_time_s - args.cool_time_min) / max(args.cool_time_max - args.cool_time_min, 1e-8)

    raw_state = np.full(state_dim, args.initial_temp, dtype=np.float32)
    actions, rewards = [], []

    for t in range(n_layers):
        with torch.no_grad():
            s_t = torch.tensor(raw_state, dtype=torch.float32, device=device).unsqueeze(0)
            s_n = (s_t - state_mean) / state_std
            z_t = surrogate.encode(s_n).squeeze(0).cpu().numpy()

        layer_token = t / max(n_layers - 1, 1)
        obs = np.concatenate([z_t, [layer_token, cool_token]]).astype(np.float32)
        a = agent.select_action(obs, explore=not greedy)

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

        print(f"    layer {t+1:2d}/{n_layers}  LP={a:7.2f}W  reward={out['reward']:+.4f}  "
              f"({elapsed:.0f}s)")

    return np.array(actions), np.array(rewards)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if shutil.which("matlab") is None:
        raise RuntimeError(
            "matlab not found on $PATH. Run `module load matlab` before this script "
            "(and make sure it's in the same shell/job — see jobs/ for an sbatch example)."
        )

    print("=" * 65)
    print("  UCPG v2 — REAL PHYSICS Evaluation (simulateHeatingCooling_v2.m)")
    print("=" * 65)
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Surrogate   : {args.surrogate}  (encoder + norm stats only)")
    print(f"  Episodes    : {args.n_episodes} greedy"
          + (f" + {args.n_episodes} stochastic" if args.also_stochastic else ""))
    print("=" * 65)

    (surrogate, state_mean, state_std, _lp_mean, _lp_std,
     _cool_mean, _cool_std, _roi) = load_two_stage_surrogate(args.surrogate, device=device)
    surrogate.eval()
    state_mean = state_mean.to(device)
    state_std  = state_std.to(device)

    agent = UCPGAgentV2.load(args.checkpoint, device=device)
    print(f"  {agent}\n")

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="ucpg_v2_real_eval_")
    os.makedirs(work_dir, exist_ok=True)
    print(f"  Scratch dir : {work_dir}\n")

    def _run_many(greedy: bool, n: int):
        all_actions, all_rewards = [], []
        for ep in range(n):
            cool_time_s = (
                float(np.random.uniform(args.cool_time_min, args.cool_time_max))
                if args.random_cool_time else args.cool_time
            )
            label = "GREEDY" if greedy else "STOCHASTIC"
            print(f"  [{label}] Episode {ep+1}/{n}  (cool_time={cool_time_s:.3f}s)")
            actions, rewards = run_episode(
                agent, surrogate, state_mean, state_std, cool_time_s, greedy, args,
                work_dir, check_mesh=(ep == 0),
            )
            print(f"    → episode return: {rewards.sum():+.4f}\n")
            all_actions.append(actions); all_rewards.append(rewards)
        return np.stack(all_actions), np.stack(all_rewards)

    t_start = time.time()
    actions_g, rewards_g = _run_many(greedy=True, n=args.n_episodes)
    actions_s, rewards_s = (None, None)
    if args.also_stochastic:
        actions_s, rewards_s = _run_many(greedy=False, n=args.n_episodes)
    total_time = time.time() - t_start

    print("=" * 65)
    print(f"  GREEDY return     : mean={rewards_g.sum(axis=1).mean():+.4f}  "
          f"std={rewards_g.sum(axis=1).std():.4f}")
    print(f"  GREEDY action     : mean={actions_g.mean():.1f}W  std={actions_g.std():.1f}W")
    if args.also_stochastic:
        print(f"  STOCHASTIC return : mean={rewards_s.sum(axis=1).mean():+.4f}  "
              f"std={rewards_s.sum(axis=1).std():.4f}")
        print(f"  STOCHASTIC action : mean={actions_s.mean():.1f}W  std={actions_s.std():.1f}W")
    print(f"  Total wall time   : {total_time/60:.1f} min")
    print("=" * 65)

    if args.results_out:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.results_out)) or ".", exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint, "surrogate": args.surrogate,
            "cool_time": args.cool_time, "random_cool_time": args.random_cool_time,
            "greedy_actions": actions_g.tolist(), "greedy_rewards": rewards_g.tolist(),
            "greedy_return_mean": float(rewards_g.sum(axis=1).mean()),
        }
        if args.also_stochastic:
            payload.update(
                stochastic_actions=actions_s.tolist(), stochastic_rewards=rewards_s.tolist(),
                stochastic_return_mean=float(rewards_s.sum(axis=1).mean()),
            )
        with open(args.results_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[evaluate_real] Results saved → {args.results_out}")

    if args.plot:
        from baselines.common.eval_harness import plot_reward_and_action_per_layer
        out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        plot_reward_and_action_per_layer(
            rewards_g, actions_g, "UCPG v2 — REAL PHYSICS (greedy)",
            os.path.join(out_dir, "eval_real_reward_and_action.png"),
        )

    if not args.work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
