"""
baselines/common/eval_harness.py
------------------------------------
Common surrogate-environment evaluation harness — every baseline (and,
optionally, the real online_RL_ucpg_v2 policy) is rolled out through the
SAME TwoStageLatentLPBFEnv instance so their reported returns are directly
comparable. This module only ever imports from surrogate_model_v3 /
online_RL_ucpg_v2; it does not modify either.

A "controller" here is any object exposing:
    act(ctx: StepContext) -> float          # laser power [W]
and, optionally, `reset()` (called at the start of every episode — used by
stateful controllers like the Kalman/particle filter to clear their belief).

StepContext gives every controller everything it might plausibly need, so
each one only reads the fields relevant to its own design:
    obs        : (obs_dim,) float32 — full online_RL_ucpg_v2 observation
                 [z_t ‖ layer_token ‖ cool_time_token] (only meaningful to
                 the RL policies, e.g. naive_pg / UCPG v2 checkpoints —
                 `z_t` is the raw normalised field, surrogate_model_v3 has
                 no latent space; see online_RL_ucpg_v2/env.py's docstring)
    z          : (latent_dim,) float32 — obs's z_t slice alone
    raw_state  : (state_dim,) float32 — DENORMALISED s_t (the pre-heat
                 field) — this is what a real deployed controller would
                 plausibly have access to (e.g. from a thermal camera), and
                 is what the offline-Q / proportional / Kalman-particle
                 baselines are trained/fit against, since none of them use
                 the RL policies' raw-field observation directly.
    layer      : int — 0-indexed layer
    cool_time  : float — this episode's (fixed) cooling duration [s]
"""

import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_v3.model import load_surrogate
from online_RL_ucpg_v2.env   import TwoStageLatentLPBFEnv
from online_RL_ucpg_v2.agent import UCPGAgentV2


@dataclass
class StepContext:
    obs:       np.ndarray
    z:         np.ndarray
    raw_state: np.ndarray
    layer:     int
    cool_time: float


class Harness:
    """
    Loads the two-stage surrogate ONCE and builds the shared
    TwoStageLatentLPBFEnv every baseline is evaluated against.
    """

    def __init__(
        self,
        surrogate_path: str,
        device:         str   = "cpu",
        T_l:            float = 2000.0,
        T_h:            float = 2800.0,
        n_layers:       int   = 12,
        initial_temp:   float = 300.0,
        mesh_path:      str   = "surrogate_model/mesh.mat",
        width:          float = 12.0,
        height:         float = 3.0,
        sq_frac_start:  float = 0.4,
        sq_frac_end:    float = 0.5,
        action_min:     float = 100.0,
        action_max:     float = 400.0,
        cool_time_min:  float = 0.05,
        cool_time_max:  float = 0.15,
    ) -> None:
        self.device = device
        (self.surrogate, self.state_mean, self.state_std, self.lp_mean, self.lp_std,
         self.cool_mean, self.cool_std) = load_surrogate(surrogate_path, device=device)
        self.surrogate.eval()

        self.env = TwoStageLatentLPBFEnv(
            surrogate=self.surrogate, state_mean=self.state_mean, state_std=self.state_std,
            lp_mean=self.lp_mean, lp_std=self.lp_std, cool_mean=self.cool_mean, cool_std=self.cool_std,
            temp_range=(T_l, T_h), n_layers=n_layers, initial_temp=initial_temp, device=device,
            mesh_path=mesh_path, width=width, height=height,
            sq_frac_start=sq_frac_start, sq_frac_end=sq_frac_end,
            action_min=action_min, action_max=action_max,
            cool_time_min=cool_time_min, cool_time_max=cool_time_max,
        )
        self.latent_dim = self.env.latent_dim
        self.n_layers   = n_layers

    def _denorm(self, z: np.ndarray) -> np.ndarray:
        """z IS the raw normalised field (no decoder — surrogate_model_v3
        has no latent space), so this is just an affine un-standardisation,
        not a network call."""
        z_t = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
        raw = z_t * self.state_std + self.state_mean
        return raw.squeeze(0).cpu().numpy()

    def _make_ctx(self, obs: np.ndarray, layer: int) -> StepContext:
        z = obs[: self.latent_dim]
        cool_tok = float(obs[-1])
        cool_time = cool_tok * (self.env.cool_time_max - self.env.cool_time_min) + self.env.cool_time_min
        return StepContext(obs=obs, z=z, raw_state=self._denorm(z), layer=layer, cool_time=cool_time)

    def run_episode(self, controller) -> dict:
        if hasattr(controller, "reset"):
            controller.reset()
        obs = self.env.reset()
        actions, rewards, u = [], [], []
        for t in range(self.n_layers):
            ctx = self._make_ctx(obs, t)
            a = float(controller.act(ctx))
            obs, reward, done, info = self.env.step(a)
            actions.append(a)
            rewards.append(reward)
            u.append(info["uncertainty"])
            if done:
                break
        return dict(actions=np.array(actions), rewards=np.array(rewards), u=np.array(u))

    def run_many(self, controller, n_episodes: int) -> dict:
        all_a, all_r, all_u = [], [], []
        for _ in range(n_episodes):
            ep = self.run_episode(controller)
            all_a.append(ep["actions"]); all_r.append(ep["rewards"]); all_u.append(ep["u"])
        return dict(actions=np.stack(all_a), rewards=np.stack(all_r), u=np.stack(all_u))


# =============================================================================
# Controller wrapper for any UCPGAgentV2-checkpoint-based policy (naive_pg
# baseline AND the real online_RL_ucpg_v2 policy both save/load in this
# format, since naive_pg reuses UCPGAgentV2 directly)
# =============================================================================

class LatentAgentController:
    """Wraps a UCPGAgentV2 checkpoint (naive_pg or the real UCPG v2 policy)
    to the Harness's act(ctx) interface. `ctx.obs` IS exactly what
    UCPGAgentV2.select_action expects — no translation needed."""

    def __init__(self, checkpoint_path: str, device: str = "cpu", greedy: bool = True):
        self.agent  = UCPGAgentV2.load(checkpoint_path, device=device)
        self.greedy = greedy

    def act(self, ctx: StepContext) -> float:
        return self.agent.select_action(ctx.obs, explore=not self.greedy)


# =============================================================================
# Metrics / reporting
# =============================================================================

def summarize(name: str, result: dict) -> dict:
    """Reduce a run_many() result to the scalar metrics used across baselines/README.md."""
    ep_return = result["rewards"].sum(axis=1)
    return dict(
        name=name,
        return_mean=float(ep_return.mean()),
        return_std=float(ep_return.std()),
        uncertainty_mean=float(result["u"].mean()),
        action_mean=float(result["actions"].mean()),
        action_std=float(result["actions"].std()),
    )


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def write_leaderboard(
    rows: List[dict], out_dir: str, n_layers: int, T_l: float, T_h: float,
    title: str = "Baseline Comparison",
    csv_name: str = "leaderboard.csv", png_name: str = "leaderboard.png",
) -> None:
    """Writes the return + physical-temperature-deviation comparison CSV and
    2-panel bar chart. Shared by evaluate_baselines.py (surrogate-driven) and
    evaluate_real_all_methods.py (real-simulator-driven) so both produce
    identically formatted, directly comparable leaderboards. `rows` are
    dicts as returned by `summarize()` (mutated in place with
    deviation_K_mean/std). deviation_K = -reward*(T_h-T_l) at every layer
    (see plot_reward_and_action_per_layer's docstring), so the mean
    per-layer deviation is recovered from the mean undiscounted return as
    -return_mean/n_layers*(T_h-T_l)."""
    T_span = T_h - T_l
    for r in rows:
        r["deviation_K_mean"] = -r["return_mean"] / n_layers * T_span
        r["deviation_K_std"]  = r["return_std"]   / n_layers * T_span

    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w") as f:
        f.write("name,return_mean,return_std,deviation_K_mean,deviation_K_std,"
                "uncertainty_mean,action_mean,action_std\n")
        for r in sorted(rows, key=lambda r: r["return_mean"], reverse=True):
            f.write(f"{r['name']},{r['return_mean']:.6f},{r['return_std']:.6f},"
                    f"{r['deviation_K_mean']:.4f},{r['deviation_K_std']:.4f},"
                    f"{r['uncertainty_mean']:.6f},{r['action_mean']:.4f},{r['action_std']:.4f}\n")
    print(f"[eval_harness] Saved → {csv_path}")

    rows_sorted = sorted(rows, key=lambda r: r["return_mean"], reverse=True)
    names = [r["name"] for r in rows_sorted]
    means = [r["return_mean"] for r in rows_sorted]
    stds  = [r["return_std"]  for r in rows_sorted]
    dev_m = [r["deviation_K_mean"] for r in rows_sorted]
    dev_s = [r["deviation_K_std"]  for r in rows_sorted]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 5))
    axL.barh(names, means, xerr=stds, color="steelblue", alpha=0.8)
    axL.set_xlabel("Undiscounted episode return (mean ± std)")
    axL.set_title("Return")
    axL.grid(True, alpha=0.3, axis="x")

    axR.barh(names, dev_m, xerr=dev_s, color="firebrick", alpha=0.8)
    axR.set_xlabel("Mean node temperature deviation from window [K]  (lower is better)")
    axR.set_title("Physical Deviation")
    axR.grid(True, alpha=0.3, axis="x")
    axR.set_yticklabels([])

    fig.suptitle(title)
    fig.tight_layout()
    png_path = os.path.join(out_dir, png_name)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[eval_harness] Saved → {png_path}")


def plot_reward_and_action_per_layer(
    rewards: np.ndarray,   # (n_ep, T)
    actions: np.ndarray,   # (n_ep, T)
    method_name: str,
    out_path: str,
    temp_range: Optional[tuple] = None,   # (T_l, T_h) — adds a temperature-deviation [K] panel
) -> None:
    """
    Per-layer reward/action trace for one baseline — identical layout to
    online_RL_ucpg_v2/evaluate.py's plot_reward_and_action_per_layer, so
    every method's plot is directly visually comparable to the RL policy's.

    `temp_range`: since reward = -meanDeviation/(T_h-T_l) (both the surrogate
    env, online_RL_ucpg_v2/env.py's _compute_reward, and the real MATLAB
    simulator's meanDeviation output use this exact normalization — see
    simulateHeatingCooling_v2.m line 100), the physical mean node-temperature
    deviation from the target window is recovered exactly as
    deviation_K = -reward * (T_h - T_l). When given, this adds a middle panel
    plotting that directly (not a secondary/twin axis, to avoid any
    sign-inversion ambiguity for the reader) — omit to keep the original
    2-panel layout unchanged for any other caller.
    """
    n_ep, T = rewards.shape
    layers  = np.arange(1, T + 1)

    n_panels = 3 if temp_range is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=True)
    ax1 = axes[0]

    for i in range(n_ep):
        ax1.plot(layers, rewards[i], alpha=0.25, linewidth=0.7, color="steelblue")
    ax1.plot(layers, rewards.mean(axis=0), color="steelblue", linewidth=2.5,
             label=f"Mean reward (n={n_ep})")
    ax1.axhline(0, color="green", linestyle="--", linewidth=1, label="Perfect (0 deviation)")
    ax1.set_ylabel("Reward (−meanDeviation, end-of-heating)")
    ax1.set_title(f"{method_name} — Per-Layer Reward")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    next_ax_idx = 1
    if temp_range is not None:
        T_l, T_h = temp_range
        dev_K = -rewards * (T_h - T_l)
        axT = axes[next_ax_idx]; next_ax_idx += 1
        for i in range(n_ep):
            axT.plot(layers, dev_K[i], alpha=0.25, linewidth=0.7, color="firebrick")
        axT.plot(layers, dev_K.mean(axis=0), color="firebrick", linewidth=2.5,
                 label=f"Mean deviation (n={n_ep})")
        axT.axhline(0, color="green", linestyle="--", linewidth=1, label="Perfect (in window)")
        axT.set_ylabel("Mean node temperature\ndeviation from window [K]")
        axT.set_title(f"{method_name} — Per-Layer Temperature Deviation")
        axT.legend(); axT.grid(True, alpha=0.3)

    ax2 = axes[next_ax_idx]
    mean_a = actions.mean(axis=0)
    std_a  = actions.std(axis=0)
    ax2.bar(layers, mean_a, alpha=0.7, color="darkorange", label="Mean LP [W]")
    ax2.errorbar(layers, mean_a, yerr=std_a, fmt="none", color="black",
                capsize=3, linewidth=1)
    ax2.set_ylabel("Laser Power [W]")
    ax2.set_xlabel("Layer")
    ax2.set_title(f"{method_name} — Action Sequence (mean ± std)")
    ax2.set_xticks(layers)
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval_harness] Saved → {out_path}")


def print_leaderboard(rows: List[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["return_mean"], reverse=True)
    header = f"{'Method':<22}{'Return (mean±std)':>22}{'Uncertainty':>14}{'Action (mean±std)':>22}"
    print(header)
    print("-" * len(header))
    for r in rows_sorted:
        print(
            f"{r['name']:<22}"
            f"{r['return_mean']:+9.4f} ± {r['return_std']:<7.4f}"
            f"{r['uncertainty_mean']:>14.5f}"
            f"{r['action_mean']:>10.1f} ± {r['action_std']:<7.1f}"
        )
