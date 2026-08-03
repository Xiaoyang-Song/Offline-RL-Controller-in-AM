"""
baselines/common/eval_harness.py
------------------------------------
Common surrogate-environment evaluation harness — every baseline (and,
optionally, the real online_RL_ucpg_v2 policy) is rolled out through the
SAME TwoStageLatentLPBFEnv instance so their reported returns are directly
comparable. This module only ever imports from surrogate_model_latent_uncertainty_v2
/ online_RL_ucpg_v2; it does not modify either.

A "controller" here is any object exposing:
    act(ctx: StepContext) -> float          # laser power [W]
and, optionally, `reset()` (called at the start of every episode — used by
stateful controllers like the Kalman/particle filter to clear their belief).

StepContext gives every controller everything it might plausibly need, so
each one only reads the fields relevant to its own design:
    obs        : (obs_dim,) float32 — full online_RL_ucpg_v2 observation
                 [z_t ‖ layer_token ‖ cool_time_token] (only meaningful to
                 latent-space policies, e.g. naive_pg / UCPG v2 checkpoints)
    z          : (latent_dim,) float32 — obs's latent slice alone
    raw_state  : (state_dim,) float32 — DECODED, denormalised s_t (the
                 pre-heat field) — this is what a real deployed controller
                 would plausibly have access to (e.g. from a thermal
                 camera), and is what the offline-Q / proportional /
                 Kalman-particle baselines are trained/fit against, since
                 none of them use the surrogate's latent space.
    layer      : int — 0-indexed layer
    cool_time  : float — this episode's (fixed) cooling duration [s]
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.train import load_two_stage_surrogate
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
         self.cool_mean, self.cool_std, _roi) = load_two_stage_surrogate(surrogate_path, device=device)
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

    @torch.no_grad()
    def _decode(self, z: np.ndarray) -> np.ndarray:
        z_t = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
        raw = self.surrogate.decode(z_t) * self.state_std + self.state_mean
        return raw.squeeze(0).cpu().numpy()

    def _make_ctx(self, obs: np.ndarray, layer: int) -> StepContext:
        z = obs[: self.latent_dim]
        cool_tok = float(obs[-1])
        cool_time = cool_tok * (self.env.cool_time_max - self.env.cool_time_min) + self.env.cool_time_min
        return StepContext(obs=obs, z=z, raw_state=self._decode(z), layer=layer, cool_time=cool_time)

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
