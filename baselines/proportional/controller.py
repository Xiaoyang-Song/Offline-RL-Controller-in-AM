"""
baselines/proportional/controller.py
----------------------------------------
Baseline 2: proportional controller.

    a_t = K_p * (T_mid - mean_ROI(s_t)) + bias

where s_t is the (raw, decoded) temperature field BEFORE this layer's laser
pass, mean_ROI is the mean temperature inside this layer's square scan
region, and T_mid = (T_l + T_h)/2 is the centre of the target window. K_p and
bias are fit ONCE via ordinary least squares against the offline dataset's
own (error, laser_power) pairs — a classical control law, not a learned
policy: no gradient descent, no reward signal, no RL of any kind.

Usage
-----
    python -m baselines.proportional.controller \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --out baselines/proportional/fitted.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.dataset_v2 import load_trajectories, split_trajectories
from baselines.common.data_utils import (
    build_offline_transitions, load_mesh_nodes, roi_masks_per_layer, roi_mean, fit_linear,
)


class ProportionalController:
    def __init__(self, K_p: float, bias: float, T_mid: float, masks: list):
        self.K_p   = K_p
        self.bias  = bias
        self.T_mid = T_mid
        self.masks = masks

    def act(self, ctx) -> float:
        mask = self.masks[min(ctx.layer, len(self.masks) - 1)]
        error = self.T_mid - roi_mean(ctx.raw_state, mask)
        return self.K_p * error + self.bias


def fit_proportional_controller(
    trajectories, mesh_path: str, T_l: float, T_h: float,
    width: float, height: float, sq_frac_start: float, sq_frac_end: float,
    n_layers: int = 12, initial_temp: float = 300.0,
) -> tuple:
    """Returns (ProportionalController, fit_info dict)."""
    data = build_offline_transitions(trajectories, initial_temp=initial_temp)
    nodes_xy = load_mesh_nodes(mesh_path)
    masks = roi_masks_per_layer(nodes_xy, width, height, sq_frac_start, sq_frac_end, n_layers)
    T_mid = (T_l + T_h) / 2.0

    errors = np.array([
        T_mid - roi_mean(data["s"][i], masks[data["layer"][i]])
        for i in range(data["s"].shape[0])
    ])
    K_p, bias, resid_std = fit_linear(errors, data["a"])

    controller = ProportionalController(K_p, bias, T_mid, masks)
    return controller, dict(K_p=K_p, bias=bias, T_mid=T_mid, resid_std=resid_std)


def save_fitted(path: str, K_p: float, bias: float, T_mid: float,
                mesh_path: str, width: float, height: float,
                sq_frac_start: float, sq_frac_end: float, n_layers: int) -> None:
    torch.save(dict(K_p=K_p, bias=bias, T_mid=T_mid, mesh_path=mesh_path, width=width, height=height,
                    sq_frac_start=sq_frac_start, sq_frac_end=sq_frac_end, n_layers=n_layers), path)


def load_proportional_controller(path: str) -> ProportionalController:
    d = torch.load(path, map_location="cpu", weights_only=False)
    nodes_xy = load_mesh_nodes(d["mesh_path"])
    masks = roi_masks_per_layer(nodes_xy, d["width"], d["height"], d["sq_frac_start"], d["sq_frac_end"], d["n_layers"])
    return ProportionalController(d["K_p"], d["bias"], d["T_mid"], masks)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline 2: fit a proportional controller from the offline dataset.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--T_l", type=float, default=2000.0)
    p.add_argument("--T_h", type=float, default=2800.0)
    p.add_argument("--mesh_path", type=str, default="surrogate_model/mesh.mat")
    p.add_argument("--width",  type=float, default=12.0)
    p.add_argument("--height", type=float, default=3.0)
    p.add_argument("--sq_frac_start", type=float, default=0.4)
    p.add_argument("--sq_frac_end",   type=float, default=0.5)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--out", type=str, default="baselines/proportional/fitted.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_trajs = load_trajectories(args.data_path)
    train_trajs, _val, _test = split_trajectories(
        all_trajs, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
    )
    _controller, info = fit_proportional_controller(
        train_trajs, args.mesh_path, args.T_l, args.T_h, args.width, args.height,
        args.sq_frac_start, args.sq_frac_end, args.n_layers, args.initial_temp,
    )
    print(f"[proportional] Fit on {len(train_trajs)} trajectories: "
          f"K_p={info['K_p']:.4f}  bias={info['bias']:.2f}  T_mid={info['T_mid']:.1f}  "
          f"residual_std={info['resid_std']:.2f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_fitted(args.out, info["K_p"], info["bias"], info["T_mid"], args.mesh_path,
               args.width, args.height, args.sq_frac_start, args.sq_frac_end, args.n_layers)
    print(f"[proportional] Saved → {args.out}")


if __name__ == "__main__":
    main()
