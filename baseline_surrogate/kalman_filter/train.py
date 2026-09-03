"""
baseline_surrogate/kalman_filter/train.py
----------------------------------------------
Fit the Kalman-filter baseline (closed form — no gradient descent, no
epochs). Named train.py to match every other baseline's CLI convention,
mirroring baselines/kalman_particle/filters.py's "fit once" pattern.

Usage:
    python -m baseline_surrogate.kalman_filter.train \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --lp_filter_min 200 --lp_filter_max 300 \\
        --out_dir baseline_surrogate/kalman_filter/runs/narrow_200_300W
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.dataset_v2 import load_trajectories, split_trajectories
from baseline_surrogate.common.data import build_single_stage_normalizers, extract_single_stage_raw_arrays
from baseline_surrogate.kalman_filter.model import fit_linear_gaussian_kalman


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit the Kalman-filter surrogate baseline.")
    p.add_argument("--data_path",     type=str, required=True)
    p.add_argument("--val_fraction",  type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--initial_temp",  type=float, default=300.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--lp_filter_min", type=float, default=None)
    p.add_argument("--lp_filter_max", type=float, default=None)

    p.add_argument("--n_components", type=int, default=64,
                   help="PCA latent dimension (matches the main surrogate's --latent_dim default).")

    p.add_argument("--out_dir", type=str, default="baseline_surrogate/kalman_filter/runs/default")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[kalman_filter.train] Output dir : {args.out_dir}")

    all_trajs = load_trajectories(args.data_path)
    train_trajs, _val, _test = split_trajectories(
        all_trajs, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
    )
    n_layers = len(train_trajs[0])

    # Kept for interoperability with the shared eval/predictor interface
    # (KalmanPredictor denormalises using these) even though PCA fitting
    # below uses raw values directly, not these z-scores.
    state_mean, state_std, lp_mean, lp_std, cool_mean, cool_std = build_single_stage_normalizers(
        train_trajs, initial_temp=args.initial_temp
    )

    lp_filter = None
    if args.lp_filter_min is not None or args.lp_filter_max is not None:
        assert args.lp_filter_min is not None and args.lp_filter_max is not None, \
            "--lp_filter_min and --lp_filter_max must be given together."
        lp_filter = (args.lp_filter_min, args.lp_filter_max)
        print(f"[kalman_filter.train] LP filter active: [{args.lp_filter_min}, {args.lp_filter_max}] W")

    data = extract_single_stage_raw_arrays(train_trajs, initial_temp=args.initial_temp, lp_filter=lp_filter)

    model = fit_linear_gaussian_kalman(
        data["s"], data["a"], data["c"], data["layer"], data["s2"],
        n_components=args.n_components, n_layers=n_layers,
    )

    ckpt_path = os.path.join(args.out_dir, "kalman_filter_fitted.pt")
    torch.save({
        "pca_components":       model.components,
        "pca_mean":             model.mean,
        "coeffs":               model.coeffs,
        "n_layers":             model.n_layers,
        "latent_dim":           model.latent_dim,
        "state_mean": state_mean, "state_std": state_std,
        "lp_mean": lp_mean, "lp_std": lp_std, "cool_mean": cool_mean, "cool_std": cool_std,
        "train_args": vars(args),
    }, ckpt_path)
    print(f"[kalman_filter.train] Saved → {ckpt_path}")


if __name__ == "__main__":
    main()
