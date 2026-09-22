"""
surrogate_model_v3/extract_dataset.py
------------------------------------------
CLI entry point that extracts + pickles the v2 (heating/cooling) LPBF
trajectories, using dataset.gather_dataset.

Kept as a separate file from dataset.py, rather than a
`if __name__ == "__main__":` block there, because `python -m
surrogate_model_v3.dataset` would re-execute dataset.py as `__main__` —
which rebinds Step.__module__ to "__main__", so pickles written that way
fail to unpickle later from train.py/evaluate.py. This file only *imports*
Step (via gather_dataset), so it never redefines it.

Note: Data/DatasetV2_layer_12_samples_5000.pkl already exists (extracted
via the sibling surrogate_model_latent_uncertainty_v2 package) and loads
fine through dataset.load_trajectories — see that module's docstring for
why the older pickle's class binding doesn't matter. Only run this script
if you need MORE trajectories than that file already has.

Usage (run from the repo root):
    python -m surrogate_model_v3.extract_dataset --n 200
"""

import argparse
import os
import pickle

import numpy as np

from surrogate_model_v3.dataset import gather_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Extract + pickle the v2 (heating/cooling) LPBF trajectories.")
    p.add_argument("--n", type=int, default=200, help="Number of trajectories to extract (first N).")
    p.add_argument("--trajectory_length", type=int, default=12, help="Number of layers per trajectory.")
    p.add_argument("--out", type=str, default="",
                   help="Output .pkl path (default: Data/DatasetV2_layer_<T>_samples_<n>.pkl).")
    args = p.parse_args()

    id_list = np.arange(1, args.n + 1, 1)
    dataset = gather_dataset(id_list, trajectory_length=args.trajectory_length)

    out_path = args.out or f"Data/DatasetV2_layer_{args.trajectory_length}_samples_{args.n}.pkl"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[extract_dataset] Saved {len(dataset)} trajectories × {args.trajectory_length} layers → {out_path}")


if __name__ == "__main__":
    main()
