"""
surrogate_model_latent_uncertainty_v2/extract_dataset_v2.py
---------------------------------------------------------------
CLI entry point that extracts + pickles the v2 (heating/cooling) LPBF
trajectories, using dataset_v2.gather_dataset_v2.

Kept as a separate file from dataset_v2.py, rather than a
`if __name__ == "__main__":` block there, because `python -m
surrogate_model_latent_uncertainty_v2.dataset_v2` would re-execute
dataset_v2.py as `__main__` — which rebinds StepV2.__module__ to "__main__",
so pickles written that way fail to unpickle later from train.py /
evaluate.py (they import dataset_v2 normally, and look for StepV2 in
surrogate_model_latent_uncertainty_v2.dataset_v2, not __main__). This file
only *imports* StepV2 (via gather_dataset_v2), so it never redefines it.

Usage (run from the repo root):
    python -m surrogate_model_latent_uncertainty_v2.extract_dataset_v2 --n 200
"""

import argparse
import os
import pickle

import numpy as np

from surrogate_model_latent_uncertainty_v2.dataset_v2 import gather_dataset_v2


def main() -> None:
    p = argparse.ArgumentParser(description="Extract + pickle the v2 (heating/cooling) LPBF trajectories.")
    p.add_argument("--n", type=int, default=200, help="Number of trajectories to extract (first N).")
    p.add_argument("--trajectory_length", type=int, default=12, help="Number of layers per trajectory.")
    p.add_argument("--out", type=str, default="",
                   help="Output .pkl path (default: Data/DatasetV2_layer_<T>_samples_<n>.pkl).")
    args = p.parse_args()

    id_list = np.arange(1, args.n + 1, 1)
    dataset = gather_dataset_v2(id_list, trajectory_length=args.trajectory_length)

    out_path = args.out or f"Data/DatasetV2_layer_{args.trajectory_length}_samples_{args.n}.pkl"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[extract_dataset_v2] Saved {len(dataset)} trajectories × {args.trajectory_length} layers → {out_path}")


if __name__ == "__main__":
    main()
