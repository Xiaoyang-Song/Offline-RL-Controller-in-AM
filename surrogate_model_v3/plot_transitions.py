"""
surrogate_model_v3/plot_transitions.py
--------------------------------------
Per-layer true-vs-predicted field plots for a few example trajectories, in the
same style as the earlier latent-surrogate `traj_000/` folders: one PNG per
layer (jet fields on the mesh, `hot` relative-error map), plus
action_sequence.png, sigma_per_layer.png and actions.txt. Two rows per PNG:
heating (top) and cooling (bottom).

Predictions are the auto-regressive rollout (only the true s_0 and the action /
cool-time schedule are given), i.e. what the surrogate does at deployment.
This is a thin wrapper around surrogate_model_v3/evaluate.py's helpers, so the
plots are identical to what `evaluate.py` writes for its example trajectories,
without the full train/val/test metric pass.

Usage (repo root):
    python -m surrogate_model_v3.plot_transitions \\
        --checkpoint surrogate_model_v3/runs/full_range/surrogate_best.pt \\
        --data_path  Data/DatasetV2_layer_12_samples_5000.pkl \\
        --out_dir    surrogate_model_v3/results_transitions
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate_model_v3.dataset import TwoStageTrajectoryDataset, load_trajectories, split_trajectories
from surrogate_model_v3.evaluate import _DropBootstrapMask, evaluate_rollout, load_mesh, plot_example_trajectory
from surrogate_model_v3.model import load_surrogate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--mesh_path", default="surrogate_model/mesh.mat")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--n_examples", type=int, default=3)
    p.add_argument("--val_fraction", type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vmin", type=float, default=300.0)
    p.add_argument("--vmax", type=float, default=5000.0)
    p.add_argument("--out_dir", default="surrogate_model_v3/results_transitions")
    p.add_argument("--device", default="")
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    model, sm, ss, lm, ls, cm, cs = load_surrogate(a.checkpoint, device)

    tr, va, te = split_trajectories(load_trajectories(a.data_path), a.val_fraction, a.test_fraction, a.seed)
    trajs = {"train": tr, "val": va, "test": te}[a.split][:a.n_examples]
    ds = TwoStageTrajectoryDataset(trajs, state_mean=sm.cpu(), state_std=ss.cpu(), lp_mean=lm, lp_std=ls,
                                   cool_mean=cm, cool_std=cs, initial_temp=a.initial_temp,
                                   n_ensemble=model.n_ensemble, bootstrap_seed=a.seed)
    loader = _DropBootstrapMask(DataLoader(ds, batch_size=len(trajs), shuffle=False, num_workers=0))

    r = evaluate_rollout(model, loader, sm, ss, lm, ls, cm, cs, device, traj_len=len(trajs[0]))
    triang = load_mesh(a.mesh_path)
    for i in range(len(trajs)):
        plot_example_trajectory(
            r["all_heat_pred"][i], r["all_heat_gt"][i], r["all_next_pred"][i], r["all_next_gt"][i],
            actions=r["all_actions"][i], cooltimes=r["all_cooltimes"][i],
            heat_epi=r["all_heat_epi"][i], heat_ale=r["all_heat_ale"][i],
            cool_epi=r["all_cool_epi"][i], cool_ale=r["all_cool_ale"][i],
            traj_idx=i, out_dir=a.out_dir, triang=triang, vmin=a.vmin, vmax=a.vmax)
    print(f"[plot_transitions] Done → {a.out_dir}")


if __name__ == "__main__":
    main()
