"""
baseline_surrogate/ablation_no_latent/evaluate_ood_ratio.py
--------------------------------------------------------------
Coverage-normalized OOD check (2x2 uncertainty summary) for the no-latent
ablation -- the same comparison as
surrogate_model_latent_uncertainty_v2/evaluate_ood_ratio.py (ratio of a
PATCHY-trained checkpoint's epistemic sigma to a FULL-range checkpoint's,
both evaluated on the same wide dataset, vs. laser power), just pointed at
NoLatentTwoStageSurrogate checkpoints instead of the encoder/decoder main
model. Reuses that package's collect_ood_samples/bin_by_action/plot_ratio/
plot_summary_2x2 unchanged -- NoLatentTwoStageSurrogate exposes the same
encode/decode/predict_heating_ensemble/n_ensemble interface those functions
call, since encode/decode are just the identity here (see model.py). Only
the checkpoint loader differs (NoLatentTwoStageSurrogate instead of
TwoStageEnsembleGaussianLatentDynamicsModel), so this file is a thin
_run_one + CLI wrapper, not a reimplementation.

Usage
-----
    python -m baseline_surrogate.ablation_no_latent.evaluate_ood_ratio \\
        --checkpoint_patchy baseline_surrogate/ablation_no_latent/runs/patchy_100-150_200-250_300-350_perturb0.1/ablation_no_latent_best.pt \\
        --checkpoint_full   baseline_surrogate/ablation_no_latent/runs/full_range/ablation_no_latent_best.pt \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --id_ranges "100-150,200-250,300-350"

`--checkpoint_full` must exist already or be trained first -- same
architecture/K/bootstrap settings as the patchy run, same --data_path, just
WITHOUT any --lp_filter_* flag:

    python -m baseline_surrogate.ablation_no_latent.train \\
        --data_path Data/DatasetV2_layer_12_samples_5000.pkl \\
        --out_dir baseline_surrogate/ablation_no_latent/runs/full_range

Outputs (--out_dir, defaults to the patchy checkpoint's directory) -- same
two files/console table as the main package's script:
  ood_epistemic_ratio_vs_action.png
  ood_uncertainty_summary_2x2.png
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from surrogate_model_latent_uncertainty_v2.dataset_v2 import load_trajectories, TwoStageLatentSurrogateDataset
from surrogate_model_latent_uncertainty_v2.train import _parse_lp_filter_ranges
from surrogate_model_latent_uncertainty_v2.evaluate_ood import collect_ood_samples, bin_by_action
from surrogate_model_latent_uncertainty_v2.evaluate_ood_ratio import plot_ratio, plot_summary_2x2

from baseline_surrogate.ablation_no_latent.model import load_no_latent_surrogate


def _run_one(checkpoint: str, trajs, args, device: str) -> dict:
    (model, state_mean, state_std, lp_mean, lp_std,
     cool_mean, cool_std) = load_no_latent_surrogate(checkpoint, device)
    print(f"[ablation_no_latent.evaluate_ood_ratio]   {checkpoint}")

    ds = TwoStageLatentSurrogateDataset(
        trajs, state_mean=state_mean.cpu(), state_std=state_std.cpu(),
        lp_mean=lp_mean, lp_std=lp_std, cool_mean=cool_mean, cool_std=cool_std,
        initial_temp=args.initial_temp, n_ensemble=model.n_ensemble, bootstrap_seed=0,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    return collect_ood_samples(model, loader, state_mean, state_std, lp_mean, lp_std, device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Coverage-normalized OOD check (no-latent ablation): ratio of a patchy-trained "
                    "checkpoint's epistemic sigma to a full-range checkpoint's, vs. laser power."
    )
    p.add_argument("--checkpoint_patchy", type=str, required=True,
                   help="Checkpoint trained on a restricted/gapped laser-power range "
                        "(ablation_no_latent/train.py's --lp_filter_min/--lp_filter_max or "
                        "--lp_filter_ranges).")
    p.add_argument("--checkpoint_full", type=str, required=True,
                   help="Checkpoint trained on the SAME --data_path with NO "
                        "--lp_filter_min/--lp_filter_max/--lp_filter_ranges.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--id_action_min", type=float, default=150.0)
    p.add_argument("--id_action_max", type=float, default=300.0)
    p.add_argument("--id_ranges", type=str, default=None,
                   help="For a GAPPED patchy checkpoint: comma-separated 'lo-hi' ranges matching "
                        "its --lp_filter_ranges exactly, e.g. '100-150,200-250,300-350'.")
    p.add_argument("--n_bins", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--initial_temp", type=float, default=300.0)
    p.add_argument("--out_dir", type=str, default="",
                   help="Defaults to the patchy checkpoint's directory.")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint_patchy))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ablation_no_latent.evaluate_ood_ratio] Output dir : {out_dir}")
    print(f"[ablation_no_latent.evaluate_ood_ratio] Device     : {device}")

    id_ranges = (_parse_lp_filter_ranges(args.id_ranges) if args.id_ranges is not None
                else [(args.id_action_min, args.id_action_max)])
    range_str = ", ".join(f"[{lo}, {hi}]" for lo, hi in id_ranges)
    print(f"[ablation_no_latent.evaluate_ood_ratio] Patchy checkpoint's ID range(s): {range_str} W")

    trajs = load_trajectories(args.data_path)

    print("[ablation_no_latent.evaluate_ood_ratio] Running PATCHY checkpoint forward pass ...")
    data_patchy = _run_one(args.checkpoint_patchy, trajs, args, device)
    print("[ablation_no_latent.evaluate_ood_ratio] Running FULL-RANGE checkpoint forward pass ...")
    data_full = _run_one(args.checkpoint_full, trajs, args, device)

    binned_patchy = bin_by_action(data_patchy, args.n_bins)
    binned_full   = bin_by_action(data_full,   args.n_bins)

    eps = 1e-8
    epi_ratio = binned_patchy["epi"] / np.clip(binned_full["epi"], eps, None)
    ale_ratio = binned_patchy["ale"] / np.clip(binned_full["ale"], eps, None)

    print(f"\n[ablation_no_latent.evaluate_ood_ratio] {'=' * 100}")
    print(f"  {'Power bin [W]':>16}  {'epi_patchy':>11}  {'epi_full':>10}  "
          f"{'epi ratio':>10}  {'ale ratio':>10}  {'region':>6}")
    for i in range(args.n_bins):
        lo, hi = binned_patchy["edges"][i], binned_patchy["edges"][i + 1]
        center = binned_patchy["centers"][i]
        is_id = any(rlo <= center <= rhi for rlo, rhi in id_ranges)
        print(f"  {lo:7.1f}-{hi:7.1f}  {binned_patchy['epi'][i]:11.5f}  "
              f"{binned_full['epi'][i]:10.5f}  {epi_ratio[i]:10.3f}  "
              f"{ale_ratio[i]:10.3f}  {'ID' if is_id else 'OOD':>6}")
    print(f"[ablation_no_latent.evaluate_ood_ratio] {'=' * 100}\n")

    plot_ratio(binned_patchy, binned_full, id_ranges,
              os.path.join(out_dir, "ood_epistemic_ratio_vs_action.png"))
    plot_summary_2x2(binned_patchy, binned_full, id_ranges,
                     os.path.join(out_dir, "ood_uncertainty_summary_2x2.png"))

    print(f"\n[ablation_no_latent.evaluate_ood_ratio] Complete. Output in: {out_dir}")


if __name__ == "__main__":
    main()
