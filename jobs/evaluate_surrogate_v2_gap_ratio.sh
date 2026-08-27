#!/bin/bash
# =============================================================================
# evaluate_surrogate_v2_gap_ratio.sh
# SLURM job script — coverage-normalized OOD check for the patchy-coverage
# two-stage surrogate: ratio of its epistemic sigma to a full-range
# checkpoint's epistemic sigma, vs. laser power (evaluate_ood_ratio.py).
# Separates "epistemic rises because it's genuinely OOD" from "epistemic
# rises because higher laser power is intrinsically harder for ANY model."
#
# Requires two checkpoints to already exist:
#   - the patchy/gapped one from jobs/train_surrogate_v2_gap_perturb.sh
#   - a full-range one trained on the SAME data_path, SAME perturb_frac/
#     bootstrap_frac (see jobs/train_surrogate_v2_full_matched.sh), with no
#     --lp_filter_* -- must be a MATCHED control, not just any full-range
#     checkpoint, or the ratio bakes in perturb/bootstrap-frac confounds
#     instead of isolating the coverage effect.
#
# This did NOT run on the login node — evaluate_ood_ratio.py loads the full
# dataset + two model forward passes and gets OOM-killed there; run it as a
# batch job instead.
#
# Submit:
#   sbatch jobs/evaluate_surrogate_v2_gap_ratio.sh
# =============================================================================

#SBATCH --job-name=eval_surrogate_v2_gap_ratio
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/eval_surrogate_v2_gap_ratio_%j.log

# ── environment ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

source ~/.bashrc
conda activate RL

echo "Python : $(which python)"
echo "============================================================"

DATA_PATH="Data/DatasetV2_layer_12_samples_5000.pkl"
CHECKPOINT_PATCHY="surrogate_model_latent_uncertainty_v2/runs/patchy_100-150_200-250_300-350_perturb0.25/two_stage_best.pt"
CHECKPOINT_FULL="surrogate_model_latent_uncertainty_v2/runs/full_matched_perturb0.25_bootstrap0.25/two_stage_best.pt"
ID_RANGES="100-150,200-250,300-350"

echo "[eval] === evaluate_ood_ratio.py (epistemic ratio: patchy / full-range) ==="
python -m surrogate_model_latent_uncertainty_v2.evaluate_ood_ratio \
    --checkpoint_patchy "$CHECKPOINT_PATCHY" \
    --checkpoint_full   "$CHECKPOINT_FULL" \
    --data_path         "$DATA_PATH" \
    --id_ranges         "$ID_RANGES"
EXIT_CODE=$?

echo "============================================================"
echo "Ratio evaluation finished: $(date)  (evaluate_ood_ratio.py=$EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
