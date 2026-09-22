#!/bin/bash
# =============================================================================
# evaluate_ood_ratio.sh
# SLURM job script — coverage-normalized OOD check for the patchy-coverage
# surrogate: ratio of its epistemic sigma to the matched full-range
# checkpoint's epistemic sigma, vs. laser power (evaluate_ood_ratio.py).
#
# Requires two checkpoints to already exist:
#   - the patchy/gapped one from jobs/train_gap_perturb.sh
#   - the MATCHED full-range one from jobs/train_full_matched.sh (same
#     perturb_frac/bootstrap_frac, no --lp_filter_*)
#
# This should NOT be run on the login node — it loads the full dataset plus
# two model forward passes and can get OOM-killed there; run it as a batch
# job instead.
#
# Submit:
#   sbatch surrogate_model_v3/jobs/evaluate_ood_ratio.sh
# =============================================================================

#SBATCH --job-name=eval_surrogate_v3_ood_ratio
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/evaluate_ood_ratio_%j.log

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
CHECKPOINT_PATCHY="surrogate_model_v3/runs/patchy_100-150_200-250_300-350_perturb0.1/surrogate_best.pt"
CHECKPOINT_FULL="surrogate_model_v3/runs/full_matched_perturb0.1_bootstrap0.5/surrogate_best.pt"
ID_RANGES="100-150,200-250,300-350"

python -m surrogate_model_v3.evaluate_ood_ratio \
    --checkpoint_patchy "$CHECKPOINT_PATCHY" \
    --checkpoint_full   "$CHECKPOINT_FULL"   \
    --data_path         "$DATA_PATH"         \
    --id_ranges         "$ID_RANGES"
EXIT_CODE=$?

echo "============================================================"
echo "Ratio evaluation finished: $(date)  (exit code: $EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
