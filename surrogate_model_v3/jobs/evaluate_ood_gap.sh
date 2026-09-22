#!/bin/bash
# =============================================================================
# evaluate_ood_gap.sh
# SLURM job script — OOD stress test (evaluate_ood.py) for the PATCHY/gapped
# checkpoint from jobs/train_gap_perturb.sh, using the same --lp_filter_ranges
# as ID.
#
# Requires the checkpoint from jobs/train_gap_perturb.sh to already exist.
#
# Submit (defaults):
#   sbatch surrogate_model_v3/jobs/evaluate_ood_gap.sh
# =============================================================================

#SBATCH --job-name=eval_surrogate_v3_ood_gap
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/evaluate_ood_gap_%j.log

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
CHECKPOINT="surrogate_model_v3/runs/patchy_100-150_200-250_300-350_perturb0.1/surrogate_best.pt"
ID_RANGES="100-150,200-250,300-350"

python -m surrogate_model_v3.evaluate_ood \
    --checkpoint "$CHECKPOINT" \
    --data_path  "$DATA_PATH"  \
    --id_ranges  "$ID_RANGES"
EXIT_CODE=$?

echo "============================================================"
echo "OOD (gapped) evaluation finished: $(date)  (exit code: $EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
