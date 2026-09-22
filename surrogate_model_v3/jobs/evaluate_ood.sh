#!/bin/bash
# =============================================================================
# evaluate_ood.sh
# SLURM job script — OOD stress test (evaluate_ood.py) for the NARROW
# (200-300W) checkpoint from jobs/train_narrow_200_300W.sh: ID vs. OOD
# epistemic/aleatoric/RMSE split against the full laser-power range.
#
# Requires the checkpoint from jobs/train_narrow_200_300W.sh to already exist.
#
# Submit (defaults):
#   sbatch surrogate_model_v3/jobs/evaluate_ood.sh
# =============================================================================

#SBATCH --job-name=eval_surrogate_v3_ood
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/evaluate_ood_%j.log

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
CHECKPOINT="surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt"

python -m surrogate_model_v3.evaluate_ood \
    --checkpoint     "$CHECKPOINT" \
    --data_path      "$DATA_PATH"  \
    --id_action_min  200           \
    --id_action_max  300
EXIT_CODE=$?

echo "============================================================"
echo "OOD evaluation finished: $(date)  (exit code: $EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
