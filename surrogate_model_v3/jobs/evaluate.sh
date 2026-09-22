#!/bin/bash
# =============================================================================
# evaluate.sh
# SLURM job script — standard evaluate.py run (single-step/rollout per-layer
# MAE/RMSE + uncertainty plots + example field plots) on the held-out test
# split of whichever checkpoint CHECKPOINT points to.
#
# Submit (defaults to the full_range checkpoint):
#   sbatch surrogate_model_v3/jobs/evaluate.sh
#
# Or point at a different checkpoint / dataset:
#   sbatch surrogate_model_v3/jobs/evaluate.sh \
#       surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt
# =============================================================================

#SBATCH --job-name=eval_surrogate_v3
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/evaluate_%j.log

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
CHECKPOINT="${1:-surrogate_model_v3/runs/full_range/surrogate_best.pt}"

echo "[eval] === evaluate.py (standard single-step/rollout metrics) ==="
python -m surrogate_model_v3.evaluate \
    --checkpoint "$CHECKPOINT" \
    --data_path  "$DATA_PATH"
EXIT_CODE=$?

echo "============================================================"
echo "Surrogate evaluation finished: $(date)  (exit code: $EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
