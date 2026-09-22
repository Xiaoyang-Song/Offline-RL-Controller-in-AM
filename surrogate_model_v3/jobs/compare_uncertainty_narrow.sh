#!/bin/bash
# =============================================================================
# compare_uncertainty_narrow.sh
# SLURM job script — runs compare_uncertainty_methods.py: diagonal vs.
# low-rank covariance, naive-sum vs. Jacobian-propagated two-stage
# combination (4 configs total), for the NARROW 200-300W-trained
# checkpoints, evaluated on the full-range test set — this is where the
# uncertainty differences between configs should be most visible, since
# everything outside [200, 300]W is genuinely OOD for both checkpoints.
#
# Requires:
#   surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt        (jobs/train_narrow_200_300W.sh)
#   surrogate_model_v3/runs/narrow_200_300W_rank8/surrogate_best.pt  (jobs/train_narrow_200_300W_rank8.sh)
#
# Submit:
#   sbatch surrogate_model_v3/jobs/compare_uncertainty_narrow.sh
# =============================================================================

#SBATCH --job-name=compare_uncertainty_v3_narrow
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/compare_uncertainty_narrow_%j.log

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

python -m surrogate_model_v3.compare_uncertainty_methods \
    --data_path           "$DATA_PATH" \
    --checkpoint_diag     surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt \
    --checkpoint_lowrank  surrogate_model_v3/runs/narrow_200_300W_rank8/surrogate_best.pt \
    --num_probes 4 \
    --id_range_min 200 --id_range_max 300 \
    --out_dir surrogate_model_v3/results_uncertainty_comparison_narrow

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Plots    : surrogate_model_v3/results_uncertainty_comparison_narrow/"
echo "============================================================"
exit $EXIT_CODE
