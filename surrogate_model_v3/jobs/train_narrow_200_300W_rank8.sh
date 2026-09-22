#!/bin/bash
# =============================================================================
# train_narrow_200_300W_rank8.sh
# SLURM job script — train the LOW-RANK covariance variant (--rank 8, see
# model.py's module docstring) on the NARROW 200-300W laser-power range.
# Matched pair with jobs/train_narrow_200_300W.sh (same data/filter/
# architecture/schedule, only --rank differs) so the two checkpoints can be
# compared directly via compare_uncertainty_narrow.sh — this is where the
# diagonal-vs-low-rank / naive-vs-propagated uncertainty differences should
# be most visible, since laser power outside [200, 300]W is genuinely OOD
# for both checkpoints.
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl (already exists).
#
# Submit:
#   sbatch surrogate_model_v3/jobs/train_narrow_200_300W_rank8.sh
# =============================================================================

#SBATCH --job-name=surrogate_v3_narrow_rank8
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/train_narrow_200_300W_rank8_%j.log

echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

source ~/.bashrc
conda activate RL

echo "Python : $(which python)"
echo "Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

DATA_PATH="Data/DatasetV2_layer_12_samples_5000.pkl"
LP_MIN=200
LP_MAX=300
RANK=8

N_ENSEMBLE=5
EPOCHS=500
BATCH_SIZE=128
LR=1e-3
PATIENCE=30

OUT_DIR="surrogate_model_v3/runs/narrow_200_300W_rank${RANK}"

python -m surrogate_model_v3.train \
    --data_path       "$DATA_PATH" \
    --lp_filter_min   $LP_MIN      \
    --lp_filter_max   $LP_MAX      \
    --rank            $RANK        \
    --n_ensemble      $N_ENSEMBLE  \
    --epochs          $EPOCHS      \
    --batch_size      $BATCH_SIZE  \
    --lr              $LR          \
    --patience        $PATIENCE    \
    --out_dir         "$OUT_DIR"   \
    "$@"

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Checkpoint: $OUT_DIR/surrogate_best.pt"
echo "============================================================"
exit $EXIT_CODE
