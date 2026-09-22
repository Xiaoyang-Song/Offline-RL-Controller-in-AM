#!/bin/bash
# =============================================================================
# train_full_range_rank8.sh
# SLURM job script — train the LOW-RANK covariance variant of the proposed
# surrogate (--rank 8, i.e. each ensemble member ALSO learns a low-rank
# aleatoric factor U in R^(D x 8), so Sigma = diag(sigma^2) + U U^T instead
# of pure diagonal — see model.py's module docstring) on the full
# laser-power range. Same data/architecture/schedule as
# jobs/train_full_range.sh otherwise, so the two checkpoints are a matched
# pair for compare_uncertainty_methods.py.
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl (already exists).
#
# Submit:
#   sbatch surrogate_model_v3/jobs/train_full_range_rank8.sh
# =============================================================================

#SBATCH --job-name=surrogate_v3_full_rank8
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/train_full_range_rank8_%j.log

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

RANK=8
N_ENSEMBLE=5
EPOCHS=500
BATCH_SIZE=128
LR=1e-3
PATIENCE=30

OUT_DIR="surrogate_model_v3/runs/full_range_rank${RANK}"

python -m surrogate_model_v3.train \
    --data_path     "$DATA_PATH"  \
    --rank          $RANK         \
    --n_ensemble    $N_ENSEMBLE   \
    --epochs        $EPOCHS       \
    --batch_size    $BATCH_SIZE   \
    --lr            $LR           \
    --patience      $PATIENCE     \
    --out_dir       "$OUT_DIR"    \
    "$@"

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Checkpoint: $OUT_DIR/surrogate_best.pt"
echo "============================================================"
exit $EXIT_CODE
