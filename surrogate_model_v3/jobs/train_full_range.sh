#!/bin/bash
# =============================================================================
# train_full_range.sh
# SLURM job script — train the primary two-stage (heating/cooling) LPBF
# surrogate on the FULL laser-power range, no coverage restriction. This is
# the canonical checkpoint for this package: no latent bottleneck, bootstrap
# Gaussian ensemble, standard N-out-of-N bootstrap (bootstrap_frac=1.0), no
# target perturbation.
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl (extract first via
# surrogate_model_v3/jobs/dataset.sh if missing — but this file already
# exists in Data/, so normally nothing to do here).
#
# Submit (defaults):
#   sbatch surrogate_model_v3/jobs/train_full_range.sh
#
# Override any parameter at submission time, e.g.:
#   sbatch surrogate_model_v3/jobs/train_full_range.sh --n_ensemble 8
# =============================================================================

#SBATCH --job-name=surrogate_v3_full
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/train_full_range_%j.log

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
echo "Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

DATA_PATH="Data/DatasetV2_layer_12_samples_5000.pkl"

N_ENSEMBLE=5
EPOCHS=500
BATCH_SIZE=128
LR=1e-3
PATIENCE=30

OUT_DIR="surrogate_model_v3/runs/full_range"

python -m surrogate_model_v3.train \
    --data_path     "$DATA_PATH"  \
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
