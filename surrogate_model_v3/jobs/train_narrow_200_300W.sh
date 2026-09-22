#!/bin/bash
# =============================================================================
# train_narrow_200_300W.sh
# SLURM job script — train a surrogate restricted to the historically-known-
# good 200-300W laser-power band, for evaluate_ood.py's ID-vs-OOD stress test
# against the full-range dataset (also matches baseline_surrogate/'s
# narrow_200_300W comparison point).
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl.
#
# Submit (defaults):
#   sbatch surrogate_model_v3/jobs/train_narrow_200_300W.sh
# =============================================================================

#SBATCH --job-name=surrogate_v3_narrow
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/train_narrow_200_300W_%j.log

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
LP_MIN=200
LP_MAX=300

N_ENSEMBLE=5
EPOCHS=500
BATCH_SIZE=128
LR=1e-3
PATIENCE=30

OUT_DIR="surrogate_model_v3/runs/narrow_200_300W"

python -m surrogate_model_v3.train \
    --data_path       "$DATA_PATH" \
    --lp_filter_min   $LP_MIN      \
    --lp_filter_max   $LP_MAX      \
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
