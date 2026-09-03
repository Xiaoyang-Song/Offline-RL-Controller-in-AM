#!/bin/bash
# =============================================================================
# train_surrogate_v2_narrow_200_300W_latent256.sh
# SLURM job script — retrain the narrow_200_300W two-stage surrogate
# (surrogate_model_latent_uncertainty_v2/runs/narrow_200_300W/two_stage_best.pt)
# with a WIDER latent space (--latent_dim 256, up from the default 64).
#
# Motivation: baseline_surrogate/ablation_no_latent (same two-stage +
# ensemble architecture, but transitions act directly on the raw 1053-dim
# field instead of a 64-dim learned latent) came out dramatically more
# accurate than the default-latent_dim=64 main surrogate on this same
# 200-300W split — see baseline_surrogate/results/. Before concluding "the
# latent bottleneck doesn't help," this checks whether that's specifically
# a 64-dim-is-too-narrow problem rather than a "latent space is bad" one —
# 64 was carried over from earlier full-range experiments and was never
# specifically tuned for this narrow slice.
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl (extract first via
# surrogate_model_latent_uncertainty_v2/jobs/dataset.sh if missing).
#
# Submit (defaults):
#   sbatch jobs/train_surrogate_v2_narrow_200_300W_latent256.sh
#
# Override any parameter at submission time, e.g. a different latent size:
#   sbatch jobs/train_surrogate_v2_narrow_200_300W_latent256.sh --latent_dim 512
# =============================================================================

#SBATCH --job-name=surrogate_v2_latent256
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/surrogate_v2_latent256_%j.log

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
LATENT_DIM=256

N_ENSEMBLE=5
EPOCHS=500
BATCH_SIZE=128
LR=1e-3
PATIENCE=30

OUT_DIR="surrogate_model_latent_uncertainty_v2/runs/narrow_200_300W_latent${LATENT_DIM}"

# ── train ─────────────────────────────────────────────────────────────────
python -m surrogate_model_latent_uncertainty_v2.train \
    --data_path       "$DATA_PATH"   \
    --lp_filter_min   $LP_MIN        \
    --lp_filter_max   $LP_MAX        \
    --latent_dim      $LATENT_DIM    \
    --n_ensemble      $N_ENSEMBLE    \
    --epochs          $EPOCHS        \
    --batch_size      $BATCH_SIZE    \
    --lr              $LR            \
    --patience        $PATIENCE      \
    --out_dir         "$OUT_DIR"     \
    "$@"

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Checkpoint: $OUT_DIR/two_stage_best.pt"
echo "============================================================"
exit $EXIT_CODE
