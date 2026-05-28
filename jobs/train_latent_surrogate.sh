#!/bin/bash
# =============================================================================
# train_latent_surrogate.sh
# SLURM job script — train the LPBF Latent-Space Dynamics surrogate model
#
# Submit:
#   sbatch jobs/train_latent_surrogate.sh
#
# Override any arg at submission time, e.g.:
#   sbatch jobs/train_latent_surrogate.sh --latent_dim 128 --epochs 300
# =============================================================================

#SBATCH --job-name=latent_surrogate
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/latent_surrogate_%j.log

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

# ── default hyper-parameters ─────────────────────────────────────────────────
# DATA_PATH="Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl"
DATA_PATH="Data/Dataset:layer_12_stepsize_10_samples_3000_150_400_lc.pkl"

# Model architecture
LATENT_DIM=32       # latent space dimension
ENC_HIDDEN=256      # encoder hidden width
ENC_DEPTH=3         # encoder residual blocks
TRANS_HIDDEN=128    # transition MLP hidden width
TRANS_DEPTH=3       # transition MLP residual blocks
DEC_HIDDEN=256      # decoder hidden width
DEC_DEPTH=3         # decoder residual blocks
DROPOUT=0.1

# Loss weights
RECON_ST_WEIGHT=1.0   # autoencoder reconstruction of s_t
RECON_ST1_WEIGHT=1.0  # predicted s_{t+1} reconstruction
NLL_WEIGHT=0.2        # β-NLL weight
NLL_BETA=0.5          # β for β-NLL: 0=full stop-grad, 0.5=balanced, 1=uniform μ gradient

# Multi-step rollout: set ROLLOUT_STEPS>1 to enable
# Recommended: 12 (full trajectory) — prevents rollout error accumulation
ROLLOUT_STEPS=12
ROLLOUT_WEIGHT=0.2

# Physics ROI — square scan region growing from initialFraction to finalFraction
# (mirrors MATLAB: fractions = linspace(0.4, 0.5, 12), centred at domain centre)
ROI_BOOST=5.0              # weight multiplier inside the square (background = 1)
ROI_INITIAL_FRACTION=0.4   # squareSideFraction at layer 0
ROI_FINAL_FRACTION=0.5     # squareSideFraction at layer 11
ROI_EDGE_SIGMA_FRAC=0.05   # soft-edge width as fraction of min(width, height)

# Training
EPOCHS=500
BATCH_SIZE=256
LR=1e-3
WEIGHT_DECAY=1e-5
PATIENCE=30
NUM_WORKERS=6

VAL_FRACTION=0.10
TEST_FRACTION=0.10
SEED=42

# ── run training ─────────────────────────────────────────────────────────────
python -m surrogate_model_latent.train \
    --data_path        "$DATA_PATH" \
    --latent_dim       $LATENT_DIM \
    --enc_hidden       $ENC_HIDDEN \
    --enc_depth        $ENC_DEPTH \
    --trans_hidden     $TRANS_HIDDEN \
    --trans_depth      $TRANS_DEPTH \
    --dec_hidden       $DEC_HIDDEN \
    --dec_depth        $DEC_DEPTH \
    --dropout          $DROPOUT \
    --recon_st_weight  $RECON_ST_WEIGHT \
    --recon_st1_weight $RECON_ST1_WEIGHT \
    --nll_weight       $NLL_WEIGHT \
    --nll_beta         $NLL_BETA \
    --rollout_steps    $ROLLOUT_STEPS \
    --rollout_weight   $ROLLOUT_WEIGHT \
    --roi_boost              $ROI_BOOST \
    --roi_initial_fraction   $ROI_INITIAL_FRACTION \
    --roi_final_fraction     $ROI_FINAL_FRACTION \
    --roi_edge_sigma_frac    $ROI_EDGE_SIGMA_FRAC \
    --epochs           $EPOCHS \
    --batch_size       $BATCH_SIZE \
    --lr               $LR \
    --weight_decay     $WEIGHT_DECAY \
    --patience         $PATIENCE \
    --num_workers      $NUM_WORKERS \
    --val_fraction     $VAL_FRACTION \
    --test_fraction    $TEST_FRACTION \
    --seed             $SEED \
    "$@"

EXIT_CODE=$?
echo "============================================================"
echo "Training finished: $(date)  (exit code: $EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
