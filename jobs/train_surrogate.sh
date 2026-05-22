#!/bin/bash
# =============================================================================
# train_surrogate.sh
# SLURM job script — train the LPBF Residual Dynamics surrogate model
#
# Submit:
#   sbatch jobs/train_surrogate.sh
#
# Override any arg at submission time, e.g.:
#   sbatch jobs/train_surrogate.sh --rollout_steps 6 --epochs 300
# =============================================================================

#SBATCH --job-name=surrogate_train
#SBATCH --account=jhjin1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/surrogate_train_%j.log

# ── environment ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

# Activate your conda / virtualenv if needed, e.g.:
# source ~/.bashrc
# conda activate <env_name>

# ── default hyper-parameters (edit here or pass via sbatch --) ───────────────
DATA_PATH="Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl"

HIDDEN=512          # hidden layer width
DEPTH=4             # number of residual blocks
DROPOUT=0.0         # dropout (0 = off)

EPOCHS=200
BATCH_SIZE=512
LR=1e-3
WEIGHT_DECAY=1e-5
PATIENCE=20         # early-stopping patience

# Multi-step rollout loss: set ROLLOUT_STEPS>1 to enable
# (e.g. 6 means model is penalised for 6-step accumulated error too)
ROLLOUT_STEPS=0     # 0 = single-step MSE only
ROLLOUT_WEIGHT=0.5  # weight of rollout loss vs 1-step loss

VAL_FRACTION=0.10
TEST_FRACTION=0.10
SEED=42

# ── run training ─────────────────────────────────────────────────────────────
python -m surrogate_model.train \
    --data_path      "$DATA_PATH" \
    --hidden         $HIDDEN \
    --depth          $DEPTH \
    --dropout        $DROPOUT \
    --epochs         $EPOCHS \
    --batch_size     $BATCH_SIZE \
    --lr             $LR \
    --weight_decay   $WEIGHT_DECAY \
    --patience       $PATIENCE \
    --rollout_steps  $ROLLOUT_STEPS \
    --rollout_weight $ROLLOUT_WEIGHT \
    --val_fraction   $VAL_FRACTION \
    --test_fraction  $TEST_FRACTION \
    --seed           $SEED \
    "$@"             # forward any extra args from sbatch command line

echo "============================================================"
echo "Training finished: $(date)"
echo "============================================================"
