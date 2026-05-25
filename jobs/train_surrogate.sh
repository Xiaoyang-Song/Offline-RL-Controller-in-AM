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
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/surrogate_train_%j.log

# ── environment ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

# Activate conda environment (required — compute nodes don't inherit shell state)
source ~/.bashrc
conda activate RL

echo "Python : $(which python)"
echo "Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

# ── default hyper-parameters (edit here or pass via sbatch --) ───────────────
DATA_PATH="Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl"

HIDDEN=512          # hidden layer width
DEPTH=5             # number of residual blocks
DROPOUT=0.0         # dropout (0 = off)

EPOCHS=400
BATCH_SIZE=512
LR=1e-3
WEIGHT_DECAY=1e-5
PATIENCE=30         # early-stopping patience (epochs without val improvement)

# Multi-step rollout loss: set ROLLOUT_STEPS>1 to enable
# (e.g. 6 means model is penalised for 6-step accumulated error too)
ROLLOUT_STEPS=0     # 0 = single-step MSE only
ROLLOUT_WEIGHT=0.5  # weight of rollout loss vs 1-step loss
NUM_WORKERS=6       # DataLoader workers (cpus-per-task=8, leave 2 for main process)

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
    --num_workers    $NUM_WORKERS \
    --rollout_steps  $ROLLOUT_STEPS \
    --rollout_weight $ROLLOUT_WEIGHT \
    --val_fraction   $VAL_FRACTION \
    --test_fraction  $TEST_FRACTION \
    --seed           $SEED \
    "$@"             # forward any extra args from: sbatch train_surrogate.sh --arg val

EXIT_CODE=$?
echo "============================================================"
echo "Training finished: $(date)  (exit code: $EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
