#!/bin/bash
# =============================================================================
# adapt_domain_surrogate.sh
# SLURM job — few-shot domain adaptation to ct_150, then evaluation plots.
#
# REQUIRED: set BASE_CHECKPOINT to the base_best.pt path from the base training
# job, either by editing this file or passing it via the command line.
#
# Submit (after base training completes):
#   sbatch jobs/adapt_domain_surrogate.sh --base_checkpoint surrogate_domain_adaptation/runs/base/<ts>/base_best.pt
#
# Or with SLURM dependency on the base job (JID = base job ID):
#   sbatch --dependency=afterok:<JID> jobs/adapt_domain_surrogate.sh \
#       --base_checkpoint surrogate_domain_adaptation/runs/base/<ts>/base_best.pt
# =============================================================================

#SBATCH --job-name=adapt_domain_surr
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/adapt_domain_surr_%j.log

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

# ── few-shot K values to sweep ────────────────────────────────────────────────
K_VALUES="5 10 20 50 100"

# ── target domain ─────────────────────────────────────────────────────────────
TARGET_CT=150
TEST_FRACTION=0.30      # 30% of ct_150 held out for evaluation

# ── adapter architecture ──────────────────────────────────────────────────────
BOTTLENECK=32           # adapter bottleneck width

# ── adaptation optimiser ──────────────────────────────────────────────────────
ADAPT_LR=1e-3
ADAPT_WEIGHT_DECAY=1e-5
ADAPT_EPOCHS=200
ADAPT_BATCH_SIZE=32
ADAPT_PATIENCE=30

# ── loss (no rollout for few-shot to avoid overfitting at small K) ────────────
RECON_ST_WEIGHT=1.0
RECON_ST1_WEIGHT=1.0
ROLLOUT_STEPS=0
ROLLOUT_WEIGHT=0.0

NUM_WORKERS=4
SEED=42

# ── parse --base_checkpoint from command line args ────────────────────────────
BASE_CHECKPOINT=""
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_checkpoint)
            BASE_CHECKPOINT="$2"; shift 2 ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [[ -z "$BASE_CHECKPOINT" ]]; then
    echo "ERROR: --base_checkpoint is required."
    echo "Usage: sbatch jobs/adapt_domain_surrogate.sh --base_checkpoint <path>"
    exit 1
fi

echo "Base checkpoint : $BASE_CHECKPOINT"
echo "K values        : $K_VALUES"
echo "Target CT       : ct_${TARGET_CT}"
echo "============================================================"

# ── Step 1: few-shot adaptation ───────────────────────────────────────────────
python -m surrogate_domain_adaptation.adapt \
    --base_checkpoint    "$BASE_CHECKPOINT" \
    --target_ct          $TARGET_CT \
    --test_fraction      $TEST_FRACTION \
    --k_values           $K_VALUES \
    --bottleneck         $BOTTLENECK \
    --adapt_lr           $ADAPT_LR \
    --adapt_weight_decay $ADAPT_WEIGHT_DECAY \
    --adapt_epochs       $ADAPT_EPOCHS \
    --adapt_batch_size   $ADAPT_BATCH_SIZE \
    --adapt_patience     $ADAPT_PATIENCE \
    --recon_st_weight    $RECON_ST_WEIGHT \
    --recon_st1_weight   $RECON_ST1_WEIGHT \
    --rollout_steps      $ROLLOUT_STEPS \
    --rollout_weight     $ROLLOUT_WEIGHT \
    --num_workers        $NUM_WORKERS \
    --seed               $SEED \
    $EXTRA_ARGS

ADAPT_EXIT=$?
if [[ $ADAPT_EXIT -ne 0 ]]; then
    echo "Adaptation failed (exit $ADAPT_EXIT). Skipping evaluate."
    exit $ADAPT_EXIT
fi

# ── Step 2: evaluation plots ──────────────────────────────────────────────────
# Derive adapt_dir from base_checkpoint path: <base_dir>/adapt_ct<TARGET_CT>
BASE_DIR=$(dirname "$BASE_CHECKPOINT")
ADAPT_DIR="${BASE_DIR}/adapt_ct${TARGET_CT}"

echo ""
echo "============================================================"
echo "Running evaluate.py on: $ADAPT_DIR"
echo "============================================================"

python -m surrogate_domain_adaptation.evaluate \
    --base_checkpoint "$BASE_CHECKPOINT" \
    --adapt_dir       "$ADAPT_DIR" \
    --target_ct       $TARGET_CT \
    --seed            $SEED \
    --num_workers     $NUM_WORKERS

EVAL_EXIT=$?
echo "============================================================"
echo "Finished: $(date)  (adapt=$ADAPT_EXIT  eval=$EVAL_EXIT)"
echo "============================================================"
exit $EVAL_EXIT
