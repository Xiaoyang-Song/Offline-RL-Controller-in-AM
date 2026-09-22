#!/bin/bash
# =============================================================================
# train_full_matched.sh
# SLURM job script — train a FULL-LASER-POWER-RANGE surrogate that is a
# MATCHED control for jobs/train_gap_perturb.sh's patchy checkpoint:
# identical perturb_frac / bootstrap_frac / architecture / training
# schedule, the ONLY difference being no --lp_filter_ranges (sees every
# laser power).
#
# Why this exists: evaluate_ood_ratio.py's epistemic_patchy / epistemic_full
# ratio is only a valid coverage-isolation test if "full" and "patchy" are
# identical in every respect EXCEPT data coverage — perturb_frac/
# bootstrap_frac inflate raw epistemic magnitude on their own, independent
# of coverage, so an unmatched comparison confounds the ratio.
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl (same as the patchy run).
#
# Submit (defaults match jobs/train_gap_perturb.sh's perturb_frac 0.1 /
# bootstrap_frac 0.5 checkpoint):
#   sbatch surrogate_model_v3/jobs/train_full_matched.sh
#
# If you retrain the patchy checkpoint with different --perturb_frac /
# --bootstrap_frac, pass the SAME values here so the two stay matched, e.g.:
#   sbatch surrogate_model_v3/jobs/train_full_matched.sh --perturb_frac 0.2
# =============================================================================

#SBATCH --job-name=surrogate_v3_full_matched
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/train_full_matched_%j.log

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

# ── MUST match the patchy checkpoint's knobs exactly (only lp_filter differs) ─
PERTURB_FRAC=0.1
PERTURB_SEED=0
BOOTSTRAP_FRAC=0.5

N_ENSEMBLE=5
EPOCHS=500
BATCH_SIZE=128
LR=1e-3
PATIENCE=30

OUT_DIR="surrogate_model_v3/runs/full_matched_perturb${PERTURB_FRAC}_bootstrap${BOOTSTRAP_FRAC}"

# ── train (no --lp_filter_ranges / --lp_filter_min / --lp_filter_max) ───────
python -m surrogate_model_v3.train \
    --data_path       "$DATA_PATH"       \
    --perturb_frac    $PERTURB_FRAC      \
    --perturb_seed    $PERTURB_SEED      \
    --bootstrap_frac  $BOOTSTRAP_FRAC    \
    --n_ensemble      $N_ENSEMBLE        \
    --epochs          $EPOCHS            \
    --batch_size      $BATCH_SIZE        \
    --lr              $LR                \
    --patience        $PATIENCE          \
    --out_dir         "$OUT_DIR"         \
    "$@"

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Checkpoint: $OUT_DIR/surrogate_best.pt"
echo "============================================================"
exit $EXIT_CODE
