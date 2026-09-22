#!/bin/bash
# =============================================================================
# train_gap_perturb.sh
# SLURM job script — train a DELIBERATELY HARD surrogate: PATCHY laser-power
# coverage (three separate training bands with two interior gaps and one
# edge gap) plus additive target noise plus a decorrelated bootstrap
# ensemble. Gives the epistemic-uncertainty channel a real signal to show
# (see evaluate_ood.py / evaluate_ood_ratio.py).
#
# Purpose: imitate limited data access to the historically-known-good
# 150-300W operating band specifically — two of the three gaps (150-200W,
# 250-300W) eat two-thirds of that band, leaving only a thin covered sliver
# (200-250W) inside it, plus a third gap at the top edge (350-400W) as a
# pure extrapolation test alongside the two interpolation tests.
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl.
#
# Submit (defaults):
#   sbatch surrogate_model_v3/jobs/train_gap_perturb.sh
#
# Override any parameter at submission time, e.g.:
#   sbatch surrogate_model_v3/jobs/train_gap_perturb.sh --perturb_frac 0.2
#
# If you change --perturb_frac/--bootstrap_frac here, pass the SAME values
# to jobs/train_full_matched.sh so the two stay a matched pair for
# evaluate_ood_ratio.py.
# =============================================================================

#SBATCH --job-name=surrogate_v3_gap
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/train_gap_perturb_%j.log

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

# ── the experiment's three "make it hard" knobs ─────────────────────────────
# Patchy coverage: data on [100,150], [200,250], [300,350] W, NOTHING in
# (150,200), (250,300), or (350,400] — two interior gaps eating two-thirds of
# the historically-known-good 150-300W band (only 200-250W stays covered
# inside it), plus a genuine edge/extrapolation gap above 350W.
LP_FILTER_RANGES="100-150,200-250,300-350"
# Additive Gaussian noise on heating/next-state TARGETS only, as a FRACTION
# of each node's own state_std. Reasonable starting range: 0.05-0.2.
PERTURB_FRAC=0.1
PERTURB_SEED=0
# Each of K=5 members bootstraps only this fraction of N (< 1.0 default)
# to decorrelate members and sharpen epistemic disagreement.
BOOTSTRAP_FRAC=0.5

N_ENSEMBLE=5
EPOCHS=500
BATCH_SIZE=128
LR=1e-3
PATIENCE=30

OUT_DIR="surrogate_model_v3/runs/patchy_100-150_200-250_300-350_perturb${PERTURB_FRAC}"

python -m surrogate_model_v3.train \
    --data_path        "$DATA_PATH"        \
    --lp_filter_ranges "$LP_FILTER_RANGES" \
    --perturb_frac     $PERTURB_FRAC       \
    --perturb_seed     $PERTURB_SEED       \
    --bootstrap_frac   $BOOTSTRAP_FRAC     \
    --n_ensemble       $N_ENSEMBLE         \
    --epochs           $EPOCHS             \
    --batch_size       $BATCH_SIZE         \
    --lr               $LR                 \
    --patience         $PATIENCE           \
    --out_dir          "$OUT_DIR"          \
    "$@"

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Checkpoint: $OUT_DIR/surrogate_best.pt"
echo "============================================================"
exit $EXIT_CODE
