#!/bin/bash
# =============================================================================
# train_surrogate_v2_gap_perturb.sh
# SLURM job script — train a DELIBERATELY HARD two-stage (heating/cooling)
# LPBF surrogate: PATCHY laser-power coverage (three separate training bands
# with two interior gaps and one edge gap) plus additive target noise plus a
# decorrelated bootstrap ensemble.
#
# Purpose: imitate limited data access to the historically-known-good
# 150-300W operating band specifically — two of the three gaps (150-200W,
# 250-300W) eat two-thirds of that band, leaving only a thin covered sliver
# (200-250W) inside it, plus a third gap at the top edge (350-400W) as a
# pure extrapolation test alongside the two interpolation tests. This gives
# the epistemic-uncertainty channel a real signal to show, so
# online_RL_ucpg_v2 (uncertainty-constrained) can be compared against
# baselines/naive_pg (no uncertainty term) on whether the uncertainty-aware
# policy actually avoids the region this surrogate can't reliably model —
# see surrogate_model_latent_uncertainty_v2/README.md's "Harder / gapped-
# surrogate experiments" section for the full rationale of each knob below.
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl — extract first via
# surrogate_model_latent_uncertainty_v2/jobs/dataset.sh if it doesn't exist.
#
# Submit (defaults):
#   sbatch jobs/train_surrogate_v2_gap_perturb.sh
#
# Override any parameter at submission time, e.g.:
#   sbatch jobs/train_surrogate_v2_gap_perturb.sh --perturb_frac 0.2
# =============================================================================

#SBATCH --job-name=surrogate_v2_gap
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/surrogate_v2_gap_%j.log

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
# of each node's own state_std (e.g. 0.1 = 10% of that node's natural
# variation) — NOT a fixed Kelvin value, since per-node std varies a lot
# across the mesh. Reasonable starting range: 0.05-0.2.
PERTURB_FRAC=0.1
PERTURB_SEED=0
# Each of K=5 members bootstraps only this fraction of N (< 1.0 default)
# to decorrelate members and sharpen epistemic disagreement.
BOOTSTRAP_FRAC=0.5

# ── model / training (unchanged defaults unless noted) ──────────────────────
N_ENSEMBLE=5
EPOCHS=500
BATCH_SIZE=128
LR=1e-3
PATIENCE=30

OUT_DIR="surrogate_model_latent_uncertainty_v2/runs/patchy_100-150_200-250_300-350_perturb${PERTURB_FRAC}"

# ── train ─────────────────────────────────────────────────────────────────
python -m surrogate_model_latent_uncertainty_v2.train \
    --data_path       "$DATA_PATH"       \
    --lp_filter_ranges "$LP_FILTER_RANGES" \
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
echo "Checkpoint: $OUT_DIR/two_stage_best.pt"
echo "============================================================"
exit $EXIT_CODE
