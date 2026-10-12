#!/bin/bash
# =============================================================================
# train_gap_100-150_350-400_perturb.sh
# SLURM job script — retrain the two-band GAP surrogate (training data only at
# 100-150 W and 350-400 W; 150-350 W is an interior gap) with a chosen target-
# noise level, then run the OOD evaluation and draw the paper figures.
#
# Same settings as the existing gap_100-150_350-400_perturb0.2_bootstrap0.5 run
# (bootstrap_frac 0.5, 5 members, 500 epochs, batch 128, lr 1e-3, patience 30,
# perturb_seed 0) — only --perturb_frac changes. perturb_frac adds Gaussian noise
# of perturb_frac x each node's state_std to the training TARGETS; at 0.2 the
# learned aleatoric sigma (~45-73 K) dominated the uncertainty plot, so this
# job is for lower levels (e.g. 0.1, 0.05).
#
# Usage (first argument = perturb_frac, default 0.1):
#   sbatch surrogate_model_v3/jobs/train_gap_100-150_350-400_perturb.sh 0.1
#   sbatch surrogate_model_v3/jobs/train_gap_100-150_350-400_perturb.sh 0.05
#
# Outputs:
#   checkpoint : surrogate_model_v3/runs/gap_100-150_350-400_perturb<P>_bootstrap0.5/
#   OOD figures: surrogate_model_v3/results_ood_gap_100-150_350-400_perturb<P>/
#                paper_fig_ood_{both,comb,heat}.{pdf,png} + ood_gap_test.npz
# =============================================================================

#SBATCH --job-name=surrogate_v3_gap_perturb
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/train_gap_100-150_350-400_perturb_%j.log

PERTURB_FRAC="${1:-0.1}"

echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Started      : $(date)"
echo "perturb_frac : $PERTURB_FRAC"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

source ~/.bashrc
conda activate RL

echo "Python : $(which python)"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

set -e

DATA_PATH="Data/DatasetV2_layer_12_samples_5000.pkl"
LP_FILTER_RANGES="100-150,350-400"
BOOTSTRAP_FRAC=0.5

RUN_DIR="surrogate_model_v3/runs/gap_100-150_350-400_perturb${PERTURB_FRAC}_bootstrap${BOOTSTRAP_FRAC}"
OOD_DIR="surrogate_model_v3/results_ood_gap_100-150_350-400_perturb${PERTURB_FRAC}"

echo "=== [1/3] train ==="
python -m surrogate_model_v3.train \
    --data_path        "$DATA_PATH"        \
    --lp_filter_ranges "$LP_FILTER_RANGES" \
    --perturb_frac     "$PERTURB_FRAC"     \
    --perturb_seed     0                   \
    --bootstrap_frac   $BOOTSTRAP_FRAC     \
    --n_ensemble       5                   \
    --epochs           500                 \
    --batch_size       128                 \
    --lr               1e-3                \
    --patience         30                  \
    --out_dir          "$RUN_DIR"

echo "=== [2/3] OOD evaluation (test split, propagated combined uncertainty) ==="
mkdir -p "$OOD_DIR"
python -m surrogate_model_v3.evaluate_ood \
    --checkpoint "$RUN_DIR/surrogate_best.pt" \
    --data_path  "$DATA_PATH"                 \
    --id_ranges  "$LP_FILTER_RANGES"          \
    --split      test                         \
    --propagate_uncertainty                   \
    --out_dir    "$OOD_DIR"                   \
    --dump_npz   "$OOD_DIR/ood_gap_test.npz"

echo "=== [3/3] paper figures ==="
for STAGE in both comb heat; do
    python -m surrogate_model_v3.paper_figure_ood \
        --npz   "$OOD_DIR/ood_gap_test.npz" \
        --stage "$STAGE"                    \
        --out   "$OOD_DIR/paper_fig_ood_${STAGE}"
done

echo "============================================================"
echo "Finished   : $(date)"
echo "Checkpoint : $RUN_DIR/surrogate_best.pt"
echo "Figures    : $OOD_DIR/"
echo "============================================================"
