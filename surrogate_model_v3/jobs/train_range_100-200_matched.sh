#!/bin/bash
# =============================================================================
# train_range_100-200_matched.sh
# SLURM job script — train the NOISE-FREE 100-200 W surrogate (training data
# only at laser power 100-200 W) with EXACTLY the settings of the paper's
# full_range / narrow_200_300W / gap matched checkpoints:
#   perturb_frac 0.0 (noise-free), bootstrap_frac 1.0, rank 0 (diagonal),
#   NO ROI loss weighting, 150 epochs, batch 256, lr 1e-3, patience 15,
#   5 members, seed 42 split.
# (The existing range_100-200_perturb0.2_bootstrap0.5 run differs: 20 % target
#  noise, bootstrap_frac 0.5, ROI weighting on, 500 epochs, batch 128, patience 30.)
#
# Then: OOD evaluation on the test split + paper_fig_ood_{both,comb,heat}.
# The RL launchers use it as scenario "range_100-200_clean":
#   bash online_RL_ucpg_v3/jobs/run_scenario_mult.sh range_100-200_clean heat
#
# Usage:
#   sbatch surrogate_model_v3/jobs/train_range_100-200_matched.sh
#
# Outputs:
#   checkpoint : surrogate_model_v3/runs/range_100-200_perturb0.0_bootstrap1.0/
#   OOD figures: surrogate_model_v3/results_ood_range_100-200_matched/
# =============================================================================

#SBATCH --job-name=surrogate_v3_range_100-200
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/train_range_100-200_matched_%j.log

echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Started      : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

source ~/.bashrc
conda activate RL

echo "Python : $(which python)"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

set -e

DATA_PATH="Data/DatasetV2_layer_12_samples_5000.pkl"
LP_FILTER_RANGES="100-200"

RUN_DIR="surrogate_model_v3/runs/range_100-200_perturb0.0_bootstrap1.0"
OOD_DIR="surrogate_model_v3/results_ood_range_100-200_matched"

echo "=== [1/2] train ==="
python -m surrogate_model_v3.train \
    --data_path        "$DATA_PATH"        \
    --lp_filter_ranges "$LP_FILTER_RANGES" \
    --perturb_frac     0.0                 \
    --perturb_seed     0                   \
    --bootstrap_frac   1.0                 \
    --rank             0                   \
    --no_roi_weights                       \
    --n_ensemble       5                   \
    --epochs           150                 \
    --batch_size       256                 \
    --lr               1e-3                \
    --patience         15                  \
    --num_workers      2                   \
    --out_dir          "$RUN_DIR"

echo "=== [2/2] OOD evaluation (test split, propagated combined uncertainty) ==="
mkdir -p "$OOD_DIR"
python -m surrogate_model_v3.evaluate_ood \
    --checkpoint "$RUN_DIR/surrogate_best.pt" \
    --data_path  "$DATA_PATH"                 \
    --id_ranges  "$LP_FILTER_RANGES"          \
    --split      test                         \
    --propagate_uncertainty                   \
    --out_dir    "$OOD_DIR"                   \
    --dump_npz   "$OOD_DIR/ood_range_test.npz"
for STAGE in both comb heat; do
    python -m surrogate_model_v3.paper_figure_ood \
        --npz   "$OOD_DIR/ood_range_test.npz" \
        --stage "$STAGE"                      \
        --out   "$OOD_DIR/paper_fig_ood_${STAGE}"
done

echo "============================================================"
echo "Finished   : $(date)"
echo "Checkpoint : $RUN_DIR/surrogate_best.pt"
echo "OOD        : $OOD_DIR/"
echo "============================================================"
