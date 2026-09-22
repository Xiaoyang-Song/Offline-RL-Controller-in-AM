#!/bin/bash
# =============================================================================
# experiment_full_range.sh
# SLURM job script — Experiment 1: train the proposed surrogate_model_v3
# (two-stage, no latent space) and the four stripped-down baselines
# (mlp, lstm, vanilla_ensemble, kalman_filter — no layer embedding, no
# residual prediction, no learned latent space) on the FULL laser-power
# range, then run summarize_results.py to produce the comparison plots
# (leaderboard, per-layer MAE/RMSE, RMSE-vs-laser-power) in
# baseline_surrogate/results_v3_full/.
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl (already exists).
#
# Submit:
#   sbatch surrogate_model_v3/jobs/experiment_full_range.sh
# =============================================================================

#SBATCH --job-name=exp_v3_full_range
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/experiment_full_range_%j.log

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

DATA="Data/DatasetV2_layer_12_samples_5000.pkl"
EPOCHS=150
PATIENCE=15
BATCH=256
WORKERS=2

set -e

echo "=== [1/6] surrogate_v3 (proposed) ==="
python -m surrogate_model_v3.train \
    --data_path "$DATA" --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH --num_workers $WORKERS \
    --out_dir surrogate_model_v3/runs/full_range

echo "=== [2/6] mlp ==="
python -m baseline_surrogate.mlp.train \
    --data_path "$DATA" --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH --num_workers $WORKERS \
    --out_dir baseline_surrogate/mlp/runs/full_range

echo "=== [3/6] lstm ==="
python -m baseline_surrogate.lstm.train \
    --data_path "$DATA" --epochs $EPOCHS --patience $PATIENCE --batch_size 128 --num_workers $WORKERS \
    --out_dir baseline_surrogate/lstm/runs/full_range

echo "=== [4/6] vanilla_ensemble ==="
python -m baseline_surrogate.vanilla_ensemble.train \
    --data_path "$DATA" --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH --num_workers $WORKERS \
    --out_dir baseline_surrogate/vanilla_ensemble/runs/full_range

echo "=== [5/6] kalman_filter ==="
python -m baseline_surrogate.kalman_filter.train \
    --data_path "$DATA" \
    --out_dir baseline_surrogate/kalman_filter/runs/full_range

echo "=== [6/6] summarize_results (comparison plots) ==="
python -m baseline_surrogate.summarize_results \
    --data_path "$DATA" \
    --surrogate_v3_checkpoint     surrogate_model_v3/runs/full_range/surrogate_best.pt \
    --mlp_checkpoint              baseline_surrogate/mlp/runs/full_range/mlp_best.pt \
    --lstm_checkpoint             baseline_surrogate/lstm/runs/full_range/lstm_best.pt \
    --vanilla_ensemble_checkpoint baseline_surrogate/vanilla_ensemble/runs/full_range/vanilla_ensemble_best.pt \
    --kalman_checkpoint           baseline_surrogate/kalman_filter/runs/full_range/kalman_filter_fitted.pt \
    --id_range_min 100 --id_range_max 400 \
    --out_dir baseline_surrogate/results_v3_full

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Plots    : baseline_surrogate/results_v3_full/"
echo "============================================================"
exit $EXIT_CODE
