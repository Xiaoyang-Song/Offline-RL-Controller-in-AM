#!/bin/bash
# =============================================================================
# experiment_narrow_200_300W.sh
# SLURM job script — Experiment 2: train the proposed surrogate_model_v3
# (two-stage, no latent space) and the four stripped-down baselines
# (mlp, lstm, vanilla_ensemble, kalman_filter — no layer embedding, no
# residual prediction, no learned latent space) on the NARROW 200-300W
# laser-power range only, then evaluate all five on the FULL-range test
# split via summarize_results.py — this is the extrapolation/OOD comparison:
# everything outside [200, 300]W is laser power none of these models were
# trained on. Plots land in baseline_surrogate/results_v3_narrow/, with the
# rmse_vs_action*.png plots shading the [200, 300]W training range so the
# OOD region is visually obvious.
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl (already exists).
#
# Submit:
#   sbatch surrogate_model_v3/jobs/experiment_narrow_200_300W.sh
# =============================================================================

#SBATCH --job-name=exp_v3_narrow
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/experiment_narrow_200_300W_%j.log

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
LP_MIN=200
LP_MAX=300
EPOCHS=150
PATIENCE=15
BATCH=256
WORKERS=2

set -e

echo "=== [1/6] surrogate_v3 (proposed) ==="
python -m surrogate_model_v3.train \
    --data_path "$DATA" --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH --num_workers $WORKERS \
    --out_dir surrogate_model_v3/runs/narrow_200_300W

echo "=== [2/6] mlp ==="
python -m baseline_surrogate.mlp.train \
    --data_path "$DATA" --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH --num_workers $WORKERS \
    --out_dir baseline_surrogate/mlp/runs/narrow_200_300W

echo "=== [3/6] lstm ==="
python -m baseline_surrogate.lstm.train \
    --data_path "$DATA" --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --epochs $EPOCHS --patience $PATIENCE --batch_size 128 --num_workers $WORKERS \
    --out_dir baseline_surrogate/lstm/runs/narrow_200_300W

echo "=== [4/6] vanilla_ensemble ==="
python -m baseline_surrogate.vanilla_ensemble.train \
    --data_path "$DATA" --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH --num_workers $WORKERS \
    --out_dir baseline_surrogate/vanilla_ensemble/runs/narrow_200_300W

echo "=== [5/6] kalman_filter ==="
python -m baseline_surrogate.kalman_filter.train \
    --data_path "$DATA" --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --out_dir baseline_surrogate/kalman_filter/runs/narrow_200_300W

echo "=== [6/6] summarize_results (comparison plots, evaluated on FULL-range test set) ==="
python -m baseline_surrogate.summarize_results \
    --data_path "$DATA" \
    --surrogate_v3_checkpoint     surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt \
    --mlp_checkpoint              baseline_surrogate/mlp/runs/narrow_200_300W/mlp_best.pt \
    --lstm_checkpoint             baseline_surrogate/lstm/runs/narrow_200_300W/lstm_best.pt \
    --vanilla_ensemble_checkpoint baseline_surrogate/vanilla_ensemble/runs/narrow_200_300W/vanilla_ensemble_best.pt \
    --kalman_checkpoint           baseline_surrogate/kalman_filter/runs/narrow_200_300W/kalman_filter_fitted.pt \
    --id_range_min $LP_MIN --id_range_max $LP_MAX \
    --out_dir baseline_surrogate/results_v3_narrow

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Plots    : baseline_surrogate/results_v3_narrow/"
echo "============================================================"
exit $EXIT_CODE
