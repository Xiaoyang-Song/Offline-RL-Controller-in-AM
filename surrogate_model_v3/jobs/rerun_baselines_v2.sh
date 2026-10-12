#!/bin/bash
# =============================================================================
# rerun_baselines_v2.sh
# SLURM job script — retrain the surrogate baselines with the revised protocol,
# re-evaluate them against the (unchanged) surrogate_v3 checkpoints, and redraw
# the paper's comparison figures.
#
# Changes vs. experiment_full_range.sh / experiment_narrow_200_300W.sh:
#   * LSTM (narrow only): --ood_mode segment — trained only on contiguous runs of
#     in-range (200-300 W) layers, hidden state reset at each run, so no
#     out-of-range laser power ever enters the network. (Previously it walked
#     full trajectories and only masked the loss.) Full range: no filter, so the
#     LSTM is unchanged and its existing checkpoint is reused.
#   * Kalman filter (both): --no_layer_bias — no layer-index conditioning, like
#     every other baseline.
#   * MLP and vanilla ensemble (both): shorter training budget
#     (BASE_EPOCHS / BASE_PATIENCE below; previously 150 / 15).
#   * surrogate_v3: NOT retrained — existing checkpoints are reused.
#
# Nothing old is overwritten except the two paper figures (tracked in git):
#   new checkpoints : baseline_surrogate/{mlp,lstm,vanilla_ensemble,kalman_filter}/runs/<case>_v2/
#   new results     : baseline_surrogate/results_v3_{full,narrow}_v2/  (+ full.npz / narrow.npz)
#   paper figures   : Document/figures/fig_per_layer_mae_full.png
#                     Document/figures/fig_rmse_vs_action_narrow.png
#
# Submit:
#   sbatch surrogate_model_v3/jobs/rerun_baselines_v2.sh
# =============================================================================

#SBATCH --job-name=baselines_v2
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32GB
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/rerun_baselines_v2_%j.log

echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc
conda activate RL
echo "Python : $(which python)"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

set -e

DATA="Data/DatasetV2_layer_12_samples_5000.pkl"
WORKERS=2

# MLP / vanilla ensemble budget (was 150 epochs, patience 15)
BASE_EPOCHS=100
BASE_PATIENCE=10
# LSTM keeps its previous budget; only the out-of-range handling changes
LSTM_EPOCHS=150
LSTM_PATIENCE=15

RAW_DIR="baseline_surrogate/results_v3_raw_v2"
mkdir -p "$RAW_DIR"

# =============================================================================
# FULL RANGE (100-400 W)
# =============================================================================
echo "=== [full 1/4] mlp ==="
python -m baseline_surrogate.mlp.train \
    --data_path "$DATA" --epochs $BASE_EPOCHS --patience $BASE_PATIENCE \
    --batch_size 256 --num_workers $WORKERS \
    --out_dir baseline_surrogate/mlp/runs/full_range_v2

echo "=== [full 2/4] vanilla_ensemble ==="
python -m baseline_surrogate.vanilla_ensemble.train \
    --data_path "$DATA" --epochs $BASE_EPOCHS --patience $BASE_PATIENCE \
    --batch_size 256 --num_workers $WORKERS \
    --out_dir baseline_surrogate/vanilla_ensemble/runs/full_range_v2

echo "=== [full 3/4] kalman_filter (no layer bias) ==="
python -m baseline_surrogate.kalman_filter.train \
    --data_path "$DATA" --no_layer_bias \
    --out_dir baseline_surrogate/kalman_filter/runs/full_range_v2

echo "=== [full 4/4] summarize (surrogate_v3 + LSTM: existing full-range checkpoints) ==="
python -m baseline_surrogate.summarize_results \
    --data_path "$DATA" \
    --surrogate_v3_checkpoint     surrogate_model_v3/runs/full_range/surrogate_best.pt \
    --mlp_checkpoint              baseline_surrogate/mlp/runs/full_range_v2/mlp_best.pt \
    --lstm_checkpoint             baseline_surrogate/lstm/runs/full_range/lstm_best.pt \
    --vanilla_ensemble_checkpoint baseline_surrogate/vanilla_ensemble/runs/full_range_v2/vanilla_ensemble_best.pt \
    --kalman_checkpoint           baseline_surrogate/kalman_filter/runs/full_range_v2/kalman_filter_fitted.pt \
    --dump_npz "$RAW_DIR/full.npz" \
    --out_dir baseline_surrogate/results_v3_full_v2

# =============================================================================
# NARROW (trained on 200-300 W, evaluated on the full 100-400 W test split)
# =============================================================================
LP_MIN=200
LP_MAX=300

echo "=== [narrow 1/5] lstm (segment mode: no out-of-range inputs) ==="
python -m baseline_surrogate.lstm.train \
    --data_path "$DATA" --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX --ood_mode segment \
    --epochs $LSTM_EPOCHS --patience $LSTM_PATIENCE --batch_size 128 --num_workers $WORKERS \
    --out_dir baseline_surrogate/lstm/runs/narrow_200_300W_v2

echo "=== [narrow 2/5] mlp ==="
python -m baseline_surrogate.mlp.train \
    --data_path "$DATA" --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --epochs $BASE_EPOCHS --patience $BASE_PATIENCE --batch_size 256 --num_workers $WORKERS \
    --out_dir baseline_surrogate/mlp/runs/narrow_200_300W_v2

echo "=== [narrow 3/5] vanilla_ensemble ==="
python -m baseline_surrogate.vanilla_ensemble.train \
    --data_path "$DATA" --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --epochs $BASE_EPOCHS --patience $BASE_PATIENCE --batch_size 256 --num_workers $WORKERS \
    --out_dir baseline_surrogate/vanilla_ensemble/runs/narrow_200_300W_v2

echo "=== [narrow 4/5] kalman_filter (no layer bias) ==="
python -m baseline_surrogate.kalman_filter.train \
    --data_path "$DATA" --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX --no_layer_bias \
    --out_dir baseline_surrogate/kalman_filter/runs/narrow_200_300W_v2

echo "=== [narrow 5/5] summarize (surrogate_v3: existing narrow checkpoint) ==="
python -m baseline_surrogate.summarize_results \
    --data_path "$DATA" \
    --surrogate_v3_checkpoint     surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt \
    --mlp_checkpoint              baseline_surrogate/mlp/runs/narrow_200_300W_v2/mlp_best.pt \
    --lstm_checkpoint             baseline_surrogate/lstm/runs/narrow_200_300W_v2/lstm_best.pt \
    --vanilla_ensemble_checkpoint baseline_surrogate/vanilla_ensemble/runs/narrow_200_300W_v2/vanilla_ensemble_best.pt \
    --kalman_checkpoint           baseline_surrogate/kalman_filter/runs/narrow_200_300W_v2/kalman_filter_fitted.pt \
    --id_range_min $LP_MIN --id_range_max $LP_MAX \
    --dump_npz "$RAW_DIR/narrow.npz" \
    --out_dir baseline_surrogate/results_v3_narrow_v2

# =============================================================================
# PAPER FIGURES
# =============================================================================
echo "=== redraw paper figures ==="
python Document/make_figures.py --raw_dir "$RAW_DIR" --out_dir Document/figures

echo "============================================================"
echo "Finished : $(date)"
echo "Results  : baseline_surrogate/results_v3_full_v2/, baseline_surrogate/results_v3_narrow_v2/"
echo "Figures  : Document/figures/fig_per_layer_mae_full.png, fig_rmse_vs_action_narrow.png"
echo "============================================================"
