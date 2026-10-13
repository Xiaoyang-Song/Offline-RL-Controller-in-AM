#!/bin/bash
# =============================================================================
# baselines_gap.sh
# SLURM job script — train the four surrogate baselines on the two-band GAP
# data (laser power 100-150 W and 350-400 W only), evaluate them next to the
# matched gap surrogate_v3 checkpoint on the full 100-400 W test split, and
# redraw the per-layer RMSE paper figure with panel (d) = gap, teacher-forced.
#
# Protocol identical to rerun_baselines_v2.sh (narrow case), only the training
# power filter changes (--lp_filter_ranges instead of --lp_filter_min/max):
#   LSTM          : --ood_mode segment (only contiguous in-range runs, no
#                   out-of-range power ever enters the network), 150 ep / pat 15
#   MLP, ensemble : 100 epochs / patience 10
#   Kalman filter : --no_layer_bias
#   surrogate_v3  : NOT trained here — uses the matched gap checkpoint from
#                   surrogate_model_v3/jobs/train_gap_matched.sh (bootstrap 1.0,
#                   noise-free, same settings as full_range / narrow).
#
# Submit AFTER train_gap_matched.sh has finished, or chain it:
#   JID=$(sbatch --parsable surrogate_model_v3/jobs/train_gap_matched.sh)
#   sbatch --dependency=afterok:$JID surrogate_model_v3/jobs/baselines_gap.sh
# (if train_gap_matched.sh already finished: sbatch surrogate_model_v3/jobs/baselines_gap.sh)
#
# Outputs (nothing existing is overwritten except the new RMSE figure):
#   checkpoints : baseline_surrogate/{mlp,lstm,vanilla_ensemble,kalman_filter}/runs/gap_100-150_350-400_v2/
#   results     : baseline_surrogate/results_v3_gap_v2/   (its rmse-vs-action plot shades the
#                 default 200-300 W band — ignore that panel, summarize_results has one ID range)
#   raw arrays  : baseline_surrogate/results_v3_raw_v2/gap.npz
#   figure      : Document/figures/fig_per_layer_rmse.{pdf,png}  (4 panels once gap.npz exists)
# =============================================================================

#SBATCH --job-name=baselines_gap
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32GB
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/baselines_gap_%j.log

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
BASE_EPOCHS=100
BASE_PATIENCE=10
LSTM_EPOCHS=150
LSTM_PATIENCE=15

RANGES="100-150,350-400"
CASE="gap_100-150_350-400_v2"
V3_CKPT="surrogate_model_v3/runs/gap_100-150_350-400_perturb0.0_bootstrap1.0/surrogate_best.pt"
RAW_DIR="baseline_surrogate/results_v3_raw_v2"

if [ ! -f "$V3_CKPT" ]; then
    echo "ERROR: $V3_CKPT not found — run surrogate_model_v3/jobs/train_gap_matched.sh first." >&2
    exit 1
fi

echo "=== [gap 1/5] lstm (segment mode: no out-of-range inputs) ==="
python -m baseline_surrogate.lstm.train \
    --data_path "$DATA" --lp_filter_ranges "$RANGES" --ood_mode segment \
    --epochs $LSTM_EPOCHS --patience $LSTM_PATIENCE --batch_size 128 --num_workers $WORKERS \
    --out_dir baseline_surrogate/lstm/runs/$CASE

echo "=== [gap 2/5] mlp ==="
python -m baseline_surrogate.mlp.train \
    --data_path "$DATA" --lp_filter_ranges "$RANGES" \
    --epochs $BASE_EPOCHS --patience $BASE_PATIENCE --batch_size 256 --num_workers $WORKERS \
    --out_dir baseline_surrogate/mlp/runs/$CASE

echo "=== [gap 3/5] vanilla_ensemble ==="
python -m baseline_surrogate.vanilla_ensemble.train \
    --data_path "$DATA" --lp_filter_ranges "$RANGES" \
    --epochs $BASE_EPOCHS --patience $BASE_PATIENCE --batch_size 256 --num_workers $WORKERS \
    --out_dir baseline_surrogate/vanilla_ensemble/runs/$CASE

echo "=== [gap 4/5] kalman_filter (no layer bias) ==="
python -m baseline_surrogate.kalman_filter.train \
    --data_path "$DATA" --lp_filter_ranges "$RANGES" --no_layer_bias \
    --out_dir baseline_surrogate/kalman_filter/runs/$CASE

echo "=== [gap 5/5] summarize (surrogate_v3: matched gap checkpoint) ==="
python -m baseline_surrogate.summarize_results \
    --data_path "$DATA" \
    --surrogate_v3_checkpoint     "$V3_CKPT" \
    --mlp_checkpoint              baseline_surrogate/mlp/runs/$CASE/mlp_best.pt \
    --lstm_checkpoint             baseline_surrogate/lstm/runs/$CASE/lstm_best.pt \
    --vanilla_ensemble_checkpoint baseline_surrogate/vanilla_ensemble/runs/$CASE/vanilla_ensemble_best.pt \
    --kalman_checkpoint           baseline_surrogate/kalman_filter/runs/$CASE/kalman_filter_fitted.pt \
    --dump_npz "$RAW_DIR/gap.npz" \
    --out_dir baseline_surrogate/results_v3_gap_v2

echo "=== per-layer RMSE figure (4 panels) ==="
python Document/make_fig_per_layer_rmse.py --raw_dir "$RAW_DIR"

echo "============================================================"
echo "Finished : $(date)"
echo "Figure   : Document/figures/fig_per_layer_rmse.{pdf,png}"
echo "============================================================"
