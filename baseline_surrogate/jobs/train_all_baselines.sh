#!/bin/bash
# =============================================================================
# train_all_baselines.sh
# SLURM job script — trains/fits all 6 baseline_surrogate/ methods (MLP,
# LSTM, Kalman filter, vanilla deep ensemble, no-two-stage ablation,
# no-latent-space ablation) on the SAME 200-300W laser-power range as the
# existing surrogate_model_latent_uncertainty_v2/runs/narrow_200_300W
# checkpoint, then runs summarize_results.py to compare all of them (plus
# that main-surrogate checkpoint) on the held-out test split.
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl (extract first via
# surrogate_model_latent_uncertainty_v2/jobs/dataset.sh if missing) and
# surrogate_model_latent_uncertainty_v2/runs/narrow_200_300W/two_stage_best.pt
# (already trained).
#
# Submit:
#   sbatch baseline_surrogate/jobs/train_all_baselines.sh
# =============================================================================

#SBATCH --job-name=baseline_surrogate
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/baseline_surrogate/jobs/baseline_surrogate_%j.log

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
LP_MIN=200
LP_MAX=300
TAG="narrow_200_300W"
SURROGATE_CKPT="surrogate_model_latent_uncertainty_v2/runs/narrow_200_300W/two_stage_best.pt"

EPOCHS=300
PATIENCE=20

# ── 1. Plain MLP ──────────────────────────────────────────────────────────
echo "[baseline_surrogate] === 1. Plain MLP ==="
python -m baseline_surrogate.mlp.train \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --epochs $EPOCHS --patience $PATIENCE \
    --out_dir "baseline_surrogate/mlp/runs/${TAG}"

# ── 2. LSTM ──────────────────────────────────────────────────────────────
echo "[baseline_surrogate] === 2. LSTM ==="
python -m baseline_surrogate.lstm.train \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --epochs $EPOCHS --patience $PATIENCE \
    --out_dir "baseline_surrogate/lstm/runs/${TAG}"

# ── 3. Kalman filter (closed-form fit, no epochs) ──────────────────────────
echo "[baseline_surrogate] === 3. Kalman filter ==="
python -m baseline_surrogate.kalman_filter.train \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --out_dir "baseline_surrogate/kalman_filter/runs/${TAG}"

# ── 4. Vanilla deep ensemble ────────────────────────────────────────────
echo "[baseline_surrogate] === 4. Vanilla deep ensemble ==="
python -m baseline_surrogate.vanilla_ensemble.train \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --epochs $EPOCHS --patience $PATIENCE \
    --out_dir "baseline_surrogate/vanilla_ensemble/runs/${TAG}"

# ── 5. Ablation: no two-stage ───────────────────────────────────────────
echo "[baseline_surrogate] === 5. Ablation: no two-stage ==="
python -m baseline_surrogate.ablation_no_two_stage.train \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --epochs $EPOCHS --patience $PATIENCE \
    --out_dir "baseline_surrogate/ablation_no_two_stage/runs/${TAG}"

# ── 6. Ablation: no latent space ────────────────────────────────────────
echo "[baseline_surrogate] === 6. Ablation: no latent space ==="
python -m baseline_surrogate.ablation_no_latent.train \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --epochs $EPOCHS --patience $PATIENCE \
    --out_dir "baseline_surrogate/ablation_no_latent/runs/${TAG}"

# ── Summarize ─────────────────────────────────────────────────────────────
echo "[baseline_surrogate] === Summarizing results ==="
python -m baseline_surrogate.summarize_results \
    --data_path "$DATA_PATH" \
    --surrogate_checkpoint             "$SURROGATE_CKPT" \
    --mlp_checkpoint                   "baseline_surrogate/mlp/runs/${TAG}/mlp_best.pt" \
    --lstm_checkpoint                  "baseline_surrogate/lstm/runs/${TAG}/lstm_best.pt" \
    --kalman_checkpoint                "baseline_surrogate/kalman_filter/runs/${TAG}/kalman_filter_fitted.pt" \
    --vanilla_ensemble_checkpoint      "baseline_surrogate/vanilla_ensemble/runs/${TAG}/vanilla_ensemble_best.pt" \
    --ablation_no_two_stage_checkpoint "baseline_surrogate/ablation_no_two_stage/runs/${TAG}/ablation_no_two_stage_best.pt" \
    --ablation_no_latent_checkpoint    "baseline_surrogate/ablation_no_latent/runs/${TAG}/ablation_no_latent_best.pt" \
    --out_dir "baseline_surrogate/results"

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Results  : baseline_surrogate/results/"
echo "============================================================"
exit $EXIT_CODE
