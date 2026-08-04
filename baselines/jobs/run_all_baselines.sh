#!/bin/bash
# =============================================================================
# run_all_baselines.sh
# SLURM job script — fits/trains baselines 1, 2, 4 (no-training baseline 3's
# constant sweep runs inside the aggregation step) against the real surrogate
# + full dataset, then aggregates everything into one leaderboard. See
# baselines/README.md for what each method is and why.
#
# Baselines 0 (naive PG) and the real UCPG v2 checkpoint are intentionally
# NOT included here for now — naive_pg hasn't been trained yet (~9h) and the
# UCPG v2 run is still in progress. Add them back later with
# --naive_pg_checkpoint / --ucpg_v2_checkpoint once ready (see
# baselines/README.md for the full command).
#
# All evaluation here is against the SURROGATE model, not the real PDE
# simulator — that comparison is a separate, not-yet-built piece (needs a
# MATLAB engine bridge; see conversation notes / ask for it when ready).
#
# Submit (defaults):
#   sbatch baselines/jobs/run_all_baselines.sh
# =============================================================================

#SBATCH --job-name=baselines
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/baselines/jobs/baselines_%j.log

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
echo "============================================================"

SURROGATE="surrogate_model_latent_uncertainty_v2/runs/20260803_184916/two_stage_best.pt"
DATA_PATH="Data/DatasetV2_layer_12_samples_5000.pkl"

OFFLINE_Q_OUT="baselines/offline_q/runs/cpu_run"
PROPORTIONAL_OUT="baselines/proportional/fitted.pt"
KALMAN_OUT="baselines/kalman_particle/fitted.pt"
RESULTS_OUT="baselines/results"

# ── 1. Offline Q-learning ────────────────────────────────────────────────────
echo "[baselines] === 1. Offline Q-learning ==="
python -m baselines.offline_q.train \
    --data_path "$DATA_PATH" \
    --epochs 50 \
    --out_dir "$OFFLINE_Q_OUT"

# ── 2. Proportional controller ──────────────────────────────────────────────
echo "[baselines] === 2. Proportional controller ==="
python -m baselines.proportional.controller \
    --data_path "$DATA_PATH" \
    --out "$PROPORTIONAL_OUT"

# ── 4. Kalman / particle filter ─────────────────────────────────────────────
echo "[baselines] === 4. Kalman / particle filter ==="
python -m baselines.kalman_particle.filters \
    --data_path "$DATA_PATH" \
    --out "$KALMAN_OUT"

# ── Aggregate (baseline 3's constant sweep runs inside this step) ──────────
echo "[baselines] === Aggregating all results ==="
python -m baselines.evaluate_baselines \
    --surrogate "$SURROGATE" \
    --offline_q_checkpoint     "$OFFLINE_Q_OUT/offline_q_best.pt" \
    --proportional_fitted      "$PROPORTIONAL_OUT" \
    --kalman_particle_fitted   "$KALMAN_OUT" \
    --n_episodes 50 \
    --out_dir "$RESULTS_OUT"

EXIT_CODE=$?
echo "============================================================"
echo "Baselines finished: $(date)  (exit code: $EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
