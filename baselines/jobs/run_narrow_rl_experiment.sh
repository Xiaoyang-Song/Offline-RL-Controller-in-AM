#!/bin/bash
# =============================================================================
# run_narrow_rl_experiment.sh
# SLURM job script — the 5-method RL/controller comparison on the narrow
# [200, 300]W surrogate, all ported to surrogate_model_v3 (no latent space):
#   1. Naive PG            — baselines/naive_pg (no uncertainty term)
#   2. Offline Q            — baselines/offline_q, FIT DATA restricted to [200,300]W
#   3. UCPG v2 (ours)       — online_RL_ucpg_v2 (uses the surrogate's uncertainty
#                             via a Lagrangian constraint, delta budget)
#   4. Proportional control — baselines/proportional, FIT DATA restricted to [200,300]W
#   5. Kalman filter control — baselines/kalman_particle, FIT DATA restricted to [200,300]W
# ...then aggregates all 5 (+ particle filter, + constant-power sweep, both
# bonus) through baselines/evaluate_baselines.py's shared environment, which
# is driven by the SAME narrow-trained surrogate — everything outside
# [200,300]W is genuinely OOD, which is exactly what should make naive PG /
# offline Q / proportional / Kalman struggle relative to UCPG's uncertainty
# constraint (see baselines/README.md's framing).
#
# Requires:
#   surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt
#     (jobs/train_narrow_200_300W.sh — already trained as of this writing)
#   Data/DatasetV2_layer_12_samples_5000.pkl (already exists)
#
# --delta (UCPG's uncertainty budget) defaults to 0.05 below, which the
# actual narrow_200_300W checkpoint's early-training J_u_hat sits just under
# (a rough calibration point, not a rigorously chosen value) — see
# online_RL_ucpg_v2/README.md's calibration procedure (a short warm-start
# run with a very large --delta, reading off J_u_hat under lambda=0) if you
# want to tune this properly before a real run.
#
# Submit (defaults):
#   sbatch baselines/jobs/run_narrow_rl_experiment.sh
#
# Override any parameter at submission time, e.g.:
#   sbatch baselines/jobs/run_narrow_rl_experiment.sh --delta 0.03
# (only forwarded to the UCPG v2 training step — see DELTA below to change
# it for that step specifically, or edit this script for the others.)
# =============================================================================

#SBATCH --job-name=narrow_rl_experiment
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/baselines/jobs/run_narrow_rl_experiment_%j.log

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

set -e

DATA_PATH="Data/DatasetV2_layer_12_samples_5000.pkl"
SURROGATE="surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt"
LP_MIN=200
LP_MAX=300
ACTION_MIN=100
ACTION_MAX=400
DELTA=0.05
N_EPISODES=50

echo "=== [1/6] Naive PG (no uncertainty) ==="
python -m baselines.naive_pg.train \
    --surrogate "$SURROGATE" \
    --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $LP_MIN --ood_max $LP_MAX \
    --out_dir baselines/naive_pg/runs/narrow_200_300W

echo "=== [2/6] Offline Q (fit data restricted to [$LP_MIN, $LP_MAX]W) ==="
python -m baselines.offline_q.train \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --out_dir baselines/offline_q/runs/narrow_200_300W

echo "=== [3/6] UCPG v2 (ours — uses uncertainty, delta=$DELTA) ==="
python -m online_RL_ucpg_v2.train \
    --surrogate "$SURROGATE" \
    --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $LP_MIN --ood_max $LP_MAX \
    --delta $DELTA \
    --out_dir online_RL_ucpg_v2/runs/narrow_200_300W \
    "$@"

echo "=== [4/6] Proportional control (fit data restricted to [$LP_MIN, $LP_MAX]W) ==="
python -m baselines.proportional.controller \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --out baselines/proportional/fitted_narrow_200_300W.pt

echo "=== [5/6] Kalman filter control (fit data restricted to [$LP_MIN, $LP_MAX]W) ==="
python -m baselines.kalman_particle.filters \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --out baselines/kalman_particle/fitted_narrow_200_300W.pt

echo "=== [6/6] Aggregate + leaderboard (shared narrow-surrogate environment) ==="
python -m baselines.evaluate_baselines \
    --surrogate "$SURROGATE" \
    --naive_pg_checkpoint    baselines/naive_pg/runs/narrow_200_300W/naive_pg_best.pt \
    --offline_q_checkpoint   baselines/offline_q/runs/narrow_200_300W/offline_q_best.pt \
    --ucpg_v2_checkpoint     online_RL_ucpg_v2/runs/narrow_200_300W/ucpg_best.pt \
    --proportional_fitted    baselines/proportional/fitted_narrow_200_300W.pt \
    --kalman_particle_fitted baselines/kalman_particle/fitted_narrow_200_300W.pt \
    --n_episodes $N_EPISODES \
    --out_dir baselines/results_narrow_200_300W

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Leaderboard : baselines/results_narrow_200_300W/leaderboard.png / .csv"
echo "============================================================"
exit $EXIT_CODE
