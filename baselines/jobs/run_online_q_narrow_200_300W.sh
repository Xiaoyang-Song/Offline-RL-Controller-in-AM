#!/bin/bash
# =============================================================================
# run_online_q_narrow_200_300W.sh
# Trains baseline 5 (online Q-learning) on the existing narrow [200,300]W
# surrogate_model_v3 checkpoint, then aggregates it alongside the OTHER
# five methods already trained/fit on that SAME surrogate (naive_pg,
# offline_q, proportional, kalman_particle, UCPG v2 — all present under
# their own runs/narrow_200_300W or fitted_narrow_200_300W.pt already), for
# a full 6-way leaderboard, in both the surrogate and real physics.
#
#   1. Train online Q-learning -> baselines/online_q/runs/narrow_200_300W/
#   2. Evaluate ALL SIX in the surrogate  (baselines.evaluate_baselines)
#   3. Evaluate ALL SIX in the REAL PDE simulator (baselines.evaluate_real_all_methods)
#
# Submit:
#   sbatch baselines/jobs/run_online_q_narrow_200_300W.sh
# Extra args are forwarded ONLY to the online_q training step, e.g.:
#   sbatch baselines/jobs/run_online_q_narrow_200_300W.sh --n_episodes 8000
# =============================================================================

#SBATCH --job-name=online_q_narrow
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/baselines/jobs/run_online_q_narrow_200_300W_%j.log

echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

module load matlab
source ~/.bashrc
conda activate RL

echo "Python : $(which python)"
echo "Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "MATLAB : $(which matlab)"
echo "============================================================"

set -e
set -o pipefail
PY="python -u"

SURROGATE="surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt"
LP_MIN=200; LP_MAX=300
ACTION_MIN=100; ACTION_MAX=400
OUT="baselines/online_q/runs/narrow_200_300W"

NAIVE_PG_CKPT="baselines/naive_pg/runs/narrow_200_300W/naive_pg_best.pt"
OFFLINE_Q_CKPT="baselines/offline_q/runs/narrow_200_300W/offline_q_best.pt"
UCPG_CKPT="online_RL_ucpg_v2/runs/narrow_200_300W/ucpg_best.pt"
PROPORTIONAL_FIT="baselines/proportional/fitted_narrow_200_300W.pt"
KALMAN_FIT="baselines/kalman_particle/fitted_narrow_200_300W.pt"

N_EPISODES=50
N_EPISODES_REAL=1

echo "=== [1/3] Train online Q-learning on the narrow [$LP_MIN, $LP_MAX]W surrogate ==="
$PY -m baselines.online_q.train \
    --surrogate "$SURROGATE" \
    --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $LP_MIN --ood_max $LP_MAX \
    --out_dir "$OUT" \
    "$@"

echo "=== [2/3] Aggregate — SURROGATE-driven leaderboard (all 6 methods) ==="
$PY -m baselines.evaluate_baselines \
    --surrogate "$SURROGATE" \
    --naive_pg_checkpoint    "$NAIVE_PG_CKPT" \
    --offline_q_checkpoint   "$OFFLINE_Q_CKPT" \
    --online_q_checkpoint    "$OUT/online_q_best.pt" \
    --proportional_fitted    "$PROPORTIONAL_FIT" \
    --kalman_particle_fitted "$KALMAN_FIT" \
    --ucpg_v2_checkpoint     "$UCPG_CKPT" \
    --n_episodes $N_EPISODES \
    --out_dir baselines/results_narrow_200_300W_with_online_q

echo "=== [3/3] Aggregate — REAL PHYSICS leaderboard (all 6 methods) ==="
$PY -m baselines.evaluate_real_all_methods \
    --surrogate "$SURROGATE" \
    --naive_pg_checkpoint    "$NAIVE_PG_CKPT" \
    --offline_q_checkpoint   "$OFFLINE_Q_CKPT" \
    --online_q_checkpoint    "$OUT/online_q_best.pt" \
    --proportional_fitted    "$PROPORTIONAL_FIT" \
    --kalman_particle_fitted "$KALMAN_FIT" \
    --ucpg_v2_checkpoint     "$UCPG_CKPT" \
    --n_episodes $N_EPISODES_REAL \
    --out_dir baselines/results_narrow_200_300W_with_online_q_real

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Online-Q checkpoint       : $OUT/online_q_best.pt"
echo "Surrogate leaderboard     : baselines/results_narrow_200_300W_with_online_q/leaderboard.png / .csv"
echo "Real-physics leaderboard  : baselines/results_narrow_200_300W_with_online_q_real/leaderboard_real.png / .csv"
echo "============================================================"
exit $EXIT_CODE
