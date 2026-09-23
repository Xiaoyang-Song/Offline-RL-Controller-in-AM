#!/bin/bash
# =============================================================================
# run_narrow_rl_experiment.sh
# SLURM job script — the 5-method RL/controller comparison on the narrow
# [200, 300]W surrogate, all ported to surrogate_model_v3 (no latent space):
#   0. Calibration          — short UCPG warm-start to pick a binding --delta
#   1. Naive PG              — baselines/naive_pg (no uncertainty term)
#   2. Offline Q              — baselines/offline_q, FIT DATA restricted to [200,300]W
#   3. UCPG v2 (ours)         — online_RL_ucpg_v2 (uses the surrogate's uncertainty
#                               via a Lagrangian constraint, delta budget)
#   4. Proportional control  — baselines/proportional, FIT DATA restricted to [200,300]W
#   5. Kalman filter control — baselines/kalman_particle, FIT DATA restricted to [200,300]W
# ...then aggregates all 5 (+ particle filter, + constant-power sweep, both
# bonus) through baselines/evaluate_baselines.py's shared SURROGATE
# environment, then re-evaluates all 5 again against the REAL PDE simulator
# via baselines/evaluate_real_all_methods.py (AFTER all training/fitting is
# complete). Both leaderboards report a physical temperature-deviation [K]
# panel alongside the abstract return, and both are driven by the SAME
# narrow-trained surrogate/baselines — everything outside [200,300]W is
# genuinely OOD, which is exactly what should make naive PG / offline Q /
# proportional / Kalman struggle relative to UCPG's uncertainty constraint
# (see baselines/README.md's framing).
#
# Requires:
#   surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt
#     (jobs/train_narrow_200_300W.sh — already trained as of this writing)
#   Data/DatasetV2_layer_12_samples_5000.pkl (already exists)
#   MATLAB module (loaded below) + LPBF-Simulation/simulation_v2 checkout
#     alongside this repo, for the real-simulator evaluation stage.
#
# --delta (UCPG's uncertainty budget) is now CALIBRATED AUTOMATICALLY at the
# start of this script (step 0): a short (40-iteration) near-random-policy
# UCPG warm-start reads off the resulting J_u_hat, and delta is set to 40%
# of it — small enough that the uncertainty constraint actually binds (per
# online_RL_ucpg_v2/README.md's calibration procedure), rather than trusting
# a fixed guess. Override CALIB_FRACTION below to loosen/tighten this.
#
# Runtime estimate (~4.2s/policy-gradient-iteration observed previously;
# real PDE solves ~30-45s/layer): calibration ~3min + naive_pg ~2.3h +
# offline_q ~1min + UCPG ~2.4h + proportional/Kalman fits ~2min +
# surrogate-driven eval (n_episodes=50, all methods) ~15min + real-simulator
# eval (n_episodes=3, all methods, ~216 MATLAB layer-solves) ~2-2.5h
# => roughly 7-8h total. --time below gives ~2x margin. NOTE: the real-sim
# stage doesn't need the GPU but runs inside this same GPU allocation for
# simplicity (single job, sequential "train everything, then evaluate
# everything") — split into a separate --partition=standard job yourself if
# you'd rather not hold a GPU idle during that stage.
#
# Submit (defaults):
#   sbatch baselines/jobs/run_narrow_rl_experiment.sh
#
# Override any parameter at submission time, e.g.:
#   sbatch baselines/jobs/run_narrow_rl_experiment.sh --delta 0.03
# (only forwarded to the UCPG v2 FULL training step, after calibration —
# passing --delta explicitly here skips the calibration step's result.)
# =============================================================================

#SBATCH --job-name=narrow_rl_experiment
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/baselines/jobs/run_narrow_rl_experiment_%j.log

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
matlab -batch "disp(version)" 2>&1 | tail -3
echo "============================================================"

set -e
set -o pipefail   # otherwise `python ... | tee log` below would mask a training failure's exit code

# Unbuffered stdout (python -u) everywhere below: without it, Python fully
# block-buffers stdout once it's redirected to a file (SLURM's --output is
# never a TTY), so a long training run's progress can go completely
# invisible in the log until either a buffer flush or process exit — exactly
# what happened last time (UCPG had reached iteration ~1200-1400/2000, with
# checkpoints and plots on disk to prove it, while the log showed zero
# "Iter" lines before the time-limit cancellation).
PY="python -u"

DATA_PATH="Data/DatasetV2_layer_12_samples_5000.pkl"
SURROGATE="surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt"
LP_MIN=200
LP_MAX=300
ACTION_MIN=100
ACTION_MAX=400
N_EPISODES=50
N_EPISODES_REAL=3
CALIB_FRACTION=0.4   # delta = CALIB_FRACTION * (near-random-policy J_u_hat)

echo "=== [0/7] Calibrating UCPG's uncertainty budget (delta) ==="
CALIB_LOG=$(mktemp)
$PY -m online_RL_ucpg_v2.train \
    --surrogate "$SURROGATE" \
    --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $LP_MIN --ood_max $LP_MAX \
    --delta 999.0 --n_iterations 40 --lambda_init 0.0 \
    --out_dir online_RL_ucpg_v2/runs/narrow_200_300W_calib \
    2>&1 | tee "$CALIB_LOG"

J_U_LAST=$(grep -oP 'J_u \K[0-9.]+' "$CALIB_LOG" | tail -1)
if [ -z "$J_U_LAST" ]; then
    echo "WARNING: could not parse J_u_hat from calibration log — falling back to delta=0.05"
    DELTA=0.05
else
    DELTA=$(python3 -c "print(round(float('$J_U_LAST') * $CALIB_FRACTION, 5))")
fi
echo "Calibration: near-random J_u_hat=$J_U_LAST  ->  delta=$DELTA (${CALIB_FRACTION} x)"
rm -f "$CALIB_LOG"
rm -rf online_RL_ucpg_v2/runs/narrow_200_300W_calib

echo "=== [1/7] Naive PG (no uncertainty) ==="
$PY -m baselines.naive_pg.train \
    --surrogate "$SURROGATE" \
    --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $LP_MIN --ood_max $LP_MAX \
    --out_dir baselines/naive_pg/runs/narrow_200_300W

echo "=== [2/7] Offline Q (fit data restricted to [$LP_MIN, $LP_MAX]W) ==="
$PY -m baselines.offline_q.train \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --out_dir baselines/offline_q/runs/narrow_200_300W

echo "=== [3/7] UCPG v2 (ours — uses uncertainty, delta=$DELTA) ==="
$PY -m online_RL_ucpg_v2.train \
    --surrogate "$SURROGATE" \
    --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $LP_MIN --ood_max $LP_MAX \
    --delta $DELTA \
    --out_dir online_RL_ucpg_v2/runs/narrow_200_300W \
    "$@"

echo "=== [4/7] Proportional control (fit data restricted to [$LP_MIN, $LP_MAX]W) ==="
$PY -m baselines.proportional.controller \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --out baselines/proportional/fitted_narrow_200_300W.pt

echo "=== [5/7] Kalman filter control (fit data restricted to [$LP_MIN, $LP_MAX]W) ==="
$PY -m baselines.kalman_particle.filters \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --out baselines/kalman_particle/fitted_narrow_200_300W.pt

echo "=== [6/7] Aggregate + leaderboard — SURROGATE-driven (shared narrow-surrogate environment) ==="
$PY -m baselines.evaluate_baselines \
    --surrogate "$SURROGATE" \
    --naive_pg_checkpoint    baselines/naive_pg/runs/narrow_200_300W/naive_pg_best.pt \
    --offline_q_checkpoint   baselines/offline_q/runs/narrow_200_300W/offline_q_best.pt \
    --ucpg_v2_checkpoint     online_RL_ucpg_v2/runs/narrow_200_300W/ucpg_best.pt \
    --proportional_fitted    baselines/proportional/fitted_narrow_200_300W.pt \
    --kalman_particle_fitted baselines/kalman_particle/fitted_narrow_200_300W.pt \
    --n_episodes $N_EPISODES \
    --out_dir baselines/results_narrow_200_300W

echo "=== [7/7] Aggregate + leaderboard — REAL PHYSICS (after ALL training/fitting above) ==="
$PY -m baselines.evaluate_real_all_methods \
    --surrogate "$SURROGATE" \
    --naive_pg_checkpoint    baselines/naive_pg/runs/narrow_200_300W/naive_pg_best.pt \
    --offline_q_checkpoint   baselines/offline_q/runs/narrow_200_300W/offline_q_best.pt \
    --ucpg_v2_checkpoint     online_RL_ucpg_v2/runs/narrow_200_300W/ucpg_best.pt \
    --proportional_fitted    baselines/proportional/fitted_narrow_200_300W.pt \
    --kalman_particle_fitted baselines/kalman_particle/fitted_narrow_200_300W.pt \
    --n_episodes $N_EPISODES_REAL \
    --out_dir baselines/results_narrow_200_300W_real

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Surrogate leaderboard : baselines/results_narrow_200_300W/leaderboard.png / .csv"
echo "Real-physics leaderboard : baselines/results_narrow_200_300W_real/leaderboard_real.png / .csv"
echo "============================================================"
exit $EXIT_CODE
