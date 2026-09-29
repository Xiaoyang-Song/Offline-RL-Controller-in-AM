#!/bin/bash
# =============================================================================
# gap_100-150_300-350_experiment.sh
# One-shot SLURM job — gapped-coverage surrogate_model_v3 experiment, targeting
# the "does uncertainty actually help" question directly:
#
#   0. Train surrogate_model_v3 on TWO disjoint laser-power islands,
#      100-150W and 300-350W, leaving 150-300W (and outside 100-350W) as a
#      genuine, un-simulated gap the model must extrapolate across. This is
#      the multi-island design that showed a real (if modest) naive_pg-vs-
#      UCPG gap on the retired latent surrogate (jobs/oneshot_gap_*.log:
#      naive_pg real return -6.49 vs UCPG -6.34) — unlike the single
#      contiguous 200-300W band, where naive_pg and UCPG were statistically
#      tied (both surrogate and real evaluation).
#   1. Full surrogate evaluation (evaluate.py: per-layer MAE/RMSE,
#      uncertainty, example fields).
#   2. OOD stress test (evaluate_ood.py --id_ranges "100-150,300-350"):
#      epistemic/aleatoric/RMSE vs. laser power (heating-only AND combined
#      heat+cool), with BOTH islands shaded — this is the "uncertainty vs.
#      action" plot.
#   3. Calibrate UCPG's uncertainty budget delta (40-iter near-random warm
#      start on this new surrogate).
#   4. Train naive PG (no uncertainty term) on this surrogate — needed as
#      the comparison point; without it there is nothing to show the
#      uncertainty constraint improves on.
#   5. Train UCPG v2 with --relative_uncertainty (see online_RL_ucpg_v2/train.py
#      — makes the Lagrange multiplier react on the RELATIVE constraint
#      violation J_u_hat/delta - 1, instead of the raw J_u_hat scale, which
#      was too small for lambda to ever move the policy noticeably).
#   6. Evaluate naive_pg + UCPG together in the surrogate
#      (baselines.evaluate_baselines, skip the 31-point constant sweep).
#   7. Evaluate naive_pg + UCPG together in the REAL PDE simulator
#      (baselines.evaluate_real_all_methods) — the actual test of the
#      hypothesis.
#
# --ood_min/--ood_max passed to the RL steps are set to the GAP ITSELF
# (150, 300), not the outer span — online_RL_ucpg_v2/train.py's OOD-fraction
# logging only supports one contiguous range, so this reports "fraction of
# chosen actions landing inside the untrained hole", the number that matters
# here (not "fraction outside [100,350]").
#
# Requires:
#   Data/DatasetV2_layer_12_samples_5000.pkl (already exists)
#   MATLAB module + LPBF-Simulation/simulation_v2 checkout alongside this
#   repo, for step 7 (real-simulator evaluation).
#
# Submit (defaults):
#   sbatch jobs/gap_100-150_300-350_experiment.sh
# Any extra args are forwarded ONLY to the UCPG v2 training step (step 5),
# e.g.:
#   sbatch jobs/gap_100-150_300-350_experiment.sh --lr_lambda 0.05
# =============================================================================

#SBATCH --job-name=gap_100-150_300-350
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/gap_100-150_300-350_%j.log

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

DATA_PATH="Data/DatasetV2_layer_12_samples_5000.pkl"
LP_RANGES="100-150,300-350"          # the two training islands
GAP_MIN=150; GAP_MAX=300             # the untrained hole between them (OOD-fraction logging only)
ACTION_MIN=100; ACTION_MAX=400       # RL action range — spans BOTH islands + the gap + the outer edges
TAG="gap_100-150_300-350"

SURROGATE_OUT="surrogate_model_v3/runs/$TAG"
SURROGATE="$SURROGATE_OUT/surrogate_best.pt"
NAIVE_PG_OUT="baselines/naive_pg/runs/$TAG"
UCPG_OUT="online_RL_ucpg_v2/runs/${TAG}_relative"
RESULTS_SURR="baselines/results_${TAG}"
RESULTS_REAL="baselines/results_${TAG}_real"

CALIB_FRACTION=0.4
N_EPISODES=50
N_EPISODES_REAL=1

echo "=== [0/7] Train surrogate_model_v3 on gapped islands $LP_RANGES ==="
$PY -m surrogate_model_v3.train \
    --data_path "$DATA_PATH" \
    --lp_filter_ranges "$LP_RANGES" \
    --n_ensemble 5 --epochs 500 --batch_size 128 --lr 1e-3 --patience 30 \
    --out_dir "$SURROGATE_OUT"

echo "=== [1/7] Full surrogate evaluation (evaluate.py) ==="
$PY -m surrogate_model_v3.evaluate \
    --checkpoint "$SURROGATE" \
    --data_path  "$DATA_PATH"

echo "=== [2/7] OOD stress test — uncertainty vs. action (evaluate_ood.py) ==="
$PY -m surrogate_model_v3.evaluate_ood \
    --checkpoint "$SURROGATE" \
    --data_path  "$DATA_PATH" \
    --id_ranges  "$LP_RANGES"

echo "=== [3/7] Calibrating UCPG's uncertainty budget (delta) ==="
CALIB_LOG=$(mktemp)
$PY -m online_RL_ucpg_v2.train \
    --surrogate "$SURROGATE" --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $GAP_MIN --ood_max $GAP_MAX \
    --delta 999.0 --n_iterations 40 --lambda_init 0.0 \
    --out_dir "online_RL_ucpg_v2/runs/${TAG}_calib" \
    2>&1 | tee "$CALIB_LOG"
J_U_LAST=$(grep -oP 'J_u \K[0-9.]+' "$CALIB_LOG" | tail -1)
if [ -z "$J_U_LAST" ]; then
    echo "WARNING: could not parse J_u_hat — falling back to delta=0.05"
    DELTA=0.05
else
    DELTA=$(python3 -c "print(round(float('$J_U_LAST') * $CALIB_FRACTION, 5))")
fi
echo "Calibration: near-random J_u_hat=$J_U_LAST  ->  delta=$DELTA (${CALIB_FRACTION}x)"
rm -f "$CALIB_LOG"; rm -rf "online_RL_ucpg_v2/runs/${TAG}_calib"

echo "=== [4/7] Naive PG (no uncertainty) ==="
$PY -m baselines.naive_pg.train \
    --surrogate "$SURROGATE" --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $GAP_MIN --ood_max $GAP_MAX \
    --out_dir "$NAIVE_PG_OUT"

echo "=== [5/7] UCPG v2 (relative uncertainty constraint, delta=$DELTA) ==="
$PY -m online_RL_ucpg_v2.train \
    --surrogate "$SURROGATE" --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $GAP_MIN --ood_max $GAP_MAX \
    --delta $DELTA --relative_uncertainty \
    --out_dir "$UCPG_OUT" \
    "$@"

echo "=== [6/7] Aggregate — SURROGATE-driven leaderboard (naive_pg vs. UCPG) ==="
$PY -m baselines.evaluate_baselines \
    --surrogate "$SURROGATE" \
    --naive_pg_checkpoint "$NAIVE_PG_OUT/naive_pg_best.pt" \
    --ucpg_v2_checkpoint  "$UCPG_OUT/ucpg_best.pt" \
    --skip_constant \
    --n_episodes $N_EPISODES \
    --out_dir "$RESULTS_SURR"

echo "=== [7/7] Aggregate — REAL PHYSICS leaderboard (naive_pg vs. UCPG) ==="
$PY -m baselines.evaluate_real_all_methods \
    --surrogate "$SURROGATE" \
    --naive_pg_checkpoint "$NAIVE_PG_OUT/naive_pg_best.pt" \
    --ucpg_v2_checkpoint  "$UCPG_OUT/ucpg_best.pt" \
    --n_episodes $N_EPISODES_REAL \
    --out_dir "$RESULTS_REAL"

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Surrogate checkpoint      : $SURROGATE"
echo "Surrogate eval plots      : $SURROGATE_OUT/  (evaluate.py) and alongside checkpoint (evaluate_ood.py)"
echo "Naive PG / UCPG checkpts  : $NAIVE_PG_OUT/naive_pg_best.pt , $UCPG_OUT/ucpg_best.pt"
echo "Surrogate leaderboard     : $RESULTS_SURR/leaderboard.png / .csv"
echo "Real-physics leaderboard  : $RESULTS_REAL/leaderboard_real.png / .csv"
echo "============================================================"
exit $EXIT_CODE
