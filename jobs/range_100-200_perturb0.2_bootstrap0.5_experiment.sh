#!/bin/bash
# =============================================================================
# range_100-200_perturb0.2_bootstrap0.5_experiment.sh
# Single-range coverage variant of
# jobs/gap_100-150_300-350_perturb0.2_bootstrap0.5_experiment.sh — the
# surrogate is trained on ONE contiguous laser-power band, 100-200W, and the
# RL policies still act over the full 100-400W range, so 200-400W is pure
# extrapolation (OOD) for the surrogate. Same perturb/bootstrap settings:
#   --perturb_frac 0.2 --bootstrap_frac 0.5  (see the reference script's header)
#
# Differences vs. the reference script (only UCPG-side, naive PG unchanged):
#   - Step 3 calibrates delta with online_RL_ucpg_v2.calibrate_delta:
#         delta = J_min + ALPHA * (J_rand - J_min)
#     (J_min = best constant-power J_u, J_rand = uniform-random-action J_u),
#     instead of 0.4 * near-random J_u, which could land below the achievable
#     floor (infeasible budget -> lambda grows without bound).
#   - UCPG uses --uncertainty epistemic (u_t = epistemic_std only; aleatoric
#     noise is ~flat across laser power and irreducible). Calibration uses the
#     same mode so delta is in the same units.
#   - --ood_min/--ood_max are the TRAINING range (100, 200), so the logged
#     "fraction of actions outside [ood_min, ood_max]" is the true OOD fraction.
#
#   0. Train surrogate_model_v3 on 100-200W, WITH perturb + decorrelated bootstrap.
#   1. Full surrogate evaluation (evaluate.py).
#   2. OOD stress test — uncertainty vs. action (evaluate_ood.py --id_ranges).
#   3. Calibrate UCPG's delta (calibrate_delta.py, epistemic, J_min/J_rand).
#   4. Train naive PG (no uncertainty term) on this surrogate.
#   5. Train UCPG v2 (--relative_uncertainty --uncertainty epistemic).
#   6. Evaluate naive_pg + UCPG together in the surrogate.
#   7. Evaluate naive_pg + UCPG together in the REAL PDE simulator.
#
# Submit (defaults):
#   sbatch jobs/range_100-200_perturb0.2_bootstrap0.5_experiment.sh
# Any extra args are forwarded ONLY to the UCPG v2 training step (step 5).
# =============================================================================

#SBATCH --job-name=r100-200_p0.2_b0.5
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/range_100-200_perturb0.2_bootstrap0.5_%j.log

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
LP_MIN=100; LP_MAX=200               # the single training band
LP_RANGES="${LP_MIN}-${LP_MAX}"
ACTION_MIN=100; ACTION_MAX=400       # RL action range — training band + extrapolation region
PERTURB_FRAC=0.2
PERTURB_SEED=0
BOOTSTRAP_FRAC=0.5
UNCERTAINTY="epistemic"              # UCPG's u_t (calibration uses the same)
ALPHA=0.4                            # delta = J_min + ALPHA * (J_rand - J_min)
TAG="range_${LP_RANGES}_perturb0.2_bootstrap0.5"

SURROGATE_OUT="surrogate_model_v3/runs/$TAG"
SURROGATE="$SURROGATE_OUT/surrogate_best.pt"
NAIVE_PG_OUT="baselines/naive_pg/runs/$TAG"
UCPG_OUT="online_RL_ucpg_v2/runs/${TAG}_relative_${UNCERTAINTY}"
CALIB_OUT="online_RL_ucpg_v2/runs/${TAG}_delta_calib_${UNCERTAINTY}"
RESULTS_SURR="baselines/results_${TAG}"
RESULTS_REAL="baselines/results_${TAG}_real"

N_EPISODES=50
N_EPISODES_REAL=1

echo "=== [0/7] Train surrogate_model_v3 on $LP_RANGES W (perturb=$PERTURB_FRAC, bootstrap_frac=$BOOTSTRAP_FRAC) ==="
$PY -m surrogate_model_v3.train \
    --data_path "$DATA_PATH" \
    --lp_filter_min $LP_MIN --lp_filter_max $LP_MAX \
    --perturb_frac $PERTURB_FRAC --perturb_seed $PERTURB_SEED \
    --bootstrap_frac $BOOTSTRAP_FRAC \
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

echo "=== [3/7] Calibrating UCPG's delta (J_min / J_rand, u=$UNCERTAINTY, alpha=$ALPHA) ==="
CALIB_LOG=$(mktemp)
$PY -m online_RL_ucpg_v2.calibrate_delta \
    --surrogate "$SURROGATE" --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --uncertainty $UNCERTAINTY --alpha $ALPHA \
    --out_dir "$CALIB_OUT" \
    2>&1 | tee "$CALIB_LOG"
DELTA=$(grep -oP '^DELTA=\K\S+' "$CALIB_LOG" | tail -1)
rm -f "$CALIB_LOG"
if [ -z "$DELTA" ]; then
    echo "ERROR: could not parse DELTA from calibrate_delta output"; exit 1
fi
echo "Calibration: delta=$DELTA  (details: $CALIB_OUT/delta_calibration.json / .png)"

echo "=== [4/7] Naive PG (no uncertainty) ==="
$PY -m baselines.naive_pg.train \
    --surrogate "$SURROGATE" --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $LP_MIN --ood_max $LP_MAX \
    --out_dir "$NAIVE_PG_OUT"

echo "=== [5/7] UCPG v2 (relative, u=$UNCERTAINTY, delta=$DELTA) ==="
$PY -m online_RL_ucpg_v2.train \
    --surrogate "$SURROGATE" --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $LP_MIN --ood_max $LP_MAX \
    --delta $DELTA --relative_uncertainty --uncertainty $UNCERTAINTY \
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
echo "Delta calibration         : $CALIB_OUT/delta_calibration.json / .png  (delta=$DELTA)"
echo "Naive PG / UCPG checkpts  : $NAIVE_PG_OUT/naive_pg_best.pt , $UCPG_OUT/ucpg_best.pt"
echo "Surrogate leaderboard     : $RESULTS_SURR/leaderboard.png / .csv"
echo "Real-physics leaderboard  : $RESULTS_REAL/leaderboard_real.png / .csv"
echo "============================================================"
exit $EXIT_CODE
