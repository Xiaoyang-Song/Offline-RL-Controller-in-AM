#!/bin/bash
# =============================================================================
# run_ucpg_relative.sh
# UCPG-only job on the narrow [200, 300]W surrogate_model_v3, using the
# RELATIVE uncertainty constraint (--relative_uncertainty): G_u/delta in the
# advantage and a dual step on (J_u_hat/delta - 1), so lambda actually binds.
#   0. Calibrate delta  (40-iteration near-random warm-start; delta = CALIB_FRACTION x J_u_hat)
#   1. Train UCPG       -> online_RL_ucpg_v2/runs/<TAG>/
#   2. Evaluate in the surrogate  (N_EPISODES)
#   3. Evaluate in the REAL PDE simulator (N_EPISODES_REAL, needs MATLAB)
# Nothing here touches the existing runs (separate out_dir <TAG>).
#
# Submit:
#   sbatch online_RL_ucpg_v2/jobs/run_ucpg_relative.sh
# Override at submission (forwarded to the training step only), e.g.:
#   sbatch online_RL_ucpg_v2/jobs/run_ucpg_relative.sh --lr_lambda 0.05 --n_iterations 3000
# =============================================================================

#SBATCH --job-name=ucpg_relative
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/online_RL_ucpg_v2/jobs/run_ucpg_relative_%j.log

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
module load matlab
source ~/.bashrc
conda activate RL
echo "Python: $(which python)  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
set -e
set -o pipefail

PY="python -u"
SURROGATE="surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt"
LP_MIN=200; LP_MAX=300
ACTION_MIN=100; ACTION_MAX=400
TAG="narrow_200_300W_relative"
OUT="online_RL_ucpg_v2/runs/$TAG"
CALIB_FRACTION=0.4       # delta = CALIB_FRACTION * (near-random-policy J_u_hat)
N_EPISODES=50
N_EPISODES_REAL=1
COOL_TIME_REAL=0.10

echo "=== [0/3] Calibrating delta ==="
CALIB_LOG=$(mktemp)
$PY -m online_RL_ucpg_v2.train \
    --surrogate "$SURROGATE" --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $LP_MIN --ood_max $LP_MAX \
    --delta 999.0 --n_iterations 40 --lambda_init 0.0 \
    --out_dir "${OUT}_calib" 2>&1 | tee "$CALIB_LOG"
J_U_LAST=$(grep -oP 'J_u \K[0-9.]+' "$CALIB_LOG" | tail -1)
if [ -z "$J_U_LAST" ]; then DELTA=0.05; else DELTA=$(python3 -c "print(round(float('$J_U_LAST') * $CALIB_FRACTION, 5))"); fi
echo "Calibration: near-random J_u_hat=$J_U_LAST -> delta=$DELTA (${CALIB_FRACTION}x)"
rm -f "$CALIB_LOG"; rm -rf "${OUT}_calib"

echo "=== [1/3] Train UCPG (relative uncertainty constraint, delta=$DELTA) ==="
$PY -m online_RL_ucpg_v2.train \
    --surrogate "$SURROGATE" --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $LP_MIN --ood_max $LP_MAX \
    --delta $DELTA --relative_uncertainty \
    --out_dir "$OUT" "$@"

echo "=== [2/3] Evaluate in the surrogate ==="
$PY -m online_RL_ucpg_v2.evaluate \
    --checkpoint "$OUT/ucpg_best.pt" --surrogate "$SURROGATE" \
    --action_min $ACTION_MIN --action_max $ACTION_MAX \
    --ood_min $LP_MIN --ood_max $LP_MAX \
    --n_episodes $N_EPISODES --plot

echo "=== [3/3] Evaluate in the REAL PDE simulator ==="
$PY -m online_RL_ucpg_v2.evaluate_real \
    --checkpoint "$OUT/ucpg_best.pt" --surrogate "$SURROGATE" \
    --cool_time $COOL_TIME_REAL --n_episodes $N_EPISODES_REAL \
    --results_out "$OUT/eval_real.json" --plot

EXIT_CODE=$?
echo "Finished: $(date)  (exit code: $EXIT_CODE)   Outputs: $OUT/"
exit $EXIT_CODE
