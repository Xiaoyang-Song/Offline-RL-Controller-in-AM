#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/train_narrow.sh
# SLURM ARRAY job — train the four v3 controllers on the narrow (200-300 W)
# surrogate, all with the SAME reward field so each pair differs only in
# uncertainty shaping:
#   task 0 : PG, plain reward      (matched naive PG)
#   task 1 : PG, shaped reward     (UCPG v3)
#   task 2 : Q,  plain reward      (matched naive online Q)
#   task 3 : Q,  shaped reward     (Q v3)
#
# Reward field: post-cooling by default. Override at submission:
#   sbatch --export=ALL,REWARD_FIELD=heat online_RL_ucpg_v3/jobs/train_narrow.sh
# Submit:
#   sbatch online_RL_ucpg_v3/jobs/train_narrow.sh
# Then the real-physics evaluation (after all 4 finish), e.g.:
#   sbatch --dependency=afterok:<JOBID> online_RL_ucpg_v3/jobs/eval_real_narrow.sh
# =============================================================================

#SBATCH --job-name=ucpg_v3_train
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/online_RL_ucpg_v3/jobs/train_narrow_%A_%a.log

REWARD_FIELD="${REWARD_FIELD:-cool}"
SURROGATE="surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt"
COMMON=(--surrogate "$SURROGATE" --action_min 100 --action_max 400 --ood_min 200 --ood_max 300
        --reward_field "$REWARD_FIELD" --seed 42)

ALGOS=(pg pg q q)
MODES=(plain shaped plain shaped)
ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
OUT="online_RL_ucpg_v3/runs/narrow_${ALGO}_${MODE}_${REWARD_FIELD}"

echo "============================================================"
echo "Job ID  : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}   Node: $SLURMD_NODENAME   Started: $(date)"
echo "Run     : algo=$ALGO  reward_mode=$MODE  reward_field=$REWARD_FIELD  ->  $OUT"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc
conda activate RL
set -e

if [ "$ALGO" = "pg" ]; then
    python -m online_RL_ucpg_v3.train_pg "${COMMON[@]}" --reward_mode "$MODE" \
        --n_iterations 2000 --n_traj 32 --use_baseline --out_dir "$OUT"
else
    python -m online_RL_ucpg_v3.train_q "${COMMON[@]}" --reward_mode "$MODE" \
        --n_episodes 3000 --out_dir "$OUT"
fi

echo "Finished: $(date)   Outputs: $OUT/"
