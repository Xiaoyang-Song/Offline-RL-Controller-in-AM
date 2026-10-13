#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/train_setting.sh — the 4 v3 runs (PG/Q x plain/shaped)
# for ANY surrogate / in-distribution range. Normally submitted by run_setting.sh.
# Env vars: TAG, SURROGATE, OOD_MIN, OOD_MAX, REWARD_FIELD (default heat).
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
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/online_RL_ucpg_v3/jobs/train_setting_%A_%a.log

F="${REWARD_FIELD:-heat}"
ALGOS=(pg pg q q); MODES=(plain shaped plain shaped)
ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}; MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
OUT="online_RL_ucpg_v3/runs/${TAG}_${ALGO}_${MODE}_${F}"
COMMON=(--surrogate "$SURROGATE" --action_min 100 --action_max 400 --ood_min "$OOD_MIN" --ood_max "$OOD_MAX"
        --reward_field "$F" --reward_mode "$MODE" --seed 42)
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $SLURMD_NODENAME $(date)"
echo "Run: $TAG  algo=$ALGO  mode=$MODE  field=$F  ID=[$OOD_MIN,$OOD_MAX] W  surrogate=$SURROGATE -> $OUT"
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc; conda activate RL; set -e
if [ "$ALGO" = "pg" ]; then
    python -m online_RL_ucpg_v3.train_pg "${COMMON[@]}" --n_iterations 2000 --n_traj 32 --use_baseline --out_dir "$OUT"
else
    python -m online_RL_ucpg_v3.train_q "${COMMON[@]}" --n_episodes 3000 --out_dir "$OUT"
fi
echo "Finished: $(date)"
