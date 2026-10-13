#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/train_sweep.sh
# SLURM ARRAY job — shaped-reward runs on the narrow (200-300 W) surrogate with
# small uncertainty penalties c (--pess_dev, K); tau unchanged (tau_k = 2).
# Env vars: PENALTIES (space-separated, e.g. "5 10 20 40"), REWARD_FIELD.
# Task i < n -> PG with PENALTIES[i];  task i >= n -> Q with PENALTIES[i-n].
# Submitted by run_v3_narrow_followup.sh with --array=0-(2n-1).
# =============================================================================
#SBATCH --job-name=ucpg_v3_sweep
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/online_RL_ucpg_v3/jobs/train_sweep_%A_%a.log

F="${REWARD_FIELD:-heat}"
read -r -a PENS <<< "${PENALTIES:-5 10 20 40}"
N=${#PENS[@]}
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt "$N" ]; then ALGO=pg; C=${PENS[$i]}; else ALGO=q; C=${PENS[$((i - N))]}; fi
OUT="online_RL_ucpg_v3/runs/narrow_${ALGO}_shaped_${F}_c${C}"
COMMON=(--surrogate surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt
        --action_min 100 --action_max 400 --ood_min 200 --ood_max 300
        --reward_field "$F" --reward_mode shaped --pess_dev "$C" --seed 42)
echo "Job ${SLURM_ARRAY_JOB_ID}_${i} on $SLURMD_NODENAME $(date)"
echo "Run: algo=$ALGO  shaped  pess_dev=$C K  field=$F  ->  $OUT"
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc; conda activate RL; set -e
if [ "$ALGO" = "pg" ]; then
    python -m online_RL_ucpg_v3.train_pg "${COMMON[@]}" --n_iterations 2000 --n_traj 32 --use_baseline --out_dir "$OUT"
else
    python -m online_RL_ucpg_v3.train_q "${COMMON[@]}" --n_episodes 3000 --out_dir "$OUT"
fi
echo "Finished: $(date)"
