#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/train_scenario.sh — PG only, reward field FIELD (heat | cool).
# Array task 0: naive PG (plain reward); task i>0: shaped PG with c = PENALTIES[i-1].
# Outputs: online_RL_ucpg_v3/runs/<SCENARIO>/<FIELD>/pg_plain/ and .../pg_c<C>/
# Env vars (set by run_scenario.sh): SCENARIO, FIELD, SURROGATE, ID_RANGES, PENALTIES.
# =============================================================================
#SBATCH --job-name=ucpg_v3_scen
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/online_RL_ucpg_v3/jobs/train_scenario_%A_%a.log

read -r -a PENS <<< "$PENALTIES"
ID_RANGES="${ID_RANGES//:/,}"          # passed as 100-150:350-400 (see run_scenario.sh)
i=$SLURM_ARRAY_TASK_ID
R="online_RL_ucpg_v3/runs/$SCENARIO/$FIELD"
if [ "$i" -eq 0 ]; then MODE=(--reward_mode plain); OUT="$R/pg_plain"
else C=${PENS[$((i - 1))]}; MODE=(--reward_mode shaped --pess_dev "$C"); OUT="$R/pg_c$C"; fi
echo "Job ${SLURM_ARRAY_JOB_ID}_${i} on $SLURMD_NODENAME $(date)"
echo "Run: $SCENARIO  ${MODE[*]}  field=$FIELD  ID=$ID_RANGES  surrogate=$SURROGATE -> $OUT"
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc; conda activate RL; set -e
python -m online_RL_ucpg_v3.train_pg --surrogate "$SURROGATE" --id_ranges "$ID_RANGES" \
    --action_min 100 --action_max 400 --reward_field "$FIELD" "${MODE[@]}" --seed 42 \
    --n_iterations 2000 --n_traj 32 --use_baseline --out_dir "$OUT"
echo "Finished: $(date)"
