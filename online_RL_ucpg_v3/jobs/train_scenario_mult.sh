#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/train_scenario_mult.sh — PG only, reward field FIELD (heat | cool),
# MULTIPLICATIVE shaping: reward = mean_i w_i exp(-d_i / W), w_i = tau^2 / (tau^2 + sigma_i^2),
# sigma = propagated UNCERTAINTY (total = epistemic + aleatoric, both propagated between layers).
# Array task 0: naive PG (plain reward); task i>0: multiplicative PG with tau_k = TAU_KS[i-1]
# (tau = tau_k x median in-distribution sigma).
# Outputs: online_RL_ucpg_v3/runs/<SCENARIO>/<FIELD>_mult_<UNCERTAINTY>/pg_plain/ and .../pg_mult_k<K>/
# Env vars (set by run_scenario_mult.sh): SCENARIO, FIELD, SURROGATE, ID_RANGES, TAU_KS, UNCERTAINTY.
# =============================================================================
#SBATCH --job-name=ucpg_v3_mult
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/online_RL_ucpg_v3/jobs/train_scenario_mult_%A_%a.log

read -r -a KS <<< "$TAU_KS"
ID_RANGES="${ID_RANGES//:/,}"          # passed as 100-150:350-400 (see run_scenario_mult.sh)
i=$SLURM_ARRAY_TASK_ID
R="online_RL_ucpg_v3/runs/$SCENARIO/${FIELD}_mult_$UNCERTAINTY"
if [ "$i" -eq 0 ]; then MODE=(--reward_mode plain); OUT="$R/pg_plain"
else K=${KS[$((i - 1))]}; MODE=(--reward_mode shaped --shaping multiplicative --tau_k "$K"); OUT="$R/pg_mult_k$K"; fi
echo "Job ${SLURM_ARRAY_JOB_ID}_${i} on $SLURMD_NODENAME $(date)"
echo "Run: $SCENARIO  ${MODE[*]}  uncertainty=$UNCERTAINTY  field=$FIELD  ID=$ID_RANGES  surrogate=$SURROGATE -> $OUT"
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc; conda activate RL; set -e
python -m online_RL_ucpg_v3.train_pg --surrogate "$SURROGATE" --id_ranges "$ID_RANGES" \
    --action_min 100 --action_max 400 --reward_field "$FIELD" "${MODE[@]}" \
    --uncertainty_mode propagated --uncertainty "$UNCERTAINTY" --seed 42 \
    --n_iterations 2000 --n_traj 32 --use_baseline --out_dir "$OUT"
echo "Finished: $(date)"
