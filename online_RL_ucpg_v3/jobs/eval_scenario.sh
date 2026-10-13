#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/eval_scenario.sh — surrogate + real-physics evaluation of
# naive PG vs shaped PG (all c) for one scenario, scored with the plain
# (unpenalised) deviation of FIELD, plus the comparison figure.
# Outputs: online_RL_ucpg_v3/results/<SCENARIO>/<FIELD>/
# Env vars (set by run_scenario.sh): SCENARIO, FIELD, SURROGATE, ID_RANGES, PENALTIES.
# =============================================================================
#SBATCH --job-name=ucpg_v3_scen_eval
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/online_RL_ucpg_v3/jobs/eval_scenario_%j.log

read -r -a PENS <<< "$PENALTIES"
ID_RANGES="${ID_RANGES//:/,}"          # passed as 100-150:350-400 (see run_scenario.sh)
R="online_RL_ucpg_v3/runs/$SCENARIO/$FIELD"
OUT="online_RL_ucpg_v3/results/$SCENARIO/$FIELD"
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
module load matlab; source ~/.bashrc; conda activate RL
set -e
echo "Started: $(date)  Node: $SLURMD_NODENAME  $SCENARIO  field=$FIELD  ID=$ID_RANGES  c: ${PENS[*]}"
METHODS=(--pg "PG plain=$R/pg_plain/pg_best.pt")
for C in "${PENS[@]}"; do METHODS+=(--pg "UCPG v3 (c=$C K)=$R/pg_c$C/pg_best.pt"); done

echo "=== [1/2] surrogate evaluation ==="
python -m online_RL_ucpg_v3.evaluate_surrogate --surrogate "$SURROGATE" --field "$FIELD" \
    --id_ranges "$ID_RANGES" --n_episodes 100 "${METHODS[@]}" --out_dir "$OUT"
python -m online_RL_ucpg_v3.plot_comparison --results_dir "$OUT" --field "$FIELD"
echo "=== [2/2] real-physics evaluation ==="
python -m online_RL_ucpg_v3.evaluate_real --surrogate "$SURROGATE" --field "$FIELD" \
    --cool_times 0.06 0.10 0.14 "${METHODS[@]}" --out_dir "$OUT"
python -m online_RL_ucpg_v3.plot_comparison --results_dir "$OUT" --field "$FIELD"
echo "Finished: $(date)   Results: $OUT/"
