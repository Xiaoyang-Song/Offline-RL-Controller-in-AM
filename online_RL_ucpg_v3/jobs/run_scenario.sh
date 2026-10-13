#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/run_scenario.sh — naive PG vs shaped PG over a range of c,
# for one coverage scenario and reward field. Run on the LOGIN node:
#   bash online_RL_ucpg_v3/jobs/run_scenario.sh <scenario> <heat|cool> [c1 c2 ...]
#   e.g. bash online_RL_ucpg_v3/jobs/run_scenario.sh range_100-200 heat 25 100 400 800
# Default c (K): heat 25 100 400 800; cool 200 400 800 1600.
# Add --dry-run as the LAST argument to print what would be submitted, submitting nothing.
# Scenarios:
#   range_100-200 : surrogate trained on 100-200 W (noise-injected, perturb 0.2)
#   gap           : NOISE-FREE surrogate trained on 100-150 + 350-400 W (perturb 0.0)
# Submits training (1 + n parallel PG runs) and the evaluation, chained.
#   runs    : online_RL_ucpg_v3/runs/<scenario>/<field>/{pg_plain, pg_c<C>}/
#   results : online_RL_ucpg_v3/results/<scenario>/<field>/  (leaderboards + comparison figure)
# =============================================================================
DRY=0
if [ "${@: -1}" = "--dry-run" ]; then DRY=1; set -- "${@:1:$(($# - 1))}"; fi
SCENARIO="$1"; FIELD="$2"
USAGE="usage: bash $0 {range_100-200|gap} {heat|cool} [c1 c2 ...] [--dry-run]"
case "$FIELD" in heat) DEF="25 100 400 800" ;; cool) DEF="200 400 800 1600" ;; *) echo "$USAGE"; exit 1 ;; esac
shift 2
PENALTIES="${*:-$DEF}"
S=surrogate_model_v3/runs
case "$SCENARIO" in
    range_100-200) SURROGATE=$S/range_100-200_perturb0.2_bootstrap0.5/surrogate_best.pt; ID_RANGES="100-200" ;;
    gap)           SURROGATE=$S/gap_100-150_350-400_perturb0.0_bootstrap0.5/surrogate_best.pt; ID_RANGES="100-150,350-400" ;;
    *) echo "$USAGE"; exit 1 ;;
esac
N=$(echo $PENALTIES | wc -w)
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
# sbatch --export splits on commas, so ranges travel as "100-150:350-400" (jobs turn ':' back into ',')
V="ALL,SCENARIO=$SCENARIO,FIELD=$FIELD,SURROGATE=$SURROGATE,ID_RANGES=${ID_RANGES//,/:},PENALTIES=$PENALTIES"

echo "scenario: $SCENARIO   field: $FIELD   ID: $ID_RANGES W   c: $PENALTIES K   (PG)"
echo "surrogate: $SURROGATE"
echo "runs: online_RL_ucpg_v3/runs/$SCENARIO/$FIELD/   results: online_RL_ucpg_v3/results/$SCENARIO/$FIELD/"
if [ "$DRY" = "1" ]; then
    echo "[dry-run] sbatch --array=0-$N --export=$V online_RL_ucpg_v3/jobs/train_scenario.sh"
    echo "[dry-run] sbatch --dependency=afterok:<train> --export=$V online_RL_ucpg_v3/jobs/eval_scenario.sh"
    exit 0
fi
T=$(sbatch --parsable --array=0-$N --export="$V" online_RL_ucpg_v3/jobs/train_scenario.sh)
E=$(sbatch --parsable --dependency=afterok:"$T" --export="$V" online_RL_ucpg_v3/jobs/eval_scenario.sh)
echo "training: $T ($((N + 1)) tasks: naive + shaped)   evaluation: $E (after training)"
