#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/run_scenario_mult.sh — naive PG vs MULTIPLICATIVE-shaped PG over a
# range of trust scales tau_k, for one coverage scenario and reward field. Run on the LOGIN node:
#   bash online_RL_ucpg_v3/jobs/run_scenario_mult.sh <scenario> <heat|cool> [k1 k2 ...]
#   e.g. bash online_RL_ucpg_v3/jobs/run_scenario_mult.sh range_100-200 heat 0.5 1 2 4
# Default tau_k: 0.5 1 2 4 (typical in-range trust w = 0.2 / 0.5 / 0.8 / 0.94).
# Shaping sigma: UNCERTAINTY env var, default total (propagated epistemic + propagated aleatoric);
# UNCERTAINTY=epistemic for the epistemic-only variant.
# Add --dry-run as the LAST argument to print what would be submitted, submitting nothing.
# Scenarios: same surrogates as run_scenario.sh.
#   runs    : online_RL_ucpg_v3/runs/<scenario>/<field>_mult_<uncertainty>/{pg_plain, pg_mult_k<K>}/
#   results : online_RL_ucpg_v3/results/<scenario>/<field>_mult_<uncertainty>/
# =============================================================================
DRY=0
if [ "${@: -1}" = "--dry-run" ]; then DRY=1; set -- "${@:1:$(($# - 1))}"; fi
SCENARIO="$1"; FIELD="$2"
USAGE="usage: bash $0 {range_100-200|range_100-200_clean|gap|full_range} {heat|cool} [k1 k2 ...] [--dry-run]"
case "$FIELD" in heat|cool) ;; *) echo "$USAGE"; exit 1 ;; esac
shift 2
TAU_KS="${*:-0.5 1 2 4}"
UNCERTAINTY="${UNCERTAINTY:-total}"
S=surrogate_model_v3/runs
case "$SCENARIO" in
    range_100-200) SURROGATE=$S/range_100-200_perturb0.2_bootstrap0.5/surrogate_best.pt; ID_RANGES="100-200" ;;
    gap)           SURROGATE=$S/gap_100-150_350-400_perturb0.0_bootstrap0.5/surrogate_best.pt; ID_RANGES="100-150,350-400" ;;
    range_100-200_clean) SURROGATE=$S/range_100-200_perturb0.0_bootstrap1.0/surrogate_best.pt; ID_RANGES="100-200" ;;
    full_range)    SURROGATE=$S/full_range/surrogate_best.pt; ID_RANGES="100-400" ;;
    *) echo "$USAGE"; exit 1 ;;
esac
[ -f "$SURROGATE" ] || { echo "surrogate checkpoint not found: $SURROGATE"; exit 1; }
N=$(echo $TAU_KS | wc -w)
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
V="ALL,SCENARIO=$SCENARIO,FIELD=$FIELD,SURROGATE=$SURROGATE,ID_RANGES=${ID_RANGES//,/:},TAU_KS=$TAU_KS,UNCERTAINTY=$UNCERTAINTY"

echo "scenario: $SCENARIO   field: $FIELD   ID: $ID_RANGES W   tau_k: $TAU_KS   sigma: propagated $UNCERTAINTY   (PG, multiplicative)"
echo "surrogate: $SURROGATE"
echo "runs: online_RL_ucpg_v3/runs/$SCENARIO/${FIELD}_mult_$UNCERTAINTY/   results: online_RL_ucpg_v3/results/$SCENARIO/${FIELD}_mult_$UNCERTAINTY/"
if [ "$DRY" = "1" ]; then
    echo "[dry-run] sbatch --array=0-$N --export=$V online_RL_ucpg_v3/jobs/train_scenario_mult.sh"
    echo "[dry-run] sbatch --dependency=afterok:<train> --export=$V online_RL_ucpg_v3/jobs/eval_scenario_mult.sh"
    exit 0
fi
T=$(sbatch --parsable --array=0-$N --export="$V" online_RL_ucpg_v3/jobs/train_scenario_mult.sh)
E=$(sbatch --parsable --dependency=afterok:"$T" --export="$V" online_RL_ucpg_v3/jobs/eval_scenario_mult.sh)
echo "training: $T ($((N + 1)) tasks: naive + multiplicative)   evaluation: $E (after training)"
