#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/run_clean_100-200_mult.sh — one-shot chain, run on the LOGIN node:
#   [1] train the NOISE-FREE 100-200 W surrogate (surrogate_model_v3/jobs/train_range_100-200_matched.sh)
#   [2] after [1]: naive PG + multiplicative-shaped PG (UCPG v3) for every tau_k, propagated TOTAL
#       uncertainty, reward field FIELD (train_scenario_mult.sh, scenario range_100-200_clean)
#   [3] after [2]: surrogate + real-physics evaluation and comparison figure (eval_scenario_mult.sh)
#
#   bash online_RL_ucpg_v3/jobs/run_clean_100-200_mult.sh [k1 k2 ...] [--dry-run]
#   e.g. bash online_RL_ucpg_v3/jobs/run_clean_100-200_mult.sh 0.5 1 2 4
# Default tau_k: 0.5 1 2 4. FIELD (env var) defaults to heat; UNCERTAINTY defaults to total.
#
#   surrogate : surrogate_model_v3/runs/range_100-200_perturb0.0_bootstrap1.0/
#   runs      : online_RL_ucpg_v3/runs/range_100-200_clean/<FIELD>_mult_<UNCERTAINTY>/{pg_plain, pg_mult_k<K>}/
#   results   : online_RL_ucpg_v3/results/range_100-200_clean/<FIELD>_mult_<UNCERTAINTY>/
# =============================================================================
DRY=0
if [ "${@: -1}" = "--dry-run" ]; then DRY=1; set -- "${@:1:$(($# - 1))}"; fi
TAU_KS="${*:-0.5 1 2 4}"
FIELD="${FIELD:-heat}"
UNCERTAINTY="${UNCERTAINTY:-total}"
SCENARIO="range_100-200_clean"
SURROGATE="surrogate_model_v3/runs/range_100-200_perturb0.0_bootstrap1.0/surrogate_best.pt"
ID_RANGES="100-200"
N=$(echo $TAU_KS | wc -w)
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
V="ALL,SCENARIO=$SCENARIO,FIELD=$FIELD,SURROGATE=$SURROGATE,ID_RANGES=$ID_RANGES,TAU_KS=$TAU_KS,UNCERTAINTY=$UNCERTAINTY"

echo "surrogate : noise-free 100-200 W -> $SURROGATE"
echo "RL        : PG plain + multiplicative PG, tau_k = $TAU_KS, sigma = propagated $UNCERTAINTY, field = $FIELD"
echo "runs      : online_RL_ucpg_v3/runs/$SCENARIO/${FIELD}_mult_$UNCERTAINTY/"
echo "results   : online_RL_ucpg_v3/results/$SCENARIO/${FIELD}_mult_$UNCERTAINTY/"
if [ "$DRY" = "1" ]; then
    echo "[dry-run] sbatch surrogate_model_v3/jobs/train_range_100-200_matched.sh"
    echo "[dry-run] sbatch --dependency=afterok:<surrogate> --array=0-$N --export=$V online_RL_ucpg_v3/jobs/train_scenario_mult.sh"
    echo "[dry-run] sbatch --dependency=afterok:<train> --export=$V online_RL_ucpg_v3/jobs/eval_scenario_mult.sh"
    exit 0
fi
if [ -f "$SURROGATE" ]; then
    echo "surrogate already exists — not retraining it"
    T=$(sbatch --parsable --array=0-$N --export="$V" online_RL_ucpg_v3/jobs/train_scenario_mult.sh)
    S="(skipped)"
else
    S=$(sbatch --parsable surrogate_model_v3/jobs/train_range_100-200_matched.sh)
    T=$(sbatch --parsable --dependency=afterok:"$S" --array=0-$N --export="$V" online_RL_ucpg_v3/jobs/train_scenario_mult.sh)
fi
E=$(sbatch --parsable --dependency=afterok:"$T" --export="$V" online_RL_ucpg_v3/jobs/eval_scenario_mult.sh)
echo "surrogate: $S   RL training: $T ($((N + 1)) tasks: naive + multiplicative)   evaluation: $E"
