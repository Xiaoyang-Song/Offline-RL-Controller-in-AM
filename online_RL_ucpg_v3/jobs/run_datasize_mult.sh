#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/run_datasize_mult.sh — DATA-SCARCITY study, run on the LOGIN node.
# For every training-set size N (full 100-400 W power range, noise-free):
#   [1] train a surrogate on N trajectories      (surrogate_model_v3/jobs/train_full_datasize.sh)
#   [2] after [1], per reward field: naive PG + multiplicative UCPG v3 for every tau_k,
#       propagated UNCERTAINTY                     (train_scenario_mult.sh, scenario full_n<N>)
#   [3] after [2]: surrogate + real-physics evaluation and comparison figure (eval_scenario_mult.sh)
#
#   bash online_RL_ucpg_v3/jobs/run_datasize_mult.sh [k1 k2 ...] [--dry-run]
# Env vars: SIZES (default "500 1000 2000"), FIELDS (default "heat"; "heat cool" for both),
#           UNCERTAINTY (default total), RANGE (full | narrow = 200-300 W, default full),
#           STEP_MATCHED (0 | 1, default 0; 1 = same gradient-step budget as the 4000-trajectory model,
#           see surrogate_model_v3/jobs/train_full_datasize.sh). Default tau_k: 0.5 1 2 4 8.
# A size whose surrogate already exists is not retrained.
#
#   surrogates : surrogate_model_v3/runs/{full_range|narrow_200_300W}_n<N>[_steps]/
#   runs       : online_RL_ucpg_v3/runs/{full|narrow}_n<N>[_steps]/<FIELD>_mult_<UNCERTAINTY>/{pg_plain, pg_mult_k<K>}/
#   results    : online_RL_ucpg_v3/results/{full|narrow}_n<N>[_steps]/<FIELD>_mult_<UNCERTAINTY>/
# The 4000-trajectory reference is the existing full_range scenario (run_scenario_mult.sh full_range ...).
# =============================================================================
DRY=0
if [ "${@: -1}" = "--dry-run" ]; then DRY=1; set -- "${@:1:$(($# - 1))}"; fi
TAU_KS="${*:-0.5 1 2 4 8}"
SIZES="${SIZES:-500 1000 2000}"
FIELDS="${FIELDS:-heat}"
UNCERTAINTY="${UNCERTAINTY:-total}"
RANGE="${RANGE:-full}"; STEP_MATCHED="${STEP_MATCHED:-0}"
case "$RANGE" in
    full)   BASE="full_range";      TAG="full";   ID="100-400" ;;
    narrow) BASE="narrow_200_300W"; TAG="narrow"; ID="200-300" ;;
    *) echo "RANGE must be full or narrow"; exit 1 ;;
esac
SUFFIX=""; [ "$STEP_MATCHED" = "1" ] && SUFFIX="_steps"
NK=$(echo $TAU_KS | wc -w)
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
run() { if [ "$DRY" = "1" ]; then echo "[dry-run] sbatch $*" >&2; echo "<job>"; else sbatch --parsable "$@"; fi; }

echo "range: $RANGE ($ID W)   sizes: $SIZES   step-matched: $STEP_MATCHED   fields: $FIELDS   tau_k: $TAU_KS   sigma: propagated $UNCERTAINTY"
for N in $SIZES; do
    SURROGATE="surrogate_model_v3/runs/${BASE}_n${N}${SUFFIX}/surrogate_best.pt"
    if [ -f "$SURROGATE" ]; then
        S=""; DEP=(); echo "N=$N: surrogate exists — not retraining"
    else
        S=$(run --export=ALL,NTRAJ=$N,RANGE=$RANGE,STEP_MATCHED=$STEP_MATCHED surrogate_model_v3/jobs/train_full_datasize.sh)
        DEP=(--dependency=afterok:$S)
    fi
    for F in $FIELDS; do
        V="ALL,SCENARIO=${TAG}_n${N}${SUFFIX},FIELD=$F,SURROGATE=$SURROGATE,ID_RANGES=$ID,TAU_KS=$TAU_KS,UNCERTAINTY=$UNCERTAINTY"
        T=$(run "${DEP[@]}" --array=0-$NK --export="$V" online_RL_ucpg_v3/jobs/train_scenario_mult.sh)
        E=$(run --dependency=afterok:$T --export="$V" online_RL_ucpg_v3/jobs/eval_scenario_mult.sh)
        echo "N=$N field=$F: surrogate ${S:-(existing)}  RL $T ($((NK + 1)) tasks)  eval $E  -> online_RL_ucpg_v3/results/${TAG}_n${N}${SUFFIX}/${F}_mult_$UNCERTAINTY/"
    done
done
