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
#           UNCERTAINTY (default total). Default tau_k: 0.5 1 2 4 8.
# A size whose surrogate already exists is not retrained.
#
#   surrogates : surrogate_model_v3/runs/full_range_n<N>/
#   runs       : online_RL_ucpg_v3/runs/full_n<N>/<FIELD>_mult_<UNCERTAINTY>/{pg_plain, pg_mult_k<K>}/
#   results    : online_RL_ucpg_v3/results/full_n<N>/<FIELD>_mult_<UNCERTAINTY>/
# The 4000-trajectory reference is the existing full_range scenario (run_scenario_mult.sh full_range ...).
# =============================================================================
DRY=0
if [ "${@: -1}" = "--dry-run" ]; then DRY=1; set -- "${@:1:$(($# - 1))}"; fi
TAU_KS="${*:-0.5 1 2 4 8}"
SIZES="${SIZES:-500 1000 2000}"
FIELDS="${FIELDS:-heat}"
UNCERTAINTY="${UNCERTAINTY:-total}"
NK=$(echo $TAU_KS | wc -w)
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
run() { if [ "$DRY" = "1" ]; then echo "[dry-run] sbatch $*" >&2; echo "<job>"; else sbatch --parsable "$@"; fi; }

echo "sizes: $SIZES   fields: $FIELDS   tau_k: $TAU_KS   sigma: propagated $UNCERTAINTY"
for N in $SIZES; do
    SURROGATE="surrogate_model_v3/runs/full_range_n$N/surrogate_best.pt"
    if [ -f "$SURROGATE" ]; then
        S=""; DEP=(); echo "N=$N: surrogate exists — not retraining"
    else
        S=$(run --export=ALL,NTRAJ=$N surrogate_model_v3/jobs/train_full_datasize.sh)
        DEP=(--dependency=afterok:$S)
    fi
    for F in $FIELDS; do
        V="ALL,SCENARIO=full_n$N,FIELD=$F,SURROGATE=$SURROGATE,ID_RANGES=100-400,TAU_KS=$TAU_KS,UNCERTAINTY=$UNCERTAINTY"
        T=$(run "${DEP[@]}" --array=0-$NK --export="$V" online_RL_ucpg_v3/jobs/train_scenario_mult.sh)
        E=$(run --dependency=afterok:$T --export="$V" online_RL_ucpg_v3/jobs/eval_scenario_mult.sh)
        echo "N=$N field=$F: surrogate ${S:-(existing)}  RL $T ($((NK + 1)) tasks)  eval $E  -> online_RL_ucpg_v3/results/full_n$N/${F}_mult_$UNCERTAINTY/"
    done
done
