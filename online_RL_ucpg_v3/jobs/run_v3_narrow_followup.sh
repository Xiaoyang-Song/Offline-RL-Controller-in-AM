#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/run_v3_narrow_followup.sh — run on the LOGIN node:
#   bash online_RL_ucpg_v3/jobs/run_v3_narrow_followup.sh [--sweep-only] [FIELD] [PENALTIES...]
#   e.g. bash online_RL_ucpg_v3/jobs/run_v3_narrow_followup.sh heat 5 10 20 40
#        bash online_RL_ucpg_v3/jobs/run_v3_narrow_followup.sh --sweep-only cool 5 10 20 40
# Defaults: FIELD=heat, PENALTIES="5 10 20 40" (K). --sweep-only skips step 1
# (use it for the second call, so the unfinished evaluations are submitted once).
#
# Submits, for the narrow (200-300 W) surrogate:
#  1. the two UNFINISHED evaluations of the existing runs (no retraining):
#       eval_real_narrow.sh for REWARD_FIELD=heat and =cool
#       -> online_RL_ucpg_v3/results_narrow_{heat,cool}/
#  2. a small-penalty sweep: PG and Q, shaped, c in PENALTIES, on FIELD
#       train_sweep.sh (2n parallel tasks)  ->  runs/narrow_{pg,q}_shaped_<FIELD>_c<C>/
#  3. its evaluation, chained after all sweep trainings succeed, next to the
#     existing plain PG / plain Q runs of FIELD (no naive retraining)
#       eval_sweep.sh  ->  online_RL_ucpg_v3/results_sweep_narrow_<FIELD>/
# =============================================================================
SWEEP_ONLY=0
if [ "$1" = "--sweep-only" ]; then SWEEP_ONLY=1; shift; fi
FIELD="${1:-heat}"; shift
PENALTIES="${*:-5 10 20 40}"
N=$(echo $PENALTIES | wc -w)
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

if [ "$SWEEP_ONLY" = "0" ]; then
    E_HEAT=$(sbatch --parsable --export=ALL,REWARD_FIELD=heat online_RL_ucpg_v3/jobs/eval_real_narrow.sh)
    E_COOL=$(sbatch --parsable --export=ALL,REWARD_FIELD=cool online_RL_ucpg_v3/jobs/eval_real_narrow.sh)
fi
T=$(sbatch --parsable --array=0-$((2 * N - 1)) --export=ALL,REWARD_FIELD="$FIELD",PENALTIES="$PENALTIES" \
    online_RL_ucpg_v3/jobs/train_sweep.sh)
E=$(sbatch --parsable --dependency=afterok:"$T" --export=ALL,REWARD_FIELD="$FIELD",PENALTIES="$PENALTIES" \
    online_RL_ucpg_v3/jobs/eval_sweep.sh)

if [ "$SWEEP_ONLY" = "0" ]; then
    echo "1. unfinished evaluations   : heat $E_HEAT, cool $E_COOL  -> results_narrow_{heat,cool}/"
else
    echo "1. unfinished evaluations   : skipped (--sweep-only)"
fi
echo "2. sweep training ($((2 * N)) tasks): $T   field=$FIELD  c = $PENALTIES K"
echo "3. sweep evaluation         : $E  (after 2)  -> results_sweep_narrow_${FIELD}/"
