#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/run_narrow.sh  — ONE command for the whole v3 experiment.
# Run it on the login node (it only submits jobs, it does not compute):
#
#   bash online_RL_ucpg_v3/jobs/run_narrow.sh            # post-cooling reward (default)
#   bash online_RL_ucpg_v3/jobs/run_narrow.sh heat       # end-of-heating reward
#
# It submits
#   1. train_narrow.sh     — array of 4 trainings in parallel (PG/Q x plain/shaped)
#   2. eval_real_narrow.sh — starts AUTOMATICALLY once all 4 trainings succeed
#                            (--dependency=afterok); surrogate + real-physics evaluation
#                            of the 4 new + 2 earlier methods, then the comparison figure.
# Results: online_RL_ucpg_v3/results_narrow_<field>/
#   leaderboard_{surrogate,real}_v3.csv, per_episode_surrogate.json, per_layer_real.json, comparison.{pdf,png}
# =============================================================================

FIELD="${1:-cool}"
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

TRAIN_ID=$(sbatch --parsable --export=ALL,REWARD_FIELD="$FIELD" online_RL_ucpg_v3/jobs/train_narrow.sh)
EVAL_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_ID" --export=ALL,REWARD_FIELD="$FIELD" \
          online_RL_ucpg_v3/jobs/eval_real_narrow.sh)

echo "reward_field   : $FIELD"
echo "training array : $TRAIN_ID  (4 tasks)"
echo "evaluation     : $EVAL_ID  (waits for all 4 trainings to succeed)"
echo "results        : online_RL_ucpg_v3/results_narrow_${FIELD}/"
