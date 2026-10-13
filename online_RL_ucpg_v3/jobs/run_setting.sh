#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/run_setting.sh — train (4 parallel) + evaluate (chained)
# v3 for one surrogate setting. Run on the login node:
#   bash online_RL_ucpg_v3/jobs/run_setting.sh range_100-150 [heat|cool]
#   bash online_RL_ucpg_v3/jobs/run_setting.sh range_100-200 [heat|cool]
# Known settings below (surrogate checkpoint + its training laser-power range).
# Results: online_RL_ucpg_v3/results_<setting>_<field>/
# =============================================================================
SETTING="$1"; FIELD="${2:-heat}"
S=surrogate_model_v3/runs
case "$SETTING" in
    range_100-150) SURROGATE=$S/range_100-150_perturb0.2_bootstrap0.5/surrogate_best.pt; LO=100; HI=150 ;;
    range_100-200) SURROGATE=$S/range_100-200_perturb0.2_bootstrap0.5/surrogate_best.pt; LO=100; HI=200 ;;
    range_300-400) SURROGATE=$S/range_300-400_perturb0.2_bootstrap0.5/surrogate_best.pt; LO=300; HI=400 ;;
    narrow_200_300W) SURROGATE=$S/narrow_200_300W/surrogate_best.pt; LO=200; HI=300 ;;
    *) echo "unknown setting '$SETTING' (range_100-150 | range_100-200 | range_300-400 | narrow_200_300W)"; exit 1 ;;
esac
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
V="ALL,TAG=$SETTING,SURROGATE=$SURROGATE,OOD_MIN=$LO,OOD_MAX=$HI,REWARD_FIELD=$FIELD"
TRAIN_ID=$(sbatch --parsable --export="$V" online_RL_ucpg_v3/jobs/train_setting.sh)
EVAL_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_ID" --export="$V" online_RL_ucpg_v3/jobs/eval_setting.sh)
echo "setting: $SETTING  field: $FIELD  ID range: [$LO, $HI] W"
echo "training array: $TRAIN_ID (4 tasks)   evaluation: $EVAL_ID (after training)"
echo "results: online_RL_ucpg_v3/results_${SETTING}_${FIELD}/"
