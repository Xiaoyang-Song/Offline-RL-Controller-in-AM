#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/eval_setting.sh — surrogate + real-physics evaluation of
# the 4 v3 runs of one setting, scored with the plain deviation of REWARD_FIELD.
# Env vars: TAG, SURROGATE, OOD_MIN, OOD_MAX, REWARD_FIELD (default heat).
# ~4 methods x ~20 min of MATLAB -> ~1.5 h. Normally submitted by run_setting.sh.
# =============================================================================
#SBATCH --job-name=ucpg_v3_eval
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/online_RL_ucpg_v3/jobs/eval_setting_%j.log

F="${REWARD_FIELD:-heat}"
R="online_RL_ucpg_v3/runs"
OUT="online_RL_ucpg_v3/results_${TAG}_${F}"
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
module load matlab; source ~/.bashrc; conda activate RL
set -e
echo "Started: $(date)  Node: $SLURMD_NODENAME  $TAG  field=$F  ID=[$OOD_MIN,$OOD_MAX] W"
METHODS=(
    --pg "PG plain=$R/${TAG}_pg_plain_$F/pg_best.pt"
    --pg "UCPG v3 shaped=$R/${TAG}_pg_shaped_$F/pg_best.pt"
    --q  "Q plain=$R/${TAG}_q_plain_$F/q_best.pt"
    --q  "Q v3 shaped=$R/${TAG}_q_shaped_$F/q_best.pt"
)
echo "=== [1/2] surrogate evaluation ==="
python -m online_RL_ucpg_v3.evaluate_surrogate --surrogate "$SURROGATE" --field "$F" \
    --ood_min "$OOD_MIN" --ood_max "$OOD_MAX" --n_episodes 100 "${METHODS[@]}" --out_dir "$OUT"
python -m online_RL_ucpg_v3.plot_comparison --results_dir "$OUT" --field "$F"
echo "=== [2/2] real-physics evaluation ==="
python -m online_RL_ucpg_v3.evaluate_real --surrogate "$SURROGATE" --field "$F" \
    --cool_times 0.06 0.10 0.14 "${METHODS[@]}" --out_dir "$OUT"
python -m online_RL_ucpg_v3.plot_comparison --results_dir "$OUT" --field "$F"
echo "Finished: $(date)   Results: $OUT/"
