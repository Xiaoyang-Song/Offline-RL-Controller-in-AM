#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/eval_real_narrow.sh
# Evaluation of the four v3 controllers plus the earlier heating-trained naive
# PG / online Q for reference, in TWO settings, always with the PLAIN
# (unpenalised) deviation of the field the v3 methods were trained on
# (REWARD_FIELD: post-cooling or end-of-heating):
#   1. surrogate (member rollouts, 100 episodes, same cool times for all) — minutes
#   2. real physics (MATLAB PDE, greedy, cool_time 0.06 / 0.10 / 0.14 s)
# then comparison.{pdf,png} with both. Results: online_RL_ucpg_v3/results_narrow_<field>/
#
# ~12 MATLAB solves per episode at ~30-45 s each -> ~20 min per method,
# ~2 h for the 6 methods below.
#
# Normally submitted for you by jobs/run_narrow.sh (chained after training).
# Manual: sbatch [--export=ALL,REWARD_FIELD=heat] online_RL_ucpg_v3/jobs/eval_real_narrow.sh
# =============================================================================

#SBATCH --job-name=ucpg_v3_real
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/online_RL_ucpg_v3/jobs/eval_real_narrow_%j.log

REWARD_FIELD="${REWARD_FIELD:-cool}"
R="online_RL_ucpg_v3/runs"
F="$REWARD_FIELD"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
module load matlab
source ~/.bashrc
conda activate RL
set -e
echo "Started: $(date)   Node: $SLURMD_NODENAME   reward_field of v3 runs: $F"

OUT="online_RL_ucpg_v3/results_narrow_$F"
SURR="surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt"
METHODS=(
    --pg "PG plain ($F)=$R/narrow_pg_plain_$F/pg_best.pt"
    --pg "UCPG v3 shaped ($F)=$R/narrow_pg_shaped_$F/pg_best.pt"
    --q  "Q plain ($F)=$R/narrow_q_plain_$F/q_best.pt"
    --q  "Q v3 shaped ($F)=$R/narrow_q_shaped_$F/q_best.pt"
    --pg "Naive PG (heat, earlier)=baselines/naive_pg/runs/narrow_200_300W/naive_pg_best.pt"
    --q  "Online Q (heat, earlier)=baselines/online_q/runs/narrow_200_300W/online_q_best.pt"
)

echo "=== [1/2] surrogate evaluation (plain deviation, 100 episodes) ==="
python -m online_RL_ucpg_v3.evaluate_surrogate --surrogate "$SURR" --field "$F" \
    --ood_min 200 --ood_max 300 --n_episodes 100 "${METHODS[@]}" --out_dir "$OUT"
python -m online_RL_ucpg_v3.plot_comparison --results_dir "$OUT" --field "$F"

echo "=== [2/2] real-physics evaluation (MATLAB, 3 episodes) ==="
python -m online_RL_ucpg_v3.evaluate_real --surrogate "$SURR" --field "$F" \
    --cool_times 0.06 0.10 0.14 "${METHODS[@]}" --out_dir "$OUT"
python -m online_RL_ucpg_v3.plot_comparison --results_dir "$OUT" --field "$F"

echo "Finished: $(date)   Results: $OUT/"
