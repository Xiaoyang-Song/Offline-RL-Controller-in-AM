#!/bin/bash
# =============================================================================
# rl_narrow/jobs/train_offline.sh — offline methods from the logged NARROW data only (200-300 W
# transitions of the training builds; no digital twin). Reward: method.tex reward with w = 1.
# Array:
#   0-8 : offline RL (rl_narrow.offline_rl), {FQI, CQL, IQL} x seeds 1-3, 50 epochs, final checkpoint
#   9   : classical controllers fit by least squares on the same data
#         (proportional controller; Kalman-filter certainty-equivalence controller)
# Normally submitted by rl_narrow/jobs/run_all.sh.
# =============================================================================
#SBATCH --job-name=rln_offline
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=24GB
#SBATCH --time=4:00:00
#SBATCH --array=0-9
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/rl_narrow/jobs/logs/train_offline_%A_%a.log

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc
conda activate RL
set -e

I=$SLURM_ARRAY_TASK_ID
DATA=Data/DatasetV2_layer_12_samples_5000.pkl
echo "task $I  node $SLURMD_NODENAME  $(date)"
if [ "$I" -lt 9 ]; then
    ALGOS=(fqi cql iql)
    ALGO=${ALGOS[$(( I / 3 ))]}
    SEED=$(( I % 3 + 1 ))
    python -m rl_narrow.offline_rl --algo "$ALGO" --seed "$SEED" --data_path "$DATA" \
        --lp_min 200 --lp_max 300 --epochs 50 --out_dir rl_narrow/runs/offline_${ALGO}_s$SEED
else
    python -m baselines.proportional.controller --data_path "$DATA" --lp_filter_min 200 --lp_filter_max 300 \
        --out rl_narrow/runs/proportional_fitted.pt
    python -m baselines.kalman_particle.filters --data_path "$DATA" --lp_filter_min 200 --lp_filter_max 300 \
        --out rl_narrow/runs/kalman_fitted.pt
fi
echo "done $(date)"
