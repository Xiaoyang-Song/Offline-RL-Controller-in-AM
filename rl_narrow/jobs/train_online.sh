#!/bin/bash
# =============================================================================
# rl_narrow/jobs/train_online.sh — online RL in the NARROW (200-300 W) digital twin, method.tex reward:
#   r_l = mean_ROI w exp(-d / W_T) on the end-of-heating field, w = tau^2 / (tau^2 + sigma^2),
#   sigma = propagated TOTAL (epistemic + aleatoric) EoH std, tau = TAU_K x median in-distribution sigma.
# TAU_K defaults to 8: in the narrow data-scarcity runs (real physics) k = 2 pulled the policy back
# toward the data range and cost ~60 K of EoH deviation, while k = 8 kept the policy near the optimum.
# Override: sbatch --export=ALL,TAU_K=4 rl_narrow/jobs/train_online.sh
# Array (3 seeds each):
#   0-2 : PG, shaped (ours)    (REINFORCE + batch-mean baseline, B = 32, 2000 iterations)
#   3-5 : online Q, shaped     (DQN on the same environment and reward, 3000 episodes)
# Normally submitted by rl_narrow/jobs/run_all.sh.
# =============================================================================
#SBATCH --job-name=rln_online
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/rl_narrow/jobs/logs/train_online_%A_%a.log

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc
conda activate RL
set -e

I=$SLURM_ARRAY_TASK_ID
SEED=$(( I % 3 + 1 ))
COMMON=(--surrogate surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt
        --action_min 100 --action_max 400 --ood_min 200 --ood_max 300
        --reward_field heat --reward_mode shaped --shaping multiplicative
        --uncertainty_mode propagated --uncertainty total --tau_k "${TAU_K:-8}" --seed "$SEED")
echo "task $I  seed $SEED  tau_k ${TAU_K:-8}  node $SLURMD_NODENAME  $(date)"

if [ "$I" -lt 3 ]; then
    python -m online_RL_ucpg_v3.train_pg "${COMMON[@]}" --n_iterations 2000 --n_traj 32 --use_baseline \
        --out_dir rl_narrow/runs/pg_shaped_s$SEED
else
    python -m online_RL_ucpg_v3.train_q "${COMMON[@]}" --n_episodes 3000 \
        --out_dir rl_narrow/runs/q_shaped_s$SEED
fi
echo "done $(date)"
