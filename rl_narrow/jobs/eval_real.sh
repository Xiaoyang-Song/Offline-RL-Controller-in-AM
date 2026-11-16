#!/bin/bash
# =============================================================================
# rl_narrow/jobs/eval_real.sh — real-physics evaluation (MATLAB PDE, heatTime 0.025 s), one method per
# array task, identical episodes for all (cooling times 0.05, 0.07, ..., 0.15 s; greedy policies).
#   0-5   online   : PG shaped (ours) / online Q shaped, seeds 1-3
#   6-14  offline  : FQI / CQL / IQL, seeds 1-3
#   15    proportional, 16 Kalman filter
#   17-47 constant : 100, 110, ..., 400 W
# ~6 x 12 MATLAB solves x ~30 s ≈ 40 min per task. Then: python -m rl_narrow.aggregate
# Normally submitted by rl_narrow/jobs/run_all.sh (after both training arrays).
# =============================================================================
#SBATCH --job-name=rln_real
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=3:00:00
#SBATCH --array=0-47
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/rl_narrow/jobs/logs/eval_real_%A_%a.log

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc
conda activate RL
module load matlab
set -e

R=rl_narrow/runs
M=()
for s in 1 2 3; do M+=("pg:PG shaped (ours) seed $s=$R/pg_shaped_s$s/pg_best.pt"); done
for s in 1 2 3; do M+=("q:Online Q shaped seed $s=$R/q_shaped_s$s/q_best.pt"); done
for a in FQI CQL IQL; do
    for s in 1 2 3; do M+=("offline:Offline $a seed $s=$R/offline_${a,,}_s$s/offline_final.pt"); done
done
M+=("prop:Proportional=$R/proportional_fitted.pt")
M+=("kalman:Kalman filter=$R/kalman_fitted.pt")
for w in $(seq 100 10 400); do M+=("const:Constant $w W=$w"); done

SPEC="${M[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID / ${#M[@]}: $SPEC   node $SLURMD_NODENAME  $(date)"
python -m rl_narrow.evaluate_real --method "$SPEC" --device cpu --out_dir rl_narrow/results_real
echo "done $(date)"
