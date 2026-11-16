#!/bin/bash
# =============================================================================
# rl_narrow/jobs/finalize.sh — after the real-physics evaluation: leaderboard + both paper figures.
# Inputs (kept for re-plotting later): rl_narrow/results_real/<method>.json (per-layer metrics) and
# <method>_fields.npz (real EoH / EoC fields of every layer and episode).
# Outputs: rl_narrow/results_real/{leaderboard_real.csv, per_layer_real.csv, constant_sweep.csv}
#          rl_narrow/results_real/figures/fig_rl_comparison.{pdf,png} + _data.json  (exact plotted values)
#          rl_narrow/results_real/figures/fig_rl_maps.{pdf,png}       + _data.npz   (exact plotted fields)
# Normally submitted by rl_narrow/jobs/run_all.sh (afterany the evaluation array).
# =============================================================================
#SBATCH --job-name=rln_final
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=0:30:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/rl_narrow/jobs/logs/finalize_%j.log

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc
conda activate RL
R=rl_narrow/results_real
mkdir -p $R/figures
python -m rl_narrow.aggregate --results_dir $R
python -m rl_narrow.plot_results --results_dir $R --out $R/figures/fig_rl_comparison
python -m rl_narrow.plot_maps --results_dir $R --out $R/figures/fig_rl_maps
echo "done $(date)"
