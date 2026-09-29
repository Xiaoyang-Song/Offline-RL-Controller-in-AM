#!/bin/bash
# =============================================================================
# sweep_diagnostic.sh
# SLURM job script — baselines.sweep_diagnostic: constant-power surrogate-vs-
# real "optimism" sweep. For each power in POWERS, rolls a constant-power
# policy through both the given surrogate and the REAL PDE simulator, and
# reports the signed gap (surrogate_return - real_return) overall AND
# per-layer. Use this BEFORE committing to a training-coverage / gap design —
# see baselines/sweep_diagnostic.py's module docstring.
#
# Real PDE solves are slow (~30-45s/layer) and this is CPU/MATLAB-only work
# (no GPU needed) — deliberately NOT run on the login node (its Arbiter
# daemon kills sustained CPU-heavy work).
#
# Requires: MATLAB module + LPBF-Simulation/simulation_v2 checkout alongside
# this repo, and the --surrogate checkpoint already trained.
#
# Submit (defaults to the existing narrow_200_300W checkpoint):
#   sbatch baselines/jobs/sweep_diagnostic.sh
# Or point at a different checkpoint / power grid (forwarded to the script):
#   sbatch baselines/jobs/sweep_diagnostic.sh \
#       --surrogate surrogate_model_v3/runs/gap_100-150_300-350/surrogate_best.pt \
#       --powers 100,130,160,190,220,250,280,310,340,370,400
# =============================================================================

#SBATCH --job-name=sweep_diagnostic
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/baselines/jobs/sweep_diagnostic_%j.log

echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

module load matlab
source ~/.bashrc
conda activate RL

echo "Python : $(which python)"
echo "MATLAB : $(which matlab)"
echo "============================================================"

set -e
set -o pipefail

DEFAULT_SURROGATE="surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt"
DEFAULT_POWERS="100,130,160,190,220,250,280,310,340,370,400"

python -u -m baselines.sweep_diagnostic \
    --surrogate "$DEFAULT_SURROGATE" \
    --powers "$DEFAULT_POWERS" \
    --n_episodes_surrogate 20 \
    --n_episodes_real 1 \
    --out_dir baselines/results_sweep_diagnostic \
    "$@"

EXIT_CODE=$?
echo "============================================================"
echo "Finished : $(date)  (exit code: $EXIT_CODE)"
echo "Outputs  : baselines/results_sweep_diagnostic/"
echo "============================================================"
exit $EXIT_CODE
