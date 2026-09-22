#!/bin/bash
# =============================================================================
# dataset.sh
# SLURM job script — extract + pickle the v2 (heating/cooling) LPBF
# trajectories from ../LPBF-Simulation/simulation_v2/RL_Dataset_v2/.
#
# NOTE: Data/DatasetV2_layer_12_samples_5000.pkl already exists (extracted
# via the sibling surrogate_model_latent_uncertainty_v2 package) and loads
# fine through surrogate_model_v3.dataset.load_trajectories — see that
# module's docstring for why the older pickle's class binding doesn't
# matter. Only submit this job if you need MORE trajectories than that file
# already has.
#
# Submit:
#   sbatch surrogate_model_v3/jobs/dataset.sh
# =============================================================================

#SBATCH --job-name=dataset_v3
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/dataset_%j.log

# ── environment ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

source ~/.bashrc
conda activate RL

echo "Python : $(which python)"
echo "============================================================"

# ── extract dataset ──────────────────────────────────────────────────────────
N=5000
TRAJECTORY_LENGTH=12
OUT="Data/DatasetV3_layer_${TRAJECTORY_LENGTH}_samples_${N}.pkl"

python -m surrogate_model_v3.extract_dataset \
    --n                  $N \
    --trajectory_length  $TRAJECTORY_LENGTH \
    --out                "$OUT"

echo "============================================================"
echo "Finished : $(date)"
echo "============================================================"
