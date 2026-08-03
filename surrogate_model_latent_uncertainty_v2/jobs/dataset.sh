#!/bin/bash
# =============================================================================
# dataset.sh
# SLURM job script — extract + pickle the v2 (heating/cooling) LPBF
# trajectories from ../LPBF-Simulation/simulation_v2/RL_Dataset_v2/.
#
# Submit:
#   sbatch surrogate_model_latent_uncertainty_v2/jobs/dataset.sh
# =============================================================================

#SBATCH --job-name=dataset_v2
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_latent_uncertainty_v2/jobs/dataset_%j.log

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
OUT="Data/DatasetV2_layer_${TRAJECTORY_LENGTH}_samples_${N}.pkl"

python -m surrogate_model_latent_uncertainty_v2.extract_dataset_v2 \
    --n                  $N \
    --trajectory_length  $TRAJECTORY_LENGTH \
    --out                "$OUT"

echo "============================================================"
echo "Finished : $(date)"
echo "============================================================"
