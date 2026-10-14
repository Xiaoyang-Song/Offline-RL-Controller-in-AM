#!/bin/bash
# =============================================================================
# train_full_datasize.sh
# SLURM job script — train a FULL-RANGE (100-400 W), noise-free surrogate on only
# NTRAJ training trajectories (data-scarcity study), with exactly the settings of the
# paper's full_range checkpoint otherwise:
#   perturb_frac 0.0, bootstrap_frac 1.0, rank 0, NO ROI loss weighting,
#   150 epochs, batch 256, lr 1e-3, patience 15, 5 members, seed 42 split.
# The subset is fixed and nested across sizes (train.py --max_train_traj); the
# val/test splits are the same as for every other model.
#
# Usually submitted by online_RL_ucpg_v3/jobs/run_datasize_mult.sh; standalone:
#   sbatch --export=ALL,NTRAJ=500 surrogate_model_v3/jobs/train_full_datasize.sh
#
# Output: surrogate_model_v3/runs/full_range_n<NTRAJ>/
# =============================================================================

#SBATCH --job-name=surrogate_v3_datasize
#SBATCH --account=jhjin1
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/train_full_datasize_%j.log

: "${NTRAJ:?set NTRAJ (number of training trajectories)}"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  $(date)   NTRAJ=$NTRAJ"
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc
conda activate RL
set -e

python -m surrogate_model_v3.train \
    --data_path      Data/DatasetV2_layer_12_samples_5000.pkl \
    --max_train_traj "$NTRAJ" \
    --perturb_frac   0.0 \
    --bootstrap_frac 1.0 \
    --rank           0   \
    --no_roi_weights     \
    --n_ensemble     5   \
    --epochs         150 \
    --batch_size     256 \
    --lr             1e-3 \
    --patience       15  \
    --num_workers    2   \
    --out_dir        "surrogate_model_v3/runs/full_range_n$NTRAJ"

echo "Finished: $(date)   Checkpoint: surrogate_model_v3/runs/full_range_n$NTRAJ/surrogate_best.pt"
