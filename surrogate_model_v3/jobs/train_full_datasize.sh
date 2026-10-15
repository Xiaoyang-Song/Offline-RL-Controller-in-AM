#!/bin/bash
# =============================================================================
# train_full_datasize.sh
# SLURM job script — train a noise-free surrogate on only NTRAJ training
# trajectories (data-scarcity study), with exactly the settings of the paper's
# full_range / narrow_200_300W checkpoints otherwise:
#   perturb_frac 0.0, bootstrap_frac 1.0, rank 0, NO ROI loss weighting,
#   batch 256, lr 1e-3, 5 members, seed 42 split.
# The subset is fixed and nested across sizes (train.py --max_train_traj); the
# val/test splits are the same as for every other model.
#
# Env vars:
#   NTRAJ         number of training trajectories (required)
#   RANGE         full (default, 100-400 W) | narrow (training transitions 200-300 W only)
#   STEP_MATCHED  0 (default): 150 epochs, patience 15 — the protocol of the first 500/1000/2000 runs.
#                 1: epochs = min(12000, 150 x 4000 / NTRAJ), patience 50 — about the same number of
#                    gradient steps as the 4000-trajectory reference, so small sets are not also
#                    under-trained (output dir gets a _steps suffix).
#
# Usually submitted by online_RL_ucpg_v3/jobs/run_datasize_mult.sh; standalone:
#   sbatch --export=ALL,NTRAJ=100,RANGE=narrow,STEP_MATCHED=1 surrogate_model_v3/jobs/train_full_datasize.sh
#
# Output: surrogate_model_v3/runs/{full_range|narrow_200_300W}_n<NTRAJ>[_steps]/
# =============================================================================

#SBATCH --job-name=surrogate_v3_datasize
#SBATCH --account=jhjin1
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/train_full_datasize_%j.log

: "${NTRAJ:?set NTRAJ (number of training trajectories)}"
RANGE="${RANGE:-full}"; STEP_MATCHED="${STEP_MATCHED:-0}"
case "$RANGE" in
    full)   FILTER=();                              BASE="full_range" ;;
    narrow) FILTER=(--lp_filter_ranges "200-300");  BASE="narrow_200_300W" ;;
    *) echo "RANGE must be full or narrow"; exit 1 ;;
esac
if [ "$STEP_MATCHED" = "1" ]; then
    EPOCHS=$(( 150 * 4000 / NTRAJ )); [ "$EPOCHS" -gt 12000 ] && EPOCHS=12000; [ "$EPOCHS" -lt 150 ] && EPOCHS=150
    PATIENCE=50; SUFFIX="_steps"
else
    EPOCHS=150; PATIENCE=15; SUFFIX=""
fi
OUT_DIR="surrogate_model_v3/runs/${BASE}_n${NTRAJ}${SUFFIX}"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  $(date)   NTRAJ=$NTRAJ  RANGE=$RANGE  epochs=$EPOCHS  patience=$PATIENCE -> $OUT_DIR"
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc
conda activate RL
set -e

python -m surrogate_model_v3.train \
    --data_path      Data/DatasetV2_layer_12_samples_5000.pkl \
    --max_train_traj "$NTRAJ" "${FILTER[@]}" \
    --perturb_frac   0.0 \
    --bootstrap_frac 1.0 \
    --rank           0   \
    --no_roi_weights     \
    --n_ensemble     5   \
    --epochs         "$EPOCHS" \
    --batch_size     256 \
    --lr             1e-3 \
    --patience       "$PATIENCE" \
    --num_workers    2   \
    --out_dir        "$OUT_DIR"

echo "Finished: $(date)   Checkpoint: $OUT_DIR/surrogate_best.pt"
