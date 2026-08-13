#!/bin/bash
# =============================================================================
# evaluate_surrogate_v2_gap.sh
# SLURM job script — evaluate the patchy-coverage two-stage surrogate
# produced by jobs/train_surrogate_v2_gap_perturb.sh:
#   1. evaluate.py      — standard single-step/rollout metrics + example
#                          field plots on the (unfiltered) held-out test split
#   2. evaluate_ood.py   — the actual point of this experiment: ID vs. OOD
#                          (the three gaps) epistemic/aleatoric/RMSE split,
#                          using the SAME --lp_filter_ranges as training
#
# Requires the checkpoint from jobs/train_surrogate_v2_gap_perturb.sh to
# already exist (chained automatically by jobs/oneshot_gap_experiment.sh).
#
# Submit (defaults):
#   sbatch jobs/evaluate_surrogate_v2_gap.sh
# =============================================================================

#SBATCH --job-name=eval_surrogate_v2_gap
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/eval_surrogate_v2_gap_%j.log

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

DATA_PATH="Data/DatasetV2_layer_12_samples_5000.pkl"
CHECKPOINT="surrogate_model_latent_uncertainty_v2/runs/patchy_100-150_200-250_300-350_perturb0.1/two_stage_best.pt"
ID_RANGES="100-150,200-250,300-350"

# ── 1. Standard evaluation (unfiltered held-out split) ───────────────────────
echo "[eval] === evaluate.py (standard single-step/rollout metrics) ==="
python -m surrogate_model_latent_uncertainty_v2.evaluate \
    --checkpoint "$CHECKPOINT" \
    --data_path  "$DATA_PATH"
EXIT_1=$?

# ── 2. ID/OOD split — the actual point of this experiment ───────────────────
echo "[eval] === evaluate_ood.py (ID vs. OOD, --id_ranges matches training) ==="
python -m surrogate_model_latent_uncertainty_v2.evaluate_ood \
    --checkpoint "$CHECKPOINT" \
    --data_path  "$DATA_PATH" \
    --id_ranges  "$ID_RANGES"
EXIT_2=$?

EXIT_CODE=0
[[ $EXIT_1 -ne 0 || $EXIT_2 -ne 0 ]] && EXIT_CODE=1

echo "============================================================"
echo "Surrogate evaluation finished: $(date)  (evaluate.py=$EXIT_1, evaluate_ood.py=$EXIT_2)"
echo "============================================================"
exit $EXIT_CODE
