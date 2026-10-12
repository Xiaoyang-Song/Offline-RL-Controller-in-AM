#!/bin/bash
# =============================================================================
# aleatoric_calibration.sh
# SLURM ARRAY job — train surrogate_v3 with known injected target noise
# (--perturb_frac ε) for several data-coverage settings, then measure how well
# the learned aleatoric σ recovers ε (paper figure: learned vs. expected noise).
#
# One array task = one (setting, ε) pair:
#   task 0-3 : full range (100-400 W)              ε = 0.01, 0.05, 0.1, 0.2
#   task 4-7 : narrow (200-300 W)                  ε = 0.01, 0.05, 0.1, 0.2
#   (the gap setting, 100-150 + 350-400 W, is trained separately:
#    runs/gap_100-150_350-400_perturb<ε>_bootstrap0.5 — its "gap" case below
#    is kept so a gap run can still be added with an explicit --array index)
#
# Every task uses the SAME training settings as those gap runs (bootstrap_frac
# 0.5, 5 members, 500 epochs, batch 128, lr 1e-3, patience 30, perturb_seed 0),
# so only the coverage setting and ε differ. These are separate models from the
# main full_range / narrow_200_300W checkpoints (which were trained without
# noise) — nothing existing is overwritten.
#
# Each task runs:
#   1. train                         -> surrogate_model_v3/runs/<setting>_perturb<ε>_bootstrap0.5/
#   2. evaluate_ood (test split, propagated)  -> <RES>/ood_test.npz
#   3. expected_aleatoric_jacobian   -> <RES>/expected_full_aleatoric.json
#   with <RES> = surrogate_model_v3/results_aleatoric_calibration/<setting>_perturb<ε>/
#
# Submit all 8:
#   sbatch surrogate_model_v3/jobs/aleatoric_calibration.sh
# Or a subset, e.g. only the narrow runs:
#   sbatch --array=4-7 surrogate_model_v3/jobs/aleatoric_calibration.sh
# =============================================================================

#SBATCH --job-name=surrogate_v3_alecal
#SBATCH --account=jhjin1
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/surrogate_model_v3/jobs/aleatoric_calibration_%A_%a.log

SETTINGS=(full full full full narrow narrow narrow narrow gap)
EPSILONS=(0.01 0.05 0.1 0.2 0.01 0.05 0.1 0.2 0.01)

SETTING=${SETTINGS[$SLURM_ARRAY_TASK_ID]}
EPS=${EPSILONS[$SLURM_ARRAY_TASK_ID]}

case $SETTING in
    full)
        NAME="full_range"
        FILTER_ARGS=()
        ID_ARGS=(--id_action_min 100 --id_action_max 400)
        ;;
    narrow)
        NAME="narrow_200_300W"
        FILTER_ARGS=(--lp_filter_ranges "200-300")
        ID_ARGS=(--id_ranges "200-300")
        ;;
    gap)
        NAME="gap_100-150_350-400"
        FILTER_ARGS=(--lp_filter_ranges "100-150,350-400")
        ID_ARGS=(--id_ranges "100-150,350-400")
        ;;
esac

RUN_DIR="surrogate_model_v3/runs/${NAME}_perturb${EPS}_bootstrap0.5"
RES_DIR="surrogate_model_v3/results_aleatoric_calibration/${NAME}_perturb${EPS}"
DATA="Data/DatasetV2_layer_12_samples_5000.pkl"

echo "============================================================"
echo "Job ID   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "Setting  : $NAME   perturb_frac = $EPS"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
source ~/.bashrc
conda activate RL
set -e
mkdir -p "$RES_DIR"

echo "=== [1/3] train ==="
python -m surrogate_model_v3.train \
    --data_path      "$DATA" \
    "${FILTER_ARGS[@]}" \
    --perturb_frac   "$EPS" \
    --perturb_seed   0 \
    --bootstrap_frac 0.5 \
    --n_ensemble     5 \
    --epochs         500 \
    --batch_size     128 \
    --lr             1e-3 \
    --patience       30 \
    --out_dir        "$RUN_DIR"

echo "=== [2/3] evaluate_ood (test split, propagated uncertainty) ==="
python -m surrogate_model_v3.evaluate_ood \
    --checkpoint "$RUN_DIR/surrogate_best.pt" \
    --data_path  "$DATA" \
    "${ID_ARGS[@]}" \
    --split test --propagate_uncertainty \
    --out_dir    "$RES_DIR" \
    --dump_npz   "$RES_DIR/ood_test.npz"

echo "=== [3/3] Jacobian-based expected aleatoric σ ==="
python -m surrogate_model_v3.expected_aleatoric_jacobian \
    --checkpoint "$RUN_DIR/surrogate_best.pt" --perturb "$EPS" \
    --data_path  "$DATA" \
    --out        "$RES_DIR/expected_full_aleatoric.json"

echo "============================================================"
echo "Finished : $(date)"
echo "Outputs  : $RUN_DIR/  and  $RES_DIR/"
echo "============================================================"
