#!/bin/bash
# =============================================================================
# run_adapted_pipeline.sh
# Full pipeline coordinator: domain-adapted online RL sweep.
#
# Run with bash (NOT sbatch) — this script submits SLURM jobs and wires
# dependencies between them:
#
#   bash jobs/run_adapted_pipeline.sh \
#       --adapt_dir     surrogate_domain_adaptation/runs/base/<ts>/adapt_ct150 \
#       --base_checkpoint surrogate_domain_adaptation/runs/base/<ts>/base_best.pt
#
# #   bash jobs/run_adapted_pipeline.sh --adapt_dir surrogate_domain_adaptation/runs/base/20260601_174626/adapt_ct150 --base_checkpoint surrogate_domain_adaptation/runs/base/20260601_174626/base_best.pt
# What it submits
# ---------------
#   Phase 1 (GPU, parallel)
#     One train_online_rl_adapted.sh job per K value.
#     K=0 trains on the unadapted base surrogate (ct=0.10 dynamics);
#     K>0 trains on the K-shot adapted surrogate (ct=0.15 dynamics).
#
#   Phase 2 (CPU, sequential, starts after ALL Phase-1 jobs finish)
#     eval_plot_adapted.sh:
#       • Runs test.py with MATLAB at coolTime=0.15 for each K
#       • Aggregates results with plot_rl_sweep.py
#
# Output layout
# -------------
#   <adapt_dir>/
#     K000/online_rl/dqn_best.pt          (K=0 baseline)
#     K005/online_rl/dqn_best.pt
#     ...
#     K100/online_rl/dqn_best.pt
#     K000/online_rl/matlab_eval.json
#     ...
#     return_vs_K.png
#     per_layer_rewards.png
#     per_layer_actions.png
#     metrics_table.txt
# =============================================================================

set -euo pipefail

REPO=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

# ── defaults ─────────────────────────────────────────────────────────────────
ADAPT_DIR=""
BASE_CHECKPOINT=""
K_LIST=(0 5 10 20 50 100)
COOL_TIME=0.15                  # MATLAB evaluation cooling time

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapt_dir)       ADAPT_DIR="$2";       shift 2 ;;
        --base_checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
        --k_values)
            shift; K_LIST=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                K_LIST+=("$1"); shift
            done ;;
        --cool_time)       COOL_TIME="$2";       shift 2 ;;
        *)  echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$ADAPT_DIR" || -z "$BASE_CHECKPOINT" ]]; then
    echo "Usage:"
    echo "  bash jobs/run_adapted_pipeline.sh \\"
    echo "       --adapt_dir     <adapt_ct150 dir> \\"
    echo "       --base_checkpoint <base_best.pt>"
    exit 1
fi

echo "======================================================================"
echo "  Adapted Online RL Pipeline"
echo "======================================================================"
echo "  Adapt dir        : $ADAPT_DIR"
echo "  Base checkpoint  : $BASE_CHECKPOINT"
echo "  K values         : ${K_LIST[*]}"
echo "  MATLAB cool_time : $COOL_TIME s"
echo "======================================================================"

# ── Phase 1: submit one GPU training job per K ────────────────────────────────
echo ""
echo "Phase 1 — submitting training jobs (GPU, in parallel)..."

JOB_IDS=()
for K in "${K_LIST[@]}"; do
    JID=$(sbatch --parsable \
        "${REPO}/jobs/train_online_rl_adapted.sh" \
        --adapt_dir       "$ADAPT_DIR" \
        --base_checkpoint "$BASE_CHECKPOINT" \
        --K               "$K")
    echo "  K=${K}  →  SLURM job ${JID}"
    JOB_IDS+=("$JID")
done

# Build afterok dependency string: afterok:JID1:JID2:...
DEP_STR="afterok"
for JID in "${JOB_IDS[@]}"; do
    DEP_STR="${DEP_STR}:${JID}"
done
echo ""
echo "  All training jobs submitted (dependency: ${DEP_STR})"

# ── Phase 2: GPU eval job (test.py, depends on all Phase-1 jobs) ─────────────
echo ""
echo "Phase 2 — submitting eval job (GPU, starts after all training jobs finish)..."

K_STR="${K_LIST[*]}"

EVAL_JID=$(sbatch --parsable \
    --dependency="${DEP_STR}" \
    "${REPO}/jobs/eval_adapted.sh" \
    --adapt_dir       "$ADAPT_DIR" \
    --base_checkpoint "$BASE_CHECKPOINT" \
    --cool_time       "$COOL_TIME" \
    --k_values        $K_STR)

echo "  Eval job         →  SLURM job ${EVAL_JID}"

# ── Phase 3: CPU plot job (plot_rl_sweep.py, depends on Phase-2) ─────────────
echo ""
echo "Phase 3 — submitting plot job (CPU, starts after eval job finishes)..."

PLOT_JID=$(sbatch --parsable \
    --dependency="afterok:${EVAL_JID}" \
    "${REPO}/jobs/plot_adapted.sh" \
    --adapt_dir "$ADAPT_DIR")

echo "  Plot job         →  SLURM job ${PLOT_JID}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  All jobs submitted.  Monitor progress:"
echo ""
echo "    squeue -u \$USER"
echo ""
echo "  Training logs (Phase 1 — GPU):"
for JID in "${JOB_IDS[@]}"; do
    echo "    ${REPO}/jobs/rl_adapted_${JID}.log"
done
echo ""
echo "  Eval log (Phase 2 — GPU):"
echo "    ${REPO}/jobs/eval_adapted_${EVAL_JID}.log"
echo ""
echo "  Plot log (Phase 3 — CPU):"
echo "    ${REPO}/jobs/plot_adapted_${PLOT_JID}.log"
echo ""
echo "  Final plots will appear in:"
echo "    ${ADAPT_DIR}/"
echo "======================================================================"
