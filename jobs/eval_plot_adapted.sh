#!/bin/bash
# =============================================================================
# eval_plot_adapted.sh
# SLURM job — MATLAB evaluation + sweep plotting for all K values.
# Submitted automatically by run_adapted_pipeline.sh after all training jobs
# finish. Do not submit this job manually unless training is already done.
#
# For each K:
#   python test.py --mode OnlineRL  --cool_time 0.15  --results_out <K>/matlab_eval.json
# Then:
#   python -m surrogate_domain_adaptation.plot_rl_sweep --adapt_dir <adapt_dir>
# =============================================================================

#SBATCH --job-name=eval_plot_adapted
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/eval_plot_adapted_%j.log

# ── environment ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

source ~/.bashrc
conda activate RL

module load matlab
echo "Python : $(which python)"
echo "MATLAB : $(which matlab)"
echo "============================================================"

# ── parse args ────────────────────────────────────────────────────────────────
ADAPT_DIR=""
BASE_CHECKPOINT=""
COOL_TIME=0.15
K_LIST=(0 5 10 20 50 100)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapt_dir)       ADAPT_DIR="$2";       shift 2 ;;
        --base_checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
        --cool_time)       COOL_TIME="$2";       shift 2 ;;
        --k_values)
            shift; K_LIST=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                K_LIST+=("$1"); shift
            done ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$ADAPT_DIR" || -z "$BASE_CHECKPOINT" ]]; then
    echo "ERROR: --adapt_dir and --base_checkpoint are required."
    exit 1
fi

echo "adapt_dir        : $ADAPT_DIR"
echo "base_checkpoint  : $BASE_CHECKPOINT"
echo "cool_time        : $COOL_TIME s"
echo "K values         : ${K_LIST[*]}"
echo "============================================================"

# ── Phase 2a: MATLAB evaluation for each K ────────────────────────────────────
echo ""
echo "Running MATLAB evaluations (coolTime=${COOL_TIME} s)..."

EVAL_FAILED=0
for K in "${K_LIST[@]}"; do
    KDIR=$(printf '%03d' "$K")
    CKPT="${ADAPT_DIR}/K${KDIR}/online_rl/dqn_best.pt"
    OUT="${ADAPT_DIR}/K${KDIR}/online_rl/matlab_eval.json"

    if [[ ! -f "$CKPT" ]]; then
        echo "  K=${K}: dqn_best.pt not found — skipping  (${CKPT})"
        EVAL_FAILED=1
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "  Evaluating K=${K}  (checkpoint: ${CKPT})"
    echo "------------------------------------------------------------"

    python test.py \
        --mode        OnlineRL \
        --checkpoint  "$CKPT" \
        --surrogate   "$BASE_CHECKPOINT" \
        --cool_time   "$COOL_TIME" \
        --results_out "$OUT"

    if [[ $? -eq 0 ]]; then
        echo "  K=${K}: MATLAB eval complete → ${OUT}"
    else
        echo "  K=${K}: MATLAB eval FAILED"
        EVAL_FAILED=1
    fi
done

# ── Phase 2b: aggregate plots ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Generating sweep plots..."
echo "============================================================"

python -m surrogate_domain_adaptation.plot_rl_sweep \
    --adapt_dir "$ADAPT_DIR"

PLOT_EXIT=$?

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Finished: $(date)"
echo "Eval failures : $EVAL_FAILED  (0 = all OK)"
echo "Plot exit code: $PLOT_EXIT"
echo ""
echo "Outputs:"
echo "  ${ADAPT_DIR}/return_vs_K.png"
echo "  ${ADAPT_DIR}/per_layer_rewards.png"
echo "  ${ADAPT_DIR}/per_layer_actions.png"
echo "  ${ADAPT_DIR}/metrics_table.txt"
echo "============================================================"

[[ $EVAL_FAILED -eq 0 && $PLOT_EXIT -eq 0 ]] && exit 0 || exit 1
