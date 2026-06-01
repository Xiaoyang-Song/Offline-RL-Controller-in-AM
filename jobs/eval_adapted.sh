#!/bin/bash
# =============================================================================
# eval_adapted.sh
# SLURM GPU job — run test.py (DQN on GPU + MATLAB on CPU) for every K value.
# Submitted automatically by run_adapted_pipeline.sh after all training jobs
# finish.  plot_adapted.sh (CPU) runs after this job completes.
# =============================================================================

#SBATCH --job-name=eval_adapted
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/eval_adapted_%j.log

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
echo "Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
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

# ── MATLAB evaluation for each K ──────────────────────────────────────────────
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
    echo "  K=${K}  checkpoint: ${CKPT}"
    echo "------------------------------------------------------------"

    python test.py \
        --mode        OnlineRL \
        --checkpoint  "$CKPT" \
        --surrogate   "$BASE_CHECKPOINT" \
        --cool_time   "$COOL_TIME" \
        --results_out "$OUT"

    if [[ $? -eq 0 ]]; then
        echo "  K=${K}: done → ${OUT}"
    else
        echo "  K=${K}: FAILED"
        EVAL_FAILED=1
    fi
done

echo ""
echo "============================================================"
echo "Finished: $(date)  (eval_failed=${EVAL_FAILED})"
echo "============================================================"
exit $EVAL_FAILED
