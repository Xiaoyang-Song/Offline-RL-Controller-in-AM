#!/bin/bash
# =============================================================================
# plot_adapted.sh
# SLURM CPU job — aggregate per-K matlab_eval.json files and produce plots.
# Submitted automatically by run_adapted_pipeline.sh after eval_adapted.sh.
# =============================================================================

#SBATCH --job-name=plot_adapted
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=0:15:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/plot_adapted_%j.log

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

# ── parse args ────────────────────────────────────────────────────────────────
ADAPT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapt_dir) ADAPT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$ADAPT_DIR" ]]; then
    echo "ERROR: --adapt_dir is required."
    exit 1
fi

echo "adapt_dir : $ADAPT_DIR"
echo "============================================================"

# ── generate sweep plots ──────────────────────────────────────────────────────
python -m surrogate_domain_adaptation.plot_rl_sweep \
    --adapt_dir "$ADAPT_DIR"

EXIT=$?
echo ""
echo "============================================================"
echo "Finished: $(date)  (exit=${EXIT})"
echo ""
echo "Outputs:"
echo "  ${ADAPT_DIR}/return_vs_K.png"
echo "  ${ADAPT_DIR}/per_layer_rewards.png"
echo "  ${ADAPT_DIR}/per_layer_actions.png"
echo "  ${ADAPT_DIR}/metrics_table.txt"
echo "============================================================"
exit $EXIT
