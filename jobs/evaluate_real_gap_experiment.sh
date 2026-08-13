#!/bin/bash
# =============================================================================
# evaluate_real_gap_experiment.sh
# SLURM job script — ground-truth check for the patchy-coverage gap
# experiment: evaluate BOTH the naive_pg and UCPG v2 checkpoints trained by
# jobs/train_ucpg_v2_gap.sh / baselines/jobs/train_naive_pg_gap.sh against
# the REAL PDE simulator (simulateHeatingCooling_v2.m, not the surrogate),
# then compare them — does the uncertainty-constrained policy actually avoid
# the surrogate's three untrained gaps (150-200W, 250-300W, 350-400W) and
# get a better REAL reward than the naive policy?
#
# Fixed-path counterpart to jobs/evaluate_real_ucpg_v2.sh (that one has
# placeholder checkpoint paths for manual one-off comparisons; this one's
# paths match jobs/train_ucpg_v2_gap.sh / baselines/jobs/train_naive_pg_gap.sh's
# fixed --out_dir exactly, so it needs no editing — built for
# jobs/oneshot_gap_experiment.sh to chain automatically).
#
# This does real PDE solves (~30-45s/layer, ~6-9 min per 12-layer episode) —
# deliberately NOT run on the login node (its Arbiter daemon kills sustained
# CPU-heavy work).
#
# Submit (defaults):
#   sbatch jobs/evaluate_real_gap_experiment.sh
# =============================================================================

#SBATCH --job-name=eval_real_gap
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/eval_real_gap_%j.log

# ── environment ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

module load matlab
source ~/.bashrc
conda activate RL

echo "Python : $(which python)"
echo "MATLAB : $(which matlab)"
echo "============================================================"

SURROGATE="surrogate_model_latent_uncertainty_v2/runs/patchy_100-150_200-250_300-350_perturb0.1/two_stage_best.pt"
NAIVE_PG_CHECKPOINT="baselines/naive_pg/runs/patchy_100-150_200-250_300-350_perturb0.1/naive_pg_best.pt"
UCPG_V2_CHECKPOINT="online_RL_ucpg_v2/runs/patchy_100-150_200-250_300-350_perturb0.1/ucpg_best.pt"
ID_RANGES="100-150,200-250,300-350"

COOL_TIME=0.10       # fixed for BOTH runs — keeps the comparison fair
N_EPISODES=5         # real PDE solves are slow; increase only if you need tighter variance estimates

RESULTS_DIR="jobs/results_gap_experiment"
mkdir -p "$RESULTS_DIR"

# ── 0. Naive PG vs. real physics ─────────────────────────────────────────────
echo "[eval_real] === Naive PG ==="
python -m online_RL_ucpg_v2.evaluate_real \
    --checkpoint "$NAIVE_PG_CHECKPOINT" \
    --surrogate  "$SURROGATE" \
    --cool_time  $COOL_TIME \
    --n_episodes $N_EPISODES \
    --results_out "$RESULTS_DIR/eval_real_naive_pg.json" \
    --plot
EXIT_1=$?

# ── 1. UCPG v2 vs. real physics ──────────────────────────────────────────────
echo "[eval_real] === UCPG v2 ==="
python -m online_RL_ucpg_v2.evaluate_real \
    --checkpoint "$UCPG_V2_CHECKPOINT" \
    --surrogate  "$SURROGATE" \
    --cool_time  $COOL_TIME \
    --n_episodes $N_EPISODES \
    --results_out "$RESULTS_DIR/eval_real_ucpg_v2.json" \
    --plot
EXIT_2=$?

# ── 2. Side-by-side comparison (see online_RL_ucpg_v2/compare_policies.py) ──
echo "[eval_real] === Comparison ==="
python -m online_RL_ucpg_v2.compare_policies \
    --results naive_pg="$RESULTS_DIR/eval_real_naive_pg.json" ucpg_v2="$RESULTS_DIR/eval_real_ucpg_v2.json" \
    --id_ranges "$ID_RANGES" \
    --out_dir online_RL_ucpg_v2/runs/compare_gap
EXIT_3=$?

EXIT_CODE=0
[[ $EXIT_1 -ne 0 || $EXIT_2 -ne 0 || $EXIT_3 -ne 0 ]] && EXIT_CODE=1

echo "============================================================"
echo "Real-physics evaluation finished: $(date)"
echo "  naive_pg eval  : exit $EXIT_1"
echo "  ucpg_v2 eval   : exit $EXIT_2"
echo "  compare_policies: exit $EXIT_3"
echo "  Summary: online_RL_ucpg_v2/runs/compare_gap/compare_summary.json"
echo "  Plot:    online_RL_ucpg_v2/runs/compare_gap/compare_returns_actions.png"
echo "============================================================"
exit $EXIT_CODE
