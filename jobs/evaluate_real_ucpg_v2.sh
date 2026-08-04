#!/bin/bash
# =============================================================================
# evaluate_real_ucpg_v2.sh
# SLURM job script — evaluate BOTH a naive_pg checkpoint and a real UCPG v2
# checkpoint against the REAL PDE simulator (simulateHeatingCooling_v2.m),
# not the neural surrogate. See online_RL_ucpg_v2/evaluate_real.py's
# docstring for the mechanism (per-layer `matlab -batch` subprocess calls).
#
# This does real PDE solves (~30-45s/layer observed, ~6-9 min per 12-layer
# episode) — deliberately NOT run on the login node (confirmed the login
# node's Arbiter daemon kills sustained CPU-heavy work; a single-layer
# smoke test at ~40s was fine, a full episode would not be).
#
# Submit (defaults):
#   sbatch jobs/evaluate_real_ucpg_v2.sh
# =============================================================================

#SBATCH --job-name=eval_real_ucpg_v2
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/eval_real_ucpg_v2_%j.log

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

SURROGATE="surrogate_model_latent_uncertainty_v2/runs/20260803_184916/two_stage_best.pt"

# ── update these to the checkpoints you actually want compared ─────────────
NAIVE_PG_CHECKPOINT="baselines/naive_pg/runs/<ts>/naive_pg_best.pt"
UCPG_V2_CHECKPOINT="online_RL_ucpg_v2/runs/20260803_210317/ucpg_best.pt"

COOL_TIME=0.10       # fixed for BOTH runs — keeps the comparison fair
N_EPISODES=1         # real PDE solves are slow; increase only if you need variance estimates

# ── 0. Naive PG vs. real physics ─────────────────────────────────────────────
echo "[eval_real] === Naive PG ==="
python -m online_RL_ucpg_v2.evaluate_real \
    --checkpoint "$NAIVE_PG_CHECKPOINT" \
    --surrogate  "$SURROGATE" \
    --cool_time  $COOL_TIME \
    --n_episodes $N_EPISODES \
    --results_out jobs/eval_real_naive_pg.json \
    --plot

# ── 1. UCPG v2 vs. real physics ──────────────────────────────────────────────
echo "[eval_real] === UCPG v2 ==="
python -m online_RL_ucpg_v2.evaluate_real \
    --checkpoint "$UCPG_V2_CHECKPOINT" \
    --surrogate  "$SURROGATE" \
    --cool_time  $COOL_TIME \
    --n_episodes $N_EPISODES \
    --results_out jobs/eval_real_ucpg_v2.json \
    --plot

EXIT_CODE=$?
echo "============================================================"
echo "Real-physics evaluation finished: $(date)  (exit code: $EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
