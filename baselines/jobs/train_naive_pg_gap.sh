#!/bin/bash
# =============================================================================
# train_naive_pg_gap.sh
# SLURM job script — train baselines/naive_pg (unconstrained continuous PG,
# no uncertainty term) against the PATCHY two-stage surrogate produced by
# jobs/train_surrogate_v2_gap_perturb.sh (trained on 100-150/200-250/
# 300-350W, gaps at 150-200/250-300/350-400W), allowed to roam the FULL
# 100-400W action range. This is the "before" half of the online_RL_ucpg_v2
# vs. naive_pg comparison — see online_RL_ucpg_v2/README.md's "Gapped
# variant" section for the full rationale, and jobs/train_ucpg_v2_gap.sh for
# the "after" half (train that job against the SAME surrogate checkpoint for
# a controlled comparison).
#
# Submit (defaults):
#   sbatch baselines/jobs/train_naive_pg_gap.sh
#
# Override any parameter at submission time, e.g.:
#   sbatch baselines/jobs/train_naive_pg_gap.sh --n_iterations 3000
# =============================================================================

#SBATCH --job-name=naive_pg_gap
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/baselines/jobs/naive_pg_gap_%j.log

# ── environment ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

source ~/.bashrc
conda activate RL

echo "Python   : $(which python)"
echo "Torch    : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA     : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

# ── Surrogate checkpoint — MUST match jobs/train_ucpg_v2_gap.sh's for a fair
# comparison. Fill in once jobs/train_surrogate_v2_gap_perturb.sh has run. ──
SURROGATE="surrogate_model_latent_uncertainty_v2/runs/patchy_100-150_200-250_300-350_perturb0.1/two_stage_best.pt"

# Fixed (non-timestamped) out_dir so downstream jobs (real-sim eval,
# compare_policies.py) can find this checkpoint without discovering a
# timestamp — see jobs/oneshot_gap_experiment.sh, which chains this.
OUT_DIR="baselines/naive_pg/runs/patchy_100-150_200-250_300-350_perturb0.1"

# ── Environment parameters (must match MATLAB paramsStruct in simulation_v2) ─
T_L=2000.0
T_H=2800.0
N_LAYERS=12
INITIAL_TEMP=300.0
MESH_PATH="surrogate_model/mesh.mat"
WIDTH=12.0
HEIGHT=3.0
SQ_FRAC_START=0.4
SQ_FRAC_END=0.5
COOL_TIME_MIN=0.05
COOL_TIME_MAX=0.15

# ── Policy allowed to roam the FULL range — includes the surrogate's three
# untrained gaps (150-200W, 250-300W, 350-400W) ─────────────────────────────
ACTION_MIN=100.0
ACTION_MAX=400.0

# ── OOD diagnostic — DISABLED (left empty) for this patchy-coverage scenario.
# baselines/naive_pg.train's --ood_min/--ood_max is a single contiguous
# range; with THREE separate gaps here it would be actively misleading. The
# authoritative gap-avoidance check is post-hoc and multi-range-aware:
# evaluate_real.py + compare_policies.py --id_ranges "100-150,200-250,300-350"
# against the real simulator.
OOD_MIN=
OOD_MAX=
OOD_ARGS=()
if [[ -n "$OOD_MIN" && -n "$OOD_MAX" ]]; then
    OOD_ARGS=(--ood_min "$OOD_MIN" --ood_max "$OOD_MAX")
fi

# ── Policy network architecture (matches train_online_rl_ucpg_v2.sh) ────────
HIDDEN=128
DEPTH=3
DROPOUT=0.0
LAYER_EMBED_DIM=8
LOG_SIGMA_INIT=-1.0
MIN_LOG_SIGMA=-3.0
MAX_LOG_SIGMA=0.0

# ── training hyperparameters ─────────────────────────────────────────────────
N_TRAJ=32
N_ITERATIONS=3000
GAMMA_R=0.99
LR_THETA=3e-4
MAX_GRAD_NORM=10.0
SEED=42

LOG_FREQ=20
SAVE_FREQ=200

python -m baselines.naive_pg.train \
    --surrogate       "$SURROGATE"      \
    --T_l             $T_L              \
    --T_h             $T_H              \
    --n_layers        $N_LAYERS         \
    --initial_temp    $INITIAL_TEMP     \
    --mesh_path       "$MESH_PATH"      \
    --width           $WIDTH            \
    --height          $HEIGHT           \
    --sq_frac_start   $SQ_FRAC_START    \
    --sq_frac_end     $SQ_FRAC_END      \
    --cool_time_min   $COOL_TIME_MIN    \
    --cool_time_max   $COOL_TIME_MAX    \
    --action_min      $ACTION_MIN       \
    --action_max      $ACTION_MAX       \
    "${OOD_ARGS[@]}"                    \
    --hidden          $HIDDEN           \
    --depth           $DEPTH            \
    --dropout         $DROPOUT          \
    --layer_embed_dim $LAYER_EMBED_DIM  \
    --log_sigma_init  $LOG_SIGMA_INIT   \
    --min_log_sigma   $MIN_LOG_SIGMA    \
    --max_log_sigma   $MAX_LOG_SIGMA    \
    --n_traj          $N_TRAJ           \
    --n_iterations    $N_ITERATIONS     \
    --gamma_r         $GAMMA_R          \
    --lr_theta        $LR_THETA         \
    --max_grad_norm   $MAX_GRAD_NORM    \
    --log_freq        $LOG_FREQ         \
    --save_freq       $SAVE_FREQ        \
    --seed            $SEED             \
    --out_dir         "$OUT_DIR"        \
    "$@"

EXIT_CODE=$?
echo "============================================================"
echo "Training finished: $(date)  (exit code: $EXIT_CODE)"
echo "Checkpoint: $OUT_DIR/naive_pg_best.pt"
echo "============================================================"
exit $EXIT_CODE
