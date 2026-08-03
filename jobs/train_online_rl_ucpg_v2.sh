#!/bin/bash
# =============================================================================
# train_online_rl_ucpg_v2.sh
# SLURM job script — Monte Carlo Uncertainty-Constrained Policy Gradient v2
# (continuous laser-power action) training for the LPBF laser-power controller.
#
# The agent interacts with the two-stage (heating/cooling) Gaussian bootstrap
# ensemble latent surrogate (surrogate_model_latent_uncertainty_v2) and learns
# a latent-space Gaussian policy that maximises reward (scored at end-of-
# heating) subject to a budget on expected cumulative epistemic + aleatoric
# uncertainty (combined across both stages). See online_RL_ucpg_v2/README.md
# for the full method.
#
# Submit (defaults):
#   sbatch jobs/train_online_rl_ucpg_v2.sh
#
# Override any parameter at submission time, e.g.:
#   sbatch jobs/train_online_rl_ucpg_v2.sh --delta 0.03 --n_iterations 5000
# =============================================================================

#SBATCH --job-name=online_rl_ucpg_v2
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/online_rl_ucpg_v2_%j.log

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

# ── Surrogate checkpoint (two-stage heating/cooling Gaussian ensemble) ───────
SURROGATE="surrogate_model_latent_uncertainty_v2/runs/<ts>/two_stage_best.pt"

# ── Environment parameters (must match MATLAB paramsStruct in simulation_v2) ─
T_L=2000.0          # lower nominal temperature bound [K] (end-of-heating target)
T_H=2800.0          # upper nominal temperature bound [K]
N_LAYERS=12         # layers per episode
INITIAL_TEMP=300.0  # initial uniform temperature [K]
MESH_PATH="surrogate_model/mesh.mat"
WIDTH=12.0
HEIGHT=3.0
SQ_FRAC_START=0.4
SQ_FRAC_END=0.5
COOL_TIME_MIN=0.05   # cool_time is a per-EPISODE nuisance variable, not an action
COOL_TIME_MAX=0.15

# ── Policy action "aim range" (continuous — NOT a hard bound, see README) ───
ACTION_MIN=100.0
ACTION_MAX=400.0

# ── Policy network architecture ───────────────────────────────────────────────
HIDDEN=128
DEPTH=3
DROPOUT=0.0
LAYER_EMBED_DIM=8
LOG_SIGMA_INIT=-1.0
MIN_LOG_SIGMA=-3.0
MAX_LOG_SIGMA=0.0

# ── UCPG hyperparameters ──────────────────────────────────────────────────────
N_TRAJ=32            # N — trajectories collected per iteration
N_ITERATIONS=3000    # K — outer iterations
GAMMA_R=0.99
GAMMA_U=0.99
DELTA=0.05           # uncertainty budget — TUNE to your surrogate's uncertainty scale
LAMBDA_INIT=0.0
LR_THETA=3e-4
LR_LAMBDA=1e-2
MAX_GRAD_NORM=10.0

SEED=42

# ── logging / checkpointing ───────────────────────────────────────────────────
LOG_FREQ=20
SAVE_FREQ=200

# ── run training ─────────────────────────────────────────────────────────────
python -m online_RL_ucpg_v2.train \
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
    --gamma_u         $GAMMA_U          \
    --delta           $DELTA            \
    --lambda_init     $LAMBDA_INIT      \
    --lr_theta        $LR_THETA         \
    --lr_lambda       $LR_LAMBDA        \
    --max_grad_norm   $MAX_GRAD_NORM    \
    --log_freq        $LOG_FREQ         \
    --save_freq       $SAVE_FREQ        \
    --seed            $SEED             \
    "$@"

EXIT_CODE=$?
echo "============================================================"
echo "Training finished: $(date)  (exit code: $EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
