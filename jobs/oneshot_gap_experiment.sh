#!/bin/bash
# =============================================================================
# oneshot_gap_experiment.sh
# ONE SLURM job that runs the FULL patchy-coverage gap experiment
# sequentially, start to finish, in a single allocation:
#   1. Train the two-stage surrogate on patchy laser-power coverage
#      (100-150/200-250/300-350W, gaps at 150-200/250-300/350-400W) with
#      target noise + a decorrelated bootstrap ensemble.
#   2. Evaluate that surrogate (standard metrics + ID/OOD gap split).
#   3. Train online_RL_ucpg_v2 (uncertainty-constrained) against it.
#   4. Train baselines/naive_pg (no uncertainty term) against it.
#   5. Evaluate BOTH trained policies against the REAL PDE simulator and
#      compare them — does the uncertainty-aware policy actually avoid the
#      three gaps and get a better REAL reward than the naive policy?
#
# See surrogate_model_latent_uncertainty_v2/README.md's "Harder / gapped-
# surrogate experiments" section and online_RL_ucpg_v2/README.md's "Gapped
# variant" section for the full rationale of every knob below. This script
# inlines the exact same parameters as the standalone per-stage scripts
# (jobs/train_surrogate_v2_gap_perturb.sh, jobs/evaluate_surrogate_v2_gap.sh,
# jobs/train_ucpg_v2_gap.sh, baselines/jobs/train_naive_pg_gap.sh,
# jobs/evaluate_real_gap_experiment.sh) — those still exist separately if you
# ever want to rerun just one stage; this file is the "run everything, wait
# once" version.
#
# Each stage's exit code is checked: if surrogate training (1), UCPG
# training (3), or naive_pg training (4) fails, the script stops immediately
# rather than continuing with a checkpoint that doesn't exist. Surrogate
# evaluation (2) is a diagnostic only — its failure is logged but does not
# stop the pipeline.
#
# Runs training on GPU and the real-simulator step (MATLAB subprocess calls,
# CPU-only) in the SAME allocation for simplicity — the GPU sits idle during
# stage 5 (~1-2h out of the total). Estimated total wall time: 6h (stage 1)
# + 1h (stage 2) + 2h (stage 3) + 2h (stage 4) + ~1.5h (stage 5) =~ 12.5h;
# --time below has buffer. Check your cluster's max walltime for the `gpu`
# partition and adjust if it's below that.
#
# Requires Data/DatasetV2_layer_12_samples_5000.pkl to already exist —
# extract first via surrogate_model_latent_uncertainty_v2/jobs/dataset.sh if
# it doesn't (that step is NOT included here since it's a one-time, reusable
# prerequisite, not specific to this experiment).
#
# Submit (defaults):
#   sbatch jobs/oneshot_gap_experiment.sh
# =============================================================================

#SBATCH --job-name=oneshot_gap
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=16:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/oneshot_gap_%j.log

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
echo "Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "MATLAB : $(which matlab)"
echo "============================================================"

# ── shared experiment identity ──────────────────────────────────────────────
DATA_PATH="Data/DatasetV2_layer_12_samples_5000.pkl"
LP_FILTER_RANGES="100-150,200-250,300-350"   # ID ranges — see script header
ID_RANGES="$LP_FILTER_RANGES"                 # same string, used by eval/compare too

SURROGATE_OUT_DIR="surrogate_model_latent_uncertainty_v2/runs/patchy_100-150_200-250_300-350_perturb0.1"
SURROGATE_CKPT="$SURROGATE_OUT_DIR/two_stage_best.pt"
UCPG_OUT_DIR="online_RL_ucpg_v2/runs/patchy_100-150_200-250_300-350_perturb0.1"
NAIVE_PG_OUT_DIR="baselines/naive_pg/runs/patchy_100-150_200-250_300-350_perturb0.1"
RESULTS_DIR="jobs/results_gap_experiment"
mkdir -p "$RESULTS_DIR"


# =============================================================================
# 1. Train the patchy-coverage surrogate
# =============================================================================
echo ""
echo "############################################################"
echo "# STAGE 1/5: Train surrogate  ($(date))"
echo "############################################################"

PERTURB_FRAC=0.1     # fraction of each node's own state_std — see dataset_v2.py
PERTURB_SEED=0
BOOTSTRAP_FRAC=0.5    # each of K=5 members bootstraps only this fraction of N
N_ENSEMBLE=5
EPOCHS=500
BATCH_SIZE=128
LR=1e-3
PATIENCE=30

python -m surrogate_model_latent_uncertainty_v2.train \
    --data_path        "$DATA_PATH"         \
    --lp_filter_ranges "$LP_FILTER_RANGES"  \
    --perturb_frac     $PERTURB_FRAC        \
    --perturb_seed     $PERTURB_SEED        \
    --bootstrap_frac   $BOOTSTRAP_FRAC      \
    --n_ensemble       $N_ENSEMBLE          \
    --epochs           $EPOCHS              \
    --batch_size       $BATCH_SIZE          \
    --lr               $LR                  \
    --patience         $PATIENCE            \
    --out_dir          "$SURROGATE_OUT_DIR"
STAGE1_EXIT=$?

if [[ $STAGE1_EXIT -ne 0 ]]; then
    echo "[oneshot] STAGE 1 FAILED (exit $STAGE1_EXIT) — aborting, nothing downstream can run."
    exit $STAGE1_EXIT
fi
echo "[oneshot] Stage 1 done. Checkpoint: $SURROGATE_CKPT"


# =============================================================================
# 2. Evaluate the surrogate (diagnostic only — failure does not stop the pipeline)
# =============================================================================
echo ""
echo "############################################################"
echo "# STAGE 2/5: Evaluate surrogate  ($(date))"
echo "############################################################"

python -m surrogate_model_latent_uncertainty_v2.evaluate \
    --checkpoint "$SURROGATE_CKPT" \
    --data_path  "$DATA_PATH"
STAGE2A_EXIT=$?

python -m surrogate_model_latent_uncertainty_v2.evaluate_ood \
    --checkpoint "$SURROGATE_CKPT" \
    --data_path  "$DATA_PATH" \
    --id_ranges  "$ID_RANGES"
STAGE2B_EXIT=$?

if [[ $STAGE2A_EXIT -ne 0 || $STAGE2B_EXIT -ne 0 ]]; then
    echo "[oneshot] STAGE 2 had a failure (evaluate.py=$STAGE2A_EXIT, evaluate_ood.py=$STAGE2B_EXIT)"
    echo "[oneshot] — continuing anyway, this stage is diagnostic-only."
else
    echo "[oneshot] Stage 2 done."
fi


# =============================================================================
# 3. Train online_RL_ucpg_v2 (uncertainty-constrained) against the surrogate
# =============================================================================
echo ""
echo "############################################################"
echo "# STAGE 3/5: Train UCPG v2  ($(date))"
echo "############################################################"

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
ACTION_MIN=100.0
ACTION_MAX=400.0
# --ood_min/--ood_max intentionally omitted: a single contiguous range can't
# represent three separate gaps without being misleading — see README.
HIDDEN=128
DEPTH=3
DROPOUT=0.0
LAYER_EMBED_DIM=8
LOG_SIGMA_INIT=-1.0
MIN_LOG_SIGMA=-3.0
MAX_LOG_SIGMA=0.0
N_TRAJ=32
N_ITERATIONS=3000
GAMMA_R=0.99
GAMMA_U=0.99
DELTA=0.05            # uncertainty budget — TUNE to this surrogate's uncertainty
                      # scale (see online_RL_ucpg_v2/README.md: warm-start with
                      # --lambda_init 0 and --n_iterations 20, read off J_u)
LAMBDA_INIT=0.0
LR_THETA=3e-4
LR_LAMBDA=1e-2
MAX_GRAD_NORM=10.0
SEED=42
LOG_FREQ=20
SAVE_FREQ=200

python -m online_RL_ucpg_v2.train \
    --surrogate       "$SURROGATE_CKPT" \
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
    --out_dir         "$UCPG_OUT_DIR"
STAGE3_EXIT=$?

if [[ $STAGE3_EXIT -ne 0 ]]; then
    echo "[oneshot] STAGE 3 (UCPG v2 training) FAILED (exit $STAGE3_EXIT) — aborting."
    exit $STAGE3_EXIT
fi
echo "[oneshot] Stage 3 done. Checkpoint: $UCPG_OUT_DIR/ucpg_best.pt"


# =============================================================================
# 4. Train baselines/naive_pg (no uncertainty term) against the SAME surrogate
# =============================================================================
echo ""
echo "############################################################"
echo "# STAGE 4/5: Train naive_pg  ($(date))"
echo "############################################################"

# Same env/architecture/training hyperparameters as stage 3 (fair comparison);
# naive_pg has no --gamma_u/--delta/--lambda_init/--lr_lambda (no uncertainty term).
python -m baselines.naive_pg.train \
    --surrogate       "$SURROGATE_CKPT" \
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
    --lr_theta        $LR_THETA         \
    --max_grad_norm   $MAX_GRAD_NORM    \
    --log_freq        $LOG_FREQ         \
    --save_freq       $SAVE_FREQ        \
    --seed            $SEED             \
    --out_dir         "$NAIVE_PG_OUT_DIR"
STAGE4_EXIT=$?

if [[ $STAGE4_EXIT -ne 0 ]]; then
    echo "[oneshot] STAGE 4 (naive_pg training) FAILED (exit $STAGE4_EXIT) — aborting."
    exit $STAGE4_EXIT
fi
echo "[oneshot] Stage 4 done. Checkpoint: $NAIVE_PG_OUT_DIR/naive_pg_best.pt"


# =============================================================================
# 5. Ground-truth check: BOTH policies against the REAL PDE simulator, then compare
# =============================================================================
echo ""
echo "############################################################"
echo "# STAGE 5/5: Real-simulator evaluation + comparison  ($(date))"
echo "############################################################"

COOL_TIME=0.10       # fixed for BOTH policies — keeps the comparison fair
N_EPISODES=5         # real PDE solves are slow; increase only if you need tighter variance estimates

echo "[oneshot] === Naive PG vs. real physics ==="
python -m online_RL_ucpg_v2.evaluate_real \
    --checkpoint "$NAIVE_PG_OUT_DIR/naive_pg_best.pt" \
    --surrogate  "$SURROGATE_CKPT" \
    --cool_time  $COOL_TIME \
    --n_episodes $N_EPISODES \
    --results_out "$RESULTS_DIR/eval_real_naive_pg.json" \
    --plot
STAGE5A_EXIT=$?

echo "[oneshot] === UCPG v2 vs. real physics ==="
python -m online_RL_ucpg_v2.evaluate_real \
    --checkpoint "$UCPG_OUT_DIR/ucpg_best.pt" \
    --surrogate  "$SURROGATE_CKPT" \
    --cool_time  $COOL_TIME \
    --n_episodes $N_EPISODES \
    --results_out "$RESULTS_DIR/eval_real_ucpg_v2.json" \
    --plot
STAGE5B_EXIT=$?

echo "[oneshot] === Comparison ==="
python -m online_RL_ucpg_v2.compare_policies \
    --results naive_pg="$RESULTS_DIR/eval_real_naive_pg.json" ucpg_v2="$RESULTS_DIR/eval_real_ucpg_v2.json" \
    --id_ranges "$ID_RANGES" \
    --out_dir online_RL_ucpg_v2/runs/compare_gap
STAGE5C_EXIT=$?

FINAL_EXIT=0
[[ $STAGE5A_EXIT -ne 0 || $STAGE5B_EXIT -ne 0 || $STAGE5C_EXIT -ne 0 ]] && FINAL_EXIT=1

echo ""
echo "============================================================"
echo "[oneshot] Full pipeline finished: $(date)"
echo "  1. surrogate train    : exit $STAGE1_EXIT"
echo "  2. surrogate eval     : exit evaluate=$STAGE2A_EXIT evaluate_ood=$STAGE2B_EXIT (diagnostic only)"
echo "  3. UCPG v2 train      : exit $STAGE3_EXIT"
echo "  4. naive_pg train     : exit $STAGE4_EXIT"
echo "  5. real eval+compare  : exit naive_pg=$STAGE5A_EXIT ucpg_v2=$STAGE5B_EXIT compare=$STAGE5C_EXIT"
echo ""
echo "  Summary: online_RL_ucpg_v2/runs/compare_gap/compare_summary.json"
echo "  Plot:    online_RL_ucpg_v2/runs/compare_gap/compare_returns_actions.png"
echo "============================================================"
exit $FINAL_EXIT
