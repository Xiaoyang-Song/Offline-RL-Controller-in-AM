#!/bin/bash
# =============================================================================
# train_online_rl_adapted.sh
# SLURM job — train online Double-DQN on a domain-adapted surrogate (ct=0.15).
#
# Run after adapt_domain_surrogate.sh has completed for all K values.
#
# Required args (pass after sbatch):
#   --adapt_dir       <path>   adapt_ct150/ directory from adapt_domain_surrogate.sh
#   --base_checkpoint <path>   base_best.pt from train_base_domain_surrogate.sh
#   --K               <int>    adaptation shots (e.g. 5, 10, 20, 50, 100)
#                              Use K=0 to train on the base (unadapted) surrogate.
#
# Submit example:
#   ADAPT_DIR=surrogate_domain_adaptation/runs/base/20260601_HHMMSS/adapt_ct150
#   BASE_CKPT=surrogate_domain_adaptation/runs/base/20260601_HHMMSS/base_best.pt
#   for K in 0 5 10 20 50 100; do
#       sbatch jobs/train_online_rl_adapted.sh \
#           --adapt_dir $ADAPT_DIR \
#           --base_checkpoint $BASE_CKPT \
#           --K $K
#   done
#
# Outputs (per K):
#   <adapt_dir>/K<K>/online_rl/dqn_best.pt
#   <adapt_dir>/K<K>/online_rl/dqn_final.pt
#   <adapt_dir>/K<K>/online_rl/{returns,q_loss,epsilon,epistemic_std}.png
#
# After all K jobs finish, evaluate in MATLAB (on a login node or compute node
# with MATLAB available) and aggregate plots:
#
#   for K in 0 5 10 20 50 100; do
#     KDIR=$(printf '%03d' $K)
#     python test.py --mode OnlineRL \
#         --checkpoint  ${ADAPT_DIR}/K${KDIR}/online_rl/dqn_best.pt \
#         --surrogate   ${BASE_CKPT} \
#         --cool_time   0.15 \
#         --results_out ${ADAPT_DIR}/K${KDIR}/online_rl/matlab_eval.json
#   done
#
#   python -m surrogate_domain_adaptation.plot_rl_sweep \
#       --adapt_dir $ADAPT_DIR
# =============================================================================

#SBATCH --job-name=rl_adapted
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/rl_adapted_%j.log

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
echo "Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

# ── parse required args ───────────────────────────────────────────────────────
ADAPT_DIR=""
BASE_CHECKPOINT=""
K_VALUE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapt_dir)       ADAPT_DIR="$2";       shift 2 ;;
        --base_checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
        --K)               K_VALUE="$2";         shift 2 ;;
        *)                 EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [[ -z "$ADAPT_DIR" || -z "$BASE_CHECKPOINT" || -z "$K_VALUE" ]]; then
    echo "ERROR: --adapt_dir, --base_checkpoint, and --K are all required."
    echo "Usage: sbatch jobs/train_online_rl_adapted.sh \\"
    echo "         --adapt_dir <path> --base_checkpoint <path> --K <int>"
    exit 1
fi

KDIR=$(printf '%03d' "$K_VALUE")
OUT_DIR="${ADAPT_DIR}/K${KDIR}/online_rl"
ADAPTED_CKPT="${ADAPT_DIR}/K${KDIR}/adapted.pt"

echo "adapt_dir         : $ADAPT_DIR"
echo "base_checkpoint   : $BASE_CHECKPOINT"
echo "K                 : $K_VALUE  (dir suffix: K${KDIR})"
echo "adapted_checkpoint: $ADAPTED_CKPT"
echo "out_dir           : $OUT_DIR"
echo "============================================================"

# ── RL hyperparameters ────────────────────────────────────────────────────────
N_EPISODES=30000
GAMMA=0.99
LR=3e-4
BATCH_SIZE=256
BUFFER_CAPACITY=100000
WARMUP_EPISODES=50
TARGET_UPDATE_FREQ=10
EPSILON_START=1.0
EPSILON_END=0.05
EPSILON_DECAY_STEPS=50000

# Q-network architecture
HIDDEN=256
DEPTH=4
LAYER_EMBED_DIM=8

# Uncertainty penalty (standard mode = no penalty)
UNCERTAINTY_MODE=standard
UNCERTAINTY_PENALTY_WEIGHT=0.0

LOG_FREQ=100
SAVE_FREQ=1000
SEED=42

# ── Step 1: train online RL ───────────────────────────────────────────────────
if [[ "$K_VALUE" -eq 0 ]]; then
    # K=0: use the base (unadapted) surrogate directly — same as standard online RL
    echo "[job] K=0: training on base (unadapted) surrogate"
    python -m online_RL.train \
        --surrogate          "$BASE_CHECKPOINT" \
        --uncertainty_mode   $UNCERTAINTY_MODE \
        --n_episodes         $N_EPISODES \
        --gamma              $GAMMA \
        --lr                 $LR \
        --batch_size         $BATCH_SIZE \
        --buffer_capacity    $BUFFER_CAPACITY \
        --warmup_episodes    $WARMUP_EPISODES \
        --target_update_freq $TARGET_UPDATE_FREQ \
        --epsilon_start      $EPSILON_START \
        --epsilon_end        $EPSILON_END \
        --epsilon_decay_steps $EPSILON_DECAY_STEPS \
        --hidden             $HIDDEN \
        --depth              $DEPTH \
        --layer_embed_dim    $LAYER_EMBED_DIM \
        --log_freq           $LOG_FREQ \
        --save_freq          $SAVE_FREQ \
        --out_dir            "$OUT_DIR" \
        --seed               $SEED \
        $EXTRA_ARGS
else
    # K>0: check adapted checkpoint exists
    if [[ ! -f "$ADAPTED_CKPT" ]]; then
        echo "ERROR: adapted checkpoint not found: $ADAPTED_CKPT"
        echo "Run adapt_domain_surrogate.sh first."
        exit 1
    fi
    echo "[job] K=${K_VALUE}: training on domain-adapted surrogate"
    python -m online_RL.train \
        --adapted_checkpoint "$ADAPTED_CKPT" \
        --base_checkpoint    "$BASE_CHECKPOINT" \
        --uncertainty_mode   $UNCERTAINTY_MODE \
        --n_episodes         $N_EPISODES \
        --gamma              $GAMMA \
        --lr                 $LR \
        --batch_size         $BATCH_SIZE \
        --buffer_capacity    $BUFFER_CAPACITY \
        --warmup_episodes    $WARMUP_EPISODES \
        --target_update_freq $TARGET_UPDATE_FREQ \
        --epsilon_start      $EPSILON_START \
        --epsilon_end        $EPSILON_END \
        --epsilon_decay_steps $EPSILON_DECAY_STEPS \
        --hidden             $HIDDEN \
        --depth              $DEPTH \
        --layer_embed_dim    $LAYER_EMBED_DIM \
        --log_freq           $LOG_FREQ \
        --save_freq          $SAVE_FREQ \
        --out_dir            "$OUT_DIR" \
        --seed               $SEED \
        $EXTRA_ARGS
fi

EXIT_CODE=$?
echo "============================================================"
echo "Training finished: $(date)  (exit code: $EXIT_CODE)"
echo "Checkpoint: ${OUT_DIR}/dqn_best.pt"
echo "============================================================"
exit $EXIT_CODE
