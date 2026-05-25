#!/bin/bash
# =============================================================================
# train_online_rl.sh
# SLURM job script — online Double-DQN training for the LPBF controller
#
# The agent interacts with the pre-trained surrogate model (no MATLAB calls)
# and learns a laser-power policy that minimises temperature deviation from
# the nominal window [T_l, T_h] across 12 build layers.
#
# Submit (defaults):
#   sbatch jobs/train_online_rl.sh
#
# Override any parameter at submission time, e.g.:
#   sbatch jobs/train_online_rl.sh --n_episodes 5000 --hidden 512 --lr 1e-4
# =============================================================================

#SBATCH --job-name=online_rl_train
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/online_rl_train_%j.log

# ── environment ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================================"

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM

# Activate conda environment (compute nodes don't inherit the login shell)
source ~/.bashrc
conda activate RL

echo "Python   : $(which python)"
echo "Torch    : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA     : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU      : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "============================================================"

# ── Surrogate checkpoint ─────────────────────────────────────────────────────
SURROGATE="surrogate_model/runs/20260521_210923/surrogate_best.pt"

# ── Environment parameters (must match MATLAB paramsStruct in test.py) ───────
T_L=2000.0          # lower nominal temperature bound [K]  (tempRange(1))
T_H=2800.0          # upper nominal temperature bound [K]  (tempRange(2))
N_LAYERS=12         # layers per episode
INITIAL_TEMP=300.0  # initial uniform temperature [K]
MESH_PATH="surrogate_model/mesh.mat"   # optional; falls back to all-nodes
WIDTH=12.0          # domain width  (paramsStruct.width)
HEIGHT=3.0          # domain height (paramsStruct.height)
SQ_FRAC_START=0.4   # squareSideFraction at layer 1  (test.py initialFraction)
SQ_FRAC_END=0.5     # squareSideFraction at last layer (test.py finalFraction)

# ── Q-network architecture ────────────────────────────────────────────────────
HIDDEN=256          # hidden layer width
DEPTH=4             # number of residual blocks
DROPOUT=0.0         # dropout inside blocks

# ── RL hyperparameters ────────────────────────────────────────────────────────
N_EPISODES=20000
GAMMA=0.99
LR=3e-4
BATCH_SIZE=256
BUFFER_CAPACITY=100000
WARMUP_EPISODES=500            # pure random exploration before learning
TARGET_UPDATE_FREQ=10         # episodes between hard target-net copies
UPDATES_PER_STEP=1            # gradient updates per env step

EPSILON_START=1.0
EPSILON_END=0.05
EPSILON_DECAY_STEPS=50000     # ~17 episodes × 12 steps = ~204 steps/episode
                              # 50000 / 204 ≈ 245 episodes of decay

SEED=42

# ── logging / checkpointing ───────────────────────────────────────────────────
LOG_FREQ=50         # print every N episodes
SAVE_FREQ=500       # save checkpoint + plots every N episodes

# ── run training ─────────────────────────────────────────────────────────────
python -m online_RL.train \
    --surrogate           "$SURROGATE"     \
    --T_l                 $T_L             \
    --T_h                 $T_H             \
    --n_layers            $N_LAYERS        \
    --initial_temp        $INITIAL_TEMP    \
    --mesh_path           "$MESH_PATH"     \
    --width               $WIDTH           \
    --height              $HEIGHT          \
    --sq_frac_start       $SQ_FRAC_START   \
    --sq_frac_end         $SQ_FRAC_END     \
    --hidden              $HIDDEN          \
    --depth               $DEPTH           \
    --dropout             $DROPOUT         \
    --n_episodes          $N_EPISODES      \
    --gamma               $GAMMA           \
    --lr                  $LR              \
    --batch_size          $BATCH_SIZE      \
    --buffer_capacity     $BUFFER_CAPACITY \
    --warmup_episodes     $WARMUP_EPISODES \
    --target_update_freq  $TARGET_UPDATE_FREQ \
    --updates_per_step    $UPDATES_PER_STEP   \
    --epsilon_start       $EPSILON_START   \
    --epsilon_end         $EPSILON_END     \
    --epsilon_decay_steps $EPSILON_DECAY_STEPS \
    --log_freq            $LOG_FREQ        \
    --save_freq           $SAVE_FREQ       \
    --seed                $SEED            \
    "$@"      # forward any extra args from: sbatch train_online_rl.sh --arg val

EXIT_CODE=$?
echo "============================================================"
echo "Training finished: $(date)  (exit code: $EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
