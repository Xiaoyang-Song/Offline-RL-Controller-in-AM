#!/bin/bash
# =============================================================================
# train_base_domain_surrogate.sh
# SLURM job — train the BASE surrogate on pooled source cooling-time domains
# (ct_050, ct_060, ct_070, ct_080, ct_090, ct_100).
#
# Submit:
#   sbatch jobs/train_base_domain_surrogate.sh
#
# After this completes, note the checkpoint path printed in the log, then run:
#   sbatch jobs/adapt_domain_surrogate.sh --base_checkpoint <path/base_best.pt>
# =============================================================================

#SBATCH --job-name=base_domain_surr
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32GB
#SBATCH --time=4:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/base_domain_surr_%j.log

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

# ── source cooling times ──────────────────────────────────────────────────────
CT_LIST="50 60 70 80 90 100"   # → 0.05 to 0.10 s

# ── model architecture (matches surrogate_model_latent defaults) ──────────────
LATENT_DIM=64
N_ENSEMBLE=5
N_LAYERS=12
LAYER_EMBED_DIM=8
ENC_HIDDEN=256
ENC_DEPTH=3
TRANS_HIDDEN=128
TRANS_DEPTH=3
DEC_HIDDEN=256
DEC_DEPTH=3
DROPOUT=0.0

# ── loss weights ──────────────────────────────────────────────────────────────
RECON_ST_WEIGHT=1.0
RECON_ST1_WEIGHT=1.0
ROLLOUT_STEPS=12          # full-trajectory rollout regularisation
ROLLOUT_WEIGHT=0.1

# ── physics ROI ───────────────────────────────────────────────────────────────
ROI_BOOST=5.0
ROI_INITIAL_FRACTION=0.4
ROI_FINAL_FRACTION=0.5
ROI_EDGE_SIGMA_FRAC=0.05

# ── training ──────────────────────────────────────────────────────────────────
EPOCHS=400
BATCH_SIZE=256
LR=1e-3
WEIGHT_DECAY=1e-5
PATIENCE=40
NUM_WORKERS=6

VAL_FRACTION=0.10
TEST_FRACTION=0.10
SEED=42

# ── run ───────────────────────────────────────────────────────────────────────
python -m surrogate_domain_adaptation.train_base \
    --ct_list            $CT_LIST \
    --latent_dim         $LATENT_DIM \
    --n_ensemble         $N_ENSEMBLE \
    --n_layers           $N_LAYERS \
    --layer_embed_dim    $LAYER_EMBED_DIM \
    --enc_hidden         $ENC_HIDDEN \
    --enc_depth          $ENC_DEPTH \
    --trans_hidden       $TRANS_HIDDEN \
    --trans_depth        $TRANS_DEPTH \
    --dec_hidden         $DEC_HIDDEN \
    --dec_depth          $DEC_DEPTH \
    --dropout            $DROPOUT \
    --recon_st_weight    $RECON_ST_WEIGHT \
    --recon_st1_weight   $RECON_ST1_WEIGHT \
    --rollout_steps      $ROLLOUT_STEPS \
    --rollout_weight     $ROLLOUT_WEIGHT \
    --roi_boost              $ROI_BOOST \
    --roi_initial_fraction   $ROI_INITIAL_FRACTION \
    --roi_final_fraction     $ROI_FINAL_FRACTION \
    --roi_edge_sigma_frac    $ROI_EDGE_SIGMA_FRAC \
    --epochs             $EPOCHS \
    --batch_size         $BATCH_SIZE \
    --lr                 $LR \
    --weight_decay       $WEIGHT_DECAY \
    --patience           $PATIENCE \
    --num_workers        $NUM_WORKERS \
    --val_fraction       $VAL_FRACTION \
    --test_fraction      $TEST_FRACTION \
    --seed               $SEED \
    "$@"

EXIT_CODE=$?
echo "============================================================"
echo "Training finished: $(date)  (exit code: $EXIT_CODE)"
echo "============================================================"
exit $EXIT_CODE
