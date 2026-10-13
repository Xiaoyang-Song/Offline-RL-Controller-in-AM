#!/bin/bash
# =============================================================================
# online_RL_ucpg_v3/jobs/eval_sweep.sh — surrogate + real-physics evaluation of
# the penalty sweep next to the existing plain PG / plain Q runs and the original
# c = 800 K shaped runs (same field), with the comparison figure,
# scored with the plain deviation of REWARD_FIELD.
# Env vars: PENALTIES, REWARD_FIELD. ~(4 + 2n) methods x ~20 min of MATLAB (~4 h for n = 4).
# =============================================================================
#SBATCH --job-name=ucpg_v3_sweep_eval
#SBATCH --account=sunwbgt0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/online_RL_ucpg_v3/jobs/eval_sweep_%j.log

F="${REWARD_FIELD:-heat}"
read -r -a PENS <<< "${PENALTIES:-5 10 20 40}"
R="online_RL_ucpg_v3/runs"
OUT="online_RL_ucpg_v3/results_sweep_narrow_$F"
SURR="surrogate_model_v3/runs/narrow_200_300W/surrogate_best.pt"
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
module load matlab; source ~/.bashrc; conda activate RL
echo "Started: $(date)   Node: $SLURMD_NODENAME   field: $F   penalties: ${PENS[*]}"

# comparison set: plain (naive) PG/Q of the same field, the sweep, and the original c = 800 K runs
METHODS=(--pg "PG plain=$R/narrow_pg_plain_$F/pg_best.pt")
for C in "${PENS[@]}"; do METHODS+=(--pg "UCPG v3 (c=$C K)=$R/narrow_pg_shaped_${F}_c$C/pg_best.pt"); done
METHODS+=(--pg "UCPG v3 (c=800 K)=$R/narrow_pg_shaped_$F/pg_best.pt")
METHODS+=(--q "Q plain=$R/narrow_q_plain_$F/q_best.pt")
for C in "${PENS[@]}"; do METHODS+=(--q "Q v3 (c=$C K)=$R/narrow_q_shaped_${F}_c$C/q_best.pt"); done
METHODS+=(--q "Q v3 (c=800 K)=$R/narrow_q_shaped_$F/q_best.pt")

echo "=== [1/2] surrogate evaluation ==="
python -m online_RL_ucpg_v3.evaluate_surrogate --surrogate "$SURR" --field "$F" \
    --ood_min 200 --ood_max 300 --n_episodes 100 "${METHODS[@]}" --out_dir "$OUT"
python -m online_RL_ucpg_v3.plot_comparison --results_dir "$OUT" --field "$F"
echo "=== [2/2] real-physics evaluation ==="
python -m online_RL_ucpg_v3.evaluate_real --surrogate "$SURR" --field "$F" \
    --cool_times 0.06 0.10 0.14 "${METHODS[@]}" --out_dir "$OUT"
python -m online_RL_ucpg_v3.plot_comparison --results_dir "$OUT" --field "$F"
echo "Finished: $(date)   Results: $OUT/"
