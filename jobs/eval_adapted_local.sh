#!/bin/bash
# Run test.py evaluation for all K values, then generate plots.
# Usage: bash jobs/eval_adapted_local.sh

cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM


ADAPT=surrogate_domain_adaptation/runs/base/20260601_174626/adapt_ct150
BASE=surrogate_domain_adaptation/runs/base/20260601_174626/base_best.pt

for K in 000 005 010 020 050 100; do
    echo "=== K=${K} ==="
    python test.py --mode OnlineRL \
        --checkpoint  $ADAPT/K${K}/online_rl/dqn_best.pt \
        --surrogate   $BASE \
        --cool_time   0.15 \
        --results_out $ADAPT/K${K}/online_rl/matlab_eval.json
done

echo "=== Generating plots ==="
python -m surrogate_domain_adaptation.plot_rl_sweep --adapt_dir $ADAPT


ADAPT=surrogate_domain_adaptation/runs/base/20260601_174626/adapt_ct150
BASE=surrogate_domain_adaptation/runs/base/20260601_174626/base_best.pt

# K=0  (base, no adaptation)
python test.py --mode OnlineRL \
    --checkpoint  $ADAPT/K000/online_rl/dqn_best.pt \
    --surrogate   $BASE --cool_time 0.15 \
    --results_out $ADAPT/K000/online_rl/matlab_eval.json

# K=5
python test.py --mode OnlineRL \
    --checkpoint  $ADAPT/K005/online_rl/dqn_best.pt \
    --surrogate   $BASE --cool_time 0.15 \
    --results_out $ADAPT/K005/online_rl/matlab_eval.json

# K=10
python test.py --mode OnlineRL \
    --checkpoint  $ADAPT/K010/online_rl/dqn_best.pt \
    --surrogate   $BASE --cool_time 0.15 \
    --results_out $ADAPT/K010/online_rl/matlab_eval.json

# K=20
python test.py --mode OnlineRL \
    --checkpoint  $ADAPT/K020/online_rl/dqn_best.pt \
    --surrogate   $BASE --cool_time 0.15 \
    --results_out $ADAPT/K020/online_rl/matlab_eval.json

# K=50
python test.py --mode OnlineRL \
    --checkpoint  $ADAPT/K050/online_rl/dqn_best.pt \
    --surrogate   $BASE --cool_time 0.15 \
    --results_out $ADAPT/K050/online_rl/matlab_eval.json

# K=100
python test.py --mode OnlineRL \
    --checkpoint  $ADAPT/K100/online_rl/dqn_best.pt \
    --surrogate   $BASE --cool_time 0.15 \
    --results_out $ADAPT/K100/online_rl/matlab_eval.json
