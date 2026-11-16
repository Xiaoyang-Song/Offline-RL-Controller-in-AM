#!/bin/bash
# =============================================================================
# rl_narrow/jobs/run_all.sh — ONE command for the whole narrow RL experiment (run on the login node;
# it only submits jobs):
#     bash rl_narrow/jobs/run_all.sh            # train everything, then evaluate in real physics
#     bash rl_narrow/jobs/run_all.sh --dry-run  # print the sbatch commands only
# 1. train_online.sh  (6 GPU tasks)   2. train_offline.sh (10 GPU tasks)
# 3. eval_real.sh     (48 CPU tasks, after 1 and 2 succeed)
# 4. finalize.sh      (after 3, whatever its exit status): leaderboard + both figures (+ plotted data)
#                     -> rl_narrow/results_real/{leaderboard_real.csv, figures/}
# =============================================================================
cd /nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM
mkdir -p rl_narrow/jobs/logs rl_narrow/runs rl_narrow/results_real
SB="sbatch --parsable"
if [ "$1" = "--dry-run" ]; then
    echo "$SB rl_narrow/jobs/train_online.sh"
    echo "$SB rl_narrow/jobs/train_offline.sh"
    echo "$SB --dependency=afterok:<ONLINE>:<OFFLINE> rl_narrow/jobs/eval_real.sh"
    echo "$SB --dependency=afterany:<EVAL> rl_narrow/jobs/finalize.sh"
    exit 0
fi
ON=$($SB --export=ALL,TAU_K="${TAU_K:-8}" rl_narrow/jobs/train_online.sh)
OFF=$($SB rl_narrow/jobs/train_offline.sh)
EV=$($SB --dependency=afterok:$ON:$OFF rl_narrow/jobs/eval_real.sh)
AG=$($SB --dependency=afterany:$EV rl_narrow/jobs/finalize.sh)
echo "online training  : $ON  (6 tasks, tau_k ${TAU_K:-8})"
echo "offline training : $OFF (10 tasks)"
echo "real evaluation  : $EV  (48 tasks, after both trainings succeed)"
echo "finalize         : $AG  -> rl_narrow/results_real/{leaderboard_real.csv, figures/}"
