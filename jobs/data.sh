#!/bin/bash
#SBATCH --job-name=data
#SBATCH --account=jhjin1
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/Offline-RL-Controller-in-AM/jobs/dataset.log

python dataset.py --mode=read --lb=150 --ub=400 --step_size=10 --n=500 --trajectory_length=12 --tag='lc'