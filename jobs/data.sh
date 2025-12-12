#!/bin/bash
#SBATCH --job-name=data
#SBATCH --account=jhjin1
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Offline-RL-Controller-in-AM/jobs/out.log

python dataset.py