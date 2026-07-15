#!/bin/bash
#SBATCH --job-name=qgan_qpu
#SBATCH --array=0-9
#SBATCH --time=08:00:00
#SBATCH --output=logs/qpu_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --comment=mem_4G

cd ~/2026/QGAN-Sleep-EEG

export SLURM_JOB_END_TIME=$(($(date +%s) + 8*3600))
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export ONLY_FOLD=$SLURM_ARRAY_TASK_ID
export ONLY_CONDITION=${ONLY_CONDITION:-qpu_noiseless}
export QPU_EPOCHS=${QPU_EPOCHS:-50}
export QPU_SHOTS=${QPU_SHOTS:-128}
export FEATURE_SET=${FEATURE_SET:-statistical}

python -u -m qgan.train_journal --mode full --conditions qpu