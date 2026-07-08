#!/bin/bash
#SBATCH --job-name=qgan_qpu
#SBATCH --array=0-9
#SBATCH --time=08:00:00
#SBATCH --output=logs/qpu_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd ~/QGAN-Sleep-EEG

# Makes the script's wall-clock guard actually functional
export SLURM_JOB_END_TIME=$(($(date +%s) + 8*3600))

export ONLY_FOLD=$SLURM_ARRAY_TASK_ID
export ONLY_CONDITION=${ONLY_CONDITION:-qpu_sim}      # confirm exact string in models_journal.py
export QPU_EPOCHS=50
export FEATURE_SET=${FEATURE_SET:-statistical}

python -m qgan.train_journal --mode full --conditions qpu