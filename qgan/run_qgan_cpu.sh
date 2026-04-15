#!/bin/bash
# ============================================================
#  run_qgan_cpu.sh — ARC job script for Conditions 1+2
#  Conditions: Simulator (noiseless) + Simulator+DataNoise
#  Submit: sbatch run_qgan_cpu.sh
# ============================================================
#SBATCH --job-name=qgan_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=logs/qgan_cpu_%j.log
#SBATCH --error=logs/qgan_cpu_%j.err

set -euo pipefail

echo "========================================================"
echo "  QGAN CPU Job — Conditions 1+2"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Node      : $SLURM_NODELIST"
echo "  Start     : $(date)"
echo "  Partition : $SLURM_JOB_PARTITION"
echo "========================================================"

# ── Environment ────────────────────────────────────────────
module load python/3.10
source activate qgan310

# Give PennyLane/PyTorch access to all 40 cores
export OMP_NUM_THREADS=38
export MKL_NUM_THREADS=38

# Wall-clock guard: compute job end timestamp so train_journal.py
# can stop cleanly before SLURM kills it mid-fold.
WALL_SECONDS=$(( $(squeue -j $SLURM_JOB_ID -h -o "%L" | awk -F: '{print $1*3600+$2*60+$3}') ))
export SLURM_JOB_END_TIME=$(( $(date +%s) + WALL_SECONDS ))
echo "  Wall-clock guard end: $(date -d @$SLURM_JOB_END_TIME)"

# ── Project setup ───────────────────────────────────────────
cd /work/$USER/QGAN_Project
mkdir -p logs

echo "  Working dir: $(pwd)"
echo "  Python:      $(python --version)"
echo "========================================================"

# ── Run ─────────────────────────────────────────────────────
CPU_EPOCHS=50 python -m qgan.train_journal --mode full --conditions cpu

echo "========================================================"
echo "  CPU job complete: $(date)"
echo "========================================================"