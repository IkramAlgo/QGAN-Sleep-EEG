#!/bin/bash
# ============================================================
#  run_qgan_qpu.sh — ARC job script for Conditions 3+4
#  Conditions: QPU-Sim (FakeNairobi, SPSA) + QPU-Sim+ZNE
#  Submit: sbatch run_qgan_qpu.sh
#  Resume (after crash): sbatch run_qgan_qpu.sh  (same command)
# ============================================================
#SBATCH --job-name=qgan_qpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --partition=compute
#SBATCH --output=logs/qgan_qpu_%j.log
#SBATCH --error=logs/qgan_qpu_%j.err

set -euo pipefail

echo "========================================================"
echo "  QGAN QPU Job — Conditions 3+4 (FakeNairobi + ZNE)"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Node      : $SLURM_NODELIST"
echo "  Start     : $(date)"
echo "  Gradient  : SPSA (h=0.05, 128 shots)"
echo "  ZNE       : scale_factors=[1,3,5], Richardson extrap."
echo "========================================================"

module load python/3.10
source activate qgan310

# Aer uses all cores internally per circuit call
export OMP_NUM_THREADS=38
export MKL_NUM_THREADS=38

# Wall-clock guard — lets the script stop before SLURM kills it
WALL_SECONDS=$(( $(squeue -j $SLURM_JOB_ID -h -o "%L" | awk -F: '{print $1*3600+$2*60+$3}') ))
export SLURM_JOB_END_TIME=$(( $(date +%s) + WALL_SECONDS ))
echo "  Wall-clock guard end: $(date -d @$SLURM_JOB_END_TIME)"

cd /work/$USER/QGAN_Project
mkdir -p logs

echo "  Working dir: $(pwd)"
echo "  Python:      $(python --version)"
echo "  Data files:"
ls /work/$USER/QGAN_Project/data/EPC*.edf 2>/dev/null || echo "  WARNING: no EDF files found in data/"
echo "========================================================"

QPU_EPOCHS=100 python -m qgan.train_journal --mode full --conditions qpu

echo "========================================================"
echo "  QPU job complete: $(date)"
echo "========================================================"