#!/bin/bash
#SBATCH --job-name=aer_4B_both
#SBATCH --output=logs/aer_4B_both_%j.out
#SBATCH --error=logs/aer_4B_both_%j.err
#SBATCH --partition=gpu_a100_il  # Use development queue for testing [cite: 244]
#SBATCH --gres=gpu:2                 # Request 2 GPUs 
#SBATCH --time=12:00:00          # dev queue limit 30 minutes [cite: 244]

# --- Container Setup (Pyxis) ---
#SBATCH --container-name=unsloth1
#SBATCH --container-mount-home
#SBATCH --container-writable
# Required mount points: include complete paths for .uc3 and .so.15 [cite: 794]
#SBATCH --container-mounts=/etc/slurm/task_prolog.uc3:/etc/slurm/task_prolog.uc3,/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm:/usr/lib64/slurm,/usr/lib64/libhwloc.so:/usr/lib64/libhwloc.so,/usr/lib64/libhwloc.so.15:/usr/lib64/libhwloc.so.15

export PROJECT_DIR="$HOME/AER"
cd "$PROJECT_DIR"
# --- Environment Isolation and Initialization ---
# 🚀 Core isolation: Disable loading packages from host ~/.local to ensure only stable versions from container are used
export PYTHONNOUSERSITE=1

# Disable supervisord loop logic from the image (resolves log spam issue)
export ENROOT_ENTRYPOINT=0

# 1. Set critical environment variables (prevent deadlock)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# If your network environment is special, you may also need to force lo, but try the above two first
# export NCCL_SOCKET_IFNAME=lo 

# 2. Debug information (optional, verify Python path is correct)
which python
python --version

# 2. Start inference (run Python directly)
export TQDM_DISABLE=1
echo "🚀 Starting Inference..."
python -u inferenceMulti.py
