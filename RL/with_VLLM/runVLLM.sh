#!/bin/bash
#SBATCH --job-name=aer_32B_NF4
#SBATCH --output=logs/aer_32B_NF4_%j.out
#SBATCH --error=logs/aer_32B_NF4_%j.err
#SBATCH --partition=gpu_h100  # Using H100 GPU partition
#SBATCH --gres=gpu:2                 # Request 2 GPUs
#SBATCH --time=36:00:00              # Maximum runtime: 36 hours

# --- Container Settings (Pyxis) ---
#SBATCH --container-name=unsloth1
#SBATCH --container-mount-home
#SBATCH --container-writable
# Required mount points: including complete paths for .uc3 and .so.15 files
#SBATCH --container-mounts=/etc/slurm/task_prolog.uc3:/etc/slurm/task_prolog.uc3,/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm:/usr/lib64/slurm,/usr/lib64/libhwloc.so:/usr/lib64/libhwloc.so,/usr/lib64/libhwloc.so.15:/usr/lib64/libhwloc.so.15

export PROJECT_DIR="$HOME/AER"
cd "$PROJECT_DIR"

# --- Environment Isolation & Initialization ---
# 🚀 Core Isolation: Prevent loading packages from host ~/.local, ensure only stable libraries in container are used
export PYTHONNOUSERSITE=1

# Disable container's built-in supervisord loop logic (fixes log spam issue)
export ENROOT_ENTRYPOINT=0

# 1. Set critical environment variables (prevent deadlocks)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# If network environment is special, may also need to force loopback interface, but try above two first
# export NCCL_SOCKET_IFNAME=lo 

# 2. Debug information (optional, verify Python path is correct)
which python
python --version

# 3. Launch training (core modification point)
# Don't use 'accelerate launch' directly, use 'python -m accelerate.commands.launch' instead
# This ensures it uses the correct interpreter found by 'which python' above
python -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes 2 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    trainRLVLLM.py
