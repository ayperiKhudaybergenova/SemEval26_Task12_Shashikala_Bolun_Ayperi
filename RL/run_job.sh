#!/bin/bash
#SBATCH --job-name=aer_resume
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem=240gb


export PROJECT_DIR="$HOME/AER"
export VENV_DIR="$PROJECT_DIR/unsloth2"
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))


source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"


echo "🚀 Node: $(hostname)"
echo "🚀 GPUs: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}')"


export DISABLE_TQDM=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16


echo "🚀 Starting Training with torchrun..."
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT trainRL.py