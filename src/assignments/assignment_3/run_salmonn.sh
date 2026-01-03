#!/bin/bash
#SBATCH --job-name=salmonn-train
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=94000mb
#SBATCH --time=30
#SBATCH --output=whisper_%j.out
#SBATCH --error=whisper_%j.err

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500       # any free port
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

module load devel/miniforge/25.3.1-python-3.12

source /opt/bwhpc/common/devel/miniforge/25.3.1-py3.12/etc/profile.d/conda.sh
conda activate salmonn310

cd "$HOME/salmonn_praktikum/SALMONN"
python run_salmonn.py
