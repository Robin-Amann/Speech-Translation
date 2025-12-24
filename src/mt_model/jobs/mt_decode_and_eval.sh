#!/bin/bash
#SBATCH --job-name=mt_eval
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --partition=cpu
set -e
module purge
source $HOME/miniconda3/bin/activate fairseq-env

DATA=$1
CKPT=$2
OUT=$3/$SLURM_JOB_ID
mkdir -p $OUT

python $HOME/mt_model/scripts/mt_decode_and_eval.py \
    --data-dir $DATA \
    --checkpoint-dir $CKPT \
    --output-dir $OUT
