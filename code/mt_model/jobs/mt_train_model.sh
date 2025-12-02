#!/bin/bash
#SBATCH --job-name=mt_train
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

set -e
module purge
module load devel/cuda/12.8
source $HOME/miniconda3/bin/activate fairseq-env

WORKDIR=$1

# copy data to $TMPDIR
cp -r $WORKDIR $TMPDIR/data-bin
mkdir -p $TMPDIR/checkpoints

python $HOME/mt_model/scripts/mt_train_model.py \
    --data-dir $TMPDIR/data-bin/iwslt14.de-en \
    --save-dir $TMPDIR/checkpoints

# copy data to back to $HOME
RESULTS=$2/$SLURM_JOB_ID
mkdir -p $RESULTS
cp -r $TMPDIR/checkpoints $RESULTS/