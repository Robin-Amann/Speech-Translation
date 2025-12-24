#!/bin/bash
#SBATCH --job-name=mt_prep
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=prep_%j.out
#SBATCH --error=prep_%j.err
#SBATCH --partition=cpu

set -e
module purge

# install fairseq
FAIRSEQ_DIR=$HOME/fairseq
if [ ! -d "$FAIRSEQ_DIR" ]; then # check if fairseq directory does not exist
    git clone https://github.com/facebookresearch/fairseq.git $FAIRSEQ_DIR
fi
source $HOME/miniconda3/bin/activate fairseq-env
pip install --editable $FAIRSEQ_DIR

# install other dependencies
pip install sacremoses sentencepiece

RAW=$1
OUT=$2
mkdir -p $OUT

python $HOME/mt_model/scripts/mt_prepare_data.py \
    --raw-dir $RAW \
    --out-dir $OUT
