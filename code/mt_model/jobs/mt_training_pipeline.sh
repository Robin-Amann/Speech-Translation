#!/bin/bash

RAW=$HOME/mt_model/training_data
OUT=$HOME/mt_model/work
jid0=$(sbatch ./mt_prepare_data.sh $RAW $OUT | awk '{print $4}')
# sbatch ./mt_prepare_data.sh $HOME/mt_model/training_data $HOME/mt_model/work

WORKDIR=$HOME/mt_model/work/data-bin
RESULTS=$HOME/mt_model/results
jid1=$(sbatch --dependency=afterok:$jid0 ./mt_train_model.sh $WORKDIR $RESULTS | awk '{print $4}')
# sbatch ./mt_train_model.sh $HOME/mt_model/work/data-bin $HOME/mt_model/results

DATA=$HOME/mt_model/work/data-bin
CKPT=/home/$USER/mt_model/results/$jid1
OUT=$HOME/mt_model/results
sbatch --dependency=afterok:$jid1 ./mt_decode_and_eval.sh $DATA $CKPT $OUT
