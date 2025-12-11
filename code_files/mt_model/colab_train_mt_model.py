### Getting Started ###
# !apt-get update -y
# !apt-get install -y python3.10 python3.10-venv python3.10-distutils python3.10-dev curl git build-essential

# # Install pip for python3.10
# !curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
# !python3.10 /tmp/get-pip.py

# !pip install pip==24.0
# !pip list -v


# Enabling and testing the GPU
import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print('Current device:', torch.cuda.get_device_name(device))
else:
    print('Failed to find GPU. Will use CPU.')
    device = 'cpu'


# Cloing fairseq repo from github
# !git clone https://github.com/facebookresearch/fairseq.git
# %cd fairseq
# !python3.10 -m pip install --editable ./

import os
os.environ['PYTHONPATH'] += ":/content/fairseq/"
# !pip install sacremoses


### Data Preperation ###
# %cd /content/fairseq/examples/translation
# !wget -nc -O sample_data.zip https://bwsyncandshare.kit.edu/s/7oo2AG8jRriLZKg/download?path=%2F&files=data.zip&downloadStartSecret=tk6qdncox5
# !unzip sample_data.zip

# !pip install sentencepiece
import sentencepiece as spm

spm.SentencePieceTrainer.train(input="sample_data/train.de-en.en,sample_data/train.de-en.de",
                               model_prefix="bpe",
                               vocab_size=10000)
print('Finished training sentencepiece model.')

# Load the trained sentencepiece model
spm_model = spm.SentencePieceProcessor(model_file="bpe.model")

for partition in ["train", "dev", "tst"]:
    for lang in ["de", "en"]:
        f_out = open(f"sample_data/spm.{partition}.de-en.{lang}", "w")

        with open(f"sample_data/{partition}.de-en.{lang}", "r") as f_in:
            for line_idx, line in enumerate(f_in.readlines()):
                # Segmented into subwords
                line_segmented = spm_model.encode(line.strip(), out_type=str)
                # Join the subwords into a string
                line_segmented = " ".join(line_segmented)
                f_out.write(line_segmented + "\n")

        f_out.close()

# Preprocess/binarize the data
TEXT="/content/fairseq/examples/translation/sample_data"
# # Binarize the data for training
# !python3.10 -m fairseq_cli.preprocess \
#     --source-lang en --target-lang de \
#     --trainpref $TEXT/spm.train.de-en \
#     --validpref $TEXT/spm.dev.de-en \
#     --testpref $TEXT/spm.tst.de-en \
#     --destdir data-bin/iwslt14.de-en \
#     --thresholdtgt 0 --thresholdsrc 0 \
#     --workers 8


### Training ###
!CUDA_VISIBLE_DEVICES=0 python3.10 -m fairseq_cli.train \
    /content/fairseq/examples/translation/data-bin/iwslt14.de-en \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --keep-last-epochs 2 \
    --max-tokens 4096 \
    --max-epoch 20 \
    --fp16

### Decoding ###
TEST_INPUT="/content/fairseq/examples/translation/sample_data/spm.tst.de-en.de"
PRED_LOG="/content/fairseq/examples/translation/en-de.decode.log"

!python3.10 -m fairseq_cli.generate /content/fairseq/examples/translation/data-bin/iwslt14.de-en \
      --task translation \
      --source-lang en \
      --target-lang de \
      --path /content/fairseq/examples/translation/checkpoints/checkpoint_best.pt \
      --batch-size 256 \
      --beam 4 \
      --remove-bpe=sentencepiece > $PRED_LOG

!grep ^H $PRED_LOG | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > ./hyp.txt
!grep ^T $PRED_LOG | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > ./ref.txt

!head ./hyp.txt
!head ./ref.txt

### Evaluation ###
!pip install sacrebleu
!cat ./hyp.txt | sacrebleu ref.txt
