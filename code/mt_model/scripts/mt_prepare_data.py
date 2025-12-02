import os
import argparse
import sentencepiece as spm
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--raw-dir", required=True)
parser.add_argument("--out-dir", required=True)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# 1. Train SPM
print("start training BPE model")
model_prefix = os.path.join(args.out_dir, "bpe")
spm.SentencePieceTrainer.train(
    input=f"{args.raw_dir}/train.de-en.en,{args.raw_dir}/train.de-en.de",
    model_prefix=model_prefix,
    vocab_size=10000
)

# 2. Apply segmentation
print("apply BPE to data")
sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
for split in ["train", "dev", "tst"]:
    for lang in ["en", "de"]:
        inp = f"{args.raw_dir}/{split}.de-en.{lang}"
        out = f"{args.out_dir}/spm.{split}.de-en.{lang}"
        with open(inp) as fin, open(out, "w") as fout:
            for line in fin:
                fout.write(" ".join(sp.encode(line.strip(), out_type=str)) + "\n")

# 3. Binarize
print("binarize")
subprocess.run([
    "python", "-m", "fairseq_cli.preprocess",
    "--source-lang", "en",
    "--target-lang", "de",
    "--trainpref", f"{args.out_dir}/spm.train.de-en",
    "--validpref", f"{args.out_dir}/spm.dev.de-en",
    "--testpref",  f"{args.out_dir}/spm.tst.de-en",
    "--destdir",   os.path.join(args.out_dir, "data-bin"),
    "--thresholdsrc", "0",
    "--thresholdtgt", "0",
    "--workers", "8"
], check=True)
