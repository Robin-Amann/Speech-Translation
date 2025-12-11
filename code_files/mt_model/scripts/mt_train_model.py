import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True)    # $TMPDIR/data-bin/iwslt14.de-en
parser.add_argument("--save-dir", required=True)    # $TMPDIR/checkpoints
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

train_cmd = [
    "python",
    "-m", "fairseq_cli.train",
    args.data_dir,
    "--arch", "transformer",
    "--share-decoder-input-output-embed",
    "--optimizer", "adam",
    "--adam-betas", "(0.9, 0.98)",
    "--clip-norm", "0.0",
    "--lr", "5e-4",
    "--lr-scheduler", "inverse_sqrt",
    "--warmup-updates", "4000",
    "--dropout", "0.3",
    "--weight-decay", "0.0001",
    "--criterion", "label_smoothed_cross_entropy",
    "--label-smoothing", "0.1",
    "--keep-last-epochs", "2",
    "--max-tokens", "4096",
    "--max-epoch", "20",
    "--fp16",
    "--save-dir", args.save_dir
]
print("train model")
subprocess.run(train_cmd, check=True)