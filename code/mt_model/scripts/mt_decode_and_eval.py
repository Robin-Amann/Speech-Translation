import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True, help="Path to binarized data")
parser.add_argument("--checkpoint-dir", required=True, help="Directory with trained checkpoints")
parser.add_argument("--output-dir", required=True, help="Directory to save logs, hyp, ref")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

pred_log = os.path.join(args.output_dir, "decode.log")
hyp_file = os.path.join(args.output_dir, "hyp.txt")
ref_file = os.path.join(args.output_dir, "ref.txt")

generate_cmd = [
    "python",
    "-m", "fairseq_cli.generate",
    args.data_dir,
    "--task", "translation",
    "--source-lang", "en",
    "--target-lang", "de",
    "--path", os.path.join(args.checkpoint_dir, "checkpoint_best.pt"),
    "--batch-size", "256",
    "--beam", "4",
    "--remove-bpe=sentencepiece"
]

# Run generation
print("run fairseq_cli.generate")
with open(pred_log, "w") as outfile:
    subprocess.run(generate_cmd, stdout=outfile, stderr=subprocess.STDOUT, check=True)

print("extract hypothesis and reference")
subprocess.run(
    f"grep ^H {pred_log} | sed 's/^H-//g' | cut -f 3 | sed 's/ ##//g' > {hyp_file}",
    shell=True,
    check=True
)
subprocess.run(
    f"grep ^T {pred_log} | sed 's/^T-//g' | cut -f 2 | sed 's/ ##//g' > {ref_file}",
    shell=True,
    check=True
)

print("compute BLEU")
bleu_cmd = ["sacrebleu", ref_file]
with open(hyp_file, "r") as hyp:
    bleu_output = subprocess.check_output(bleu_cmd, input=hyp.read().encode())

print("sacrebleu:", bleu_output.decode())