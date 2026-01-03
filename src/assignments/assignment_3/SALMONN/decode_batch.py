import os
import json
import argparse
import torch
from transformers import WhisperFeatureExtractor
from omegaconf import OmegaConf

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair in xxx=yyy format",
)
parser.add_argument("--data", type=str, required=True, help="path to test json")
parser.add_argument("--out-hyps", type=str, required=True, help="where to save hypotheses")
parser.add_argument("--out-refs", type=str, default="refs.txt", help="where to save references")
args = parser.parse_args()

# load config and model (same as cli_inference.py)
cfg = Config(args)
model = SALMONN.from_config(cfg.config.model)
model.to(args.device)
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

# load test data
with open(args.data, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# if the file has {"annotation": [...]} structure, unwrap it
test_data = test_data["annotation"]

refs = []
hyps = []

for sample in test_data:
    wav_path = sample["path"]
    if not os.path.isabs(wav_path):
        wav_path = os.path.join(os.getcwd(), wav_path)

    prompt_txt = "Listen to the speech and translate it into German."
    samples = prepare_one_sample(wav_path, wav_processor)
    prompt = [
        cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt_txt.strip())
    ]

    with torch.cuda.amp.autocast(dtype=torch.float16):
        out = model.generate(samples, cfg.config.generate, prompts=prompt)[0]

    refs.append(sample["text"].strip())
    hyps.append(out.strip())

# write references and hypotheses
with open(args.out_refs, "w", encoding="utf-8") as f:
    for r in refs:
        f.write(r + "\n")

with open(args.out_hyps, "w", encoding="utf-8") as f:
    for h in hyps:
        f.write(h + "\n")

print("Wrote", args.out_refs, "and", args.out_hyps)
