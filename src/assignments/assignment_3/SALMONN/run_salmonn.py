import os

TEST_JSON = "data/europarl_test.json"

print("== Decoding test set with raw SALMONN ==")
os.system(
    f"python decode_batch.py --cfg-path configs/decode_config.yaml "
    f"--data {TEST_JSON} --out-hyps hyps_raw.txt --out-refs refs.txt"
)

print("== Starting to finetune SALMONN ==")
os.system("python train.py --cfg-path configs/config.yaml")

print("== Decoding test set with finetuned SALMONN ==")
os.system(
    f"python decode_batch.py --cfg-path configs/decode_config.yaml "
    f"--data {TEST_JSON} --out-hyps hyps_ft.txt --out-refs refs.txt"
)

print("== Evaluating BLEU before/after finetuning ==")
os.system("python ../eval_bleu.py")
