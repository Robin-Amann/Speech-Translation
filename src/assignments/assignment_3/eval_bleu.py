import sacrebleu

refs_path = "refs.txt"
raw_path = "hyps_raw.txt"
ft_path  = "hyps_ft.txt"

with open(refs_path, encoding="utf-8") as f:
    refs = [l.strip() for l in f]
with open(raw_path, encoding="utf-8") as f:
    hyps_raw = [l.strip() for l in f]
with open(ft_path, encoding="utf-8") as f:
    hyps_ft = [l.strip() for l in f]

bleu_raw = sacrebleu.corpus_bleu(hyps_raw, [refs])
bleu_ft  = sacrebleu.corpus_bleu(hyps_ft,  [refs])

print(f"BLEU raw: {bleu_raw.score:.2f}")
print(f"BLEU finetuned: {bleu_ft.score:.2f}")
