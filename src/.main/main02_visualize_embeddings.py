### generate embeddings ###
import src.utils as utils
import src.utils_file as utils_file
import src.magic_strings as magic_strings
import src.asr_model.data_preperation.torgo as torgo
from src.asr_model.whisper_hyper.skripts.mixture_model import generate_embeddings

print("load data")
dataset = utils_file.read_dict(magic_strings.TORGO_PATH_LOCAL / "torgo.txt")
print("generate embeddings")
dataset = generate_embeddings(dataset)
# map1: 74.09 examples/s
# map2: 4.40 s/example
dataset = dataset.to_list()

utils_file.write_dict(magic_strings.TORGO_PATH_LOCAL / "torgo.txt", dataset)
# IMPORTANT: all code after this uses those embeddings



### visualize embeddings ###
import src.magic_strings as magic_strings
import src.asr_model.data_preperation.torgo as torgo
from src.asr_model.whisper_hyper.skripts.mixture_model import visualize_embeddings

dataset = utils_file.read_dict(magic_strings.TORGO_PATH_LOCAL / "torgo.txt")
speaker_ids, embeddings = map(list, zip(*[(entry["speaker_id"], entry["embedding"]) for entry in dataset]))
visualize_embeddings(speaker_ids, embeddings)


### fit GMM ###
import src.utils as utils
import src.asr_model.data_preperation.torgo as torgo
import src.magic_strings as magic_strings
from src.asr_model.whisper_hyper.skripts.mixture_model import fit_BGMM
from src.asr_model.whisper_hyper.skripts.mixture_model import visualize_embeddings
import joblib
import os
import numpy as np

dataset = utils_file.read_dict(magic_strings.TORGO_PATH_LOCAL / "torgo.txt")
embeddings = [ entry["embedding"] for entry in dataset ]
embeddings_array = np.asarray(embeddings)

model = fit_BGMM(embeddings_array, max_components=10)
labels = model.predict(embeddings_array)

# print(model.weight_concentration_)
os.makedirs("./data/mixture_models/", exist_ok=True)
joblib.dump(model, "./data/mixture_models/bgmm.joblib")

# Use it again
# model = joblib.load("./data/mixture_models/bgmm.joblib")
# labels = model.predict(embeddings_array)


### UMAP parameters ###
for n_neighbors in [2, 5, 10, 20, 50, 100, 200] : # 1 is to little
    for min_dist in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1] : # 2 is to much
        print(n_neighbors, min_dist)
        visualize_embeddings(labels.tolist(), embeddings, n_neighbors=n_neighbors, min_dist=min_dist, save_dir="./results/embeddings/UMAP_parameters", filename=f"UMAP_{n_neighbors}_{min_dist}.png")

n_neighbors = 10
min_dist = 0.2
visualize_embeddings(labels.tolist(), embeddings, n_neighbors=n_neighbors, min_dist=min_dist, save_dir="./results/embeddings", filename=f"UMAP_{n_neighbors}_{min_dist}.png")


### test how the embedding space groups embeddings. test for
# - speaker
# - words
# - source

import src.utils as utils
import src.magic_strings as magic_strings
import src.asr_model.data_preperation.torgo as torgo
from collections import Counter

dataset = utils_file.read_dict(magic_strings.TORGO_PATH_LOCAL / "torgo.txt")


# inspect how often prompts were prompted
prompts = [ entry["prompt"] for entry in dataset ]
counts: dict[str, int] = dict(Counter(prompts))

k = 3
sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
top_k = dict(sorted_items[:k])
for prompt, count in top_k.items() :
    print(prompt.ljust(60), count)

count_count = dict(Counter(list(counts.values())))
count_count = dict(sorted(count_count.items(), key=lambda item: item[0])) 
print(count_count)
# top 3
# yet he still thinks as swiftly as ever.                      11
# Except in the winter when the ooze or snow or ice prevents,  10
# The quick brown fox jumps over the lazy dog.                 10
# distribution of how often prompts where prompted ( number of times: number of different prompts)
# {1: 46, 2: 19, 3: 11, 4: 3, 5: 5, 6: 20, 7: 36, 8: 12, 9: 7, 10: 5, 11: 1}

# give each embedding corresponding to a certain prompt the label 1 else 0
from src.asr_model.whisper_hyper.skripts.mixture_model import visualize_embeddings

selected_prompts = [ 
    "yet he still thinks as swiftly as ever.", 
    "Except in the winter when the ooze or snow or ice prevents,", 
    "The quick brown fox jumps over the lazy dog."
    ] 
for prompt in selected_prompts :
    labels, embeddings = map(list, zip(*[((1 if entry["prompt"].strip() == prompt else 0), entry["embedding"]) for entry in dataset]))
    visualize_embeddings(labels, embeddings, save_dir="./results/embeddings/prompts/", filename=f"{prompt}.png", show=False)

# alright should be good


# now do the same for speaker
speaker_ids, embeddings = map(list, zip(*[(entry["speaker_id"], entry["embedding"]) for entry in dataset]))
visualize_embeddings(speaker_ids, embeddings, save_dir="./results/embeddings/", filename="UMAP_10_0.2_speaker_id.png")
# ok there is a clear correlation between speaker_id and cohort but this makes sense. same speaker stays in the same cohort.

# now for the source
labels, embeddings = map(list, zip(*[((1 if entry["source"].strip() == "headMic" else 0), entry["embedding"]) for entry in dataset]))
visualize_embeddings(labels, embeddings, save_dir="./results/embeddings/", filename=f"UMAP_10_0.2_source.png")