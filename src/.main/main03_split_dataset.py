# this propabably depends on the dataset
# in torgo the cohorts and not clear since there are only 3 that make sense. 
# we don't have enough data for each cohort (10) and speaker.
# but in the real dataset we need different cohorts, speaker that are present in some cohorts and in some not. 
# each train / dev / test split should be random but repeatable.
# i wrote some functions for that but i am not happy with them.

# this code below just devides the data by cohort and speaker and sees how much data of what we have

### group data based on speaker and cohort ###
import src.utils_file as utils_file
import src.magic_strings as magic_strings
import src.asr_model.data_preperation.torgo as torgo
import joblib
import numpy as np

dataset = utils_file.read_dict(magic_strings.TORGO_PATH_LOCAL / "torgo.txt")

embeddings = [ entry["embedding"] for entry in dataset ]
embeddings_array = np.asarray(embeddings)

loaded_model = joblib.load("./data/mixture_models/bgmm.joblib")
labels = loaded_model.predict(embeddings_array)

for entry, label in zip(dataset, labels.tolist()) :
    entry["cohort"] = label


def group_files(dataset: list[dict], split_keys: list[str]) :
    if len(split_keys) == 0 :
        return dataset
    key, remaining_keys = split_keys[0], split_keys[1:]

    groups: dict[str, list[dict]] = dict()
    for entry in dataset :
        k = entry[key]
        if k not in groups :
            groups[k] = []
        groups[k].append(entry)
    
    for k, values in groups.items() :
        groups[k] = group_files(values, remaining_keys)
    return groups


groups = group_files(dataset, ["cohort", "speaker_id"])

matrix = [ [ 0 for _ in range(10) ] for _ in range(8) ]   # matrix[speaker][cohort]
speaker_map = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05" ]
speaker_map = { k: v for v, k in enumerate(speaker_map) }

for cohort in groups :
    for speaker_id in groups[cohort] :
        matrix[speaker_map[speaker_id]][int(cohort)] = len(groups[cohort][speaker_id])
        # print(speaker_id, cohort, len(groups[speaker_id][cohort]))

print(matrix)
# [ 3,  0, 11,  1,  0,  3,  1,  1,  0, 0]
# [ 0,  0, 14,  5, 31,  7, 60,  1, 21, 0]
# [ 0,  0,  2,  1, 54,  2, 11,  0, 31, 0]
# [20,  2, 26, 22,  2,  7,  2,  8,  0, 0]
# [26,  2, 19, 24,  0, 13,  1,  7,  0, 0]
# [ 0,  0,  0,  0, 57,  0, 26,  0, 12, 0]
# [14, 13,  7, 13,  0, 17,  1, 21,  0, 0]
# [11, 10,  3, 28,  0, 40,  0, 23,  0, 9]