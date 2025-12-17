### preprocess torgo ###
import code_files.utils_file as utils_file
import code_files.magic_strings as magic_strings
import code_files.asr_model.data_preperation.torgo as torgo

dataset = torgo.load_original()
# prompt file has no wav file (data\datasets\TORGO\everything\F\F03\Session2\prompts\0117.txt)
# prompt file has no wav file (data\datasets\TORGO\everything\M\M01\Session2_3\prompts\0219.txt)
# prompt file has no wav file (data\datasets\TORGO\everything\M\M04\Session1\prompts\0066.txt)

dataset = torgo.filter_original(dataset)
torgo.analyze_original(dataset)
# speaker #sentences
# F01     20
# F03     139
# F04     101
# M01     89
# M02     92
# M03     95
# M04     86
# M05     124
# ------------------
# sum     746

# copy wav files
import os
from pathlib import Path
import shutil
target_dir = Path(magic_strings.TORGO_PATH_LOCAL) / "wav"
os.makedirs(target_dir, exist_ok=True)
for entry in dataset :
    name = f"{entry["speaker_id"]}_{entry["session"]}_{id}.wav"
    if (target_dir / name).is_file() :
        print("file already exists", name)
    shutil.copy2(entry["wav_file_path"], (target_dir / name))
    entry["wav_file_path"] = str(target_dir / name)

# save dataset
utils_file.write_dict(magic_strings.TORGO_PATH_LOCAL / "torgo.txt", dataset)