import src.utils_file as utils_file
import src.magic_strings as magic_strings
from datasets import Dataset, Audio, DatasetDict

### load dataset
dataset = utils_file.read_dict(magic_strings.TORGO_PATH_LOCAL / "torgo.txt")[:5]

### dataset to Dataset
hf_dataset: Dataset = Dataset.from_dict({
    "audio": [ entry["wav_file_path"] for entry in dataset ],
    "text": [ entry["prompt"] for entry in dataset ]
})

### cast audio
hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=16000)) # providing a sampling_rate is optional

# now the audio data is more usable
print(hf_dataset)
# Dataset({
#     features: ['audio', 'text'],
#     num_rows: 5
# })
print(hf_dataset[0])
# {'audio': <datasets.features._torchcodec.AudioDecoder object at 0x000001CF3C548740>, 'text': 'Except in the winter when the ooze or snow or ice prevents,'}
print(hf_dataset[0]["audio"]["array"])
# [-2.6245117e-03 -2.8686523e-03 -1.5258789e-03 ... -3.0517578e-05 -1.5258789e-04 -6.1035156e-05]
print(type(hf_dataset[0]["audio"]["array"]))
# <class 'numpy.ndarray'>
print(hf_dataset[0]["audio"]["sampling_rate"])
# 16000


### split dataset
# if the original dataset hay key "split" you can split direct otherwise
split1: DatasetDict = hf_dataset.train_test_split(test_size=0.8) # 80 / 20
train, temp = split1["train"], split1["test"]
split2 = temp.train_test_split(test_size=0.5) # 50 / 50

split_dataset = DatasetDict({
    "train": train,             # 80 %
    "dev":  split2["train"],    # 10 %
    "test": split2["test"]      # 10 %
})

