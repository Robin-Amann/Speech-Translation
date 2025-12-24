import src.asr_model.whisper_lora.scripts.whisper_lora as whisper_lora

import src.utils_file as utils_file
import src.magic_strings as magic_strings
from datasets import Dataset, Audio, DatasetDict

### load dataset
dataset = utils_file.read_dict(magic_strings.TORGO_PATH_LOCAL / "torgo.txt")[:10]

### dataset to Dataset
hf_dataset: Dataset = Dataset.from_dict({
    "audio": [ entry["wav_file_path"] for entry in dataset ],
    "text": [ entry["prompt"] for entry in dataset ]
}).cast_column("audio", Audio(sampling_rate=16000))

split1: DatasetDict = hf_dataset.train_test_split(test_size=0.8) # 80 / 20
train, temp = split1["train"], split1["test"]
split2 = temp.train_test_split(test_size=0.5) # 50 / 50

split_dataset = DatasetDict({
    "train": train,             # 80 %
    "dev":  split2["train"],    # 10 %
    "test": split2["test"]      # 10 %
})

split_dataset = whisper_lora.prepare_data(split_dataset)
print(split_dataset)

whisper_lora.finetune_lora(split_dataset)