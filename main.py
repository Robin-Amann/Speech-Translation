import src.asr_model.whisper_lora.scripts.whisper_lora as whisper_lora
from datasets import load_dataset, Dataset


dataset = load_dataset("<username>/my_dataset")

audio_dataset = Dataset.from_dict({"audio": ["path/to/audio_1", "path/to/audio_2", ..., "path/to/audio_n"]}).cast_column("audio", Audio())
audio_dataset[0]["audio"]