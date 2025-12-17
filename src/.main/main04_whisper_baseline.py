from src.asr_model.whisper.scripts.inference import inference

data = [
    "./data/datasets/TORGO/wav/F01_Session1_0008.wav",
    "./data/datasets/TORGO/wav/F01_Session1_0014.wav"
    ]

result = inference(data)



