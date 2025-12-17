import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from src.magic_strings import WHISPER_V3_MODEL_NAME, WHISPER_V3_CHECKPOINT_DIR


def inference(data: str | list[str], batch_size=8, target_language="english") -> str | list[str]:
    """data is a single path or a list of paths to audio files"""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_V3_MODEL_NAME, 
        cache_dir=WHISPER_V3_CHECKPOINT_DIR,
        dtype=dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(WHISPER_V3_MODEL_NAME)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
    )

    result = pipe(data, batch_size=batch_size, generate_kwargs={"language": target_language})

    if type(data) == str :
        return result["text"]
    else :
        return [ item["text"] for item in result]

    # result = pipe(sample, return_timestamps=True)
    # print(result["chunks"])
    # # word level
    # result = pipe(sample, return_timestamps="word")
    # print(result["chunks"])




