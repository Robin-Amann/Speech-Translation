# https://huggingface.co/blog/fine-tune-whisper

# !pip install --upgrade --quiet pip
# !pip install --upgrade --quiet datasets[audio] transformers accelerate evaluate jiwer tensorboard

import torch
from typing import Any, Dict, List, Union
from dataclasses import dataclass
from datasets import DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, pipeline
from peft import PeftModel, LoraConfig, get_peft_model
import evaluate
from src.magic_strings import WHISPER_V3_MODEL_NAME, WHISPER_V3_LORA_CHECKPOINT_DIR, WHISPER_V3_CHECKPOINT_DIR
from src.training_args import whisper_local_steps, whisper_cluster_steps
from transformers import Seq2SeqTrainer
from datasets.features._torchcodec import AudioDecoder

### Define a Data Collator
# see the blogpost
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
            labels_batch.attention_mask = labels_batch.attention_mask[:, 1:]    # (*1)

        batch["labels"] = labels
        batch["decoder_attention_mask"] = labels_batch.attention_mask           # (*2)
        
        return batch
    

def prepare_data(dataset: DatasetDict) :
    """insert a dataset dict with a train / dev / test split.\\
    each split needs to have the same columns and at least\\
    - "filepath": contains the filepath to a audiofile\\
    - "text": the label corresponding to the audiofile
    """

    # Prepare Feature Extractor, Tokenizer and Data
    # The ASR pipeline can be de-composed into three stages:
    # 1. A feature extractor which pre-processes the raw audio-inputs
    #    The Whisper feature extractor performs two operations:
    #    1. Pads / truncates the audio inputs to 30s: any audio inputs shorter than 30s are padded to 30s with silence (zeros), and those longer that 30s are truncated to 30s
    #    2. Converts the audio inputs to _log-Mel spectrogram_ input features, a visual representation of the audio and the form of the input expected by the Whisper model
    # 2. The model which performs the sequence-to-sequence mapping
    # 3. A tokenizer which post-processes the model outputs to text format    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_V3_MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(WHISPER_V3_MODEL_NAME, language="english", task="transcribe")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Now we can write a function to prepare our data ready for the model:
    def prepare_dataset(batch):
        # 1. We load and resample the audio data by calling `batch["audio"]`. As explained above, 🤗 Datasets performs any necessary resampling operations on the fly.
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # 2. We use the feature extractor to compute the log-Mel spectrogram input features from our 1-dimensional audio array.
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # 3. We encode the transcriptions to label ids through the use of the tokenizer.
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch

    # remove_columns removes all existing columns from the dataset. this encures that only "audio", "input_features" and "labels" remain
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])
    return dataset
    

def finetune_lora(dataset: DatasetDict, local=True) :

    ### Load a Pre-Trained Checkpoint
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_V3_MODEL_NAME, cache_dir=WHISPER_V3_CHECKPOINT_DIR)
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Create the LoRA configuration
    config = LoraConfig(
        r=2,                        # rank of BA
        lora_alpha=16,              # Typical values: 8, 16, 32
        lora_dropout=0.05,          # dropout during learning
        target_modules=["fc1"],     # Important: matches the module suffix
        bias="none",                # none | lora_only | all
        # task_type="SEQ_2_SEQ_LM"  # (*3) # "SEQ_2_SEQ_LM" | "CAUSAL_LM" | "TOKEN_CLS" | "QUESTION_ANS" | "FEATURE_EXTRACTION" | "IMAGE_CLASSIFICATION" (Whisper is a sequence-to-sequence language model)
    )
    model.enable_input_require_grads() # (*2)
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()
    lora_model.save_pretrained(WHISPER_V3_LORA_CHECKPOINT_DIR) # initial save before training

    processor = WhisperProcessor.from_pretrained(WHISPER_V3_MODEL_NAME, cache_dir=WHISPER_V3_CHECKPOINT_DIR, language="english", task="transcribe")
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=lora_model.config.decoder_start_token_id,
    )

    metric = evaluate.load("wer")

    # We then simply have to define a function that takes our model predictions and returns the WER metric. 
    # This function, called `compute_metrics`, first replaces `-100` with the `pad_token_id` in the `label_ids` 
    # (undoing the step we applied in the data collator to ignore padded tokens correctly in the loss).
    # It then decodes the predicted and label ids to strings. 
    # Finally, it computes the WER between the predictions and reference labels:
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    if local :
        training_args = whisper_local_steps(WHISPER_V3_LORA_CHECKPOINT_DIR)
    else :
        training_args = whisper_cluster_steps(WHISPER_V3_LORA_CHECKPOINT_DIR)
        
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=lora_model,
        train_dataset=dataset["train"],
        eval_dataset={"validation": dataset["dev"], "test": dataset["test"]},
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def inference(data: str | list[str] | AudioDecoder | list[AudioDecoder], checkpoint: str="checkpoint-10", batch_size=8, target_language="english") -> str | list[str]:
    """data is a single path or a list of paths to audio files"""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # load checkpoint
    base = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_V3_MODEL_NAME, 
        cache_dir=WHISPER_V3_CHECKPOINT_DIR,
        dtype=dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    
    model = PeftModel.from_pretrained(base, WHISPER_V3_LORA_CHECKPOINT_DIR + f"/{checkpoint}")

    model.to(device)

    processor = WhisperProcessor.from_pretrained(
        WHISPER_V3_MODEL_NAME, 
        cache_dir=WHISPER_V3_CHECKPOINT_DIR, 
        language=target_language, 
        task="transcribe"
    )

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
    )

    result = pipe(data, batch_size=batch_size, generate_kwargs={"language": target_language})

    print(result)
    if type(data) != list :
        return result["text"]
    else :
        return [ item["text"] for item in result]


# Create the LoRA configuration
# W = W + alpha / r * BA
# standard initialization
# - A
#   - Initialized from a standard Kaiming/He uniform distribution (the same initialization used for normal linear layers).
#   - This provides small but non-zero initial weights.
# - B
#   - Always initialized to zeros

# fc1 [nn layer] is transformed to
# fc1 (wrapper)
#     base_layer (original layer)
#     lora_dropout (wrapper)
#         lora_dropout.default (for training)
#     lora_A (wrapper)
#         lora_A.default (trainable)
#     lora_B (wrapper)
#         lora_B.default (trainable)
#     lora_embedding_A (for embeddings, useless for me)
#     lora_embedding_B (for embeddings, useless for me)
#     lora_magnitude_vector (for training)


# warning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
# `use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`...

# (*1)
# - error: 
#   The attention mask is not set and cannot be inferred from input because pad token is 
#   same as eos token. As a consequence, you may observe unexpected behavior. Please pass your 
#   input's `attention_mask` to obtain reliable results.
# - problem: 
#   - Whisper’s tokenizer uses the same token ID for pad_token and eos_token
#   - The model cannot infer padding positions from token IDs alone
#   - The transformer backend cannot construct an attention mask implicitly
# - solution
#   - Pass the decoder attention mask explicitly to the model.
#   - The decoder now receives an explicit binary mask
#   - Padding positions are unambiguously ignored
#   - The model no longer attempts to infer padding from token identity
#   - Numerical behavior becomes deterministic and stable

# (*2)
# error: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# solution: https://github.com/huggingface/peft/issues/137

# (*3)
# problem: WhisperForConditionalGeneration.forward() got an unexpected keyword argument 'input_ids'
# solution: https://github.com/huggingface/peft/issues/1988
