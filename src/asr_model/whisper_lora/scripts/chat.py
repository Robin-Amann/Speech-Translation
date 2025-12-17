import torch
from torch.utils.data import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import evaluate
import numpy as np

wer_metric = evaluate.load("wer")


class WhisperDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        audio = self.processor.feature_extractor(
            sample["audio"],
            sampling_rate=16000,
            return_tensors="pt"
        )

        labels = self.processor.tokenizer(
            sample["text"],
            return_tensors="pt"
        ).input_ids

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_features": audio.input_features[0],
            "labels": labels[0]
        }


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}


def train_whisper_lora(train, dev, test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    global processor
    processor = WhisperProcessor.from_pretrained(WHISPER_V3_MODEL_NAME)

    base_model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_V3_MODEL_NAME,
        torch_dtype=dtype
    )

    lora_config = LoraConfig(
        r=2,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["fc1"],
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    train_ds = WhisperDataset(train, processor)
    dev_ds = WhisperDataset(dev, processor)

    args = TrainingArguments(
        output_dir=WHISPER_V3_LORA_CHECKPOINT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        learning_rate=1e-4,
        fp16=torch.cuda.is_available(),
        num_train_epochs=5,
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(WHISPER_V3_LORA_CHECKPOINT_DIR)

    return model




def transcribe(audio_paths, language="english"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = WhisperProcessor.from_pretrained(WHISPER_V3_MODEL_NAME)

    base = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_V3_MODEL_NAME,
        torch_dtype=dtype
    )

    model = PeftModel.from_pretrained(base, WHISPER_V3_LORA_CHECKPOINT_DIR)
    model.to(device)
    model.eval()

    single = isinstance(audio_paths, str)
    audio_paths = [audio_paths] if single else audio_paths

    results = []
    for path in audio_paths:
        inputs = processor(
            path,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                language=language
            )

        text = processor.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        results.append(text)

    return results[0] if single else results
