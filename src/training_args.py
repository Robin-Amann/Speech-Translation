from transformers import Seq2SeqTrainingArguments

# when setting training arguments, keep in mind
# - memory (capacity/efficiency)
# - precision
# - quality
# - speed

# all arguments: https://huggingface.co/docs/transformers/main_classes/trainer

def whisper_local_steps(output_dir) -> Seq2SeqTrainingArguments :
    "for running whisper locally just to check if it works. dataset size should be around 10 or 20."
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,          # where to save the checkpoints
        per_device_train_batch_size=4,  # samples per batch (train) per device. The global batch size is computed as: per_device_train_batch_size * number_of_devices in multi-GPU or distributed setups.
        gradient_accumulation_steps=1,  # number of batches before parameter update. simulates larger batch size without the need for more memory
                                        # effective batch = per_device_train_batch_size * #devices * gradient_accumulation_steps
        learning_rate=1e-5,             
        warmup_steps=2,                # steps in which learning rate goes from 0 to 100%
        max_steps=10,                  # total number of optimizer update steps to perform
        gradient_checkpointing=True,    # Activates recomputation of intermediate activations during backpropagation to reduce GPU memory usage. 
                                        # This trades increased computation time for lower memory consumption.
        fp16=True,                      # half-precision (16-bit floating point)
        eval_strategy ="steps",         # no | steps | epoch
        per_device_eval_batch_size=4,   # samples per batch (eval) per device
        predict_with_generate=True,     # Whether to use generate to calculate generative metrics (required for generative metrics)
        generation_max_length=100,      # The max_length to use on each evaluation loop when predict_with_generate=True
        save_steps=2,                   # Number of updates steps before two checkpoint saves if save_strategy="steps"
        eval_steps=2,                   # Number of update steps between two evaluations if eval_strategy="steps"
        logging_steps=1,               
        report_to="none",                
        load_best_model_at_end=True,     
        metric_for_best_model="eval_validation_wer",     
        greater_is_better=False,        # smaller wer is better than larger wer
    )

def whisper_cluster_steps(output_dir) -> Seq2SeqTrainingArguments :
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,  # Directory to save model checkpoints
        per_device_train_batch_size=8,  # Batch size per GPU
        per_device_eval_batch_size=8,   # Batch size for evaluation
        gradient_accumulation_steps=1,  # Accumulate gradients for effective batch size
        
        learning_rate=1e-4, # Learning rate
        warmup_steps=500,  # Number of warmup steps for learning rate scheduler
        max_steps=10000,

        gradient_checkpointing=True,
        fp16=True,

        eval_strategy="steps",  # Evaluate every few steps
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=50,
        remove_unused_columns=False, # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above

        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        report_to=["tensorboard"],
    )

# https://colab.research.google.com/github/Vaibhavs10/fast-whisper-finetuning/blob/main/Whisper_w_PEFT.ipynb#scrollTo=0ae3e9af-97b7-4aa0-ae85-20b23b5bcb3a

