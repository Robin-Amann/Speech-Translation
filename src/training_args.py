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
        per_device_train_batch_size=4,  # samples per batch (train)
        gradient_accumulation_steps=1,  # number of batches before parameter update. simulates larger batch size without the need for more memory
                                        # effective batch = per_device_train_batch_size * #devices * gradient_accumulation_steps
        learning_rate=1e-5,             
        warmup_steps=2,                # steps in which learning rate goes from 0 to 100%
        max_steps=10,                  # total number of optimizer update steps to perform
        gradient_checkpointing=True,    # Activates recomputation of intermediate activations during backpropagation to reduce GPU memory usage. 
                                        # This trades increased computation time for lower memory consumption.
        fp16=True,                      # half-precision (16-bit floating point)
        eval_strategy ="steps",         # no | steps | epoch
        per_device_eval_batch_size=4,   # samples per batch (eval)
        predict_with_generate=True,     # Whether to use generate to calculate generative metrics (required for generative metrics)
        generation_max_length=100,      # The max_length to use on each evaluation loop when predict_with_generate=True
        save_steps=2,                
        eval_steps=2,                 
        logging_steps=1,               
        report_to="none",                
        load_best_model_at_end=True,     
        metric_for_best_model="eval_validation_wer",     
        greater_is_better=False,        # smaller wer is better than larger wer
    )

def whisper_cluster_steps(output_dir) -> Seq2SeqTrainingArguments :
    return whisper_local_steps(output_dir)



