from transformers import WhisperForConditionalGeneration
from peft import PeftModel, LoraConfig, get_peft_model


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", cache_dir="./asr_model/checkpoints")

# Find target module names
for name, module in model.named_modules():
    if "fc1" in name:
        print(f"Module: {name}", module.in_features, module.out_features)

# Create the LoRA configuration
# W = W + alpha / r * BA
# standard initialization
# - A
#   - Initialized from a standard Kaiming/He uniform distribution (the same initialization used for normal linear layers).
#   - This provides small but non-zero initial weights.
# - B
#   - Always initialized to zeros
config = LoraConfig(
    r=2,                        # rank of BA
    lora_alpha=16,              # Typical values: 8, 16, 32
    lora_dropout=0.05,          # dropout during learning
    target_modules=["fc1"],     # Important: matches the module suffix
    bias="none",                # none | lora_only | all
    task_type="SEQ_2_SEQ_LM"    # "SEQ_2_SEQ_LM" | "CAUSAL_LM" | "TOKEN_CLS" | "QUESTION_ANS" | "FEATURE_EXTRACTION" | "IMAGE_CLASSIFICATION" (Whisper is a sequence-to-sequence language model)
)

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

# Inject LoRA adapters into those layers
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()
lora_model.save_pretrained("./asr_model/lora_init")

base = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    cache_dir="./asr_model/checkpoints"
)
lora_loaded = PeftModel.from_pretrained(base, "./asr_model/lora_init")