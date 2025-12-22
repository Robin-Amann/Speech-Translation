import torch
from torch import nn
from transformers import PreTrainedModel, WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperModel, WhisperConfig
from typing import Optional, List, Dict, Any
from peft import LoraConfig, get_peft_model
from typing import Type
from src.magic_strings import WHISPER_V3_MODEL_NAME
from torch.utils.hooks import RemovableHandle

class HyperLoRAWhisperASRModel(PreTrainedModel):

    def __init__(self, config: WhisperConfig, Hypermodel_cls: Type[nn.Module], **hypermodel_kwargs):
        super().__init__(config)

        self.speech_encoder = WhisperModel.from_pretrained(WHISPER_V3_MODEL_NAME).encoder

        # freeze speech encoder
        for p in self.speech_encoder.parameters():
            p.requires_grad = False

        # load base model
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_V3_MODEL_NAME)
        lora_config = LoraConfig(
            r=2,                        # rank of BA
            lora_alpha=16,              # Typical values: 8, 16, 32
            lora_dropout=0.05,          # dropout during learning
            target_modules=["fc1"],     # Important: matches the module suffix
            bias="none",                # none | lora_only | all
            task_type="SEQ_2_SEQ_LM"    # "SEQ_2_SEQ_LM" | "CAUSAL_LM" | "TOKEN_CLS" | "QUESTION_ANS" | "FEATURE_EXTRACTION" | "IMAGE_CLASSIFICATION" (Whisper is a sequence-to-sequence language model)
        )
        self.base_model = get_peft_model(model, lora_config)
        
        # get all LoRA layers and make sure they are the correct size
        self.lora_modules: List[nn.Module] = [ module for name, module in self.base_model.named_modules() if "fc1" in name and hasattr(module, "lora_A") ]

        A: nn.Linear
        B: nn.Linear
        A, B = self.lora_modules[0].lora_A.default, self.lora_modules[0].lora_B.default
        dim_in, r, dim_out = A.in_features, A.out_features, B.out_features
        assert A.out_features == B.in_features
        for module in self.lora_modules :
            A, B = module.lora_A.default, module.lora_B.default
            assert A.in_features == dim_in and A.out_features == r
            assert B.in_features == r and B.out_features == dim_out

        # freeze LoRA adapters
        for module in self.lora_modules:
            module.lora_A.default.requires_grad = False
            module.lora_B.default.requires_grad = False

        speech_embedding_dim = self.speech_encoder.config.d_model  # should be output size of encoder
        oh_dim = len(self.lora_modules)
        self.hypermodel = Hypermodel_cls(
            embedding_dim=speech_embedding_dim,
            context_dim=oh_dim,
            lora_dim=(dim_in, r, dim_out),
            **hypermodel_kwargs
        )

        self.feature_extractor_speech_encoder = WhisperFeatureExtractor.from_pretrained(WHISPER_V3_MODEL_NAME)
        self.feature_extractor_base = WhisperFeatureExtractor.from_pretrained(WHISPER_V3_MODEL_NAME)

        self.hooks = []


    def compute_speech_embedding(self, audio: Optional[torch.Tensor] = None, input_features: Optional[torch.Tensor] = None, sampling_rate: int = 16000) -> torch.Tensor:
        if audio is None and input_features is None :
            raise ValueError("Provide audio to compute speech embedding.")
        
        device = next(self.speech_encoder.parameters()).device  # why not self.speech_encoder.device ?
        
        if input_features is None :
            # TODO: check if that is necessary
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio  # assume numpy already

            # The processor returns a dict with input_features key
            inputs = self.feature_extractor_speech_encoder(audio_np, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            input_features = inputs["input_features"].to(device)  # shape (B, seq_len, feature_dim)

        if input_features.device != device:
            input_features = input_features.to(device)
        
        encoder_outputs = self.speech_encoder(input_features)  # BaseModelOutputWithPooling etc.
        # Some encoder outputs have last_hidden_state as .last_hidden_state, or .hidden_states
        if hasattr(encoder_outputs, "last_hidden_state") :
            hidden_states = encoder_outputs.last_hidden_state # (B, T, d_model)
        else :
            hidden_states = encoder_outputs[0] # (B, T, d_model)

        # Average over time dimension -> (B, d_model)
        embedding = hidden_states.mean(dim=1)

        return embedding


    def _create_adapter_one_hot(self, pos_idx: int, device: torch.device) -> torch.Tensor:
        oh = torch.zeros(len(self.lora_modules), device=device, dtype=torch.float32)
        oh[pos_idx] = 1.0
        return oh


    def _make_hook(self, new_A: nn.Parameter, new_B: nn.Parameter):
        def hook(module, module_input, module_output):
            module.lora_A.default = new_A
            module.lora_B.default = new_B
            return module_output

        return hook


    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        input_features_speech_embedding: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        sampling_rate: int = 16000,
        **generate_kwargs,
    ) -> Dict[str, Any]:
        if audio is None and input_features is None :
            raise ValueError("Unable to build inputs for whisper_small. Provide audio or input_features.")

        device = next(self.parameters()).device

        # 1) compute speech embedding
        speech_embedding = self.compute_speech_embedding(audio, input_features_speech_embedding, sampling_rate)
        if speech_embedding.device != device:
            speech_embedding = speech_embedding.to(device)
        
        # 2) For each fc1 module, compute weights via hypermodel and register hooks
        self.hooks: list[RemovableHandle] = []
        for idx, module in enumerate(self.lora_modules) :
            oh = self._create_adapter_one_hot(idx, device=device)
            # TODO: Make sure your hypermodel output tensors (new_A, new_B) require gradients, because you want gradients to flow through the hypernetwork.
            new_A, new_B = self.hypermodel(speech_embedding, oh)
            hook_fn = self._make_hook(new_A, new_B)
            h = module.register_forward_hook(hook_fn)
            self.hooks.append(h)
  
        # 3) Run forward (generation/training)
        if input_features :
            model_kwargs = {"input_features": input_features.to(device)}
        else :
            # TODO: check if that is necessary
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio
            inputs = self.feature_extractor_base(audio_np, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            input_features = inputs["input_features"].to(device)
            model_kwargs = {"input_features": input_features}

        if labels :
            model_kwargs["labels"] = labels.to(device)

        outputs = self.base_model(**model_kwargs, **generate_kwargs)

        # 4) Remove hooks
        for h in self.hooks:
            h.remove()
        self.hooks = []

        return outputs



# training loop
optimizer = torch.optim.AdamW(
    self.hypermodel.parameters(),
    lr=5e-5,
)

outputs = model(
    audio=batch_audio,
    labels=batch_labels,
)

loss = outputs.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()



# permanent hooks alternative:
#     def _create_permanent_hook(self, idx: int):
#         module = self.lora_modules[idx]

#         def hook(module, module_input, module_output):
#             # The forward pass will set these dynamically
#             oh = self._create_adapter_one_hot(idx, device=module.lora_A.default.device)
#             new_A, new_B = self.hypermodel(self.current_speech_embedding, oh)
#             with torch.no_grad():
#                 module.lora_A.default.copy_(new_A)
#                 module.lora_B.default.copy_(new_B)
#             return module_output

#         return module.register_forward_hook(hook)

#     in __init__:
#         self.hooks = [self._create_permanent_hook(idx) for idx in range(len(self.lora_modules))]

#     for each forward pass set self.current_speech_embedding
    


