import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperModel,
)
from typing import Optional, List, Dict, Any
from peft import LoraConfig, get_peft_model
from asr_model.skripts.hypermodel import HyperModel


class CustomASRConfig(PretrainedConfig):

    def __init__(
        self,
        base_model_name: str = "openai/whisper-small",
        speech_encoder_model_name: str = "openai/whisper-large-v2",
        hyper_hidden_dim: int = 1024,
        freeze_speech_encoder: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.speech_encoder_model_name = speech_encoder_model_name
        self.hyper_hidden_dim = hyper_hidden_dim
        self.freeze_speech_encoder = freeze_speech_encoder

        
class HyperLoRAWhisperASRModel(PreTrainedModel):

    def __init__(self, config: CustomASRConfig):
        super().__init__(config)

        # Load the whisper-large encoder
        self.speech_encoder = WhisperModel.from_pretrained(config.speech_encoder_model_name).encoder
        if config.freeze_speech_encoder:
            for p in self.speech_encoder.parameters():
                p.requires_grad = False

        model = WhisperForConditionalGeneration.from_pretrained(config.base_model_name)
        config = LoraConfig(
            r=2,                        # rank of BA
            lora_alpha=16,              # Typical values: 8, 16, 32
            lora_dropout=0.05,          # dropout during learning
            target_modules=["fc1"],     # Important: matches the module suffix
            bias="none",                # none | lora_only | all
            task_type="SEQ_2_SEQ_LM"    # "SEQ_2_SEQ_LM" | "CAUSAL_LM" | "TOKEN_CLS" | "QUESTION_ANS" | "FEATURE_EXTRACTION" | "IMAGE_CLASSIFICATION" (Whisper is a sequence-to-sequence language model)
        )
        self.base_model = get_peft_model(model, config)
        self.lora_modules: List[nn.Module] = []
        for name, module in self.base_model.named_modules():
            if "fc1" in name and hasattr(module, "lora_A"):
                self.lora_modules.append(module)     

        A : nn.Linear = self.lora_modules[0].lora_A.default
        B : nn.Linear = self.lora_modules[0].lora_B.default
        in_f_A = A.in_features
        out_f_A = A.out_features
        in_f_B = B.in_features
        out_f_B = B.out_features
        
        for module in self.lora_modules :
            A : nn.Linear = module.lora_A.default
            B : nn.Linear = module.lora_B.default
            assert A.in_features == in_f_A and A.out_features == out_f_A
            assert B.in_features == in_f_B and B.out_features == out_f_B
        assert out_f_A == in_f_B
        
        speech_embedding_dim = self.speech_encoder.config.d_model  # should be output size of encoder
        oh_dim = len(self.lora_modules)
        self.hypermodel = HyperModel(             
            embedding_dim= speech_embedding_dim,
            context_dim= oh_dim,
            hidden_dim= config.hyper_hidden_dim,
            in_f_A= in_f_A,
            r= out_f_A,
            out_f_B= out_f_B,
            n_experts= 5 # TODO
        )
        self.feature_extractor_encoder = WhisperFeatureExtractor.from_pretrained(config.speech_encoder_model_name)
        self.feature_extractor_base = WhisperFeatureExtractor.from_pretrained(config.base_model_name)

        self.hooks = []


    def compute_speech_embedding(self, audio: torch.Tensor, sampling_rate: int = 16000) -> torch.Tensor:
        if not audio :
            raise ValueError("Provide audio to compute speech embedding.")
        
        device = next(self.speech_encoder.parameters()).device
            
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio  # assume numpy already

        # The processor returns a dict with input_features key
        inputs = self.feature_extractor_encoder(audio_np, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
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
            module.lora_A.default.data.copy_(new_A)
            module.lora_B.default.data.copy_(new_B) # TODO: maybe unsave
            # try this
            # with torch.no_grad():
            #     module.lora_A.default.copy_(new_A)

            return module_output

        return hook


    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        sampling_rate: int = 16000,
        **generate_kwargs,
    ) -> Dict[str, Any]:
        """
        Forward API:
          - Provide either `audio` (raw waveform Tensor[B, samples]) OR `input_features` (precomputed),
            plus (optionally) labels for computing loss (the base-model forward will compute loss).
        Returns whatever WhisperForConditionalGeneration returns (loss/logits/...), so it can be trained
        with standard HF Trainer or manual loss/backprop.
        """
        device = next(self.parameters()).device

        # 1) compute speech embedding
        speech_embedding = self.compute_speech_embedding(audio, input_features, sampling_rate)
        if speech_embedding.device != device:
            speech_embedding = speech_embedding.to(device)
        
        # 2) For each fc1 module, compute weights via hypermodel.
        weights: List[tuple[torch.Tensor, torch.Tensor]] = []
        for idx in range(len(self.lora_modules)):
            oh = self._create_adapter_one_hot(idx, device=device)
            lora_weights = self.hypermodel(speech_embedding, oh)
            weights.append(lora_weights)

        # 3) Register forward hooks on every fc1 module to use base_weight + delta
        self.hooks = []
        for module, (new_A, new_B) in zip(self.lora_modules, weights):            
            hook_fn = self._make_hook(new_A, new_B)
            h = module.register_forward_hook(hook_fn)
            self.hooks.append(h)
 
        # 4) Run whisper-small forward (generation/training)
        device = next(self.base_model.parameters()).device
        if audio:
            # compute input_features via feature_extractor (we already computed encoder embedding but need input_features for small model call)
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio
            inputs = self.feature_extractor_base(audio_np, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            small_input_features = inputs["input_features"].to(device)
            model_kwargs = {"input_features": small_input_features}
        elif input_features:
            model_kwargs = {"input_features": input_features.to(device)}
        else:
            raise ValueError("Unable to build inputs for whisper_small. Provide audio or input_features.")

        # If labels provided, include labels to compute loss.
        if labels :
            model_kwargs["labels"] = labels.to(device)

        outputs = self.base_model(**model_kwargs, **generate_kwargs)

        # 5) Remove hooks
        for h in self.hooks:
            h.remove()
        self.hooks = []

        return outputs


# Example usage:
if __name__ == "__main__":
    # Demonstration (not runnable in a plain environment unless transformers & model weights are available)
    config = CustomASRConfig(
        base_model_name="openai/whisper-small",
        speech_encoder_model_name="openai/whisper-large-v2",
        hyper_hidden_dim=512,
    )
    model = HyperLoRAWhisperASRModel(config)

    # To load from hub/local path:
    # model = HyperLoRAASRModel.from_pretrained("/path/to/saved/model_or_hf_repo_id")

    # Example training stub (batch_size==1 for this simple implementation)
    dummy_audio = torch.randn(1, 16000)  # one second of fake audio at 16 kHz
    output = model(audio=dummy_audio)
    print("Output keys:", output.keys())
