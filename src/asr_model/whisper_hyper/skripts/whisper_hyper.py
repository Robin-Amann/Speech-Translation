import torch
from torch import nn
from transformers import PreTrainedModel, WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperModel, WhisperConfig
from typing import Optional, List, Dict, Any
from peft import LoraConfig, get_peft_model
from typing import Type
from src.magic_strings import WHISPER_V3_MODEL_NAME
from torch.utils.hooks import RemovableHandle
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer, WhisperDecoderLayer

# the idea here is to reimplement LoRA so that it is batch friendly
# instead of implementing y = (W + AB) * x we implement
# y_b = W * x_b  + (B_b * A_b) * x_b 
class FunctionalLoRALinear(nn.Module):

    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.base_linear = base_linear

        for p in self.base_linear.parameters():
            p.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # https://stackoverflow.com/questions/57540745/what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch 
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        # "If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer, 
        # you should register them as buffers. Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to u…"
        # persistent: whether the buffer is part of this module's state_dict (when the model is saved) 
        self.register_buffer("A", None, persistent=False)
        self.register_buffer("B", None, persistent=False)

    def forward(
        self,
        x: torch.Tensor,        # [B, T, in_f]
    ) -> torch.Tensor:
        # W * x_b
        y = self.base_linear(x)  # [B, T, out_f]
        if self.A is None or self.B is None:
            return y
        # (B_b * A_b) * x_b 
        tmp = torch.bmm(x, self.A)    # [B, T, r]
        delta = torch.bmm(tmp, self.B)  # [B, T, out_f]
        # this should be the same but I don't completely understand it
        # delta = torch.einsum("btd,bdr,bro->bto", x, A, B)

        return y + delta


class HyperLoRAWhisperASRModel(PreTrainedModel):

    def __init__(self, config: WhisperConfig, Hypermodel_cls: Type[nn.Module], lora_rank: int, **hypermodel_kwargs):
        super().__init__(config)

        self.speech_encoder = WhisperModel.from_pretrained(WHISPER_V3_MODEL_NAME).encoder
        for p in self.speech_encoder.parameters():
            p.requires_grad = False

        self.base_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_V3_MODEL_NAME)
        for p in self.base_model.parameters():
            p.requires_grad = False

        # replace fc1 layer with lora adapter
        for layer in self.base_model.get_encoder().layers :
            layer: WhisperEncoderLayer = layer
            lora = FunctionalLoRALinear(layer.fc1)
            layer.fc1 = lora

        for layer in self.base_model.get_decoder().layers :
            layer: WhisperDecoderLayer = layer
            lora = FunctionalLoRALinear(layer.fc1)
            layer.fc1 = lora

        # see whisper implementation
        f_in = config.d_model
        f_out = config.encoder_ffn_dim
        assert config.encoder_ffn_dim == config.decoder_ffn_dim
        
        self.hypermodel = Hypermodel_cls(
            embedding_dim=config.d_model,
            num_modules=len(self.base_model.get_encoder().layers) + len(self.base_model.get_decoder().layers),
            lora_dim=(f_in, lora_rank, f_out),
            **hypermodel_kwargs,
        )

        self.feature_extractor_speech_encoder = WhisperFeatureExtractor.from_pretrained(WHISPER_V3_MODEL_NAME)
        self.feature_extractor_base = WhisperFeatureExtractor.from_pretrained(WHISPER_V3_MODEL_NAME)


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

        speech_embedding = self.compute_speech_embedding(audio, input_features_speech_embedding, sampling_rate)
        if speech_embedding.device != device:
            speech_embedding = speech_embedding.to(device)
        
        new_A, new_B = self.hypermodel(speech_embedding)
        # A_all: [M, B, in_f, r]
        # B_all: [M, B, r, out_f]

        for i, layer in self.base_model.get_encoder().layers + self.base_model.get_decoder().layers :
            layer.fc1.A = new_A[i]
            layer.fc1.B = new_B[i]
            # lora.A = lora.A.to(x.device, x.dtype)
            # lora.B = lora.B.to(x.device, x.dtype)


        if input_features is not None :
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

        if labels is not None :
            model_kwargs["labels"] = labels.to(device)

        outputs = self.base_model(**model_kwargs, **generate_kwargs)

        return outputs
    



# from transformers import WhisperForConditionalGeneration
# from src.magic_strings import WHISPER_V3_MODEL_NAME, WHISPER_V3_CHECKPOINT_DIR

# model = WhisperForConditionalGeneration.from_pretrained(WHISPER_V3_MODEL_NAME, cache_dir=WHISPER_V3_CHECKPOINT_DIR)

# for name, module in model.named_modules():
#     print(name)

# whisper large v3 modules:
# model
#     model.encoder
#         model.encoder.conv1
#         model.encoder.conv2
#         model.encoder.embed_positions
#         model.encoder.layers [n:0-31]
#             model.encoder.layers.n
#                 model.encoder.layers.n.self_attn
#                     model.encoder.layers.n.self_attn.k_proj
#                     model.encoder.layers.n.self_attn.v_proj
#                     model.encoder.layers.n.self_attn.q_proj
#                     model.encoder.layers.n.self_attn.out_proj
#                     model.encoder.layers.n.self_attn_layer_norm
#                 model.encoder.layers.n.activation_fn
#                 model.encoder.layers.n.fc1
#                 model.encoder.layers.n.fc2
#                 model.encoder.layers.n.final_layer_norm
#             model.encoder.layer_norm
#     model.decoder
#         model.decoder.embed_tokens
#         model.decoder.embed_positions
#         model.decoder.layers [n:0-31]
#             model.decoder.layers.n
#                 model.decoder.layers.n.self_attn
#                     model.decoder.layers.n.self_attn.k_proj
#                     model.decoder.layers.n.self_attn.v_proj
#                     model.decoder.layers.n.self_attn.q_proj
#                     model.decoder.layers.n.self_attn.out_proj
#                 model.decoder.layers.n.activation_fn
#                 model.decoder.layers.n.self_attn_layer_norm
#                 model.decoder.layers.n.encoder_attn
#                     model.decoder.layers.n.encoder_attn.k_proj
#                     model.decoder.layers.n.encoder_attn.v_proj
#                     model.decoder.layers.n.encoder_attn.q_proj
#                     model.decoder.layers.n.encoder_attn.out_proj
#                 model.decoder.layers.n.encoder_attn_layer_norm
#                 model.decoder.layers.n.fc1
#                 model.decoder.layers.n.fc2
#                 model.decoder.layers.n.final_layer_norm
#         model.decoder.layer_norm
# proj_out