import torch
from torch import nn, Tensor


# Hypermodel needs to have the following parameters (in that order)
# - embedding_dim=speech_embedding_dim,
# - context_dim=oh_dim,
# - lora_dim=(dim_in, r, dim_out),
# - **hypermodel_kwargs

# forward method
# - hypermodel(speech_embedding, oh)


class MoE_HyperModel(nn.Module):
    """
    Mixture-of-Experts hypernetwork that generates LoRA adapters.
    lora_dim = (f_in, r, f_out)
    """

    def __init__(self, embedding_dim: int, context_dim: int, lora_dim: tuple[int, int, int], n_experts: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.f_in, self.r, self.f_out = lora_dim
        self.n_experts = n_experts

        self.gating_model = nn.Sequential(
            nn.Linear(embedding_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_experts),
            nn.Softmax(dim=-1)
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim + context_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, 512),
            )
            for _ in range(n_experts)
        ])
        
        self.decoder_A = nn.Linear(512, self.f_in * self.r)
        self.decoder_B = nn.Linear(512, self.r * self.f_out)


    def _create_adapter_one_hot(self, pos_idx: int, device: torch.device) -> torch.Tensor:
        oh = torch.zeros(self.context_dim, device=device, dtype=torch.float32)
        oh[pos_idx] = 1.0
        return oh

    # output [M, B, in_f, r], [M, B, in_f, r]
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: speech embedding [B, embedding_dim]

        Returns
            :A: [context_dim, B, f_in, r]
            :B: [context_dim, B, r, f_out]
        """

        device = s.device
        weights: Tensor = self.gating_model(s)  # [B, n_experts]

        outputs: list[Tensor] = []  # length context_dim [B, 512]
        for idx in range(self.context_dim) :
            c = self._create_adapter_one_hot(idx, device) # [context_dim]
            if s.dim == 1 :
                c = c.unsqueeze(dim=0).repeat(s.dim, 1) # [B, context_dim]
            x = torch.cat([s, c], dim=-1) # [B, embedding_dim + context_dim]
            
            # expert -> [B, 512]            
            expert_outputs = torch.stack((expert(x) for expert in self.experts), dim=-1) # [B, 512, n_experts]
            z = (weights[:, None] * expert_outputs).sum(dim=-1) # [B, 512]
            outputs.append(z)
        
        Z = torch.stack(outputs, dim=0) # [context_dim, B, 512]

        A: Tensor = self.decoder_A(Z) # [context_dim, B, f_in * r]
        B: Tensor = self.decoder_B(Z) # [context_dim, B, r * f_out]
        return A.reshape(self.f_in, self.r), B.reshape(self.r, self.f_out)  # [context_dim, B, in_f, r], [context_dim, B, in_f, r]