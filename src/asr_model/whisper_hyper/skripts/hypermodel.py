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

    def __init__(self, embedding_dim: int, context_dim: int, lora_dim: tuple[int, int, int], n_experts: int):
        "lora_dim = (in, r, out)"

        super().__init__()

        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.lora_dim = lora_dim

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
        
        self.decoder_A = nn.Linear(512, self.lora_dim[0] * self.lora_dim[1])
        self.decoder_B = nn.Linear(512, self.lora_dim[1] * self.lora_dim[2])


    def forward(self, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, c], dim=-1)

        weights: Tensor = self.gating_model(s)
        expert_outputs = torch.stack((expert(x) for expert in self.experts), dim=0)
        z = (weights[:, None] * expert_outputs).sum(dim=0)

        A: Tensor = self.decoder_A(z)
        B: Tensor = self.decoder_B(z)
        return A.reshape(self.lora_dim[0], self.lora_dim[1]), B.reshape(self.lora_dim[1], self.lora_dim[2])


    def supervised_step(
        self,
        speech_embedding: torch.Tensor,
        context: torch.Tensor,
        target_A: torch.Tensor,
        target_B: torch.Tensor,
    ) -> torch.Tensor:
        """
        speech_embedding: (B, D)
        context: (B, C)
        target_A: (B, in, r)
        target_B: (B, r, out)
        """

        pred_A, pred_B = self(speech_embedding, context)

        loss_A = torch.mean((pred_A - target_A) ** 2)
        loss_B = torch.mean((pred_B - target_B) ** 2)

        return loss_A + loss_B


# training loop
optimizer = torch.optim.Adam(hypermodel.parameters(), lr=1e-4)

for batch in dataloader:
    s, c, A_gt, B_gt = batch
    loss = hypermodel.supervised_step(s, c, A_gt, B_gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class HyperModel(nn.Module) :
    pass