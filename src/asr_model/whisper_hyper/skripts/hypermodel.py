import torch
from torch import nn, Tensor


class MoE_HyperModel(nn.Module):

    def __init__(self, embedding_dim: int, context_dim: int, hidden_dim: int, in_f_A: int,  r: int, out_f_B: int, n_experts: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.in_f_A = in_f_A
        self.r = r
        self.out_f_B = out_f_B

        self.gating_model = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
            nn.Softmax()
        )

        self.experts = []
        for _ in range(n_experts) :
            expert = nn.Sequential(
                nn.Linear(embedding_dim + context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.experts.append(expert)
        
        self.decoder_A = nn.Linear(hidden_dim, in_f_A * r)
        self.decoder_B = nn.Linear(hidden_dim, r * out_f_B)


    def forward(self, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, c], dim=-1)
        weights: Tensor = self.gating_model(s)


        expert_outputs: list[Tensor] = [ expert(x) for expert in self.experts ]
        z: Tensor = weights.dot(expert_outputs) # TODO

        A: Tensor = self.decoder_A(z)
        B: Tensor = self.decoder_B(z)
        return A.reshape(self.in_f_A, self.r), B.reshape(self.r, self.out_f_B)    



class HyperModel(nn.Module) :
    pass