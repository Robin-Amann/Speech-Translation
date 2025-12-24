import torch

# scalar weights (n,)
weights = torch.tensor([0.2, 0.5, 0.3])

# expert outputs: list of 1-D tensors, each shape (d,)
expert_outputs = [
    torch.tensor([1.0, 0.0, 2.0, 1.0]),
    torch.tensor([0.0, 1.0, 1.0, 0.0]),
    torch.tensor([2.0, 1.0, 0.0, 1.0]),
]


z = (weights[:, None] * torch.stack(expert_outputs, dim=0)).sum(dim=0)

print(z)