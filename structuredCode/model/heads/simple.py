import torch
import torch.nn as nn

class SimpleHead(nn.Module):
    """
    Single linear layer: feat_dim → 1.
    Matches the paper's setup — just replace the classifier with one output.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)