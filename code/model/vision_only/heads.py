import torch
import torch.nn as nn
# ── Regression heads ──────────────────────────────────────────────────────────

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


class MLPHead(nn.Module):
    """
    Linear → ReLU → Dropout → Linear(1).
    No final activation — raw scalar output.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

