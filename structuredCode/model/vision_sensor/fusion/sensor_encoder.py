import torch.nn as nn
import torch

class SensorEncoder(nn.Module):
    """
    Lightweight sensor feature projection for intermediate/late fusion.

    Architecture: Linear(input, hidden) → ReLU → Linear(hidden, output)

    Design rationale:
      - No BatchNorm: 647 training samples, BN statistics would be noisy
      - No Dropout: the simple linear head already constrains capacity;
        adding dropout in the encoder over-regularises
      - Two layers (not one): allows learning nonlinear feature interactions
        (e.g., Fx_mid_energy x Fy_std) before fusion with image embedding
      - Hidden dim 64: proportional to input (25-40 features), avoids
        creating a bottleneck or excess capacity

    The sensor sanity check showed massive overfitting with Ridge (train 21.7
    vs test 56 µm). This encoder is deliberately minimal to avoid amplifying
    that tendency when combined with a 1280-dim image backbone.
    """

    def __init__(
        self,
        input_dim:  int = 40,
        hidden_dim: int = 64,
        output_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

