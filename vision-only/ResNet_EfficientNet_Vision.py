"""
MATWI — Vision-Only Wear Estimation Model
===========================================
Supports two backbones for systematic comparison:

  1. ResNet50       — paper replication baseline (De Pauw et al., 2023)
  2. EfficientNetV2-S — our improved backbone (Stage 1 of project plan)

Both use the same regression head so any performance difference is
attributable to the backbone alone.

Task:
    Regression — predict flank wear VB (normalised [0,1]) from a single
    cropped image of the cutting edge.
    Target = wear_µm / 1000.  Multiply predictions by 1000 for µm.

Architecture:
    backbone (timm, pretrained) → global avg pool → feature vector
    → RegressionHead(feat_dim → hidden → 1)

    No activation on the output — predictions can go slightly outside [0,1]
    during training.  Clamp when reporting µm if needed.

    The image embedding (backbone output) is returned alongside predictions
    for downstream use in Stage 3 (PINN physics loss).

Usage:
    from model_vision import MATWIVisionModel

    # Paper replication
    model = MATWIVisionModel(backbone="resnet50")

    # Our improved backbone
    model = MATWIVisionModel(backbone="efficientnetv2_s")

    out = model(images)          # images: (B, 3, H, W)
    out["wear"]                  # (B, 1) predictions
    out["image_embed"]           # (B, feat_dim) backbone features
"""

import torch
import torch.nn as nn
from typing import Literal

try:
    import timm
except ImportError:
    raise ImportError("timm is required.  pip install timm")


# ── Backbone registry ─────────────────────────────────────────────────────────

BACKBONES = {
    "resnet50": {
        "timm_name":    "resnet50.a1_in1k",
        "feat_dim":     2048,
        "default_size": 224,
    },
    "efficientnetv2_s": {
        "timm_name":    "tf_efficientnetv2_s.in21k_ft_in1k",
        "feat_dim":     1280,
        "default_size": 384,
    },
}


# ── Regression head ───────────────────────────────────────────────────────────

class RegressionHead(nn.Module):
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


# ── Main model ────────────────────────────────────────────────────────────────

class MATWIVisionModel(nn.Module):
    """
    Vision-only wear estimation model.

    Parameters
    ----------
    backbone : "resnet50" | "efficientnetv2_s"
        Which backbone to use.
        "resnet50"          — matches the paper (feat_dim=2048, input 224×224)
        "efficientnetv2_s"  — our upgrade   (feat_dim=1280, input 384×384)

    pretrained : bool
        Load ImageNet pretrained weights.  Default True.

    head_hidden_dim : int
        Hidden dim of the regression head MLP.  Default 256.

    dropout_backbone : float
        Dropout rate inside the timm backbone.  Default 0.3.

    dropout_head : float
        Dropout rate in the regression head.  Default 0.4.
    """

    def __init__(
        self,
        backbone:         Literal["resnet50", "efficientnetv2_s"] = "resnet50",
        pretrained:       bool  = True,
        head_hidden_dim:  int   = 256,
        dropout_backbone: float = 0.3,
        dropout_head:     float = 0.4,
    ):
        super().__init__()

        if backbone not in BACKBONES:
            raise ValueError(
                f"backbone must be one of {list(BACKBONES.keys())}, got '{backbone}'"
            )

        self.backbone_name = backbone
        info = BACKBONES[backbone]

        # ── Backbone ──────────────────────────────────────────────────────────
        # num_classes=0 removes the classifier head → returns pooled features
        self.backbone = timm.create_model(
            info["timm_name"],
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout_backbone,
        )
        self.feat_dim = info["feat_dim"]

        # ── Regression head ───────────────────────────────────────────────────
        self.regression_head = RegressionHead(
            input_dim=self.feat_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout_head,
        )

        # ── Summary ───────────────────────────────────────────────────────────
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MATWIVisionModel] backbone={backbone} ({info['timm_name']}) | "
              f"feat_dim={self.feat_dim} | pretrained={pretrained}")
        print(f"  Total params    : {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images : (B, 3, H, W) float32

        Returns
        -------
        dict with:
            "wear"        : (B, 1) raw scalar predictions
            "image_embed" : (B, feat_dim) backbone features
        """
        image_embed = self.backbone(images)              # (B, feat_dim)
        wear        = self.regression_head(image_embed)   # (B, 1)
        return {"wear": wear, "image_embed": image_embed}


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for bb in ["resnet50", "efficientnetv2_s"]:
        info = BACKBONES[bb]
        model = MATWIVisionModel(backbone=bb, pretrained=False)
        x = torch.randn(2, 3, info["default_size"], info["default_size"])
        out = model(x)
        print(f"  {bb}: wear={out['wear'].shape}, "
              f"embed={out['image_embed'].shape}\n")