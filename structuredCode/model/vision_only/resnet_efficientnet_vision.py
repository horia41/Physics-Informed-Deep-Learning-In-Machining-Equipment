"""
MATWI — Vision-Only Wear Estimation Model
===========================================
Supports two backbones and two head types for systematic comparison.

Backbones:
  1. ResNet50       — paper replication baseline (De Pauw et al., 2023)
  2. EfficientNetV2-S — our improved backbone (Stage 1)

Head types:
  "simple"  — Linear(feat_dim, 1). Matches the paper's description:
              "adjusted the output classifier to output only one class prediction"
  "mlp"     — Linear → ReLU → Dropout → Linear(1). Richer but adds parameters.

Usage:
    model = MATWIVisionModel(backbone="resnet50", head_type="simple")     # paper
    model = MATWIVisionModel(backbone="efficientnetv2_s", head_type="mlp") # ours
"""

import torch
import torch.nn as nn
from typing import Literal
from model.heads .simple import SimpleHead
from model.heads.mlp import MLPHead

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

# ── Main model ────────────────────────────────────────────────────────────────

class MATWIVisionModel(nn.Module):
    """
    Vision-only wear estimation model.

    Parameters
    ----------
    backbone : "resnet50" | "efficientnetv2_s"
    head_type : "simple" | "mlp"
        "simple" — Linear(feat_dim, 1). Paper replication.
        "mlp"    — Linear → ReLU → Dropout → Linear(1). Our experiments.
    pretrained : bool
    head_hidden_dim : int   (only used when head_type="mlp")
    dropout_backbone : float
    dropout_head : float    (only used when head_type="mlp")
    """

    def __init__(
        self,
        backbone:         Literal["resnet50", "efficientnetv2_s"] = "resnet50",
        head_type:        Literal["simple", "mlp"] = "simple",
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
        self.head_type     = head_type
        info = BACKBONES[backbone]

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            info["timm_name"],
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout_backbone,
        )
        self.feat_dim = info["feat_dim"]

        # ── Regression head ───────────────────────────────────────────────────
        if head_type == "simple":
            self.regression_head = SimpleHead(self.feat_dim)
        elif head_type == "mlp":
            self.regression_head = MLPHead(
                input_dim=self.feat_dim,
                hidden_dim=head_hidden_dim,
                dropout=dropout_head,
            )
        else:
            raise ValueError(f"head_type must be 'simple' or 'mlp', got '{head_type}'")

        # ── Summary ───────────────────────────────────────────────────────────
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MATWIVisionModel] backbone={backbone} | head={head_type} | "
              f"feat_dim={self.feat_dim} | pretrained={pretrained}")
        print(f"  Total params    : {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        image_embed = self.backbone(images)
        wear        = self.regression_head(image_embed)
        return {"wear": wear, "image_embed": image_embed}