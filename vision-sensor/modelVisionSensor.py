"""
MATWI — Multimodal Wear Estimation Model
==========================================
EfficientNetV2-S backbone (timm) with three sensor fusion strategies.

Task:
    Regression — predict flank wear VB (normalised, [0,1]) from:
      - A single cropped image of the cutting edge (384×384 px)
      - 40 engineered sensor features from the same milling pass
    Target is wear_µm / 1000. Multiply predictions by 1000 for µm reporting.

Fusion strategies (set via fusion_mode):
    "early"         — 40 sensor features concatenated directly to the 1280-dim
                      backbone output. No sensor encoder. Simple, fast.
    "intermediate"  — sensor features first encoded by a small MLP (40→128→128),
                      then concatenated with the 1280-dim backbone output.
                      Main approach from the project plan.
    "late"          — image branch and sensor branch each produce an independent
                      wear prediction, combined by a learned 2→1 MLP combiner.

Backbone:
    efficientnetv2_s from timm, pretrained on ImageNet-21k → ImageNet-1k.
    Input: 384×384 RGB. Feature dim: 1280 (after global average pooling).
    Full fine-tuning from the start (deep transfer learning).

Usage:
    from model import MATWIWearModel

    # Image only (Stage 1 compatible, no sensor input used)
    model = MATWIWearModel(fusion_mode="intermediate", use_sensors=False)

    # Intermediate fusion (Stage 2 main)
    model = MATWIWearModel(fusion_mode="intermediate", use_sensors=True)

    # Early fusion
    model = MATWIWearModel(fusion_mode="early", use_sensors=True)

    # Late fusion
    model = MATWIWearModel(fusion_mode="late", use_sensors=True)

    # Forward pass — see training script for full usage
    out = model(images, sensor_features)
    # out["wear"]        → (B,1) raw linear predictions, normalised [0,1]
    # out["image_embed"] → (B,1280) backbone features, used by PINN loss
    # out["sensor_embed"]→ (B,128) encoded sensor features (intermediate/late)
    # out["pred_image"]  → (B,1) image branch prediction (late fusion only)
    # out["pred_sensor"] → (B,1) sensor branch prediction (late fusion only)
"""

import torch
import torch.nn as nn
from typing import Literal, Optional

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required. Install with: pip install timm"
    )

# ── Constants ─────────────────────────────────────────────────────────────────
BACKBONE_NAME   = "efficientnetv2_s"   # timm model name
BACKBONE_FEATDIM = 1280                # output dim after global avg pool
N_SENSOR_FEATURES = 40                # from DatasetClass_VisionSensors


# ══════════════════════════════════════════════════════════════════════════════
# Sub-modules
# ══════════════════════════════════════════════════════════════════════════════

class SensorEncoder(nn.Module):
    """
    Small MLP that projects 40 sensor features into a richer embedding.
    Used in intermediate fusion.

    Architecture: 40 → 128 → 128
    Each layer: Linear → BatchNorm1d → ReLU → Dropout(0.3)
    """
    def __init__(
        self,
        input_dim:  int = N_SENSOR_FEATURES,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
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


class RegressionHead(nn.Module):
    """
    Regression head: Linear → ReLU → Dropout → Linear (no final activation).

    Output is a raw scalar per sample — matches the paper's approach.
    No sigmoid or ReLU on the output so predictions can go slightly outside
    [0,1] during training; clamp when reporting µm values if needed.
    """
    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = 256,
        dropout:    float = 0.4,
    ):
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
        return self.net(x)   # (B, 1)


class LateFusionCombiner(nn.Module):
    """
    Learned combiner for late fusion.
    Takes two scalar predictions (image branch + sensor branch) and
    produces a single final prediction via a small MLP.

    Architecture: 2 → 16 → 1 (no final activation)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        # Initialise close to simple average so training starts stable
        with torch.no_grad():
            nn.init.kaiming_normal_(self.net[0].weight)
            nn.init.zeros_(self.net[0].bias)
            # Final layer: initialise close to equal weighting of both branches
            self.net[2].weight.data.fill_(0.5)
            nn.init.zeros_(self.net[2].bias)

    def forward(self, pred_image: torch.Tensor, pred_sensor: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pred_image, pred_sensor], dim=1)   # (B, 2)
        return self.net(x)                                 # (B, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Main model
# ══════════════════════════════════════════════════════════════════════════════

class MATWIWearModel(nn.Module):
    """
    Multimodal wear estimation model for the MATWI dataset.

    Parameters
    ----------
    fusion_mode : "early" | "intermediate" | "late"
        How image and sensor features are combined. See module docstring.

    use_sensors : bool
        If False, sensor input is ignored and only images are used.
        Equivalent to Stage 1 (vision-only baseline).
        When False, fusion_mode is irrelevant.

    sensor_hidden_dim : int
        Hidden and output dimension of the SensorEncoder MLP.
        Only used when fusion_mode="intermediate".
        Default 128.

    head_hidden_dim : int
        Hidden dimension of the regression head MLP.
        Default 256.

    dropout_backbone : float
        Dropout applied inside timm's classifier (overrides timm default).
        Default 0.3.

    dropout_head : float
        Dropout in the regression head.
        Default 0.4.

    pretrained : bool
        Load ImageNet pretrained weights for the backbone.
        Default True. Set False for ablations or testing.
    """

    def __init__(
        self,
        fusion_mode:       Literal["early", "intermediate", "late"] = "intermediate",
        use_sensors:       bool = True,
        sensor_hidden_dim: int = 128,
        head_hidden_dim:   int = 256,
        dropout_backbone:  float = 0.3,
        dropout_head:      float = 0.4,
        pretrained:        bool = True,
    ):
        super().__init__()

        self.fusion_mode  = fusion_mode
        self.use_sensors  = use_sensors

        # ── Backbone ──────────────────────────────────────────────────────────
        # timm's efficientnetv2_s with the classifier head removed.
        # num_classes=0 makes timm return the pooled feature vector (1280-dim)
        # instead of class logits.
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=pretrained,
            num_classes=0,          # removes classifier, returns 1280-dim features
            drop_rate=dropout_backbone,
        )

        # ── Fusion-specific modules ───────────────────────────────────────────
        if use_sensors:
            if fusion_mode == "early":
                # No sensor encoder — raw 40-dim features go straight to head
                head_input_dim = BACKBONE_FEATDIM + N_SENSOR_FEATURES  # 1320

            elif fusion_mode == "intermediate":
                self.sensor_encoder = SensorEncoder(
                    input_dim  = N_SENSOR_FEATURES,
                    hidden_dim = sensor_hidden_dim,
                    output_dim = sensor_hidden_dim,
                    dropout    = dropout_head,
                )
                head_input_dim = BACKBONE_FEATDIM + sensor_hidden_dim  # 1408

            elif fusion_mode == "late":
                # Sensor branch: its own independent regression head
                self.sensor_encoder = SensorEncoder(
                    input_dim  = N_SENSOR_FEATURES,
                    hidden_dim = sensor_hidden_dim,
                    output_dim = sensor_hidden_dim,
                    dropout    = dropout_head,
                )
                self.sensor_head  = RegressionHead(sensor_hidden_dim, head_hidden_dim, dropout_head)
                self.late_combiner = LateFusionCombiner()
                head_input_dim = BACKBONE_FEATDIM   # image head operates on backbone only

            else:
                raise ValueError(f"fusion_mode must be 'early', 'intermediate', or 'late'. Got '{fusion_mode}'")
        else:
            head_input_dim = BACKBONE_FEATDIM   # 1280, image only

        # ── Regression head (image branch, or sole head for early/intermediate) ─
        self.regression_head = RegressionHead(head_input_dim, head_hidden_dim, dropout_head)

        # ── Summary ───────────────────────────────────────────────────────────
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MATWIWearModel] backbone={BACKBONE_NAME} | fusion={fusion_mode if use_sensors else 'none (image only)'} | "
              f"pretrained={pretrained}")
        print(f"  Total params    : {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images:          torch.Tensor,
        sensor_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images : (B, 3, 384, 384) float32
            Cropped, normalised tool images.

        sensor_features : (B, 40) float32 or None
            Standardised sensor feature vectors.
            Required when use_sensors=True, ignored when use_sensors=False.

        Returns
        -------
        dict with keys:
            "wear"          : (B, 1) raw scalar predictions, normalised [0,1]
            "image_embed"   : (B, 1280) backbone features — used by PINN loss
                              and intermediate/late fusion sensor branch
            "sensor_embed"  : (B, sensor_hidden_dim) or None
                              Encoded sensor features (intermediate/late only)
            "pred_image"    : (B, 1) image-branch prediction (late fusion only)
            "pred_sensor"   : (B, 1) sensor-branch prediction (late fusion only)
        """
        # ── Image branch ─────────────────────────────────────────────────────
        image_embed = self.backbone(images)   # (B, 1280)

        out = {"image_embed": image_embed, "sensor_embed": None,
               "pred_image": None, "pred_sensor": None}

        # ── Fusion ───────────────────────────────────────────────────────────
        if not self.use_sensors or sensor_features is None:
            wear = self.regression_head(image_embed)

        elif self.fusion_mode == "early":
            # Concatenate raw sensor features directly to backbone output
            fused = torch.cat([image_embed, sensor_features], dim=1)   # (B, 1320)
            wear  = self.regression_head(fused)

        elif self.fusion_mode == "intermediate":
            # Encode sensors, concatenate with image embedding
            sensor_embed = self.sensor_encoder(sensor_features)        # (B, 128)
            fused        = torch.cat([image_embed, sensor_embed], dim=1)  # (B, 1408)
            wear         = self.regression_head(fused)
            out["sensor_embed"] = sensor_embed

        elif self.fusion_mode == "late":
            # Image branch prediction
            pred_image = self.regression_head(image_embed)             # (B, 1)

            # Sensor branch prediction
            sensor_embed = self.sensor_encoder(sensor_features)        # (B, 128)
            pred_sensor  = self.sensor_head(sensor_embed)              # (B, 1)

            # Combine
            wear = self.late_combiner(pred_image, pred_sensor)         # (B, 1)

            out["sensor_embed"] = sensor_embed
            out["pred_image"]   = pred_image
            out["pred_sensor"]  = pred_sensor

        out["wear"] = wear
        return out