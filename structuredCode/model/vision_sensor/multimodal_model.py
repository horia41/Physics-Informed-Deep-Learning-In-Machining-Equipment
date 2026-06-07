"""
MATWI — Multimodal Wear Estimation Model (v2)
================================================
Updated based on vision ablation findings:
  - Simple linear heads throughout (no MLP) — critical for 647-sample dataset
  - Lightweight SensorEncoder for intermediate/late fusion
  - Three fusion modes: early, intermediate, late

Architecture decisions informed by:
  - Vision ablation: simple head (Linear→1) beat MLP head (42→19 µm)
  - Sensor sanity check: 40 features overfit in Ridge (train 21.7 vs test 56 µm)
    → minimal capacity for sensor branch to avoid amplifying overfitting

Reference baseline: efficientnetv2_dataset_MSE = 19.0 µm (vision-only, simple head)

Fusion modes:
  "early"        — sensor features concatenated directly to 1280-dim backbone
                   output. Head: Linear(1280 + n_feat, 1). Simplest integration.
  "intermediate" — sensor features projected by lightweight encoder (2-layer MLP,
                   no BN, no Dropout), then concatenated with backbone output.
                   Head: Linear(1280 + 64, 1).
  "late"         — separate image and sensor prediction branches combined by
                   a learned 2→1 linear combiner. Included for comparison only;
                   expected to underperform given weak standalone sensor signal.

Usage:
    from modelVisionSensor import MATWIMultimodalModel

    # Vision-only control (on 647-sample multimodal subset)
    model = MATWIMultimodalModel(use_sensors=False)

    # Early fusion with 25 features
    model = MATWIMultimodalModel(fusion_mode="early", n_sensor_features=25)

    # Intermediate fusion with all 40 features
    model = MATWIMultimodalModel(fusion_mode="intermediate", n_sensor_features=40)

    out = model(images, sensor_features)
    # out["wear"]         → (B, 1) predictions
    # out["image_embed"]  → (B, 1280) backbone features (for future PINN loss)
    # out["sensor_embed"] → (B, 64) encoded sensor features (intermediate/late)
    # out["pred_image"]   → (B, 1) image-branch prediction (late only)
    # out["pred_sensor"]  → (B, 1) sensor-branch prediction (late only)
"""

import torch
import torch.nn as nn
from typing import Literal, Optional
from model.vision_sensor.fusion.sensor_encoder import SensorEncoder

try:
    import timm
except ImportError:
    raise ImportError("timm is required.  pip install timm")

BACKBONE_NAME    = "tf_efficientnetv2_s.in21k_ft_in1k"
BACKBONE_FEATDIM = 1280

class MATWIMultimodalModel(nn.Module):
    """
    Multimodal wear estimation model with simple linear heads.

    Parameters
    ----------
    fusion_mode : "early" | "intermediate" | "late"
        How image and sensor features are combined.
    use_sensors : bool
        If False, sensor input is ignored. Used for the vision-only control
        experiment on the 647-sample multimodal subset.
    n_sensor_features : int
        Number of sensor features (25 for raw/top25, 40 for all40).
        Must match the feature_set used in the dataset class.
    sensor_encoder_dim : int
        Hidden and output dimension of the SensorEncoder.
        Only used for intermediate and late fusion.
    dropout_backbone : float
        Dropout applied inside timm backbone (overrides timm default).
    pretrained : bool
        Load ImageNet-21k→1k pretrained weights for the backbone.
    """

    def __init__(
        self,
        fusion_mode:        Literal["early", "intermediate", "late"] = "early",
        use_sensors:        bool  = True,
        n_sensor_features:  int   = 40,
        sensor_encoder_dim: int   = 64,
        dropout_backbone:   float = 0.3,
        pretrained:         bool  = True,
    ):
        super().__init__()

        self.fusion_mode = fusion_mode
        self.use_sensors = use_sensors

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=pretrained,
            num_classes=0,          # removes classifier → returns 1280-dim features
            drop_rate=dropout_backbone,
        )

        # ── Fusion-specific modules ───────────────────────────────────────────
        if use_sensors:
            if fusion_mode == "early":
                # Direct concatenation → single linear head
                # No sensor encoder — raw/selected features go straight to head
                self.head = nn.Linear(BACKBONE_FEATDIM + n_sensor_features, 1)

            elif fusion_mode == "intermediate":
                # Lightweight encoder → concatenation → single linear head
                self.sensor_encoder = SensorEncoder(
                    input_dim  = n_sensor_features,
                    hidden_dim = sensor_encoder_dim,
                    output_dim = sensor_encoder_dim,
                )
                self.head = nn.Linear(BACKBONE_FEATDIM + sensor_encoder_dim, 1)

            elif fusion_mode == "late":
                # Separate branches → learned combiner
                self.sensor_encoder = SensorEncoder(
                    input_dim  = n_sensor_features,
                    hidden_dim = sensor_encoder_dim,
                    output_dim = sensor_encoder_dim,
                )
                self.image_head  = nn.Linear(BACKBONE_FEATDIM, 1)
                self.sensor_head = nn.Linear(sensor_encoder_dim, 1)
                # Combiner initialised as simple average (0.5 + 0.5)
                self.combiner = nn.Linear(2, 1)
                with torch.no_grad():
                    self.combiner.weight.data.fill_(0.5)
                    nn.init.zeros_(self.combiner.bias)

            else:
                raise ValueError(
                    f"fusion_mode must be 'early', 'intermediate', or 'late'. "
                    f"Got '{fusion_mode}'"
                )
        else:
            # Vision-only: single linear head on backbone output
            self.head = nn.Linear(BACKBONE_FEATDIM, 1)

        # ── Summary ───────────────────────────────────────────────────────────
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode_str = fusion_mode if use_sensors else "none (vision-only)"
        n_feat_str = str(n_sensor_features) if use_sensors else "0"
        print(f"[MATWIMultimodalModel] fusion={mode_str} | "
              f"n_sensor_feat={n_feat_str} | pretrained={pretrained}")
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
        sensor_features : (B, n_sensor_features) float32 or None

        Returns
        -------
        dict with keys: wear, image_embed, sensor_embed, pred_image, pred_sensor
        """
        # ── Image branch ─────────────────────────────────────────────────────
        image_embed = self.backbone(images)   # (B, 1280)

        out = {
            "image_embed":  image_embed,
            "sensor_embed": None,
            "pred_image":   None,
            "pred_sensor":  None,
        }

        # ── Fusion ───────────────────────────────────────────────────────────
        if not self.use_sensors or sensor_features is None:
            # Vision-only path
            wear = self.head(image_embed)

        elif self.fusion_mode == "early":
            fused = torch.cat([image_embed, sensor_features], dim=1)
            wear  = self.head(fused)

        elif self.fusion_mode == "intermediate":
            sensor_embed = self.sensor_encoder(sensor_features)
            fused        = torch.cat([image_embed, sensor_embed], dim=1)
            wear         = self.head(fused)
            out["sensor_embed"] = sensor_embed

        elif self.fusion_mode == "late":
            pred_image   = self.image_head(image_embed)
            sensor_embed = self.sensor_encoder(sensor_features)
            pred_sensor  = self.sensor_head(sensor_embed)
            wear         = self.combiner(
                torch.cat([pred_image, pred_sensor], dim=1)
            )
            out["sensor_embed"] = sensor_embed
            out["pred_image"]   = pred_image
            out["pred_sensor"]  = pred_sensor

        out["wear"] = wear
        return out

