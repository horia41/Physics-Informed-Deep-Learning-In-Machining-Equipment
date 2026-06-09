"""
MATWI — Multimodal Wear Estimation Model (v3)
================================================
Updated for Stage 3 / task 3 (improving fusion WITHOUT Taylor's equation).

New in v3 (vs v2):
  - Modality dropout: training-time flag that randomly zeroes the entire
    sensor feature vector for a fraction of samples. Works with ANY fusion
    mode. Targets the "modality competition" problem identified in Stage 2.
  - Gated fusion: a new fusion_mode that learns a per-sample scalar gate,
    letting the model down-weight unreliable sensor input and fall back to
    vision when sensors are noisy.

Carried over from v2:
  - Simple linear heads throughout (no MLP) — critical for the small dataset
  - Lightweight SensorEncoder (Linear -> ReLU -> Linear, no BN, no Dropout)
  - Fusion modes: early, intermediate, late

Reference baselines:
  - vision-only, full 664 samples            = 19.0 µm test MAE
  - vision-only, 647-sample multimodal subset = 22.4 µm test MAE  <-- compare here
  - best Stage 2 fusion (intermediate, top25) = 29.1 µm test MAE

Fusion modes:
  "early"        - sensor features concatenated directly to the 1280-dim
                   backbone output. Head: Linear(1280 + n_feat, 1).
  "intermediate" - sensor features projected by the SensorEncoder, then
                   concatenated with the backbone output.
                   Head: Linear(1280 + encoder_dim, 1).
  "late"         - separate image and sensor prediction branches combined by
                   a learned 2->1 linear combiner.
  "gated"        - sensor features encoded, then scaled by a learned per-sample
                   gate in [0,1] before concatenation. The gate sees both
                   modalities and can suppress the sensor branch per sample.
                   Head: Linear(1280 + encoder_dim, 1).

Modality dropout (modality_dropout_p > 0):
  During training only, each sample's sensor vector is zeroed with probability
  modality_dropout_p. No inverted-dropout scaling is applied (we want the model
  to see genuine "sensors absent" inputs, matching what missing data looks like
  at inference). Has no effect when use_sensors=False or during eval.

Usage:
    from modelVisionSensor import MATWIMultimodalModel

    # Vision-only control (647-sample subset)
    model = MATWIMultimodalModel(use_sensors=False)

    # Gated fusion, 25 features, with 30% modality dropout
    model = MATWIMultimodalModel(
        fusion_mode="gated", n_sensor_features=25, modality_dropout_p=0.3)

    out = model(images, sensor_features)
    # out["wear"]         -> (B, 1) predictions
    # out["image_embed"]  -> (B, 1280) backbone features
    # out["sensor_embed"] -> (B, encoder_dim) or None
    # out["gate"]         -> (B, 1) gate values in [0,1] (gated fusion only)
    # out["pred_image"]   -> (B, 1) image-branch prediction (late only)
    # out["pred_sensor"]  -> (B, 1) sensor-branch prediction (late only)
"""

import torch
import torch.nn as nn
from typing import Literal, Optional

try:
    import timm
except ImportError:
    raise ImportError("timm is required.  pip install timm")


# ── Constants ─────────────────────────────────────────────────────────────────

BACKBONE_NAME    = "tf_efficientnetv2_s.in21k_ft_in1k"
BACKBONE_FEATDIM = 1280

VALID_FUSION_MODES = ("early", "intermediate", "late", "gated")


# ══════════════════════════════════════════════════════════════════════════════
# Sub-modules
# ══════════════════════════════════════════════════════════════════════════════

class SensorEncoder(nn.Module):
    """
    Lightweight sensor feature projection for intermediate / late / gated fusion.

    Architecture: Linear(input, hidden) -> ReLU -> Linear(hidden, output)

    Deliberately minimal: no BatchNorm (647 samples make BN statistics noisy),
    no Dropout (the simple linear head already constrains capacity). Two layers
    allow some nonlinear feature interaction before fusion.
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


# ══════════════════════════════════════════════════════════════════════════════
# Main model
# ══════════════════════════════════════════════════════════════════════════════

class MATWIMultimodalModel(nn.Module):
    """
    Multimodal wear estimation model with simple linear heads.

    Parameters
    ----------
    fusion_mode : "early" | "intermediate" | "late" | "gated"
        How image and sensor features are combined.
    use_sensors : bool
        If False, sensor input is ignored (vision-only control). When False,
        fusion_mode and modality_dropout_p are irrelevant.
    n_sensor_features : int
        Number of sensor features (25 for raw25/top25, 40 for all40).
        Must match the feature_set used in the dataset class.
    sensor_encoder_dim : int
        Hidden / output dim of the SensorEncoder (intermediate / late / gated).
    modality_dropout_p : float
        Probability of zeroing a sample's whole sensor vector during training.
        0.0 disables it. Applied during training only.
    dropout_backbone : float
        Dropout inside the timm backbone.
    pretrained : bool
        Load ImageNet-21k -> 1k pretrained backbone weights.
    """

    def __init__(
        self,
        fusion_mode:        Literal["early", "intermediate", "late", "gated"] = "early",
        use_sensors:        bool  = True,
        n_sensor_features:  int   = 40,
        sensor_encoder_dim: int   = 64,
        modality_dropout_p: float = 0.0,
        dropout_backbone:   float = 0.3,
        pretrained:         bool  = True,
    ):
        super().__init__()

        if use_sensors and fusion_mode not in VALID_FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {VALID_FUSION_MODES}, got '{fusion_mode}'"
            )
        if not (0.0 <= modality_dropout_p < 1.0):
            raise ValueError(
                f"modality_dropout_p must be in [0, 1), got {modality_dropout_p}"
            )

        self.fusion_mode        = fusion_mode
        self.use_sensors        = use_sensors
        self.modality_dropout_p = modality_dropout_p

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=pretrained,
            num_classes=0,          # removes classifier -> 1280-dim features
            drop_rate=dropout_backbone,
        )

        # ── Fusion-specific modules ───────────────────────────────────────────
        if use_sensors:
            if fusion_mode == "early":
                # Raw features concatenated straight to the backbone output
                self.head = nn.Linear(BACKBONE_FEATDIM + n_sensor_features, 1)

            elif fusion_mode == "intermediate":
                self.sensor_encoder = SensorEncoder(
                    input_dim  = n_sensor_features,
                    hidden_dim = sensor_encoder_dim,
                    output_dim = sensor_encoder_dim,
                )
                self.head = nn.Linear(BACKBONE_FEATDIM + sensor_encoder_dim, 1)

            elif fusion_mode == "late":
                self.sensor_encoder = SensorEncoder(
                    input_dim  = n_sensor_features,
                    hidden_dim = sensor_encoder_dim,
                    output_dim = sensor_encoder_dim,
                )
                self.image_head  = nn.Linear(BACKBONE_FEATDIM, 1)
                self.sensor_head = nn.Linear(sensor_encoder_dim, 1)
                self.combiner    = nn.Linear(2, 1)
                with torch.no_grad():
                    self.combiner.weight.data.fill_(0.5)
                    nn.init.zeros_(self.combiner.bias)

            elif fusion_mode == "gated":
                # Encode sensors, then scale the embedding by a learned
                # per-sample gate before concatenating with the image.
                self.sensor_encoder = SensorEncoder(
                    input_dim  = n_sensor_features,
                    hidden_dim = sensor_encoder_dim,
                    output_dim = sensor_encoder_dim,
                )
                # Gate sees both modalities -> one scalar in [0,1] per sample.
                self.gate = nn.Linear(BACKBONE_FEATDIM + sensor_encoder_dim, 1)
                with torch.no_grad():
                    nn.init.zeros_(self.gate.bias)   # sigmoid(0)=0.5 -> neutral start
                self.head = nn.Linear(BACKBONE_FEATDIM + sensor_encoder_dim, 1)
        else:
            # Vision-only control
            self.head = nn.Linear(BACKBONE_FEATDIM, 1)

        # ── Summary ───────────────────────────────────────────────────────────
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode_str   = fusion_mode if use_sensors else "none (vision-only)"
        n_feat_str = str(n_sensor_features) if use_sensors else "0"
        md_str     = f"{modality_dropout_p}" if use_sensors else "n/a"
        print(f"[MATWIMultimodalModel] fusion={mode_str} | "
              f"n_sensor_feat={n_feat_str} | modality_dropout_p={md_str} | "
              f"pretrained={pretrained}")
        print(f"  Total params    : {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

    # ── Modality dropout ──────────────────────────────────────────────────────

    def _apply_modality_dropout(self, sensor_features: torch.Tensor) -> torch.Tensor:
        """
        Zero each sample's whole sensor vector with prob modality_dropout_p.
        Training only. No inverted-dropout scaling — we want genuine
        "sensors absent" inputs (matches missing data at inference).
        """
        if not self.training or self.modality_dropout_p <= 0.0:
            return sensor_features
        keep = (torch.rand(sensor_features.size(0), 1,
                           device=sensor_features.device)
                >= self.modality_dropout_p).to(sensor_features.dtype)
        return sensor_features * keep

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images:          torch.Tensor,
        sensor_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        images          : (B, 3, 384, 384) float32
        sensor_features : (B, n_sensor_features) float32 or None

        Returns dict: wear, image_embed, sensor_embed, gate, pred_image, pred_sensor
        """
        # ── Image branch ─────────────────────────────────────────────────────
        image_embed = self.backbone(images)   # (B, 1280)

        out = {
            "image_embed":  image_embed,
            "sensor_embed": None,
            "gate":         None,
            "pred_image":   None,
            "pred_sensor":  None,
        }

        # ── Vision-only path ─────────────────────────────────────────────────
        if not self.use_sensors or sensor_features is None:
            out["wear"] = self.head(image_embed)
            return out

        # ── Modality dropout (training only) ─────────────────────────────────
        sensor_features = self._apply_modality_dropout(sensor_features)

        # ── Fusion ───────────────────────────────────────────────────────────
        if self.fusion_mode == "early":
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

        elif self.fusion_mode == "gated":
            sensor_embed = self.sensor_encoder(sensor_features)
            gate_input   = torch.cat([image_embed, sensor_embed], dim=1)
            gate         = torch.sigmoid(self.gate(gate_input))    # (B, 1)
            gated_sensor = gate * sensor_embed                     # broadcast
            fused        = torch.cat([image_embed, gated_sensor], dim=1)
            wear         = self.head(fused)
            out["sensor_embed"] = sensor_embed
            out["gate"]         = gate

        out["wear"] = wear
        return out


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Model architecture test ===\n")

    configs = [
        ("vision-only",        dict(use_sensors=False)),
        ("early-25",           dict(fusion_mode="early",        n_sensor_features=25)),
        ("intermediate-40",    dict(fusion_mode="intermediate", n_sensor_features=40)),
        ("late-25",            dict(fusion_mode="late",         n_sensor_features=25)),
        ("gated-25",           dict(fusion_mode="gated",        n_sensor_features=25)),
        ("gated-25 + mdrop",   dict(fusion_mode="gated",        n_sensor_features=25,
                                    modality_dropout_p=0.3)),
        ("intermediate + mdrop", dict(fusion_mode="intermediate", n_sensor_features=25,
                                      modality_dropout_p=0.3)),
    ]

    for label, kwargs in configs:
        model = MATWIMultimodalModel(**kwargs, pretrained=False)
        n_feat = kwargs.get("n_sensor_features", 40)
        use_s  = kwargs.get("use_sensors", True)
        images  = torch.randn(4, 3, 384, 384)
        sensors = torch.randn(4, n_feat) if use_s else None

        # train mode (exercises modality dropout)
        model.train()
        out_train = model(images, sensors)
        # eval mode
        model.eval()
        out_eval = model(images, sensors)

        gate_str = ("gate=%.3f" % out_eval["gate"].mean().item()
                    if out_eval["gate"] is not None else "gate=n/a")
        print(f"  {label:22s}: wear={tuple(out_eval['wear'].shape)}  {gate_str}")
        print()
