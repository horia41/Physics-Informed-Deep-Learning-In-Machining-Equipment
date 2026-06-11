from dataclasses import dataclass

@dataclass
class MultimodalExperimentConfig:
    name:               str
    fusion_mode:        str       # "none", "early", "intermediate", "late"
    feature_set:        str       # "none", "raw25", "top25", "all40"
    use_sensors:        bool
    # Task-3 knobs (default off → reproduces the original Stage 2 grid)
    modality_dropout_p: float = 0.0
    # Fixed settings (same for all experiments)
    set_range:          str   = "1-13"
    image_size:         tuple = (384, 384)
    epochs:             int   = 17
    lr:                 float = 3e-4
    normalisation:      str   = "dataset"
    loss:               str   = "MSE"
    weight_decay:       float = 1e-4
    batch_size:         int   = 32
    sensor_encoder_dim: int   = 64
    dropout_backbone:   float = 0.3
    gate_aircuts:       bool  = False   # remove tool-approach/retraction phases


BASE_EXPERIMENTS = [
    # ── Vision-only control (on 647-sample multimodal subset) ──
    MultimodalExperimentConfig(
        name         = "vision_only_647",
        fusion_mode  = "none",
        feature_set  = "none",
        use_sensors  = False,
    ),

    # ── Early fusion ──
    MultimodalExperimentConfig(
        name         = "early_raw25",
        fusion_mode  = "early",
        feature_set  = "raw25",
        use_sensors  = True,
    ),
    MultimodalExperimentConfig(
        name         = "early_top25",
        fusion_mode  = "early",
        feature_set  = "top25",
        use_sensors  = True,
    ),
    MultimodalExperimentConfig(
        name         = "early_all40",
        fusion_mode  = "early",
        feature_set  = "all40",
        use_sensors  = True,
    ),

    # ── Intermediate fusion ──
    MultimodalExperimentConfig(
        name         = "intermediate_raw25",
        fusion_mode  = "intermediate",
        feature_set  = "raw25",
        use_sensors  = True,
    ),
    MultimodalExperimentConfig(
        name         = "intermediate_top25",
        fusion_mode  = "intermediate",
        feature_set  = "top25",
        use_sensors  = True,
    ),
    MultimodalExperimentConfig(
        name         = "intermediate_all40",
        fusion_mode  = "intermediate",
        feature_set  = "all40",
        use_sensors  = True,
    ),

    # ── Late fusion ──
    MultimodalExperimentConfig(
        name         = "late_raw25",
        fusion_mode  = "late",
        feature_set  = "raw25",
        use_sensors  = True,
    ),
    MultimodalExperimentConfig(
        name         = "late_top25",
        fusion_mode  = "late",
        feature_set  = "top25",
        use_sensors  = True,
    ),
    MultimodalExperimentConfig(
        name         = "late_all40",
        fusion_mode  = "late",
        feature_set  = "all40",
        use_sensors  = True,
    ),
]

AIR_CUT_EXPERIMENTS = [
    # ── Air-cut ablation ───────────────────────────────────────────────────
    # Gated twins of the strongest fusion configs. Each is identical to its
    # ungated counterpart above except gate_aircuts=True (sensor features are
    # computed on the tool-engaged portion only, with length-invariant
    # relative band energy). Compare *_gated vs the base name to isolate the
    # air-cut effect. all40 is included because the relative-energy FFT fix
    # matters most for the frequency features that gating most affects.
    MultimodalExperimentConfig(
        name         = "intermediate_top25_gated",
        fusion_mode  = "intermediate",
        feature_set  = "top25",
        use_sensors  = True,
        gate_aircuts = True,
    ),
    MultimodalExperimentConfig(
        name         = "early_top25_gated",
        fusion_mode  = "early",
        feature_set  = "top25",
        use_sensors  = True,
        gate_aircuts = True,
    ),
    MultimodalExperimentConfig(
        name         = "intermediate_all40_gated",
        fusion_mode  = "intermediate",
        feature_set  = "all40",
        use_sensors  = True,
        gate_aircuts = True,
    )
]

TASK3_EXPERIMENTS = [
    MultimodalExperimentConfig(
        name               = "t3_intermediate_top25",
        fusion_mode        = "intermediate",
        feature_set        = "top25",
        use_sensors        = True,
        modality_dropout_p = 0.0,
    ),
    MultimodalExperimentConfig(
        name               = "t3_intermediate_top25_md30",
        fusion_mode        = "intermediate",
        feature_set        = "top25",
        use_sensors        = True,
        modality_dropout_p = 0.3,
    ),
    MultimodalExperimentConfig(
        name               = "t3_gated_top25",
        fusion_mode        = "gated",
        feature_set        = "top25",
        use_sensors        = True,
        modality_dropout_p = 0.0,
    ),
    MultimodalExperimentConfig(
        name               = "t3_gated_top25_md30",
        fusion_mode        = "gated",
        feature_set        = "top25",
        use_sensors        = True,
        modality_dropout_p = 0.3,
    ),
]

TASK3_AIR_CUT_EXPERIMENTS = [
    # Air-cut variant of the flagship gated model: identical to
    # t3_gated_top25_md30 except sensor features are computed on the
    # tool-engaged window only (relative band energy). Compare the two to
    # isolate the air-cut effect on the best fusion model.
    MultimodalExperimentConfig(
        name               = "t3_gated_top25_md30_gated",
        fusion_mode        = "gated",
        feature_set        = "top25",
        use_sensors        = True,
        modality_dropout_p = 0.3,
        gate_aircuts       = True,
    )
]

MULTIMODAL_EXPERIMENTS_V1 = BASE_EXPERIMENTS + AIR_CUT_EXPERIMENTS
MULTIMODAL_EXPERIMENTS_V2 = BASE_EXPERIMENTS + TASK3_EXPERIMENTS
MULTIMODAL_EXPERIMENTS_V3 = BASE_EXPERIMENTS + TASK3_EXPERIMENTS + TASK3_AIR_CUT_EXPERIMENTS