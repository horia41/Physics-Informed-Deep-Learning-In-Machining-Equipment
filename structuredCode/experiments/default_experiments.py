from vision_sensor_experiment import ExperimentConfig

DEFAULT_EXPERIMENTS = [
    # ── Vision-only control (on 647-sample multimodal subset) ──
    ExperimentConfig(
        name         = "vision_only_647",
        fusion_mode  = "none",
        feature_set  = "none",
        use_sensors  = False,
    ),

    # ── Early fusion ──
    ExperimentConfig(
        name         = "early_raw25",
        fusion_mode  = "early",
        feature_set  = "raw25",
        use_sensors  = True,
    ),
    ExperimentConfig(
        name         = "early_top25",
        fusion_mode  = "early",
        feature_set  = "top25",
        use_sensors  = True,
    ),
    ExperimentConfig(
        name         = "early_all40",
        fusion_mode  = "early",
        feature_set  = "all40",
        use_sensors  = True,
    ),

    # ── Intermediate fusion ──
    ExperimentConfig(
        name         = "intermediate_raw25",
        fusion_mode  = "intermediate",
        feature_set  = "raw25",
        use_sensors  = True,
    ),
    ExperimentConfig(
        name         = "intermediate_top25",
        fusion_mode  = "intermediate",
        feature_set  = "top25",
        use_sensors  = True,
    ),
    ExperimentConfig(
        name         = "intermediate_all40",
        fusion_mode  = "intermediate",
        feature_set  = "all40",
        use_sensors  = True,
    ),

    # ── Late fusion ──
    ExperimentConfig(
        name         = "late_raw25",
        fusion_mode  = "late",
        feature_set  = "raw25",
        use_sensors  = True,
    ),
    ExperimentConfig(
        name         = "late_top25",
        fusion_mode  = "late",
        feature_set  = "top25",
        use_sensors  = True,
    ),
    ExperimentConfig(
        name         = "late_all40",
        fusion_mode  = "late",
        feature_set  = "all40",
        use_sensors  = True,
    ),
]

AIRCUT_EXPERIMENTS = [
    # ── Air-cut ablation ───────────────────────────────────────────────────
    # Gated twins of the strongest fusion configs. Each is identical to its
    # ungated counterpart above except gate_aircuts=True (sensor features are
    # computed on the tool-engaged portion only, with length-invariant
    # relative band energy). Compare *_gated vs the base name to isolate the
    # air-cut effect. all40 is included because the relative-energy FFT fix
    # matters most for the frequency features that gating most affects.
    ExperimentConfig(
        name         = "intermediate_top25_gated",
        fusion_mode  = "intermediate",
        feature_set  = "top25",
        use_sensors  = True,
        gate_aircuts = True,
    ),
    ExperimentConfig(
        name         = "early_top25_gated",
        fusion_mode  = "early",
        feature_set  = "top25",
        use_sensors  = True,
        gate_aircuts = True,
    ),
    ExperimentConfig(
        name         = "intermediate_all40_gated",
        fusion_mode  = "intermediate",
        feature_set  = "all40",
        use_sensors  = True,
        gate_aircuts = True,
    )
]

# ── Task-3 grid: fusion improvements WITHOUT Taylor ───────────────────────────
# 2×2 controlled grid (modality dropout × {intermediate, gated}) on the best
# feature set from Stage 2 (top25). Each cell differs from its neighbours by
# exactly one variable. Compare against vision_only_647 (22.4 µm reference).
#
#   modality_dropout_p │ fusion_mode
#   ───────────────────┼──────────────
#         0.0          │ intermediate   (= Stage 2 best, re-run as task-3 anchor)
#         0.3          │ intermediate
#         0.0          │ gated
#         0.3          │ gated

TASK3_EXPERIMENTS = [
    ExperimentConfig(
        name               = "t3_intermediate_top25",
        fusion_mode        = "intermediate",
        feature_set        = "top25",
        use_sensors        = True,
        modality_dropout_p = 0.0,
    ),
    ExperimentConfig(
        name               = "t3_intermediate_top25_md30",
        fusion_mode        = "intermediate",
        feature_set        = "top25",
        use_sensors        = True,
        modality_dropout_p = 0.3,
    ),
    ExperimentConfig(
        name               = "t3_gated_top25",
        fusion_mode        = "gated",
        feature_set        = "top25",
        use_sensors        = True,
        modality_dropout_p = 0.0,
    ),
    ExperimentConfig(
        name               = "t3_gated_top25_md30",
        fusion_mode        = "gated",
        feature_set        = "top25",
        use_sensors        = True,
        modality_dropout_p = 0.3,
    ),
]