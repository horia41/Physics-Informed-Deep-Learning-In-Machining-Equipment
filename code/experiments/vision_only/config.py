@dataclass
class VisionExperimentConfig:
    name:          str
    backbone:      str
    set_range:     str
    image_size:    tuple[int, int]
    epochs:        int
    lr:            float
    normalisation: str   = "imagenet"
    loss:          str   = "L1"         # "L1" or "MSE"
    head_type:     str   = "simple"     # "simple" or "mlp"
    augment:       bool  = False
    weight_decay:  float = 1e-4
    batch_size:    int   = 32
    use_scheduler: bool  = False

    # ── Experiment grid ───────────────────────────────────────────────────────────
# ResNet50 ablations: 2 norms × 2 losses = 4, all simple head, fixed LR
# EfficientNetV2: our best setup for reference

DEFAULT_VISION_EXPERIMENTS = [
    # ── ResNet50 paper replication ablations ──
    VisionExperimentConfig(
        name          = "resnet50_imagenet_L1",
        backbone      = "resnet50",
        set_range     = "1-13",
        image_size    = (224, 224),
        epochs        = 17,
        lr            = 3e-4,
        normalisation = "imagenet",
        loss          = "L1",
        head_type     = "simple",
        use_scheduler = False,
    ),
    VisionExperimentConfig(
        name          = "resnet50_imagenet_MSE",
        backbone      = "resnet50",
        set_range     = "1-13",
        image_size    = (224, 224),
        epochs        = 17,
        lr            = 3e-4,
        normalisation = "imagenet",
        loss          = "MSE",
        head_type     = "simple",
        use_scheduler = False,
    ),
    VisionExperimentConfig(
        name          = "resnet50_dataset_L1",
        backbone      = "resnet50",
        set_range     = "1-13",
        image_size    = (224, 224),
        epochs        = 17,
        lr            = 3e-4,
        normalisation = "dataset",
        loss          = "L1",
        head_type     = "simple",
        use_scheduler = False,
    ),
    VisionExperimentConfig(
        name          = "resnet50_dataset_MSE",
        backbone      = "resnet50",
        set_range     = "1-13",
        image_size    = (224, 224),
        epochs        = 17,
        lr            = 3e-4,
        normalisation = "dataset",
        loss          = "MSE",
        head_type     = "simple",
        use_scheduler = False,
    ),

    # ── EfficientNetV2-S, simple head, fixed LR (the 2×2 norm×loss grid) ──
    # These four reproduce §5.1 of the README, incl. the headline best result
    # efficientnetv2_dataset_MSE (19.0 µm). They were previously missing from
    # this grid even though the README reports them.
    VisionExperimentConfig(
        name          = "efficientnetv2_dataset_MSE",
        backbone      = "efficientnetv2_s",
        set_range     = "1-13",
        image_size    = (384, 384),
        epochs        = 17,
        lr            = 3e-4,
        normalisation = "dataset",
        loss          = "MSE",
        head_type     = "simple",
        use_scheduler = False,
    ),
    VisionExperimentConfig(
        name          = "efficientnetv2_imagenet_MSE",
        backbone      = "efficientnetv2_s",
        set_range     = "1-13",
        image_size    = (384, 384),
        epochs        = 17,
        lr            = 3e-4,
        normalisation = "imagenet",
        loss          = "MSE",
        head_type     = "simple",
        use_scheduler = False,
    ),
    VisionExperimentConfig(
        name          = "efficientnetv2_dataset_L1",
        backbone      = "efficientnetv2_s",
        set_range     = "1-13",
        image_size    = (384, 384),
        epochs        = 17,
        lr            = 3e-4,
        normalisation = "dataset",
        loss          = "L1",
        head_type     = "simple",
        use_scheduler = False,
    ),
    VisionExperimentConfig(
        name          = "efficientnetv2_imagenet_L1",
        backbone      = "efficientnetv2_s",
        set_range     = "1-13",
        image_size    = (384, 384),
        epochs        = 17,
        lr            = 3e-4,
        normalisation = "imagenet",
        loss          = "L1",
        head_type     = "simple",
        use_scheduler = False,
    ),

    # ── EfficientNetV2 MLP-head + OneCycle reference (rank 5 in §5.1) ──
    VisionExperimentConfig(
        name          = "efficientnetv2_imagenet_L1_mlp_sched",
        backbone      = "efficientnetv2_s",
        set_range     = "1-13",
        image_size    = (384, 384),
        epochs        = 17,
        lr            = 3e-4,
        normalisation = "imagenet",
        loss          = "L1",
        head_type     = "mlp",
        use_scheduler = True,
    ),
]
