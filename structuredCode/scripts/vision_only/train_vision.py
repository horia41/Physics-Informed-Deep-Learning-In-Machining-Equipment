"""
MATWI — Vision Baseline Ablation Training Script
==================================================
Runs a grid of experiments to find which undocumented paper settings
explain the gap between our ResNet50 replication (42 µm) and their
reported result (30 µm).

Ablation grid (9 experiments, reproducing §5.1 of the README):
  ResNet50, simple head, fixed LR:
    1. resnet50_imagenet_L1
    2. resnet50_imagenet_MSE
    3. resnet50_dataset_L1
    4. resnet50_dataset_MSE
  EfficientNetV2-S, simple head, fixed LR (2×2 norm × loss):
    5. efficientnetv2_dataset_MSE      <-- headline best (19.0 µm)
    6. efficientnetv2_imagenet_MSE
    7. efficientnetv2_dataset_L1
    8. efficientnetv2_imagenet_L1
  EfficientNetV2-S, MLP head + OneCycleLR (reference):
    9. efficientnetv2_imagenet_L1_mlp_sched

All experiments use:
  - Sets 1-13, paper split, 664 training images
  - No augmentation, no oversampling
  - AdamW (wd=1e-4), batch_size=32, 17 epochs
  - wear_cap=450, no zero-wear imputation, seed=42

Designed for SLURM (Snellius): use --only <name> to run one experiment per job.

Usage:
  # Run all 5 experiments sequentially:
  python train_vision.py --data-dir ./data/matwi ...

  # Run a single experiment (for SLURM array jobs):
  python train_vision.py --only resnet50_imagenet_L1 ...

  # List available experiment names:
  python train_vision.py --list-experiments
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
from experiments.vision_only_experiment import run_experiment, ExperimentConfig
from experiments.utils import set_seed, resolve_device

# ResNet50 ablations: 2 norms × 2 losses = 4, all simple head, fixed LR
# EfficientNetV2: our best setup for reference

DEFAULT_EXPERIMENTS = [
    # ── ResNet50 paper replication ablations ──
    ExperimentConfig(
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
    ExperimentConfig(
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
    ExperimentConfig(
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
    ExperimentConfig(
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
    ExperimentConfig(
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
    ExperimentConfig(
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
    ExperimentConfig(
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
    ExperimentConfig(
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
    ExperimentConfig(
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

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MATWI vision baseline ablation: head/norm/loss grid."
    )
    p.add_argument("--data-dir",    type=Path, required=True)
    p.add_argument("--labels-csv",  type=Path, required=True)
    p.add_argument("--sets-csv",    type=Path, required=True)
    p.add_argument("--output-dir",  type=Path, required=True)

    p.add_argument("--only", type=str, default=None,
                   help="Run only one experiment by name.")
    p.add_argument("--list-experiments", action="store_true",
                   help="Print available experiment names and exit.")

    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--device",           type=str,   default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-pretrained",    action="store_true")
    p.add_argument("--head-hidden-dim",  type=int,   default=256)
    p.add_argument("--dropout-backbone", type=float, default=0.3)
    p.add_argument("--dropout-head",     type=float, default=0.4)
    p.add_argument("--log-every",        type=int,   default=0)
    return p

def write_comparison(results: list[dict], output_dir: Path) -> None:
    rows = []
    for res in results:
        cfg = res["config"]
        for split, m in res["final_results"].items():
            rows.append({
                "experiment":                  res["name"],
                "backbone":                    cfg["backbone"],
                "head_type":                   cfg["head_type"],
                "normalisation":               cfg["normalisation"],
                "loss":                        cfg["loss"],
                "scheduler":                   cfg["use_scheduler"],
                "eval_split":                  split,
                "best_epoch":                  res["best_epoch"],
                "best_val_mae_um":             res["best_val_mae_um"],
                "mae_overall_um":              m["mae_overall_um"],
                "mae_flank_wear_um":           m["mae_flank_wear_um"],
                "mae_adhesion_um":             m["mae_adhesion_um"],
                "mae_flank_wear+adhesion_um":  m["mae_flank_wear+adhesion_um"],
                "n_total":                     m["n_total"],
            })

    # Paper reference
    rows.append({
        "experiment": "paper_baseline", "backbone": "resnet50",
        "head_type": "simple", "normalisation": "?", "loss": "?",
        "scheduler": False, "eval_split": "test", "best_epoch": "—",
        "best_val_mae_um": "—", "mae_overall_um": 30.0,
        "mae_flank_wear_um": 14.0, "mae_adhesion_um": 39.0,
        "mae_flank_wear+adhesion_um": 91.0, "n_total": "—",
    })

    df = pd.DataFrame(rows)
    path = output_dir / "comparison_summary.csv"
    df.to_csv(path, index=False)

    print(f"\n{'='*90}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*90}")
    # Show test results only for a clean overview
    test_df = df[df["eval_split"] == "test"]
    print(test_df.to_string(index=False))
    print(f"\n  Full results (all splits) → {path}")

def main() -> None:
    args = build_parser().parse_args()

    if args.list_experiments:
        print("Available experiments:")
        for cfg in DEFAULT_EXPERIMENTS:
            print(f"  {cfg.name:40s}  backbone={cfg.backbone}  head={cfg.head_type}  "
                  f"norm={cfg.normalisation}  loss={cfg.loss}  sched={cfg.use_scheduler}")
        sys.exit(0)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    experiments = []
    for cfg in DEFAULT_EXPERIMENTS:
        if args.only and cfg.name != args.only:
            continue
        experiments.append(cfg)

    if not experiments:
        available = [c.name for c in DEFAULT_EXPERIMENTS]
        print(f"No experiment matched --only='{args.only}'.")
        print(f"Available: {available}")
        sys.exit(1)

    print(f"Device       : {device}")
    print(f"Experiments  : {[c.name for c in experiments]}")
    print(f"Output dir   : {output_dir}")

    all_results = []
    for cfg in experiments:
        set_seed(args.seed)
        result = run_experiment(cfg, args, device, output_dir)
        all_results.append(result)

    write_comparison(all_results, output_dir)


if __name__ == "__main__":
    main()