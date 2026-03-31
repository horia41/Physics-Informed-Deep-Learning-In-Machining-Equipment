"""
MATWI — Vision Baseline Training Script
=========================================
Produces the comparison ladder needed to validate the pipeline and
quantify each component's contribution:

  Experiment 1  resnet50,        1-13, 224×224  →  paper replication
  Experiment 2  efficientnetv2,  1-13, 384×384  →  backbone upgrade

Both use the paper's settings where known:
  - LR 3e-4, 17 epochs, no augmentation, ImageNet normalisation
  - L1 loss (MAE), AdamW optimiser
  - Sets 1-13, paper train/val/test split
  - wear_cap 450 µm, no zero-wear imputation

Expected result for Experiment 1:
  Overall ~30 µm | Flank ~14 µm | Adhesion ~39 µm | F+A ~91 µm
  (matching De Pauw et al. 2023, Table 3 — regression baseline)

If Experiment 1 is far from these numbers, the pipeline has an issue
that must be debugged before proceeding to sensor fusion or PINN.

Outputs (under --output-dir/):
  {experiment_name}/history.csv      — per-epoch train/val metrics
  {experiment_name}/best_model.pt    — best checkpoint by val MAE
  {experiment_name}/results.json     — final val/test MAE breakdown
  comparison_summary.csv             — side-by-side comparison table

Usage:
  python train_vision_baseline.py \\
    --data-dir   ./data/matwi \\
    --labels-csv ./data/matwi/labels.csv \\
    --sets-csv   ./data/matwi/sets.csv \\
    --output-dir ./runs/vision_baseline

  # Run only the paper replication:
  python train_vision_baseline.py ... --only resnet50_1-13

  # Run only EfficientNet:
  python train_vision_baseline.py ... --only efficientnetv2_1-13
"""

from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Local imports (resolve relative to this script) ──────────────────────────

import importlib.util

_HERE = Path(__file__).resolve().parent


def _load_module(name: str, candidates: list[Path]):
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(name, path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
                return mod
    raise FileNotFoundError(
        f"Could not find '{name}'. Searched:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


_DS = _load_module("dataset_module", [
    _HERE / "DatasetClass_Vision.py",
    _HERE / "DatasetClassVision.py",
])
_MDL = _load_module("model_module", [
    _HERE / "ResNet_EfficientNet_Vision.py",
    _HERE / "modelVision.py",
])

MATWIVisionDataset = _DS.MATWIVisionDataset
MATWIVisionModel   = _MDL.MATWIVisionModel
BACKBONES          = _MDL.BACKBONES


# ── Constants ─────────────────────────────────────────────────────────────────

WEAR_CAP    = 450.0
WEAR_TYPES  = ["flank_wear", "adhesion", "flank_wear+adhesion"]


@dataclass
class ExperimentConfig:
    """One row of the comparison ladder."""
    name:          str
    backbone:      str
    set_range:     str
    image_size:    tuple[int, int]
    epochs:        int
    lr:            float
    normalisation: str   = "imagenet"
    augment:       bool  = False
    weight_decay:  float = 1e-4
    batch_size:    int   = 32
    use_scheduler: bool = True


# The two key experiments
DEFAULT_EXPERIMENTS = [
    ExperimentConfig(
        name       = "resnet50_1-13",
        backbone   = "resnet50",
        set_range  = "1-13",
        image_size = (224, 224),
        epochs     = 17,
        lr         = 3e-4,
        use_scheduler = False,
    ),
    ExperimentConfig(
        name       = "efficientnetv2_1-13",
        backbone   = "efficientnetv2_s",
        set_range  = "1-13",
        image_size = (384, 384),
        epochs     = 17,
        lr         = 3e-4,
        use_scheduler = True,
    ),
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


def predictions_to_um(pred: torch.Tensor) -> torch.Tensor:
    """Convert normalised predictions to µm, clamped to [0, WEAR_CAP]."""
    return (pred.squeeze(-1) * 1000.0).clamp(0.0, WEAR_CAP)


def compute_mae(
    pred_um:   np.ndarray,
    target_um: np.ndarray,
    types:     list[str],
) -> dict[str, Any]:
    """Compute overall and per-wear-type MAE in µm."""
    abs_err = np.abs(pred_um - target_um)
    out: dict[str, Any] = {
        "mae_overall_um": float(abs_err.mean()) if len(abs_err) else math.nan,
        "n_total":        len(target_um),
    }
    for wt in WEAR_TYPES:
        mask = np.array([t == wt for t in types], dtype=bool)
        out[f"mae_{wt}_um"] = float(abs_err[mask].mean()) if mask.any() else math.nan
        out[f"n_{wt}"]      = int(mask.sum())
    return out


def fmt_metrics(m: dict[str, Any]) -> str:
    return (
        f"overall={m['mae_overall_um']:.2f}µm  "
        f"flank={m['mae_flank_wear_um']:.2f}  "
        f"adh={m['mae_adhesion_um']:.2f}  "
        f"f+a={m['mae_flank_wear+adhesion_um']:.2f}  "
        f"(n={m['n_total']})"
    )


# ── Data ──────────────────────────────────────────────────────────────────────

def build_loaders(
    cfg:        ExperimentConfig,
    data_dir:   Path,
    labels_csv: Path,
    sets_csv:   Path,
    num_workers: int,
) -> dict[str, DataLoader]:
    """Build train / val / test loaders.  Unseen only for 1-13."""
    common = dict(
        data_dir         = data_dir,
        labels_csv       = labels_csv,
        sets_csv         = sets_csv,
        set_range        = cfg.set_range,
        normalisation    = cfg.normalisation,
        wear_cap         = WEAR_CAP,
        impute_zero_wear = False,
        image_size       = cfg.image_size,
    )

    # Training set
    train_ds = MATWIVisionDataset(**common, split="train", augment=cfg.augment)
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size  = cfg.batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = True,
        ),
    }

    # Eval splits
    eval_splits = ["val", "test"]
    if cfg.set_range == "1-13":
        eval_splits.append("unseen")

    for split in eval_splits:
        ds = MATWIVisionDataset(**common, split=split, augment=False)
        if len(ds) == 0:
            print(f"  [warn] split='{split}' is empty, skipping.")
            continue
        loaders[split] = DataLoader(
            ds,
            batch_size  = cfg.batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

    sizes = "  |  ".join(f"{k}: {len(v.dataset)}" for k, v in loaders.items())
    print(f"  Loaders: {sizes}")
    return loaders


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    criterion: nn.Module,
    device:    torch.device,
    log_every: int = 0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae  = 0.0
    n = 0

    for step, batch in enumerate(loader, 1):
        images    = batch["image"].to(device, non_blocking=True)
        target    = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out  = model(images)
        loss = criterion(out["wear"], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        pred_um   = predictions_to_um(out["wear"].detach())
        batch_mae = torch.abs(pred_um - target_um).mean().item()
        bsz       = images.size(0)

        total_loss += loss.item() * bsz
        total_mae  += batch_mae  * bsz
        n          += bsz

        if log_every > 0 and step % log_every == 0:
            lr = (scheduler.get_last_lr()[0] if scheduler
                  else optimizer.param_groups[0]["lr"])
            print(f"    step {step:04d}/{len(loader)}  "
                  f"loss={loss.item():.5f}  mae={batch_mae:.1f}µm  lr={lr:.2e}")

    return {"loss": total_loss / max(n, 1), "mae_um": total_mae / max(n, 1)}


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> dict[str, Any]:
    model.eval()
    all_pred, all_target, all_types = [], [], []
    total_loss = 0.0
    n = 0

    for batch in loader:
        images    = batch["image"].to(device, non_blocking=True)
        target    = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].cpu().numpy()

        out  = model(images)
        loss = criterion(out["wear"], target)

        all_pred.append(predictions_to_um(out["wear"]).cpu().numpy())
        all_target.append(target_um)
        all_types.extend(list(batch["type"]))

        bsz         = images.size(0)
        total_loss  += loss.item() * bsz
        n           += bsz

    if not all_pred:
        empty = {f"mae_{wt}_um": math.nan for wt in WEAR_TYPES}
        empty.update({"mae_overall_um": math.nan, "n_total": 0, "loss": math.nan})
        return empty

    metrics = compute_mae(np.concatenate(all_pred),
                          np.concatenate(all_target),
                          all_types)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


# ── Single experiment ─────────────────────────────────────────────────────────

def run_experiment(
    cfg:        ExperimentConfig,
    args:       argparse.Namespace,
    device:     torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT: {cfg.name}")
    print(f"  backbone={cfg.backbone}  set_range={cfg.set_range}  "
          f"image={cfg.image_size}  epochs={cfg.epochs}  lr={cfg.lr}")
    print(f"{'='*80}")

    run_dir = output_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = build_loaders(
        cfg         = cfg,
        data_dir    = args.data_dir,
        labels_csv  = args.labels_csv,
        sets_csv    = args.sets_csv,
        num_workers = args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MATWIVisionModel(
        backbone         = cfg.backbone,
        pretrained       = not args.no_pretrained,
        head_hidden_dim  = args.head_hidden_dim,
        dropout_backbone = args.dropout_backbone,
        dropout_head     = args.dropout_head,
    ).to(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
    )

    if cfg.use_scheduler:
        steps_per_epoch = len(loaders["train"])
        total_steps = cfg.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        print(f"  Scheduler: OneCycleLR (max_lr={cfg.lr})")
    else:
        scheduler = None
        print(f"  Scheduler: None (fixed LR={cfg.lr})")

    criterion = nn.L1Loss()

    # ── Training loop ─────────────────────────────────────────────────────────
    history:     list[dict] = []
    best_state:  Optional[dict] = None
    best_val_mae = float("inf")
    best_epoch   = -1

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, loaders["train"], optimizer, scheduler,
            criterion, device, log_every=args.log_every,
        )
        val_m = evaluate(model, loaders["val"], criterion, device)
        elapsed = time.time() - t0

        lr_now = (scheduler.get_last_lr()[0] if scheduler
                  else optimizer.param_groups[0]["lr"])

        print(
            f"  [{epoch:02d}/{cfg.epochs}]  "
            f"train loss={train_m['loss']:.5f} mae={train_m['mae_um']:.1f}µm  |  "
            f"val {fmt_metrics(val_m)}  "
            f"lr={lr_now:.2e}  ({elapsed:.1f}s)"
        )

        history.append({
            "epoch":                          epoch,
            "train_loss":                     train_m["loss"],
            "train_mae_um":                   train_m["mae_um"],
            "val_loss":                       val_m["loss"],
            "val_mae_overall_um":             val_m["mae_overall_um"],
            "val_mae_flank_wear_um":          val_m["mae_flank_wear_um"],
            "val_mae_adhesion_um":            val_m["mae_adhesion_um"],
            "val_mae_flank_wear+adhesion_um": val_m["mae_flank_wear+adhesion_um"],
            "lr":                             lr_now,
            "seconds":                        elapsed,
        })

        # Checkpoint best
        if val_m["mae_overall_um"] < best_val_mae:
            best_val_mae = val_m["mae_overall_um"]
            best_epoch   = epoch
            best_state   = deepcopy(model.state_dict())
            torch.save({
                "epoch":            epoch,
                "model_state_dict": best_state,
                "val_mae_um":       best_val_mae,
                "backbone":         cfg.backbone,
                "set_range":        cfg.set_range,
                "image_size":       cfg.image_size,
            }, run_dir / "best_model.pt")
            print(f"    ✓ new best val_mae={best_val_mae:.2f}µm")

    # ── Save history ──────────────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # ── Final evaluation with best model ──────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best model (epoch {best_epoch}, "
              f"val={best_val_mae:.2f}µm)")

    print(f"\n  --- Final results: {cfg.name} ---")
    final_results: dict[str, dict] = {}
    for split in [k for k in ("val", "test", "unseen") if k in loaders]:
        m = evaluate(model, loaders[split], criterion, device)
        final_results[split] = m
        print(f"  [{split:7s}] {fmt_metrics(m)}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    with open(run_dir / "results.json", "w") as fh:
        json.dump({
            "experiment":      cfg.name,
            "backbone":        cfg.backbone,
            "set_range":       cfg.set_range,
            "image_size":      list(cfg.image_size),
            "epochs":          cfg.epochs,
            "lr":              cfg.lr,
            "best_epoch":      best_epoch,
            "best_val_mae_um": best_val_mae,
            "splits": {
                split: {k: v for k, v in m.items()
                        if not isinstance(v, np.ndarray)}
                for split, m in final_results.items()
            },
        }, fh, indent=2)

    return {
        "name":            cfg.name,
        "backbone":        cfg.backbone,
        "set_range":       cfg.set_range,
        "best_epoch":      best_epoch,
        "best_val_mae_um": best_val_mae,
        "final_results":   final_results,
    }


# ── Comparison summary ────────────────────────────────────────────────────────

def write_comparison(results: list[dict], output_dir: Path) -> None:
    """Write a CSV + printed table comparing all experiments."""
    rows = []
    for res in results:
        for split, m in res["final_results"].items():
            rows.append({
                "experiment":                  res["name"],
                "backbone":                    res["backbone"],
                "set_range":                   res["set_range"],
                "eval_split":                  split,
                "best_epoch":                  res["best_epoch"],
                "best_val_mae_um":             res["best_val_mae_um"],
                "mae_overall_um":              m["mae_overall_um"],
                "mae_flank_wear_um":           m["mae_flank_wear_um"],
                "mae_adhesion_um":             m["mae_adhesion_um"],
                "mae_flank_wear+adhesion_um":  m["mae_flank_wear+adhesion_um"],
                "n_total":                     m["n_total"],
            })

    # Add paper reference row for context
    rows.append({
        "experiment":                  "paper_regression_baseline",
        "backbone":                    "resnet50",
        "set_range":                   "1-13",
        "eval_split":                  "test",
        "best_epoch":                  "—",
        "best_val_mae_um":             "—",
        "mae_overall_um":              30.0,
        "mae_flank_wear_um":           14.0,
        "mae_adhesion_um":             39.0,
        "mae_flank_wear+adhesion_um":  91.0,
        "n_total":                     "—",
    })

    df = pd.DataFrame(rows)
    path = output_dir / "comparison_summary.csv"
    df.to_csv(path, index=False)

    print(f"\n{'='*90}")
    print("  COMPARISON SUMMARY  (paper reference included for context)")
    print(f"{'='*90}")
    print(df.to_string(index=False))
    print(f"\n  Saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MATWI vision baseline: ResNet50 (paper) vs EfficientNetV2 (ours)."
    )
    p.add_argument("--data-dir",    type=Path, required=True)
    p.add_argument("--labels-csv",  type=Path, required=True)
    p.add_argument("--sets-csv",    type=Path, required=True)
    p.add_argument("--output-dir",  type=Path, required=True)

    p.add_argument("--only", type=str, default=None,
                   help="Run only one experiment by name, e.g. 'resnet50_1-13'")
    p.add_argument("--epochs",    type=int,   default=None,
                   help="Override epoch count for all experiments.")
    p.add_argument("--lr",        type=float, default=None,
                   help="Override learning rate for all experiments.")
    p.add_argument("--batch-size", type=int,  default=None,
                   help="Override batch size for all experiments.")

    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--device",           type=str,   default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-pretrained",    action="store_true")
    p.add_argument("--head-hidden-dim",  type=int,   default=256)
    p.add_argument("--dropout-backbone", type=float, default=0.3)
    p.add_argument("--dropout-head",     type=float, default=0.4)
    p.add_argument("--log-every",        type=int,   default=0,
                   help="Print step-level logs every N steps (0 = epoch only)")
    return p


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def main() -> None:
    args       = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    # ── Build experiment list ─────────────────────────────────────────────────
    experiments = []
    for cfg in DEFAULT_EXPERIMENTS:
        if args.only and cfg.name != args.only:
            continue
        # Apply CLI overrides
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.lr is not None:
            cfg.lr = args.lr
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        experiments.append(cfg)

    if not experiments:
        available = [c.name for c in DEFAULT_EXPERIMENTS]
        print(f"No experiment matched --only='{args.only}'.")
        print(f"Available: {available}")
        sys.exit(1)

    print(f"Device       : {device}")
    print(f"Experiments  : {[c.name for c in experiments]}")
    print(f"Output dir   : {output_dir}")
    print(f"Seed         : {args.seed}")

    # ── Run experiments ───────────────────────────────────────────────────────
    all_results = []
    for cfg in experiments:
        set_seed(args.seed)   # reset seed for each experiment for reproducibility
        result = run_experiment(cfg, args, device, output_dir)
        all_results.append(result)

    # ── Comparison ────────────────────────────────────────────────────────────
    write_comparison(all_results, output_dir)


if __name__ == "__main__":
    main()