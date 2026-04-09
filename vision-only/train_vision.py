"""
MATWI — Vision Baseline Ablation Training Script
==================================================
Runs a grid of experiments to find which undocumented paper settings
explain the gap between our ResNet50 replication (42 µm) and their
reported result (30 µm).

Ablation grid for ResNet50 paper replication (all use simple head, fixed LR):
  1. resnet50_imagenet_L1   — our previous setup
  2. resnet50_imagenet_MSE  — alternative loss
  3. resnet50_dataset_L1    — alternative normalisation
  4. resnet50_dataset_MSE   — alternative norm + loss

Plus our EfficientNetV2 reference (mlp head, OneCycleLR):
  5. efficientnetv2_imagenet_L1_mlp

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
import json
import math
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Local imports ─────────────────────────────────────────────────────────────

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
])
_MDL = _load_module("model_module", [
    _HERE / "ResNet_EfficientNet_Vision.py",
])

MATWIVisionDataset = _DS.MATWIVisionDataset
MATWIVisionModel   = _MDL.MATWIVisionModel
BACKBONES          = _MDL.BACKBONES


# ── Constants ─────────────────────────────────────────────────────────────────

WEAR_CAP    = 450.0
WEAR_TYPES  = ["flank_wear", "adhesion", "flank_wear+adhesion"]


# ── Experiment config ─────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
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

    # ── EfficientNetV2 reference (our best) ──
    ExperimentConfig(
        name          = "efficientnetv2_imagenet_L1_mlp",
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


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


def predictions_to_um(pred: torch.Tensor) -> torch.Tensor:
    return (pred.squeeze(-1) * 1000.0).clamp(0.0, WEAR_CAP)


def compute_mae(
    pred_um:   np.ndarray,
    target_um: np.ndarray,
    types:     list[str],
) -> dict[str, Any]:
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


def make_criterion(loss_name: str) -> nn.Module:
    if loss_name == "L1":
        return nn.L1Loss()
    elif loss_name == "MSE":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Use 'L1' or 'MSE'.")


# ── Data ──────────────────────────────────────────────────────────────────────

def build_loaders(
    cfg:         ExperimentConfig,
    data_dir:    Path,
    labels_csv:  Path,
    sets_csv:    Path,
    num_workers: int,
) -> dict[str, DataLoader]:
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

    eval_splits = ["val", "test"]
    if cfg.set_range == "1-13":
        eval_splits.append("unseen")

    for split in eval_splits:
        ds = MATWIVisionDataset(**common, split=split, augment=False)
        if len(ds) == 0:
            print(f"  [warn] split='{split}' empty, skipping.")
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
    print(f"  backbone={cfg.backbone}  head={cfg.head_type}  norm={cfg.normalisation}  "
          f"loss={cfg.loss}  lr={cfg.lr}  sched={cfg.use_scheduler}")
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
        head_type        = cfg.head_type,
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
        total_steps     = cfg.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr           = cfg.lr,
            total_steps      = total_steps,
            pct_start        = 0.3,
            anneal_strategy  = "cos",
            div_factor       = 25.0,
            final_div_factor = 1e4,
        )
        print(f"  Scheduler: OneCycleLR")
    else:
        scheduler = None
        print(f"  Scheduler: None (fixed LR={cfg.lr})")

    criterion = make_criterion(cfg.loss)
    print(f"  Loss: {cfg.loss} → {criterion}")

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

        if val_m["mae_overall_um"] < best_val_mae:
            best_val_mae = val_m["mae_overall_um"]
            best_epoch   = epoch
            best_state   = deepcopy(model.state_dict())
            torch.save({
                "epoch":            epoch,
                "model_state_dict": best_state,
                "val_mae_um":       best_val_mae,
                "config":           cfg.__dict__,
            }, run_dir / "best_model.pt")
            print(f"    ✓ new best val_mae={best_val_mae:.2f}µm")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # ── Final evaluation ──────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best model (epoch {best_epoch}, val={best_val_mae:.2f}µm)")

    print(f"\n  --- Final results: {cfg.name} ---")
    final_results: dict[str, dict] = {}
    for split in [k for k in ("val", "test", "unseen") if k in loaders]:
        m = evaluate(model, loaders[split], criterion, device)
        final_results[split] = m
        print(f"  [{split:7s}] {fmt_metrics(m)}")

    with open(run_dir / "results.json", "w") as fh:
        json.dump({
            "experiment":      cfg.name,
            "config":          cfg.__dict__,
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
        "config":          cfg.__dict__,
        "best_epoch":      best_epoch,
        "best_val_mae_um": best_val_mae,
        "final_results":   final_results,
    }


# ── Comparison summary ────────────────────────────────────────────────────────

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


# ── CLI ───────────────────────────────────────────────────────────────────────

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


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


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