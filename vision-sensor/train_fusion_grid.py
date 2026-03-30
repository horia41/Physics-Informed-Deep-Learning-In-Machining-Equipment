#!/usr/bin/env python3
"""
Grid training script — MATWI multimodal wear estimation.
=========================================================

Sweeps all meaningful combinations of:
  - fusion_mode      : early | intermediate | late          (3 values)
  - set_range        : 1-13  | 1-17                         (2 values)
  - augment          : False | True                         (2 values)
  - augment_strategy : uniform | oversample_adhesion        (2 values,
                       but oversample_adhesion is skipped when augment=False)
  - epochs           : 17    | 34                           (2 values)

Fixed for all runs (not swept):
  - normalisation    : dataset
  - wear_cap         : 450.0 µm
  - impute_zero_wear : True
  - image_size       : (384, 384)
  - batch_size       : 16

Total runs after deduplication: 36
  augment=False → only "uniform"  (1 strategy × 3 fusion × 2 range × 2 epochs = 12)
  augment=True  → both strategies (2 strategy × 3 fusion × 2 range × 2 epochs = 24)

Sensor scaler optimisation:
  The scaler depends only on set_range (not fusion/aug/epochs).
  It is fitted once per set_range and reused across all runs sharing that range.

Outputs per run (under --output-dir/<run_tag>/):
  history.csv        — per-epoch train/val metrics
  results.json       — final metrics on val / test / unseen
  checkpoints/best.pt — best model by val overall MAE

Global outputs (under --output-dir/):
  summary_all_runs.csv          — all runs, all splits
  ranking_val.csv               — sorted by val MAE
  ranking_test.csv              — sorted by test MAE
  failures.json                 — any crashed runs

Run example:
  python train_fusion_grid.py \
    --data-dir   ./data/matwi \
    --labels-csv ./data/matwi/labels.csv \
    --sets-csv   ./data/matwi/sets.csv \
    --output-dir ./runs/grid_v1 \
    --lr         3e-4 \
    --num-workers 4
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================================
# Dynamic imports — keeps the script independent of install location
# ============================================================================

def _load_module(module_name: str, candidates: Sequence[Path]):
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            return mod, path
    raise FileNotFoundError(
        f"Could not find '{module_name}'. Searched:\n" +
        "\n".join(f"  {p}" for p in candidates)
    )


_HERE = Path(__file__).resolve().parent

_DATASET_MOD, _DATASET_PATH = _load_module("dataset_module", [
    _HERE / "DatasetClass_VisionSensors.py",
    _HERE / "DatasetClassVisionSensors.py",
    _HERE / "dataset_class.py",
])
_MODEL_MOD, _MODEL_PATH = _load_module("model_module", [
    _HERE / "model.py",
    _HERE / "modelVisionSensor.py",
    _HERE / "modelVisionSensor (1).py",
])

MATWIMultimodalDataset  = _DATASET_MOD.MATWIMultimodalDataset
SensorScaler            = _DATASET_MOD.SensorScaler
MATWIWearModel          = _MODEL_MOD.MATWIWearModel


# ============================================================================
# Constants (fixed across all runs)
# ============================================================================

WEAR_CAP         = 450.0
IMAGE_SIZE       = (384, 384)
NORMALISATION    = "dataset"
IMPUTE_ZERO_WEAR = True
WEAR_TYPES       = ["flank_wear", "adhesion", "flank_wear+adhesion"]

# Sweep axes
FUSION_MODES = ["early", "intermediate", "late"]
SET_RANGES   = ["1-13", "1-17"]
EPOCH_VALUES = [17, 34]


# ============================================================================
# Config dataclasses
# ============================================================================

@dataclass(frozen=True)
class RunConfig:
    """Identifies one experiment. All swept dimensions are here."""
    fusion_mode:       str
    set_range:         str
    augment:           bool
    augment_strategy:  str
    epochs:            int

    def tag(self) -> str:
        aug_str = f"aug1_{self.augment_strategy}" if self.augment else "aug0"
        return (
            f"fusion-{self.fusion_mode}"
            f"__sets-{self.set_range}"
            f"__{aug_str}"
            f"__ep{self.epochs}"
        )


@dataclass
class TrainConfig:
    """Hyperparameters and infrastructure settings, constant across runs."""
    data_dir:          Path
    labels_csv:        Path
    sets_csv:          Path
    output_dir:        Path
    batch_size:        int   = 16
    num_workers:       int   = 4
    learning_rate:     float = 3e-4
    weight_decay:      float = 1e-4
    seed:              int   = 42
    device:            str   = "auto"
    pretrained:        bool  = True
    head_hidden_dim:   int   = 256
    sensor_hidden_dim: int   = 128
    dropout_backbone:  float = 0.3
    dropout_head:      float = 0.4
    log_every:         int   = 20


# ============================================================================
# Search space
# ============================================================================

def build_search_space() -> List[RunConfig]:
    """
    Returns all 36 meaningful RunConfig combinations.

    Deduplication rule: when augment=False, augment_strategy is always
    "uniform" regardless of what was requested — oversampling without
    augmentation still affects sampling but is a meaningless distinction
    since the images are identical. We test it only when augment=True.
    """
    configs: List[RunConfig] = []
    for fusion, set_range, augment, strategy, epochs in itertools.product(
        FUSION_MODES, SET_RANGES, [False, True],
        ["uniform", "oversample_adhesion"], EPOCH_VALUES
    ):
        # Skip oversample_adhesion when augment=False (duplicate of uniform)
        if not augment and strategy == "oversample_adhesion":
            continue
        configs.append(RunConfig(
            fusion_mode      = fusion,
            set_range        = set_range,
            augment          = augment,
            augment_strategy = strategy,
            epochs           = epochs,
        ))
    return configs


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True   # faster on fixed-size inputs


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(arg)


def make_run_dirs(output_dir: Path, cfg: RunConfig) -> Dict[str, Path]:
    run_dir  = output_dir / cfg.tag()
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return {"run": run_dir, "ckpt": ckpt_dir}


def predictions_to_um(pred: torch.Tensor) -> torch.Tensor:
    """Convert normalised prediction to µm, clamped to [0, WEAR_CAP]."""
    return (pred.squeeze(1) * 1000.0).clamp(0.0, WEAR_CAP)


def compute_mae_by_type(
    pred_um:   np.ndarray,
    target_um: np.ndarray,
    types:     Sequence[str],
) -> Dict[str, Any]:
    """
    Compute overall MAE and per-wear-type MAE in µm.
    Returns nan for a type with no samples (rather than crashing).
    """
    abs_err = np.abs(pred_um - target_um)
    metrics: Dict[str, Any] = {
        "mae_overall_um": float(abs_err.mean()) if len(abs_err) else math.nan,
        "n_total":        int(len(target_um)),
        "loss":           math.nan,   # filled by caller
    }
    for wt in WEAR_TYPES:
        mask = np.array([t == wt for t in types], dtype=bool)
        key  = f"mae_{wt}_um"
        metrics[key]            = float(abs_err[mask].mean()) if mask.any() else math.nan
        metrics[f"n_{wt}"]      = int(mask.sum())
    return metrics


# ============================================================================
# Scaler cache — fitted once per set_range, reused across all matching runs
# ============================================================================

_SCALER_CACHE: Dict[str, SensorScaler] = {}


def get_or_fit_scaler(
    set_range:  str,
    train_cfg:  TrainConfig,
    output_dir: Path,
) -> SensorScaler:
    """
    Return a cached SensorScaler for this set_range, or fit one on the
    training set and cache it. Saves the scaler to disk for reproducibility.

    The scaler depends only on set_range (not fusion/aug/epochs) because:
      - Sensor features are extracted from raw CSV files, not images
      - Augmentation does not affect sensor values
      - All training sets with the same set_range share the same raw samples
    """
    if set_range in _SCALER_CACHE:
        return _SCALER_CACHE[set_range]

    print(f"\n[scaler] Fitting sensor scaler for set_range={set_range} ...")
    scaler_path = output_dir / f"sensor_scaler_{set_range.replace('-','_')}.pkl"

    if scaler_path.exists():
        print(f"[scaler] Loading cached scaler from {scaler_path}")
        scaler = SensorScaler.load(scaler_path)
        _SCALER_CACHE[set_range] = scaler
        return scaler

    # Build a minimal training dataset just for scaler fitting
    train_ds = MATWIMultimodalDataset(
        data_dir         = train_cfg.data_dir,
        labels_csv       = train_cfg.labels_csv,
        sets_csv         = train_cfg.sets_csv,
        split            = "train",
        set_range        = set_range,
        normalisation    = NORMALISATION,
        augment          = False,        # augmentation doesn't affect sensor features
        augment_strategy = "uniform",
        wear_cap         = WEAR_CAP,
        impute_zero_wear = IMPUTE_ZERO_WEAR,
        image_size       = IMAGE_SIZE,
        sensor_scaler    = None,
    )
    scaler = train_ds.fit_sensor_scaler(save_path=scaler_path, verbose=True)
    _SCALER_CACHE[set_range] = scaler
    print(f"[scaler] Scaler fitted and saved → {scaler_path}")
    return scaler


# ============================================================================
# Dataloaders
# ============================================================================

def build_loaders(
    run_cfg:   RunConfig,
    train_cfg: TrainConfig,
    scaler:    SensorScaler,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test loaders (and unseen if set_range='1-13').

    Note on set_range='1-17':
      Sets 14-17 are in the training split, so there is no unseen set.
      The dataset class returns an empty dataset for split='unseen' in
      this case — we skip it rather than evaluate on 0 samples.
    """
    common = dict(
        data_dir         = train_cfg.data_dir,
        labels_csv       = train_cfg.labels_csv,
        sets_csv         = train_cfg.sets_csv,
        set_range        = run_cfg.set_range,
        normalisation    = NORMALISATION,
        wear_cap         = WEAR_CAP,
        impute_zero_wear = IMPUTE_ZERO_WEAR,
        image_size       = IMAGE_SIZE,
        sensor_scaler    = scaler,
    )

    # Training loader
    train_ds = MATWIMultimodalDataset(
        **common,
        split            = "train",
        augment          = run_cfg.augment,
        augment_strategy = run_cfg.augment_strategy,
    )

    if run_cfg.augment and run_cfg.augment_strategy == "oversample_adhesion":
        train_loader = DataLoader(
            train_ds,
            batch_size  = train_cfg.batch_size,
            sampler     = train_ds.get_sampler(),
            num_workers = train_cfg.num_workers,
            pin_memory  = True,
            drop_last   = True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size  = train_cfg.batch_size,
            shuffle     = True,
            num_workers = train_cfg.num_workers,
            pin_memory  = True,
            drop_last   = True,
        )

    loaders: Dict[str, DataLoader] = {"train": train_loader}

    # Val / test / unseen (no augmentation, no shuffle)
    eval_splits = ["val", "test"]
    if run_cfg.set_range == "1-13":
        eval_splits.append("unseen")   # sets 14-17 only available when not in training

    for split in eval_splits:
        ds = MATWIMultimodalDataset(
            **common,
            split            = split,
            augment          = False,
            augment_strategy = "uniform",
        )
        if len(ds) == 0:
            print(f"  [warn] split='{split}' has 0 samples for {run_cfg.tag()}, skipping.")
            continue
        loaders[split] = DataLoader(
            ds,
            batch_size  = train_cfg.batch_size,
            shuffle     = False,
            num_workers = train_cfg.num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

    n_train = len(train_ds)
    print(f"  Loaders: " + " | ".join(
        f"{k}={len(v.dataset)}" for k, v in loaders.items()
    ))
    return loaders


# ============================================================================
# Train / evaluate
# ============================================================================

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scheduler:  torch.optim.lr_scheduler.OneCycleLR,
    criterion:  nn.Module,
    device:     torch.device,
    log_every:  int,
    epoch:      int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()

    running_loss   = 0.0
    running_mae_um = 0.0
    n_samples      = 0

    for step, batch in enumerate(loader, start=1):
        images  = batch["image"].to(device, non_blocking=True)
        sensors = batch["sensor_features"].to(device, non_blocking=True)
        target  = batch["wear"].to(device, non_blocking=True).unsqueeze(1)   # (B,1)
        target_um = batch["wear_raw"].to(device, non_blocking=True)          # (B,)  µm

        optimizer.zero_grad(set_to_none=True)
        out  = model(images, sensors)
        pred = out["wear"]                          # (B,1)
        loss = criterion(pred, target)
        loss.backward()

        # Gradient clipping — prevents occasional large steps early in training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()   # OneCycleLR steps per batch, not per epoch

        pred_um   = predictions_to_um(pred.detach())
        batch_mae = torch.abs(pred_um - target_um).mean().item()
        bsz       = images.size(0)

        running_loss   += loss.item() * bsz
        running_mae_um += batch_mae   * bsz
        n_samples      += bsz

        if log_every > 0 and step % log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"    [train] epoch {epoch:03d}/{total_epochs} "
                f"step {step:04d}/{len(loader):04d}  "
                f"loss={loss.item():.5f}  "
                f"batch_mae={batch_mae:.2f}µm  "
                f"lr={current_lr:.2e}"
            )

    n = max(n_samples, 1)
    return {
        "loss":   running_loss   / n,
        "mae_um": running_mae_um / n,
    }


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Dict[str, Any]:
    model.eval()

    all_pred_um:   List[np.ndarray] = []
    all_target_um: List[np.ndarray] = []
    all_types:     List[str]        = []
    running_loss = 0.0
    n_samples    = 0

    for batch in loader:
        images    = batch["image"].to(device, non_blocking=True)
        sensors   = batch["sensor_features"].to(device, non_blocking=True)
        target    = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].cpu().numpy()

        out  = model(images, sensors)
        pred = out["wear"]
        loss = criterion(pred, target)

        pred_um = predictions_to_um(pred).cpu().numpy()

        all_pred_um.append(pred_um)
        all_target_um.append(target_um)
        all_types.extend(list(batch["type"]))

        bsz           = images.size(0)
        running_loss += loss.item() * bsz
        n_samples    += bsz

    if not all_pred_um:
        return {"mae_overall_um": math.nan, "loss": math.nan, "n_total": 0}

    pred_arr   = np.concatenate(all_pred_um)
    target_arr = np.concatenate(all_target_um)

    metrics         = compute_mae_by_type(pred_arr, target_arr, all_types)
    metrics["loss"] = running_loss / max(n_samples, 1)
    metrics["pred_um"]   = pred_arr    # kept for potential downstream analysis
    metrics["target_um"] = target_arr
    metrics["types"]     = all_types
    return metrics


# ============================================================================
# Single experiment
# ============================================================================

def run_experiment(
    run_cfg:   RunConfig,
    train_cfg: TrainConfig,
    device:    torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 100)
    print(f"  RUN: {run_cfg.tag()}")
    print("=" * 100)

    dirs   = make_run_dirs(output_dir, run_cfg)
    scaler = get_or_fit_scaler(run_cfg.set_range, train_cfg, output_dir)
    loaders = build_loaders(run_cfg, train_cfg, scaler)

    # ── Model ────────────────────────────────────────────────────────────────
    model = MATWIWearModel(
        fusion_mode       = run_cfg.fusion_mode,
        use_sensors       = True,
        sensor_hidden_dim = train_cfg.sensor_hidden_dim,
        head_hidden_dim   = train_cfg.head_hidden_dim,
        dropout_backbone  = train_cfg.dropout_backbone,
        dropout_head      = train_cfg.dropout_head,
        pretrained        = train_cfg.pretrained,
    ).to(device)

    # ── Optimiser + OneCycleLR ────────────────────────────────────────────────
    # Single learning rate for all parameters (full fine-tuning from the start).
    # OneCycleLR: linear warmup for 30% of training, then cosine annealing.
    # steps_per_epoch uses the actual loader length (accounts for drop_last).
    steps_per_epoch = len(loaders["train"])
    total_steps     = run_cfg.epochs * steps_per_epoch

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = train_cfg.learning_rate,
        weight_decay = train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr       = train_cfg.learning_rate,
        total_steps  = total_steps,
        pct_start    = 0.3,          # 30% warmup
        anneal_strategy = "cos",
        div_factor   = 25.0,         # initial_lr = max_lr / 25
        final_div_factor = 1e4,      # final_lr  = initial_lr / 10000
    )

    criterion = nn.L1Loss()   # MAE loss, matches paper

    # ── Training loop ────────────────────────────────────────────────────────
    history:    List[Dict[str, Any]] = []
    best_state: Optional[Dict]       = None
    best_val_mae = float("inf")
    best_epoch   = -1

    for epoch in range(1, run_cfg.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model        = model,
            loader       = loaders["train"],
            optimizer    = optimizer,
            scheduler    = scheduler,
            criterion    = criterion,
            device       = device,
            log_every    = train_cfg.log_every,
            epoch        = epoch,
            total_epochs = run_cfg.epochs,
        )
        val_metrics = evaluate(
            model     = model,
            loader    = loaders["val"],
            criterion = criterion,
            device    = device,
        )

        elapsed = time.time() - t0
        val_mae = val_metrics["mae_overall_um"]

        print(
            f"  [epoch {epoch:03d}/{run_cfg.epochs}] "
            f"train_loss={train_metrics['loss']:.5f}  "
            f"train_mae={train_metrics['mae_um']:.2f}µm  |  "
            f"val_mae={val_mae:.2f}µm  "
            f"flank={val_metrics['mae_flank_wear_um']:.2f}  "
            f"adh={val_metrics['mae_adhesion_um']:.2f}  "
            f"f+a={val_metrics['mae_flank_wear+adhesion_um']:.2f}  "
            f"({elapsed:.1f}s)"
        )

        record = {
            "epoch":                         epoch,
            "train_loss":                    train_metrics["loss"],
            "train_mae_um":                  train_metrics["mae_um"],
            "val_loss":                      val_metrics["loss"],
            "val_mae_overall_um":            val_mae,
            "val_mae_flank_wear_um":         val_metrics["mae_flank_wear_um"],
            "val_mae_adhesion_um":           val_metrics["mae_adhesion_um"],
            "val_mae_flank_wear+adhesion_um": val_metrics["mae_flank_wear+adhesion_um"],
            "lr":                            scheduler.get_last_lr()[0],
            "seconds":                       elapsed,
        }
        history.append(record)

        # Save best checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            best_state   = {
                "epoch":            epoch,
                "model_state_dict": deepcopy(model.state_dict()),
                "val_mae_um":       best_val_mae,
                "run_config":       asdict(run_cfg),
                "train_config":     {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(train_cfg).items()
                },
            }
            ckpt_path = dirs["ckpt"] / "best.pt"
            torch.save(best_state, ckpt_path)
            print(f"    [ckpt] New best val_mae={best_val_mae:.2f}µm → {ckpt_path}")

    # ── Save training history ─────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(dirs["run"] / "history.csv", index=False)

    # ── Final evaluation with best model ─────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
        print(f"\n  Restored best model from epoch {best_epoch} "
              f"(val_mae={best_val_mae:.2f}µm)")

    split_results: Dict[str, Dict[str, Any]] = {}
    eval_splits = [k for k in ("val", "test", "unseen") if k in loaders]

    for split in eval_splits:
        metrics = evaluate(
            model     = model,
            loader    = loaders[split],
            criterion = criterion,
            device    = device,
        )
        split_results[split] = metrics
        print(
            f"  [final {split:6s}] "
            f"overall={metrics['mae_overall_um']:.2f}µm  "
            f"flank={metrics['mae_flank_wear_um']:.2f}  "
            f"adh={metrics['mae_adhesion_um']:.2f}  "
            f"f+a={metrics['mae_flank_wear+adhesion_um']:.2f}  "
            f"(n={metrics['n_total']})"
        )

    # ── Save results JSON ─────────────────────────────────────────────────────
    serialisable = {
        "run_config":      asdict(run_cfg),
        "best_epoch":      best_epoch,
        "best_val_mae_um": best_val_mae,
        "splits": {
            split: {k: v for k, v in m.items()
                    if k not in ("pred_um", "target_um", "types")}
            for split, m in split_results.items()
        },
    }
    with open(dirs["run"] / "results.json", "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)

    return {
        "config":          run_cfg,
        "best_epoch":      best_epoch,
        "best_val_mae_um": best_val_mae,
        "split_results":   split_results,
        "run_dir":         str(dirs["run"]),
    }


# ============================================================================
# Global summary
# ============================================================================

def write_summary(results: List[Dict[str, Any]], output_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []

    for res in results:
        cfg: RunConfig = res["config"]
        base = {
            **asdict(cfg),
            "best_epoch":      res["best_epoch"],
            "best_val_mae_um": res["best_val_mae_um"],
            "run_dir":         res["run_dir"],
        }
        for split, metrics in res["split_results"].items():
            row = dict(base)
            row["eval_split"] = split
            row.update({
                "mae_overall_um":              metrics["mae_overall_um"],
                "mae_flank_wear_um":           metrics["mae_flank_wear_um"],
                "mae_adhesion_um":             metrics["mae_adhesion_um"],
                "mae_flank_wear+adhesion_um":  metrics["mae_flank_wear+adhesion_um"],
                "n_total":                     metrics["n_total"],
                "n_flank_wear":                metrics.get("n_flank_wear", math.nan),
                "n_adhesion":                  metrics.get("n_adhesion", math.nan),
                "n_flank_wear+adhesion":       metrics.get("n_flank_wear+adhesion", math.nan),
                "val_loss":                    metrics.get("loss", math.nan),
            })
            rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    summary_path = output_dir / "summary_all_runs.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n[summary] {summary_path}")

    for split in df["eval_split"].dropna().unique():
        split_df = (df[df["eval_split"] == split]
                    .sort_values("mae_overall_um", ascending=True)
                    .copy())
        ranking_path = output_dir / f"ranking_{split}.csv"
        split_df.to_csv(ranking_path, index=False)
        top = split_df.iloc[0]
        print(f"[summary] ranking_{split}.csv  "
              f"best: {top['fusion_mode']} / sets{top['set_range']} / "
              f"aug={top['augment']} / {top['augment_strategy']} / "
              f"ep{top['epochs']}  →  mae={top['mae_overall_um']:.2f}µm")


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MATWI multimodal wear estimation — grid training sweep."
    )
    p.add_argument("--data-dir",   type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--sets-csv",   type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       type=str,   default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-pretrained", action="store_true",
                   help="Disable ImageNet pretrained weights.")
    p.add_argument("--head-hidden-dim",   type=int,   default=256)
    p.add_argument("--sensor-hidden-dim", type=int,   default=128)
    p.add_argument("--dropout-backbone",  type=float, default=0.3)
    p.add_argument("--dropout-head",      type=float, default=0.4)
    p.add_argument("--log-every",         type=int,   default=20,
                   help="Print training progress every N steps (0 = silent).")

    # Subset sweep for quick testing
    p.add_argument("--dry-run", action="store_true",
                   help="Run only the first configuration and exit.")
    p.add_argument("--fusion-modes", type=str, default=None,
                   help="Comma-separated subset of fusion modes to run, e.g. 'early,intermediate'.")
    p.add_argument("--set-ranges", type=str, default=None,
                   help="Comma-separated subset of set ranges, e.g. '1-13'.")
    return p


def main() -> None:
    args   = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainConfig(
        data_dir          = args.data_dir.resolve(),
        labels_csv        = args.labels_csv.resolve(),
        sets_csv          = args.sets_csv.resolve(),
        output_dir        = output_dir,
        batch_size        = args.batch_size,
        num_workers       = args.num_workers,
        learning_rate     = args.lr,
        weight_decay      = args.weight_decay,
        seed              = args.seed,
        device            = args.device,
        pretrained        = not args.no_pretrained,
        head_hidden_dim   = args.head_hidden_dim,
        sensor_hidden_dim = args.sensor_hidden_dim,
        dropout_backbone  = args.dropout_backbone,
        dropout_head      = args.dropout_head,
        log_every         = args.log_every,
    )

    set_seed(train_cfg.seed)
    device = resolve_device(train_cfg.device)

    print(f"Dataset module : {_DATASET_PATH}")
    print(f"Model module   : {_MODEL_PATH}")
    print(f"Device         : {device}")
    print(f"Output dir     : {output_dir}")
    print(f"Fixed settings : wear_cap={WEAR_CAP}µm | image={IMAGE_SIZE} | "
          f"norm={NORMALISATION} | impute_zero={IMPUTE_ZERO_WEAR}")

    # Build and optionally filter the search space
    search_space = build_search_space()

    if args.fusion_modes:
        keep = {x.strip() for x in args.fusion_modes.split(",")}
        search_space = [c for c in search_space if c.fusion_mode in keep]
    if args.set_ranges:
        keep = {x.strip() for x in args.set_ranges.split(",")}
        search_space = [c for c in search_space if c.set_range in keep]
    if args.dry_run:
        search_space = search_space[:1]

    print(f"\nTotal runs: {len(search_space)}")
    for i, cfg in enumerate(search_space, 1):
        print(f"  [{i:03d}] {cfg.tag()}")

    results:  List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, cfg in enumerate(search_space, 1):
        print(f"\n\n##### {i}/{len(search_space)} #####")
        try:
            result = run_experiment(cfg, train_cfg, device, output_dir)
            results.append(result)
        except KeyboardInterrupt:
            print("\n[interrupted] Saving partial summary before exit.")
            break
        except Exception as exc:
            import traceback
            print(f"[ERROR] {cfg.tag()}\n{traceback.format_exc()}")
            failures.append({"config": asdict(cfg), "error": str(exc)})
            with open(output_dir / "failures.json", "w") as fh:
                json.dump(failures, fh, indent=2)
        finally:
            # Write summary after every run so partial results are always available
            write_summary(results, output_dir)

    write_summary(results, output_dir)

    if failures:
        print(f"\n{len(failures)} run(s) failed. See {output_dir / 'failures.json'}")
    else:
        print(f"\nAll {len(results)} runs completed successfully.")


if __name__ == "__main__":
    main()
