"""
MATWI — Sensor Fusion Ablation Training Script
=================================================
Controlled experiment: does adding sensor features improve over vision-only?

Grid: 3 fusion modes × 3 feature sets + 1 vision-only control = 10 experiments.
All experiments share identical settings except the sensor integration.

Reference baseline:
  efficientnetv2_dataset_MSE (vision-only, 664 samples) = 19.0 µm

Experiments:
  #0  vision_only_647       — vision-only control on 647-sample multimodal subset
  #1  early_raw25           — early fusion, 25 time-domain features
  #2  early_top25           — early fusion, 25 Ridge-selected features
  #3  early_all40           — early fusion, all 40 features
  #4  intermediate_raw25    — intermediate fusion, 25 time-domain features
  #5  intermediate_top25    — intermediate fusion, 25 Ridge-selected features
  #6  intermediate_all40    — intermediate fusion, all 40 features
  #7  late_raw25            — late fusion, 25 time-domain features
  #8  late_top25            — late fusion, 25 Ridge-selected features
  #9  late_all40            — late fusion, all 40 features

Fixed settings (matching vision-only reference):
  - EfficientNetV2-S, 384×384, dataset normalisation
  - MSE loss, fixed LR 3e-4, AdamW (wd=1e-4)
  - 17 epochs, batch_size=32, no augmentation, no oversampling
  - Sets 1-13, wear_cap=450, seed=42

Designed for SLURM (Snellius): use --only <name> to run one experiment per job.

Usage:
  python train_vision_sensor.py --data-dir ./data/matwi ... --only early_all40
  python train_vision_sensor.py --list-experiments
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
from typing import Any, Optional

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
    _HERE / "DatasetClass_VisionSensors.py",
])
_MDL = _load_module("model_module", [
    _HERE / "modelVisionSensor.py",
])

MATWIMultimodalDataset = _DS.MATWIMultimodalDataset
SensorScaler           = _DS.SensorScaler
FEATURE_SETS           = _DS.FEATURE_SETS
MATWIMultimodalModel   = _MDL.MATWIMultimodalModel


# ── Constants ─────────────────────────────────────────────────────────────────

WEAR_CAP   = 450.0
WEAR_TYPES = ["flank_wear", "adhesion", "flank_wear+adhesion"]


# ── Experiment config ─────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name:               str
    fusion_mode:        str       # "none", "early", "intermediate", "late"
    feature_set:        str       # "none", "raw25", "top25", "all40"
    use_sensors:        bool
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


# ── Experiment grid ───────────────────────────────────────────────────────────

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
    cfg:         ExperimentConfig,
    data_dir:    Path,
    labels_csv:  Path,
    sets_csv:    Path,
    num_workers: int,
) -> tuple[dict[str, DataLoader], Optional[SensorScaler]]:
    """
    Build train/val/test DataLoaders using the multimodal dataset class.

    For all experiments (including vision-only control), we use
    MATWIMultimodalDataset to ensure identical sample populations
    (647 train / 300 val / 247 test).

    Returns (loaders_dict, fitted_scaler_or_None).
    """
    # For vision-only control, feature_set doesn't matter but we still
    # need a valid value — use "all40" (features are extracted but ignored)
    feature_set = cfg.feature_set if cfg.use_sensors else "all40"

    common = dict(
        data_dir         = data_dir,
        labels_csv       = labels_csv,
        sets_csv         = sets_csv,
        set_range        = cfg.set_range,
        normalisation    = cfg.normalisation,
        wear_cap         = WEAR_CAP,
        impute_zero_wear = False,
        image_size       = cfg.image_size,
        feature_set      = feature_set,
    )

    # Build training set and fit scaler
    train_ds = MATWIMultimodalDataset(
        **common,
        split         = "train",
        augment       = False,
        sensor_scaler = None,
    )
    print(f"\n  Fitting sensor scaler on {len(train_ds)} training samples...")
    scaler = train_ds.fit_sensor_scaler()

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

    # Val / test loaders
    for split in ["val", "test"]:
        ds = MATWIMultimodalDataset(
            **common,
            split         = split,
            augment       = False,
            sensor_scaler = scaler,
        )
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
    return loaders, scaler


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    use_sensors: bool,
    log_every:  int = 0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae  = 0.0
    n = 0

    for step, batch in enumerate(loader, 1):
        images    = batch["image"].to(device, non_blocking=True)
        target    = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].to(device, non_blocking=True)

        sensor_features = None
        if use_sensors:
            sensor_features = batch["sensor_features"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out  = model(images, sensor_features)
        loss = criterion(out["wear"], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pred_um   = predictions_to_um(out["wear"].detach())
        batch_mae = torch.abs(pred_um - target_um).mean().item()
        bsz       = images.size(0)

        total_loss += loss.item() * bsz
        total_mae  += batch_mae  * bsz
        n          += bsz

        if log_every > 0 and step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"    step {step:04d}/{len(loader)}  "
                  f"loss={loss.item():.5f}  mae={batch_mae:.1f}µm  lr={lr:.2e}")

    return {"loss": total_loss / max(n, 1), "mae_um": total_mae / max(n, 1)}


@torch.no_grad()
def evaluate(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    use_sensors: bool,
) -> dict[str, Any]:
    model.eval()
    all_pred, all_target, all_types = [], [], []
    total_loss = 0.0
    n = 0

    for batch in loader:
        images    = batch["image"].to(device, non_blocking=True)
        target    = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].cpu().numpy()

        sensor_features = None
        if use_sensors:
            sensor_features = batch["sensor_features"].to(device, non_blocking=True)

        out  = model(images, sensor_features)
        loss = criterion(out["wear"], target)

        all_pred.append(predictions_to_um(out["wear"]).cpu().numpy())
        all_target.append(target_um)
        all_types.extend(list(batch["type"]))

        bsz        = images.size(0)
        total_loss += loss.item() * bsz
        n          += bsz

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
    print(f"  fusion={cfg.fusion_mode}  features={cfg.feature_set}  "
          f"sensors={cfg.use_sensors}  loss={cfg.loss}  lr={cfg.lr}")
    print(f"{'='*80}")

    run_dir = output_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders, scaler = build_loaders(
        cfg         = cfg,
        data_dir    = args.data_dir,
        labels_csv  = args.labels_csv,
        sets_csv    = args.sets_csv,
        num_workers = args.num_workers,
    )

    # Save scaler for reproducibility
    if scaler is not None:
        scaler.save(run_dir / "sensor_scaler.pkl")

    # ── Determine n_sensor_features from the dataset ─────────────────────────
    train_ds = loaders["train"].dataset
    n_sensor_features = train_ds.n_selected_features

    # ── Model ─────────────────────────────────────────────────────────────────
    model_kwargs = dict(
        use_sensors        = cfg.use_sensors,
        pretrained         = not args.no_pretrained,
        dropout_backbone   = cfg.dropout_backbone,
    )
    if cfg.use_sensors:
        model_kwargs.update(
            fusion_mode        = cfg.fusion_mode,
            n_sensor_features  = n_sensor_features,
            sensor_encoder_dim = cfg.sensor_encoder_dim,
        )

    model = MATWIMultimodalModel(**model_kwargs).to(device)

    # ── Optimiser (fixed LR, no scheduler — matches vision reference) ────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
    )
    print(f"  Optimiser: AdamW (lr={cfg.lr}, wd={cfg.weight_decay})")
    print(f"  Scheduler: None (fixed LR={cfg.lr})")

    criterion = nn.MSELoss()
    print(f"  Loss: {cfg.loss} → {criterion}")

    # ── Training loop ─────────────────────────────────────────────────────────
    history:     list[dict] = []
    best_state:  Optional[dict] = None
    best_val_mae = float("inf")
    best_epoch   = -1

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, loaders["train"], optimizer, criterion,
            device, cfg.use_sensors, log_every=args.log_every,
        )
        val_m = evaluate(
            model, loaders["val"], criterion, device, cfg.use_sensors,
        )
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]

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
        print(f"\n  Restored best model (epoch {best_epoch}, "
              f"val={best_val_mae:.2f}µm)")

    print(f"\n  --- Final results: {cfg.name} ---")
    final_results: dict[str, dict] = {}
    for split in [k for k in ("val", "test") if k in loaders]:
        m = evaluate(model, loaders[split], criterion, device, cfg.use_sensors)
        final_results[split] = m
        print(f"  [{split:7s}] {fmt_metrics(m)}")

    # ── Save results ──────────────────────────────────────────────────────────
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
                "fusion_mode":                 cfg["fusion_mode"],
                "feature_set":                 cfg["feature_set"],
                "use_sensors":                 cfg["use_sensors"],
                "eval_split":                  split,
                "best_epoch":                  res["best_epoch"],
                "best_val_mae_um":             res["best_val_mae_um"],
                "mae_overall_um":              m["mae_overall_um"],
                "mae_flank_wear_um":           m["mae_flank_wear_um"],
                "mae_adhesion_um":             m["mae_adhesion_um"],
                "mae_flank_wear+adhesion_um":  m["mae_flank_wear+adhesion_um"],
                "n_total":                     m["n_total"],
            })

    # Reference rows
    rows.append({
        "experiment": "vision_only_664 (reference)", "fusion_mode": "none",
        "feature_set": "none", "use_sensors": False, "eval_split": "test",
        "best_epoch": 14, "best_val_mae_um": "—",
        "mae_overall_um": 19.0, "mae_flank_wear_um": 16.6,
        "mae_adhesion_um": 37.2, "mae_flank_wear+adhesion_um": 23.2,
        "n_total": 254,
    })
    rows.append({
        "experiment": "paper_baseline", "fusion_mode": "none",
        "feature_set": "none", "use_sensors": False, "eval_split": "test",
        "best_epoch": "—", "best_val_mae_um": "—",
        "mae_overall_um": 30.0, "mae_flank_wear_um": 14.0,
        "mae_adhesion_um": 39.0, "mae_flank_wear+adhesion_um": 91.0,
        "n_total": 254,
    })

    df = pd.DataFrame(rows)
    path = output_dir / "comparison_summary.csv"
    df.to_csv(path, index=False)

    print(f"\n{'='*100}")
    print("  SENSOR FUSION ABLATION — TEST SET RESULTS (sorted by overall MAE)")
    print(f"{'='*100}")
    test_df = df[df["eval_split"] == "test"].copy()
    test_df = test_df.sort_values("mae_overall_um")
    cols = ["experiment", "fusion_mode", "feature_set",
            "mae_overall_um", "mae_flank_wear_um", "mae_adhesion_um",
            "mae_flank_wear+adhesion_um", "best_epoch"]
    print(test_df[cols].to_string(index=False))
    print(f"\n  Full results (all splits) → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MATWI sensor fusion ablation: "
                    "3 fusion modes × 3 feature sets + vision-only control."
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
            print(f"  {cfg.name:25s}  fusion={cfg.fusion_mode:15s}  "
                  f"features={cfg.feature_set:6s}  sensors={cfg.use_sensors}")
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