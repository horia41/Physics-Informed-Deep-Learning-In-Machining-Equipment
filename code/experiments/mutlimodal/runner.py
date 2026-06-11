from __future__ import annotations
import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.mutlimodal.config import ExperimentConfig
from dataset.transform.sensor_scaler import SensorScaler
from dataset.multimodal import MATWIMultimodalDatasetV1, MATWIMultimodalDatasetV2
from constants.matwi_dataset_constants import WEAR_CAP, WEAR_TYPES
from utils.experiment import predictions_to_um, seed_worker
from metrics.fmt import fmt_metrics_v1
from metrics.mae import compute_mae_v1
from model.vision_sensor.multimodal_v1 import MATWIMultimodalModelV1
from model.vision_sensor.multimodal_v2 import MATWIMultimodalModelV2

# ── Data ──────────────────────────────────────────────────────────────────────

def build_loaders_v1(
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
        gate_aircuts     = cfg.gate_aircuts,
    )

    # Build training set and fit scaler
    train_ds = MATWIMultimodalDatasetV1(
        **common,
        split         = "train",
        augment       = False,
        sensor_scaler = None,
    )
    print(f"\n  Fitting sensor scaler on {len(train_ds)} training samples...")
    scaler = train_ds.fit_sensor_scaler()

    # Reproducible shuffle order tied to the global seed set before this call.
    g = torch.Generator()
    g.manual_seed(torch.initial_seed() % 2**32)
    # persistent_workers keeps the per-worker sensor-feature cache alive across
    # epochs; without it, workers are re-spawned every epoch and the ~99k-row
    # sensor CSVs are re-parsed from disk each time.
    persistent = num_workers > 0

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size        = cfg.batch_size,
            shuffle           = True,
            num_workers       = num_workers,
            pin_memory        = True,
            drop_last         = True,
            worker_init_fn    = seed_worker,
            generator         = g,
            persistent_workers = persistent,
        ),
    }

    # Val / test loaders
    for split in ["val", "test"]:
        ds = MATWIMultimodalDatasetV1(
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
            batch_size        = cfg.batch_size,
            shuffle           = False,
            num_workers       = num_workers,
            pin_memory        = True,
            drop_last         = False,
            worker_init_fn    = seed_worker,
            persistent_workers = persistent,
        )

    sizes = "  |  ".join(f"{k}: {len(v.dataset)}" for k, v in loaders.items())
    print(f"  Loaders: {sizes}")
    return loaders, scaler

def build_loaders_v2(
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
    train_ds = MATWIMultimodalDatasetV2(
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
        ds = MATWIMultimodalDatasetV2(
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

    metrics = compute_mae_v1(np.concatenate(all_pred),
                          np.concatenate(all_target),
                          all_types)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


# ── Single experiment ─────────────────────────────────────────────────────────

def run_experiment_v1(
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
    loaders, scaler = build_loaders_v1(
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

    model = MATWIMultimodalModelV1(**model_kwargs).to(device)

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
            f"val {fmt_metrics_v1(val_m)}  "
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
        print(f"  [{split:7s}] {fmt_metrics_v1(m)}")

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

def run_experiment_v2(
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
    loaders, scaler = build_loaders_v2(
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
            modality_dropout_p = cfg.modality_dropout_p,
        )

    model = MATWIMultimodalModelV2(**model_kwargs).to(device)

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
            f"val {fmt_metrics_v1(val_m)}  "
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
        print(f"  [{split:7s}] {fmt_metrics_v1(m)}")

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
