"""
MATWI - Two-Stage Residual Fusion Training
==========================================

Implements Option 4 (recommended):
  Stage A: train vision-only model to convergence.
  Stage B: freeze Stage A and train sensor residual model to predict:
           residual = wear_true - wear_vision
  Inference: wear_final = wear_vision + residual_pred

This script is intentionally standalone so it can run end-to-end in one call,
including training and final evaluation.

Usage:
  python train_two_stage_residual.py \
      --data-dir /path/to/matwi \
      --labels-csv /path/to/matwi/labels.csv \
      --sets-csv /path/to/matwi/sets.csv \
      --output-dir /path/to/runs/two_stage_residual
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


_DS = _load_module("dataset_module_twostage", [
    _HERE / "DatasetClass_VisionSensors.py",
])
_MDL = _load_module("model_module_twostage", [
    _HERE / "modelVisionSensor.py",
])

MATWIMultimodalDataset = _DS.MATWIMultimodalDataset
SensorScaler = _DS.SensorScaler
MATWIMultimodalModel = _MDL.MATWIMultimodalModel
SensorEncoder = _MDL.SensorEncoder


WEAR_CAP = 450.0
WEAR_TYPES = ["flank_wear", "adhesion", "flank_wear+adhesion"]


@dataclass
class TrainConfig:
    name: str = "two_stage_residual"
    set_range: str = "1-13"
    image_size: tuple[int, int] = (384, 384)
    normalisation: str = "dataset"
    feature_set: str = "top25"

    stage_a_epochs: int = 17
    stage_b_epochs: int = 17

    batch_size: int = 32
    lr_stage_a: float = 3e-4
    lr_stage_b: float = 3e-4
    weight_decay: float = 1e-4

    dropout_backbone: float = 0.3
    sensor_encoder_dim: int = 64


class SensorResidualModel(nn.Module):
    """
    Residual regressor trained in Stage B.

    Modes:
      - sensors_only: residual = f(sensor_features)
      - hybrid:       residual = f(sensor_features, frozen image_embed)
    """

    def __init__(
        self,
        n_sensor_features: int,
        sensor_encoder_dim: int = 64,
        residual_input: Literal["sensors_only", "hybrid"] = "sensors_only",
        image_embed_dim: int = 1280,
        modality_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.residual_input = residual_input
        self.modality_dropout_p = float(modality_dropout_p)

        self.sensor_encoder = SensorEncoder(
            input_dim=n_sensor_features,
            hidden_dim=sensor_encoder_dim,
            output_dim=sensor_encoder_dim,
        )

        head_in = sensor_encoder_dim
        if residual_input == "hybrid":
            head_in += image_embed_dim

        self.head = nn.Linear(head_in, 1)

    def _apply_modality_dropout(self, sensor_features: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout_p <= 0.0:
            return sensor_features

        keep_mask = (
            torch.rand(sensor_features.size(0), 1, device=sensor_features.device)
            >= self.modality_dropout_p
        ).float()
        return sensor_features * keep_mask

    def forward(
        self,
        sensor_features: torch.Tensor,
        image_embed: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        sensor_features = self._apply_modality_dropout(sensor_features)
        sensor_embed = self.sensor_encoder(sensor_features)

        if self.residual_input == "hybrid":
            if image_embed is None:
                raise ValueError("image_embed is required for residual_input='hybrid'")
            fused = torch.cat([sensor_embed, image_embed], dim=1)
        else:
            fused = sensor_embed

        residual = self.head(fused)
        return {
            "residual": residual,
            "sensor_embed": sensor_embed,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def predictions_to_um(pred_norm: torch.Tensor) -> torch.Tensor:
    return (pred_norm.squeeze(-1) * 1000.0).clamp(0.0, WEAR_CAP)


def compute_mae(
    pred_um: np.ndarray,
    target_um: np.ndarray,
    types: list[str],
) -> dict[str, Any]:
    abs_err = np.abs(pred_um - target_um)
    out: dict[str, Any] = {
        "mae_overall_um": float(abs_err.mean()) if len(abs_err) else math.nan,
        "n_total": len(target_um),
    }
    for wt in WEAR_TYPES:
        mask = np.array([t == wt for t in types], dtype=bool)
        out[f"mae_{wt}_um"] = float(abs_err[mask].mean()) if mask.any() else math.nan
        out[f"n_{wt}"] = int(mask.sum())
    return out


def fmt_metrics(m: dict[str, Any]) -> str:
    return (
        f"overall={m['mae_overall_um']:.2f}um  "
        f"flank={m['mae_flank_wear_um']:.2f}  "
        f"adh={m['mae_adhesion_um']:.2f}  "
        f"f+a={m['mae_flank_wear+adhesion_um']:.2f}  "
        f"(n={m['n_total']})"
    )


def build_loaders(
    cfg: TrainConfig,
    data_dir: Path,
    labels_csv: Path,
    sets_csv: Path,
    num_workers: int,
) -> tuple[dict[str, DataLoader], SensorScaler]:
    common = dict(
        data_dir=data_dir,
        labels_csv=labels_csv,
        sets_csv=sets_csv,
        set_range=cfg.set_range,
        normalisation=cfg.normalisation,
        wear_cap=WEAR_CAP,
        impute_zero_wear=False,
        image_size=cfg.image_size,
        feature_set=cfg.feature_set,
    )

    train_ds = MATWIMultimodalDataset(
        **common,
        split="train",
        augment=False,
        sensor_scaler=None,
    )
    print(f"\nFitting sensor scaler on {len(train_ds)} training samples...")
    scaler = train_ds.fit_sensor_scaler()

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    }

    for split in ["val", "test"]:
        ds = MATWIMultimodalDataset(
            **common,
            split=split,
            augment=False,
            sensor_scaler=scaler,
        )
        if len(ds) == 0:
            print(f"[warn] split='{split}' empty, skipping.")
            continue
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    sizes = "  |  ".join(f"{k}: {len(v.dataset)}" for k, v in loaders.items())
    print(f"Loaders: {sizes}")
    return loaders, scaler


def train_stage_a_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(images, None)
        loss = criterion(out["wear"], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pred_um = predictions_to_um(out["wear"].detach())
        batch_mae = torch.abs(pred_um - target_um).mean().item()
        bsz = images.size(0)

        total_loss += loss.item() * bsz
        total_mae += batch_mae * bsz
        n += bsz

    return {
        "loss": total_loss / max(n, 1),
        "mae_um": total_mae / max(n, 1),
    }


@torch.no_grad()
def evaluate_stage_a(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_pred, all_target, all_types = [], [], []
    total_loss = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].cpu().numpy()

        out = model(images, None)
        loss = criterion(out["wear"], target)

        all_pred.append(predictions_to_um(out["wear"]).cpu().numpy())
        all_target.append(target_um)
        all_types.extend(list(batch["type"]))

        bsz = images.size(0)
        total_loss += loss.item() * bsz
        n += bsz

    if not all_pred:
        return {
            "mae_overall_um": math.nan,
            "mae_flank_wear_um": math.nan,
            "mae_adhesion_um": math.nan,
            "mae_flank_wear+adhesion_um": math.nan,
            "n_total": 0,
            "loss": math.nan,
        }

    metrics = compute_mae(np.concatenate(all_pred), np.concatenate(all_target), all_types)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def _load_state_dict_from_checkpoint(ckpt_path: Path, model: nn.Module, strict: bool = True) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    model.load_state_dict(state_dict, strict=strict)


def run_stage_a(
    cfg: TrainConfig,
    args: argparse.Namespace,
    device: torch.device,
    loaders: dict[str, DataLoader],
    run_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    stage_a_dir = run_dir / "stage_a"
    stage_a_dir.mkdir(parents=True, exist_ok=True)

    model = MATWIMultimodalModel(
        use_sensors=False,
        pretrained=not args.no_pretrained,
        dropout_backbone=cfg.dropout_backbone,
    ).to(device)

    if args.stage_a_checkpoint is not None and args.skip_stage_a_train:
        print(f"\n[Stage A] Loading checkpoint and skipping training: {args.stage_a_checkpoint}")
        _load_state_dict_from_checkpoint(args.stage_a_checkpoint, model, strict=True)
    else:
        if args.stage_a_checkpoint is not None:
            print(f"\n[Stage A] Warm-starting from checkpoint: {args.stage_a_checkpoint}")
            _load_state_dict_from_checkpoint(args.stage_a_checkpoint, model, strict=True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr_stage_a,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.MSELoss()

        history: list[dict[str, Any]] = []
        best_state: Optional[dict[str, torch.Tensor]] = None
        best_val_mae = float("inf")
        best_epoch = -1

        print(f"\n{'=' * 80}")
        print("STAGE A - Vision-only training")
        print(f"{'=' * 80}")

        for epoch in range(1, cfg.stage_a_epochs + 1):
            t0 = time.time()
            train_m = train_stage_a_one_epoch(
                model=model,
                loader=loaders["train"],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            val_m = evaluate_stage_a(
                model=model,
                loader=loaders["val"],
                criterion=criterion,
                device=device,
            )
            elapsed = time.time() - t0

            print(
                f"[{epoch:02d}/{cfg.stage_a_epochs}] "
                f"train loss={train_m['loss']:.5f} mae={train_m['mae_um']:.2f}um  |  "
                f"val {fmt_metrics(val_m)}  (loss={val_m['loss']:.5f})  "
                f"({elapsed:.1f}s)"
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_mae_um": train_m["mae_um"],
                "val_loss": val_m["loss"],
                "val_mae_overall_um": val_m["mae_overall_um"],
                "val_mae_flank_wear_um": val_m["mae_flank_wear_um"],
                "val_mae_adhesion_um": val_m["mae_adhesion_um"],
                "val_mae_flank_wear+adhesion_um": val_m["mae_flank_wear+adhesion_um"],
                "seconds": elapsed,
            })

            if val_m["mae_overall_um"] < best_val_mae:
                best_val_mae = val_m["mae_overall_um"]
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "val_mae_um": best_val_mae,
                    "config": asdict(cfg),
                    "stage": "A",
                }, stage_a_dir / "best_model.pt")
                print(f"  new best val_mae={best_val_mae:.2f}um")

        pd.DataFrame(history).to_csv(stage_a_dir / "history.csv", index=False)

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[Stage A] Restored best model from epoch {best_epoch}")

    criterion = nn.MSELoss()
    final_metrics: dict[str, Any] = {}
    for split in [k for k in ("val", "test") if k in loaders]:
        m = evaluate_stage_a(model, loaders[split], criterion, device)
        final_metrics[split] = m
        print(f"[Stage A final/{split}] {fmt_metrics(m)}")

    with open(stage_a_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump({
            "stage": "A",
            "mode": "vision_only",
            "config": asdict(cfg),
            "checkpoint_source": str(args.stage_a_checkpoint) if args.stage_a_checkpoint else None,
            "splits": final_metrics,
        }, fh, indent=2)

    return model, final_metrics


def train_stage_b_one_epoch(
    vision_model: nn.Module,
    residual_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    residual_input: Literal["sensors_only", "hybrid"],
) -> dict[str, float]:
    vision_model.eval()
    residual_model.train()

    total_residual_loss = 0.0
    total_final_mae = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        sensors = batch["sensor_features"].to(device, non_blocking=True)
        target = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].to(device, non_blocking=True)

        with torch.no_grad():
            vision_out = vision_model(images, None)
            vision_pred = vision_out["wear"].detach()
            image_embed = vision_out["image_embed"].detach()

        residual_target = target - vision_pred

        optimizer.zero_grad(set_to_none=True)
        if residual_input == "hybrid":
            residual_out = residual_model(sensors, image_embed=image_embed)
        else:
            residual_out = residual_model(sensors, image_embed=None)

        residual_pred = residual_out["residual"]
        residual_loss = criterion(residual_pred, residual_target)
        residual_loss.backward()
        torch.nn.utils.clip_grad_norm_(residual_model.parameters(), max_norm=1.0)
        optimizer.step()

        final_pred = vision_pred + residual_pred.detach()
        final_pred_um = predictions_to_um(final_pred)
        batch_mae = torch.abs(final_pred_um - target_um).mean().item()

        bsz = images.size(0)
        total_residual_loss += residual_loss.item() * bsz
        total_final_mae += batch_mae * bsz
        n += bsz

    return {
        "residual_loss": total_residual_loss / max(n, 1),
        "final_mae_um": total_final_mae / max(n, 1),
    }


@torch.no_grad()
def evaluate_stage_b(
    vision_model: nn.Module,
    residual_model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    residual_input: Literal["sensors_only", "hybrid"],
) -> dict[str, Any]:
    vision_model.eval()
    residual_model.eval()

    all_final_pred_um, all_target_um, all_types = [], [], []
    all_vision_pred_um = []
    total_residual_loss = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        sensors = batch["sensor_features"].to(device, non_blocking=True)
        target = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].cpu().numpy()

        vision_out = vision_model(images, None)
        vision_pred = vision_out["wear"]
        image_embed = vision_out["image_embed"]

        if residual_input == "hybrid":
            residual_out = residual_model(sensors, image_embed=image_embed)
        else:
            residual_out = residual_model(sensors, image_embed=None)

        residual_pred = residual_out["residual"]
        residual_target = target - vision_pred
        residual_loss = criterion(residual_pred, residual_target)

        final_pred = vision_pred + residual_pred

        all_final_pred_um.append(predictions_to_um(final_pred).cpu().numpy())
        all_vision_pred_um.append(predictions_to_um(vision_pred).cpu().numpy())
        all_target_um.append(target_um)
        all_types.extend(list(batch["type"]))

        bsz = images.size(0)
        total_residual_loss += residual_loss.item() * bsz
        n += bsz

    if not all_final_pred_um:
        return {
            "residual_loss": math.nan,
            "fusion_mae_overall_um": math.nan,
            "vision_mae_overall_um": math.nan,
            "gain_over_vision_um": math.nan,
            "n_total": 0,
        }

    final_pred = np.concatenate(all_final_pred_um)
    vision_pred = np.concatenate(all_vision_pred_um)
    target = np.concatenate(all_target_um)

    fusion_metrics = compute_mae(final_pred, target, all_types)
    vision_metrics = compute_mae(vision_pred, target, all_types)

    out: dict[str, Any] = {
        "residual_loss": total_residual_loss / max(n, 1),
        "n_total": fusion_metrics["n_total"],
        "fusion_mae_overall_um": fusion_metrics["mae_overall_um"],
        "vision_mae_overall_um": vision_metrics["mae_overall_um"],
        "gain_over_vision_um": vision_metrics["mae_overall_um"] - fusion_metrics["mae_overall_um"],

        "fusion_mae_flank_wear_um": fusion_metrics["mae_flank_wear_um"],
        "fusion_mae_adhesion_um": fusion_metrics["mae_adhesion_um"],
        "fusion_mae_flank_wear+adhesion_um": fusion_metrics["mae_flank_wear+adhesion_um"],

        "vision_mae_flank_wear_um": vision_metrics["mae_flank_wear_um"],
        "vision_mae_adhesion_um": vision_metrics["mae_adhesion_um"],
        "vision_mae_flank_wear+adhesion_um": vision_metrics["mae_flank_wear+adhesion_um"],
    }
    return out


def run_stage_b(
    cfg: TrainConfig,
    args: argparse.Namespace,
    device: torch.device,
    loaders: dict[str, DataLoader],
    vision_model: nn.Module,
    run_dir: Path,
) -> dict[str, Any]:
    stage_b_dir = run_dir / "stage_b"
    stage_b_dir.mkdir(parents=True, exist_ok=True)

    train_ds = loaders["train"].dataset
    n_sensor_features = train_ds.n_selected_features

    residual_model = SensorResidualModel(
        n_sensor_features=n_sensor_features,
        sensor_encoder_dim=cfg.sensor_encoder_dim,
        residual_input=args.residual_input,
        image_embed_dim=1280,
        modality_dropout_p=args.modality_dropout_p,
    ).to(device)

    optimizer = torch.optim.AdamW(
        residual_model.parameters(),
        lr=cfg.lr_stage_b,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.MSELoss()

    history: list[dict[str, Any]] = []
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_val_fusion_mae = float("inf")
    best_epoch = -1

    print(f"\n{'=' * 80}")
    print("STAGE B - Sensor residual learning")
    print(f"residual_input={args.residual_input}  modality_dropout_p={args.modality_dropout_p}")
    print(f"{'=' * 80}")

    for epoch in range(1, cfg.stage_b_epochs + 1):
        t0 = time.time()
        train_m = train_stage_b_one_epoch(
            vision_model=vision_model,
            residual_model=residual_model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            residual_input=args.residual_input,
        )
        val_m = evaluate_stage_b(
            vision_model=vision_model,
            residual_model=residual_model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            residual_input=args.residual_input,
        )
        elapsed = time.time() - t0

        print(
            f"[{epoch:02d}/{cfg.stage_b_epochs}] "
            f"train residual_loss={train_m['residual_loss']:.5f} final_mae={train_m['final_mae_um']:.2f}um  |  "
            f"val fusion_mae={val_m['fusion_mae_overall_um']:.2f}um "
            f"vision_mae={val_m['vision_mae_overall_um']:.2f}um "
            f"gain={val_m['gain_over_vision_um']:.2f}um  "
            f"(residual_loss={val_m['residual_loss']:.5f})  ({elapsed:.1f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_residual_loss": train_m["residual_loss"],
            "train_final_mae_um": train_m["final_mae_um"],
            "val_residual_loss": val_m["residual_loss"],
            "val_fusion_mae_overall_um": val_m["fusion_mae_overall_um"],
            "val_vision_mae_overall_um": val_m["vision_mae_overall_um"],
            "val_gain_over_vision_um": val_m["gain_over_vision_um"],
            "seconds": elapsed,
        })

        if val_m["fusion_mae_overall_um"] < best_val_fusion_mae:
            best_val_fusion_mae = val_m["fusion_mae_overall_um"]
            best_epoch = epoch
            best_state = deepcopy(residual_model.state_dict())
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_state,
                "val_fusion_mae_um": best_val_fusion_mae,
                "config": asdict(cfg),
                "residual_input": args.residual_input,
                "modality_dropout_p": args.modality_dropout_p,
                "stage": "B",
            }, stage_b_dir / "best_model.pt")
            print(f"  new best val fusion_mae={best_val_fusion_mae:.2f}um")

    pd.DataFrame(history).to_csv(stage_b_dir / "history.csv", index=False)

    if best_state is not None:
        residual_model.load_state_dict(best_state)
        print(f"[Stage B] Restored best residual model from epoch {best_epoch}")

    final_metrics: dict[str, Any] = {}
    for split in [k for k in ("val", "test") if k in loaders]:
        m = evaluate_stage_b(
            vision_model=vision_model,
            residual_model=residual_model,
            loader=loaders[split],
            criterion=criterion,
            device=device,
            residual_input=args.residual_input,
        )
        final_metrics[split] = m
        print(
            f"[Stage B final/{split}] "
            f"fusion={m['fusion_mae_overall_um']:.2f}um  "
            f"vision={m['vision_mae_overall_um']:.2f}um  "
            f"gain={m['gain_over_vision_um']:.2f}um"
        )

    with open(stage_b_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump({
            "stage": "B",
            "mode": "residual",
            "residual_input": args.residual_input,
            "modality_dropout_p": args.modality_dropout_p,
            "config": asdict(cfg),
            "best_epoch": best_epoch,
            "best_val_fusion_mae_um": best_val_fusion_mae,
            "splits": final_metrics,
        }, fh, indent=2)

    return final_metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-stage residual fusion: vision baseline + sensor residual correction"
    )
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--sets-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--run-name", type=str, default="two_stage_residual")
    p.add_argument("--feature-set", type=str, default="top25", choices=["raw25", "top25", "all40"])
    p.add_argument("--normalisation", type=str, default="dataset", choices=["dataset", "imagenet"])
    p.add_argument("--set-range", type=str, default="1-13", choices=["1-13", "1-17"])

    p.add_argument("--stage-a-epochs", type=int, default=17)
    p.add_argument("--stage-b-epochs", type=int, default=17)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-stage-a", type=float, default=3e-4)
    p.add_argument("--lr-stage-b", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout-backbone", type=float, default=0.3)
    p.add_argument("--sensor-encoder-dim", type=int, default=64)

    p.add_argument("--residual-input", type=str, default="sensors_only", choices=["sensors_only", "hybrid"])
    p.add_argument("--modality-dropout-p", type=float, default=0.0)

    p.add_argument("--stage-a-checkpoint", type=Path, default=None)
    p.add_argument("--skip-stage-a-train", action="store_true")

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-pretrained", action="store_true")
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

    cfg = TrainConfig(
        name=args.run_name,
        set_range=args.set_range,
        image_size=(384, 384),
        normalisation=args.normalisation,
        feature_set=args.feature_set,
        stage_a_epochs=args.stage_a_epochs,
        stage_b_epochs=args.stage_b_epochs,
        batch_size=args.batch_size,
        lr_stage_a=args.lr_stage_a,
        lr_stage_b=args.lr_stage_b,
        weight_decay=args.weight_decay,
        dropout_backbone=args.dropout_backbone,
        sensor_encoder_dim=args.sensor_encoder_dim,
    )

    run_dir = args.output_dir.resolve() / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"Device      : {device}")
    print(f"Run dir     : {run_dir}")
    print(f"Feature set : {cfg.feature_set}")
    print(f"Residual in : {args.residual_input}")

    loaders, scaler = build_loaders(
        cfg=cfg,
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        sets_csv=args.sets_csv,
        num_workers=args.num_workers,
    )

    scaler.save(run_dir / "sensor_scaler.pkl")

    vision_model, stage_a_metrics = run_stage_a(
        cfg=cfg,
        args=args,
        device=device,
        loaders=loaders,
        run_dir=run_dir,
    )

    stage_b_metrics = run_stage_b(
        cfg=cfg,
        args=args,
        device=device,
        loaders=loaders,
        vision_model=vision_model,
        run_dir=run_dir,
    )

    summary = {
        "run_name": cfg.name,
        "config": asdict(cfg),
        "residual_input": args.residual_input,
        "modality_dropout_p": args.modality_dropout_p,
        "stage_a": stage_a_metrics,
        "stage_b": stage_b_metrics,
    }

    with open(run_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    rows = []
    for split in stage_b_metrics:
        rows.append({
            "run_name": cfg.name,
            "split": split,
            "residual_input": args.residual_input,
            "feature_set": cfg.feature_set,
            "modality_dropout_p": args.modality_dropout_p,
            "fusion_mae_overall_um": stage_b_metrics[split]["fusion_mae_overall_um"],
            "vision_mae_overall_um": stage_b_metrics[split]["vision_mae_overall_um"],
            "gain_over_vision_um": stage_b_metrics[split]["gain_over_vision_um"],
        })
    pd.DataFrame(rows).to_csv(run_dir / "comparison_stage_b.csv", index=False)

    print("\n" + "=" * 100)
    print("Two-stage residual training finished")
    print("=" * 100)
    for split in sorted(stage_b_metrics.keys()):
        m = stage_b_metrics[split]
        print(
            f"[{split}] fusion={m['fusion_mae_overall_um']:.2f}um  "
            f"vision={m['vision_mae_overall_um']:.2f}um  "
            f"gain={m['gain_over_vision_um']:.2f}um"
        )


if __name__ == "__main__":
    main()
"""
MATWI — Two-Stage Residual Fusion Training
==========================================

Implements Option 4 (competition-free residual fusion):
  Stage A: train vision-only model (or load checkpoint) on multimodal subset
  Stage B: freeze Stage A, train sensor residual model on
           residual = wear_true - wear_vision_pred

Final inference:
  wear_final = wear_vision + residual_pred

This script is standalone and intended for cluster usage.
One invocation runs both stages and final evaluation end-to-end.
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
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local dynamic imports (same pattern used elsewhere in this repo)
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


_DS = _load_module("dataset_module_two_stage", [_HERE / "DatasetClass_VisionSensors.py"])
_MDL = _load_module("model_module_two_stage", [_HERE / "modelVisionSensor.py"])

MATWIMultimodalDataset = _DS.MATWIMultimodalDataset
SensorScaler = _DS.SensorScaler
MATWIMultimodalModel = _MDL.MATWIMultimodalModel
SensorEncoder = _MDL.SensorEncoder
BACKBONE_FEATDIM = _MDL.BACKBONE_FEATDIM


WEAR_CAP = 450.0
WEAR_TYPES = ["flank_wear", "adhesion", "flank_wear+adhesion"]


@dataclass
class TwoStageConfig:
    name: str
    set_range: str = "1-13"
    image_size: tuple[int, int] = (384, 384)
    normalisation: str = "dataset"
    feature_set: str = "top25"
    batch_size: int = 32
    stage_a_epochs: int = 17
    stage_b_epochs: int = 17
    stage_a_lr: float = 3e-4
    stage_b_lr: float = 3e-4
    stage_a_weight_decay: float = 1e-4
    stage_b_weight_decay: float = 1e-4
    dropout_backbone: float = 0.3
    sensor_encoder_dim: int = 64


class ResidualSensorModel(nn.Module):
    """Sensor branch for Stage B residual prediction."""

    def __init__(
        self,
        n_sensor_features: int,
        sensor_encoder_dim: int = 64,
        residual_input: Literal["sensors_only", "hybrid"] = "sensors_only",
        sensor_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.residual_input = residual_input
        self.sensor_dropout = nn.Dropout(p=sensor_dropout_p)

        self.sensor_encoder = SensorEncoder(
            input_dim=n_sensor_features,
            hidden_dim=sensor_encoder_dim,
            output_dim=sensor_encoder_dim,
        )

        if residual_input == "hybrid":
            self.image_proj = nn.Linear(BACKBONE_FEATDIM, sensor_encoder_dim)
            head_in = sensor_encoder_dim * 2
        else:
            self.image_proj = None
            head_in = sensor_encoder_dim

        self.head = nn.Linear(head_in, 1)

    def forward(
        self,
        sensor_features: torch.Tensor,
        image_embed: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        s = self.sensor_dropout(sensor_features)
        sensor_embed = self.sensor_encoder(s)

        if self.residual_input == "hybrid":
            if image_embed is None:
                raise ValueError("hybrid residual_input requires image_embed")
            image_feat = torch.relu(self.image_proj(image_embed))
            fused = torch.cat([sensor_embed, image_feat], dim=1)
        else:
            fused = sensor_embed

        residual_pred = self.head(fused)
        return {
            "residual_pred": residual_pred,
            "sensor_embed": sensor_embed,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def predictions_to_um(pred: torch.Tensor) -> torch.Tensor:
    return (pred.squeeze(-1) * 1000.0).clamp(0.0, WEAR_CAP)


def compute_mae(
    pred_um: np.ndarray,
    target_um: np.ndarray,
    types: list[str],
) -> dict[str, Any]:
    abs_err = np.abs(pred_um - target_um)
    out: dict[str, Any] = {
        "mae_overall_um": float(abs_err.mean()) if len(abs_err) else math.nan,
        "n_total": len(target_um),
    }
    for wt in WEAR_TYPES:
        mask = np.array([t == wt for t in types], dtype=bool)
        out[f"mae_{wt}_um"] = float(abs_err[mask].mean()) if mask.any() else math.nan
        out[f"n_{wt}"] = int(mask.sum())
    return out


def fmt_metrics(m: dict[str, Any]) -> str:
    return (
        f"overall={m['mae_overall_um']:.2f}um  "
        f"flank={m['mae_flank_wear_um']:.2f}  "
        f"adh={m['mae_adhesion_um']:.2f}  "
        f"f+a={m['mae_flank_wear+adhesion_um']:.2f}  "
        f"(n={m['n_total']})"
    )


def build_loaders(
    cfg: TwoStageConfig,
    data_dir: Path,
    labels_csv: Path,
    sets_csv: Path,
    num_workers: int,
) -> tuple[dict[str, DataLoader], SensorScaler]:
    common = dict(
        data_dir=data_dir,
        labels_csv=labels_csv,
        sets_csv=sets_csv,
        set_range=cfg.set_range,
        normalisation=cfg.normalisation,
        wear_cap=WEAR_CAP,
        impute_zero_wear=False,
        image_size=cfg.image_size,
        feature_set=cfg.feature_set,
    )

    train_ds = MATWIMultimodalDataset(
        **common,
        split="train",
        augment=False,
        sensor_scaler=None,
    )
    print(f"\n  Fitting sensor scaler on {len(train_ds)} training samples...")
    scaler = train_ds.fit_sensor_scaler()

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    }

    for split in ["val", "test"]:
        ds = MATWIMultimodalDataset(
            **common,
            split=split,
            augment=False,
            sensor_scaler=scaler,
        )
        if len(ds) == 0:
            continue
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    sizes = "  |  ".join(f"{k}: {len(v.dataset)}" for k, v in loaders.items())
    print(f"  Loaders: {sizes}")
    return loaders, scaler


def train_stage_a_epoch(
    vision_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    vision_model.train()
    total_loss = 0.0
    total_mae = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = vision_model(images, None)
        loss = criterion(out["wear"], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vision_model.parameters(), max_norm=1.0)
        optimizer.step()

        pred_um = predictions_to_um(out["wear"].detach())
        batch_mae = torch.abs(pred_um - target_um).mean().item()
        bsz = images.size(0)

        total_loss += loss.item() * bsz
        total_mae += batch_mae * bsz
        n += bsz

    return {
        "loss": total_loss / max(n, 1),
        "mae_um": total_mae / max(n, 1),
    }


@torch.no_grad()
def evaluate_stage_a(
    vision_model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    vision_model.eval()
    all_pred, all_target, all_types = [], [], []
    total_loss = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].cpu().numpy()

        out = vision_model(images, None)
        loss = criterion(out["wear"], target)

        all_pred.append(predictions_to_um(out["wear"]).cpu().numpy())
        all_target.append(target_um)
        all_types.extend(list(batch["type"]))

        bsz = images.size(0)
        total_loss += loss.item() * bsz
        n += bsz

    metrics = compute_mae(np.concatenate(all_pred), np.concatenate(all_target), all_types)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def train_stage_b_epoch(
    vision_model: nn.Module,
    residual_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    residual_input: Literal["sensors_only", "hybrid"],
) -> dict[str, float]:
    vision_model.eval()
    residual_model.train()

    total_loss = 0.0
    total_residual_mae_um = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        sensors = batch["sensor_features"].to(device, non_blocking=True)
        target = batch["wear"].to(device, non_blocking=True).unsqueeze(1)

        with torch.no_grad():
            v_out = vision_model(images, None)
            vision_pred = v_out["wear"]
            image_embed = v_out["image_embed"]

        residual_target = target - vision_pred

        optimizer.zero_grad(set_to_none=True)
        r_out = residual_model(
            sensors,
            image_embed=image_embed if residual_input == "hybrid" else None,
        )
        residual_pred = r_out["residual_pred"]

        loss = criterion(residual_pred, residual_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual_model.parameters(), max_norm=1.0)
        optimizer.step()

        residual_target_um = (residual_target.squeeze(1) * 1000.0)
        residual_pred_um = (residual_pred.detach().squeeze(1) * 1000.0)
        batch_residual_mae = torch.abs(residual_pred_um - residual_target_um).mean().item()

        bsz = images.size(0)
        total_loss += loss.item() * bsz
        total_residual_mae_um += batch_residual_mae * bsz
        n += bsz

    return {
        "loss": total_loss / max(n, 1),
        "residual_mae_um": total_residual_mae_um / max(n, 1),
    }


@torch.no_grad()
def evaluate_stage_b(
    vision_model: nn.Module,
    residual_model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    residual_input: Literal["sensors_only", "hybrid"],
) -> dict[str, Any]:
    vision_model.eval()
    residual_model.eval()

    all_final_pred, all_vision_pred, all_target, all_types = [], [], [], []
    total_residual_loss = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        sensors = batch["sensor_features"].to(device, non_blocking=True)
        target = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].cpu().numpy()

        v_out = vision_model(images, None)
        vision_pred = v_out["wear"]
        image_embed = v_out["image_embed"]

        r_out = residual_model(
            sensors,
            image_embed=image_embed if residual_input == "hybrid" else None,
        )
        residual_pred = r_out["residual_pred"]
        residual_target = target - vision_pred
        residual_loss = criterion(residual_pred, residual_target)

        final_pred = vision_pred + residual_pred

        all_vision_pred.append(predictions_to_um(vision_pred).cpu().numpy())
        all_final_pred.append(predictions_to_um(final_pred).cpu().numpy())
        all_target.append(target_um)
        all_types.extend(list(batch["type"]))

        bsz = images.size(0)
        total_residual_loss += residual_loss.item() * bsz
        n += bsz

    target_np = np.concatenate(all_target)
    types = all_types

    vision_metrics = compute_mae(np.concatenate(all_vision_pred), target_np, types)
    final_metrics = compute_mae(np.concatenate(all_final_pred), target_np, types)

    out = {
        "vision": vision_metrics,
        "final": final_metrics,
        "residual_loss": total_residual_loss / max(n, 1),
        "gain_over_vision_um": vision_metrics["mae_overall_um"] - final_metrics["mae_overall_um"],
    }
    return out


def run_two_stage(
    cfg: TwoStageConfig,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    print("\n" + "=" * 90)
    print(f"  TWO-STAGE RUN: {cfg.name}")
    print("=" * 90)
    print(f"  feature_set={cfg.feature_set} | residual_input={args.residual_input} | "
          f"sensor_dropout_p={args.sensor_dropout_p}")

    run_dir = output_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, scaler = build_loaders(
        cfg=cfg,
        data_dir=args.data_dir,
        labels_csv=args.labels_csv,
        sets_csv=args.sets_csv,
        num_workers=args.num_workers,
    )
    scaler.save(run_dir / "sensor_scaler.pkl")

    n_sensor_features = loaders["train"].dataset.n_selected_features

    criterion = nn.MSELoss()

    # Stage A: vision-only
    stage_a_path = run_dir / "stage_a_best.pt"
    vision_model = MATWIMultimodalModel(
        use_sensors=False,
        pretrained=not args.no_pretrained,
        dropout_backbone=cfg.dropout_backbone,
    ).to(device)

    stage_a_history: list[dict[str, Any]] = []

    if args.stage_a_checkpoint:
        print(f"\n  [Stage A] Loading checkpoint: {args.stage_a_checkpoint}")
        ckpt = torch.load(args.stage_a_checkpoint, map_location=device)
        vision_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        best_stage_a_epoch = ckpt.get("epoch", -1)
        best_stage_a_val_mae = ckpt.get("val_mae_um", math.nan)
    else:
        print("\n  [Stage A] Training vision-only model...")
        opt_a = torch.optim.AdamW(
            vision_model.parameters(),
            lr=cfg.stage_a_lr,
            weight_decay=cfg.stage_a_weight_decay,
        )
        best_stage_a_state: Optional[dict[str, Any]] = None
        best_stage_a_val_mae = float("inf")
        best_stage_a_epoch = -1

        for epoch in range(1, cfg.stage_a_epochs + 1):
            t0 = time.time()
            tr = train_stage_a_epoch(vision_model, loaders["train"], opt_a, criterion, device)
            va = evaluate_stage_a(vision_model, loaders["val"], criterion, device)
            elapsed = time.time() - t0

            print(f"    [A {epoch:02d}/{cfg.stage_a_epochs}] train_loss={tr['loss']:.5f} "
                  f"train_mae={tr['mae_um']:.2f}um | val {fmt_metrics(va)} "
                  f"loss={va['loss']:.5f} ({elapsed:.1f}s)")

            row = {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_mae_um": tr["mae_um"],
                "val_loss": va["loss"],
                "val_mae_overall_um": va["mae_overall_um"],
                "val_mae_flank_wear_um": va["mae_flank_wear_um"],
                "val_mae_adhesion_um": va["mae_adhesion_um"],
                "val_mae_flank_wear+adhesion_um": va["mae_flank_wear+adhesion_um"],
                "seconds": elapsed,
            }
            stage_a_history.append(row)

            if va["mae_overall_um"] < best_stage_a_val_mae:
                best_stage_a_val_mae = va["mae_overall_um"]
                best_stage_a_epoch = epoch
                best_stage_a_state = deepcopy(vision_model.state_dict())
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_stage_a_state,
                        "val_mae_um": best_stage_a_val_mae,
                        "config": cfg.__dict__,
                    },
                    stage_a_path,
                )
                print(f"      new best stage A val_mae={best_stage_a_val_mae:.2f}um")

        if best_stage_a_state is None:
            raise RuntimeError("Stage A failed to produce a checkpoint")
        vision_model.load_state_dict(best_stage_a_state)

    pd.DataFrame(stage_a_history).to_csv(run_dir / "stage_a_history.csv", index=False)

    # Freeze Stage A
    for p in vision_model.parameters():
        p.requires_grad = False

    # Stage B: residual model
    print("\n  [Stage B] Training residual model...")
    residual_model = ResidualSensorModel(
        n_sensor_features=n_sensor_features,
        sensor_encoder_dim=cfg.sensor_encoder_dim,
        residual_input=args.residual_input,
        sensor_dropout_p=args.sensor_dropout_p,
    ).to(device)

    opt_b = torch.optim.AdamW(
        residual_model.parameters(),
        lr=cfg.stage_b_lr,
        weight_decay=cfg.stage_b_weight_decay,
    )

    stage_b_history: list[dict[str, Any]] = []
    best_stage_b_state: Optional[dict[str, Any]] = None
    best_stage_b_val_final_mae = float("inf")
    best_stage_b_epoch = -1

    for epoch in range(1, cfg.stage_b_epochs + 1):
        t0 = time.time()
        tr = train_stage_b_epoch(
            vision_model,
            residual_model,
            loaders["train"],
            opt_b,
            criterion,
            device,
            residual_input=args.residual_input,
        )
        va = evaluate_stage_b(
            vision_model,
            residual_model,
            loaders["val"],
            criterion,
            device,
            residual_input=args.residual_input,
        )
        elapsed = time.time() - t0

        final_val = va["final"]["mae_overall_um"]
        vision_val = va["vision"]["mae_overall_um"]

        print(f"    [B {epoch:02d}/{cfg.stage_b_epochs}] train_res_loss={tr['loss']:.5f} "
              f"train_res_mae={tr['residual_mae_um']:.2f}um | "
              f"val vision={vision_val:.2f}um final={final_val:.2f}um "
              f"gain={va['gain_over_vision_um']:+.2f}um ({elapsed:.1f}s)")

        stage_b_history.append(
            {
                "epoch": epoch,
                "train_residual_loss": tr["loss"],
                "train_residual_mae_um": tr["residual_mae_um"],
                "val_vision_mae_overall_um": vision_val,
                "val_final_mae_overall_um": final_val,
                "val_gain_over_vision_um": va["gain_over_vision_um"],
                "val_residual_loss": va["residual_loss"],
                "seconds": elapsed,
            }
        )

        if final_val < best_stage_b_val_final_mae:
            best_stage_b_val_final_mae = final_val
            best_stage_b_epoch = epoch
            best_stage_b_state = deepcopy(residual_model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_stage_b_state,
                    "val_final_mae_um": best_stage_b_val_final_mae,
                    "config": cfg.__dict__,
                    "residual_input": args.residual_input,
                    "sensor_dropout_p": args.sensor_dropout_p,
                },
                run_dir / "stage_b_best.pt",
            )
            print(f"      new best stage B val_final_mae={best_stage_b_val_final_mae:.2f}um")

    pd.DataFrame(stage_b_history).to_csv(run_dir / "stage_b_history.csv", index=False)

    if best_stage_b_state is None:
        raise RuntimeError("Stage B failed to produce a checkpoint")
    residual_model.load_state_dict(best_stage_b_state)

    # Final evaluation on val/test
    print("\n  --- Final evaluation (best Stage B checkpoint) ---")
    final_splits: dict[str, Any] = {}
    for split in [k for k in ("val", "test") if k in loaders]:
        m = evaluate_stage_b(
            vision_model,
            residual_model,
            loaders[split],
            criterion,
            device,
            residual_input=args.residual_input,
        )
        final_splits[split] = m
        print(f"  [{split:5s}] vision={m['vision']['mae_overall_um']:.2f}um  "
              f"final={m['final']['mae_overall_um']:.2f}um  "
              f"gain={m['gain_over_vision_um']:+.2f}um")

    with open(run_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "experiment": cfg.name,
                "mode": "two_stage_residual",
                "config": cfg.__dict__,
                "residual_branch": {
                    "residual_input": args.residual_input,
                    "sensor_dropout_p": args.sensor_dropout_p,
                },
                "stage_a": {
                    "checkpoint": str(stage_a_path if not args.stage_a_checkpoint else args.stage_a_checkpoint),
                    "best_epoch": best_stage_a_epoch,
                    "best_val_mae_um": best_stage_a_val_mae,
                    "trained_here": args.stage_a_checkpoint is None,
                },
                "stage_b": {
                    "best_epoch": best_stage_b_epoch,
                    "best_val_final_mae_um": best_stage_b_val_final_mae,
                },
                "splits": final_splits,
            },
            fh,
            indent=2,
        )

    return {
        "name": cfg.name,
        "stage_a_best_val_mae_um": best_stage_a_val_mae,
        "stage_b_best_val_final_mae_um": best_stage_b_val_final_mae,
        "splits": final_splits,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and evaluate two-stage residual fusion (vision + sensor residual)."
    )

    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--sets-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--run-name", type=str, default="two_stage_residual")
    p.add_argument("--feature-set", type=str, default="top25", choices=["raw25", "top25", "all40"])

    p.add_argument("--residual-input", type=str, default="sensors_only", choices=["sensors_only", "hybrid"])
    p.add_argument("--sensor-dropout-p", type=float, default=0.0)

    p.add_argument("--stage-a-epochs", type=int, default=17)
    p.add_argument("--stage-b-epochs", type=int, default=17)
    p.add_argument("--stage-a-lr", type=float, default=3e-4)
    p.add_argument("--stage-b-lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sensor-encoder-dim", type=int, default=64)

    p.add_argument("--set-range", type=str, default="1-13", choices=["1-13", "1-17"])
    p.add_argument("--normalisation", type=str, default="dataset", choices=["dataset", "imagenet"])

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--stage-a-checkpoint", type=Path, default=None,
                   help="If provided, skip Stage A training and load this vision checkpoint.")

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

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    cfg = TwoStageConfig(
        name=args.run_name,
        set_range=args.set_range,
        normalisation=args.normalisation,
        feature_set=args.feature_set,
        batch_size=args.batch_size,
        stage_a_epochs=args.stage_a_epochs,
        stage_b_epochs=args.stage_b_epochs,
        stage_a_lr=args.stage_a_lr,
        stage_b_lr=args.stage_b_lr,
        sensor_encoder_dim=args.sensor_encoder_dim,
    )

    print(f"Device     : {device}")
    print(f"Output dir : {output_dir}")
    print(f"Run name   : {cfg.name}")

    _ = run_two_stage(cfg, args, device, output_dir)


if __name__ == "__main__":
    main()
