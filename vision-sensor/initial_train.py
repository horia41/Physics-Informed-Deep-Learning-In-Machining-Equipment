# this file performs a first training attempt for model vision-sensor
# will experiment over all 3 fusion strategies and number of sets used 1-13 and 1-17
# see what is fixed below

"""
MATWI multimodal wear estimation — simple training script.
===========================================================
Runs two experiments back-to-back:
  1. set_range = 1-13  (paper split, 647 train samples)
  2. set_range = 1-17  (extended, 1108 train samples)

Everything else is fixed:
  - normalisation    : dataset
  - augment          : False
  - wear_cap         : 450.0 µm
  - image_size       : (384, 384)
  - impute_zero_wear : True
  - batch_size       : 16
  - loss             : L1 (MAE)

Configurable via argparse:
  - fusion_mode  : early | intermediate | late
  - epochs       : any int
  - lr, seed, etc.

Outputs (under --output-dir/):
  {set_range}/history.csv      — per-epoch metrics
  {set_range}/results.json     — final val/test/unseen MAE
  {set_range}/best_model.pt    — best checkpoint by val MAE
  comparison_summary.csv       — side-by-side comparison of both runs

Run example:
  python train.py \
    --data-dir   ./data/matwi \
    --labels-csv ./data/matwi/labels.csv \
    --sets-csv   ./data/matwi/sets.csv \
    --output-dir ./runs/intermediate_no_aug \
    --fusion-mode intermediate \
    --epochs 30
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def _load_module(name: str, candidates: Sequence[Path]):
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        f"Could not find '{name}'. Searched:\n" +
        "\n".join(f"  {p}" for p in candidates)
    )


_HERE = Path(__file__).resolve().parent

_DS  = _load_module("dataset_module", [
    _HERE / "DatasetClass_VisionSensors.py",
    _HERE / "DatasetClassVisionSensors.py",
])
_MDL = _load_module("model_module", [
    _HERE / "model.py",
    _HERE / "modelVisionSensor.py",
])

MATWIMultimodalDataset = _DS.MATWIMultimodalDataset
SensorScaler           = _DS.SensorScaler
MATWIWearModel         = _MDL.MATWIWearModel


# ── Fixed settings ─────────────────────────────────────────────────────────────

WEAR_CAP         = 450.0
IMAGE_SIZE       = (384, 384)
NORMALISATION    = "imagenet"
IMPUTE_ZERO_WEAR = False
BATCH_SIZE       = 16
WEAR_TYPES       = ["flank_wear", "adhesion", "flank_wear+adhesion"]
SET_RANGES       = ["1-13", "1-17"]


# ── Utilities ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


def predictions_to_um(pred: torch.Tensor) -> torch.Tensor:
    return (pred.squeeze(1) * 1000.0).clamp(0.0, WEAR_CAP)


def compute_mae(
    pred_um:   np.ndarray,
    target_um: np.ndarray,
    types:     List[str],
) -> Dict[str, Any]:
    abs_err = np.abs(pred_um - target_um)
    out: Dict[str, Any] = {
        "mae_overall_um": float(abs_err.mean()) if len(abs_err) else math.nan,
        "n_total":        int(len(target_um)),
    }
    for wt in WEAR_TYPES:
        mask = np.array([t == wt for t in types], dtype=bool)
        out[f"mae_{wt}_um"] = float(abs_err[mask].mean()) if mask.any() else math.nan
        out[f"n_{wt}"]      = int(mask.sum())
    return out


def print_metrics(label: str, m: Dict[str, Any]) -> None:
    print(
        f"  [{label:10s}] "
        f"overall={m['mae_overall_um']:.2f}µm  "
        f"flank={m['mae_flank_wear_um']:.2f}  "
        f"adh={m['mae_adhesion_um']:.2f}  "
        f"f+a={m['mae_flank_wear+adhesion_um']:.2f}  "
        f"(n={m['n_total']})"
    )


# ── Data ───────────────────────────────────────────────────────────────────────

def build_loaders(
    set_range:   str,
    scaler:      SensorScaler,
    data_dir:    Path,
    labels_csv:  Path,
    sets_csv:    Path,
    num_workers: int,
) -> Dict[str, DataLoader]:
    """
    Build loaders for train / val / test, and unseen only when set_range='1-13'
    (when set_range='1-17', sets 14-17 are already in training — nothing unseen).
    """
    common = dict(
        data_dir         = data_dir,
        labels_csv       = labels_csv,
        sets_csv         = sets_csv,
        set_range        = set_range,
        normalisation    = NORMALISATION,
        # augment          = False,
        # augment_strategy = "uniform",
        wear_cap         = WEAR_CAP,
        impute_zero_wear = IMPUTE_ZERO_WEAR,
        image_size       = IMAGE_SIZE,
        sensor_scaler    = scaler,
    )

    train_ds = MATWIMultimodalDataset(**common, split="train", augment= True, augment_strategy = "oversample_adhesion")
    loaders  = {
        "train": DataLoader(
            train_ds,
            batch_size  = BATCH_SIZE,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = True,
        )
    }

    eval_splits = ["val", "test"] + (["unseen"] if set_range == "1-13" else [])
    for split in eval_splits:
        ds = MATWIMultimodalDataset(**common, split=split)
        if len(ds) == 0:
            print(f"  [warn] split='{split}' is empty for set_range={set_range}, skipping.")
            continue
        loaders[split] = DataLoader(
            ds,
            batch_size  = BATCH_SIZE,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

    print(f"  Loaders: " + "  |  ".join(
        f"{k}: {len(v.dataset)}" for k, v in loaders.items()
    ))
    return loaders


def fit_scaler(
    set_range:  str,
    data_dir:   Path,
    labels_csv: Path,
    sets_csv:   Path,
    save_path:  Path,
) -> SensorScaler:
    """Fit sensor scaler on the training set, save to disk."""
    if save_path.exists():
        print(f"  [scaler] Loading existing scaler from {save_path}")
        return SensorScaler.load(save_path)

    print(f"  [scaler] Fitting on set_range={set_range} training set...")
    ds = MATWIMultimodalDataset(
        data_dir         = data_dir,
        labels_csv       = labels_csv,
        sets_csv         = sets_csv,
        split            = "train",
        set_range        = set_range,
        normalisation    = NORMALISATION,
        augment          = False,
        augment_strategy = "uniform",
        wear_cap         = WEAR_CAP,
        impute_zero_wear = IMPUTE_ZERO_WEAR,
        image_size       = IMAGE_SIZE,
        sensor_scaler    = None,
    )
    scaler = ds.fit_sensor_scaler(save_path=save_path, verbose=True)
    return scaler


# ── Train / evaluate ───────────────────────────────────────────────────────────

def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    scheduler:    torch.optim.lr_scheduler.OneCycleLR,
    criterion:    nn.Module,
    device:       torch.device,
    epoch:        int,
    total_epochs: int,
    log_every:    int,
) -> Dict[str, float]:
    model.train()

    total_loss   = 0.0
    total_mae_um = 0.0
    n            = 0

    for step, batch in enumerate(loader, 1):
        images    = batch["image"].to(device, non_blocking=True)
        sensors   = batch["sensor_features"].to(device, non_blocking=True)
        target    = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out  = model(images, sensors)
        loss = criterion(out["wear"], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # scheduler.step()

        pred_um   = predictions_to_um(out["wear"].detach())
        batch_mae = torch.abs(pred_um - target_um).mean().item()
        bsz       = images.size(0)

        total_loss   += loss.item() * bsz
        total_mae_um += batch_mae   * bsz
        n            += bsz

        if log_every > 0 and step % log_every == 0:
            print(
                f"    step {step:04d}/{len(loader):04d}  "
                f"loss={loss.item():.5f}  "
                f"mae={batch_mae:.2f}µm  "
                # f"lr={scheduler.get_last_lr()[0]:.2e}"
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    return {"loss": total_loss / max(n, 1), "mae_um": total_mae_um / max(n, 1)}


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Dict[str, Any]:
    model.eval()

    all_pred:   List[np.ndarray] = []
    all_target: List[np.ndarray] = []
    all_types:  List[str]        = []
    total_loss = 0.0
    n          = 0

    for batch in loader:
        images    = batch["image"].to(device, non_blocking=True)
        sensors   = batch["sensor_features"].to(device, non_blocking=True)
        target    = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].cpu().numpy()

        out  = model(images, sensors)
        loss = criterion(out["wear"], target)

        all_pred.append(predictions_to_um(out["wear"]).cpu().numpy())
        all_target.append(target_um)
        all_types.extend(list(batch["type"]))

        bsz         = images.size(0)
        total_loss += loss.item() * bsz
        n          += bsz

    if not all_pred:
        return {"mae_overall_um": math.nan, "n_total": 0}

    pred_arr   = np.concatenate(all_pred)
    target_arr = np.concatenate(all_target)
    metrics    = compute_mae(pred_arr, target_arr, all_types)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


# ── Single set_range run ───────────────────────────────────────────────────────

def run_one(
    set_range:    str,
    fusion_mode:  str,
    epochs:       int,
    args:         argparse.Namespace,
    device:       torch.device,
    output_dir:   Path,
) -> Dict[str, Any]:
    print(f"\n{'='*80}")
    print(f"  set_range={set_range}  |  fusion={fusion_mode}  |  epochs={epochs}")
    print(f"{'='*80}")

    run_dir = output_dir / set_range.replace("-", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Scaler
    scaler = fit_scaler(
        set_range  = set_range,
        data_dir   = args.data_dir,
        labels_csv = args.labels_csv,
        sets_csv   = args.sets_csv,
        save_path  = run_dir / "sensor_scaler.pkl",
    )

    # Loaders
    loaders = build_loaders(
        set_range   = set_range,
        scaler      = scaler,
        data_dir    = args.data_dir,
        labels_csv  = args.labels_csv,
        sets_csv    = args.sets_csv,
        num_workers = args.num_workers,
    )

    # Model
    model = MATWIWearModel(
        fusion_mode       = fusion_mode,
        use_sensors       = True,
        sensor_hidden_dim = args.sensor_hidden_dim,
        head_hidden_dim   = args.head_hidden_dim,
        dropout_backbone  = args.dropout_backbone,
        dropout_head      = args.dropout_head,
        pretrained        = not args.no_pretrained,
    ).to(device)

    # Optimiser + OneCycleLR
    steps_per_epoch = len(loaders["train"])
    total_steps     = epochs * steps_per_epoch

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr           = args.lr,
    #     weight_decay = args.weight_decay,
    # )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr           = args.lr,
    #     total_steps      = total_steps,
    #     pct_start        = 0.3,
    #     anneal_strategy  = "cos",
    #     div_factor       = 25.0,
    #     final_div_factor = 1e4,
    # )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # No scheduler — fixed LR for all epochs, matching the paper's approach
    scheduler = None

    criterion = nn.L1Loss()

    # Training loop
    history:    List[Dict[str, Any]] = []
    best_state: Optional[Dict]       = None
    best_val_mae = float("inf")
    best_epoch   = -1

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\n[epoch {epoch:03d}/{epochs}]  set_range={set_range}")

        train_m = train_one_epoch(
            model        = model,
            loader       = loaders["train"],
            optimizer    = optimizer,
            scheduler    = scheduler,
            criterion    = criterion,
            device       = device,
            epoch        = epoch,
            total_epochs = epochs,
            log_every    = args.log_every,
        )
        val_m = evaluate(model, loaders["val"], criterion, device)
        elapsed = time.time() - t0

        print(
            f"  train  loss={train_m['loss']:.5f}  mae={train_m['mae_um']:.2f}µm  |  "
            f"val  mae={val_m['mae_overall_um']:.2f}µm  "
            f"flank={val_m['mae_flank_wear_um']:.2f}  "
            f"adh={val_m['mae_adhesion_um']:.2f}  "
            f"f+a={val_m['mae_flank_wear+adhesion_um']:.2f}  "
            f"({elapsed:.1f}s)"
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
            # "lr":                             scheduler.get_last_lr()[0],
            "seconds":                        elapsed,
            "lr": args.lr
        })

        if val_m["mae_overall_um"] < best_val_mae:
            best_val_mae = val_m["mae_overall_um"]
            best_epoch   = epoch
            best_state   = deepcopy(model.state_dict())
            ckpt_path    = run_dir / "best_model.pt"
            torch.save({
                "epoch":            epoch,
                "model_state_dict": best_state,
                "val_mae_um":       best_val_mae,
                "fusion_mode":      fusion_mode,
                "set_range":        set_range,
            }, ckpt_path)
            print(f"  [ckpt] best val_mae={best_val_mae:.2f}µm at epoch {epoch}")

    # Save history
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # Final evaluation with best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best model (epoch {best_epoch}, val={best_val_mae:.2f}µm)")

    print(f"\n  --- Final results (set_range={set_range}) ---")
    final_results: Dict[str, Dict] = {}
    for split in [k for k in ("val", "test", "unseen") if k in loaders]:
        m = evaluate(model, loaders[split], criterion, device)
        final_results[split] = m
        print_metrics(split, m)

    # Save results JSON
    with open(run_dir / "results.json", "w") as fh:
        json.dump({
            "set_range":       set_range,
            "fusion_mode":     fusion_mode,
            "epochs":          epochs,
            "best_epoch":      best_epoch,
            "best_val_mae_um": best_val_mae,
            "splits": {
                split: {k: v for k, v in m.items()
                        if not isinstance(v, np.ndarray)}
                for split, m in final_results.items()
            },
        }, fh, indent=2)

    return {
        "set_range":       set_range,
        "best_epoch":      best_epoch,
        "best_val_mae_um": best_val_mae,
        "final_results":   final_results,
    }


# ── Comparison summary ─────────────────────────────────────────────────────────

def write_comparison(
    results:    List[Dict[str, Any]],
    output_dir: Path,
    fusion_mode: str,
    epochs:     int,
) -> None:
    rows = []
    for res in results:
        for split, m in res["final_results"].items():
            rows.append({
                "set_range":                   res["set_range"],
                "fusion_mode":                 fusion_mode,
                "epochs":                      epochs,
                "eval_split":                  split,
                "best_epoch":                  res["best_epoch"],
                "best_val_mae_um":             res["best_val_mae_um"],
                "mae_overall_um":              m["mae_overall_um"],
                "mae_flank_wear_um":           m["mae_flank_wear_um"],
                "mae_adhesion_um":             m["mae_adhesion_um"],
                "mae_flank_wear+adhesion_um":  m["mae_flank_wear+adhesion_um"],
                "n_total":                     m["n_total"],
            })

    df = pd.DataFrame(rows)
    path = output_dir / "comparison_summary.csv"
    df.to_csv(path, index=False)

    print(f"\n{'='*80}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"\nSaved → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MATWI simple training: set_range 1-13 vs 1-17."
    )
    p.add_argument("--data-dir",    type=Path, required=True)
    p.add_argument("--labels-csv",  type=Path, required=True)
    p.add_argument("--sets-csv",    type=Path, required=True)
    p.add_argument("--output-dir",  type=Path, required=True)
    p.add_argument("--fusion-mode", type=str,  required=True,
                   choices=["early", "intermediate", "late"])
    p.add_argument("--epochs",      type=int,  required=True)

    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight-decay",     type=float, default=1e-4)
    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--device",           type=str,   default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-pretrained",    action="store_true")
    p.add_argument("--head-hidden-dim",  type=int,   default=256)
    p.add_argument("--sensor-hidden-dim",type=int,   default=128)
    p.add_argument("--dropout-backbone", type=float, default=0.3)
    p.add_argument("--dropout-head",     type=float, default=0.4)
    p.add_argument("--log-every",        type=int,   default=20,
                   help="Print step-level log every N steps (0 = epoch-level only).")
    p.add_argument("--only-set-range",   type=str,   default=None,
                   choices=["1-13", "1-17"],
                   help="Run only one set_range instead of both.")
    return p


def main() -> None:
    args       = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device(args.device if args.device != "auto" else "mps"))

    print(f"Device      : {device}")
    print(f"Fusion mode : {args.fusion_mode}")
    print(f"Epochs      : {args.epochs}")
    print(f"Output dir  : {output_dir}")
    print(f"Fixed       : norm={NORMALISATION} | wear_cap={WEAR_CAP} | "
          f"image={IMAGE_SIZE} | impute_zero={IMPUTE_ZERO_WEAR}")

    ranges_to_run = (
        [args.only_set_range] if args.only_set_range else SET_RANGES
    )

    all_results = []
    for set_range in ranges_to_run:
        result = run_one(
            set_range   = set_range,
            fusion_mode = args.fusion_mode,
            epochs      = args.epochs,
            args        = args,
            device      = device,
            output_dir  = output_dir,
        )
        all_results.append(result)

    if len(all_results) > 1:
        write_comparison(all_results, output_dir, args.fusion_mode, args.epochs)


if __name__ == "__main__":
    main()