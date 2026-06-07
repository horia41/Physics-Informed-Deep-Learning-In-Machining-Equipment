import argparse
import json
import math
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
from model.vision_only.resnet_efficientnet_vision import MATWIVisionModel
from experiments.utils import predictions_to_um, seed_worker
from metrics.fmt import fmt_metrics
from metrics.mae import compute_mae
from structuredCode.metrics.loss.builder import make_criterion
from structuredCode.constants import WEAR_TYPES,WEAR_CAP
from dataset.matwi.dataset_vision import MATWIVisionDataset

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

def _train_one_epoch(
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

    # Reproducible shuffle order, tied to the global seed set before this call.
    g = torch.Generator()
    g.manual_seed(torch.initial_seed() % 2**32)
    persistent = num_workers > 0

    train_ds = MATWIVisionDataset(**common, split="train", augment=cfg.augment)
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

    # Sets 14-17 (RVS 304) are only available as a held-out generalisation
    # split when they are NOT in training, i.e. set_range="1-13". For "1-17"
    # they are folded into train, so there is nothing held out to evaluate.
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
    return loaders

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

        train_m = _train_one_epoch(
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

