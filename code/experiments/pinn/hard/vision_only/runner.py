import torch
import time
import numpy as np
import pandas as pd
import json
from pathlib import Path
from copy import deepcopy
import torch.nn as nn

from constants.matwi_dataset_constants import PINN_HARD_TRAIN_SETS
from experiments.pinn.hard.vision_only.config import HardConfig
from experiments.vision_only.config import VisionExperimentConfig
from experiments.vision_only.runner import build_loaders
from metrics.mae import compute_mae_v1
from metrics.fmt import fmt_metrics_v1
from utils.experiment import predictions_to_um
from model.pinn.constraint.hard import HardTaylorConstraint, recommend_C
from model.vision_only.vision_model import MATWIVisionModel

def _predict(model, hard, images, set_ids, image_ids):
    """Return (pred_norm, corr_um). hard=None ⇒ unconstrained control."""
    out = model(images)
    raw = out["wear"]
    if hard is None:
        return raw.squeeze(-1) if raw.dim() > 1 else raw, None
    pred_norm, corr_um, _ = hard(raw, set_ids, image_ids)
    return pred_norm, corr_um

def train_one_epoch_hard(model, hard, loader, optimizer, scheduler, criterion, device):
    model.train()
    tot_loss = tot_mae = 0.0; n = 0
    corr_abs_sum = 0.0; sat_sum = 0
    for batch in loader:
        images    = batch["image"].to(device, non_blocking=True)
        target    = batch["wear"].to(device, non_blocking=True)
        target_um = batch["wear_raw"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred_norm, corr_um = _predict(model, hard, images, batch["set"], batch["image_id"])
        loss = criterion(pred_norm, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bsz = images.size(0)
        pred_um = predictions_to_um(pred_norm.detach())
        tot_mae  += torch.abs(pred_um - target_um).mean().item() * bsz
        tot_loss += loss.item() * bsz; n += bsz
        if corr_um is not None and hard is not None:
            corr_abs_sum += corr_um.detach().abs().mean().item() * bsz
            sat_sum += (corr_um.detach().abs() > 0.95 * hard.C_um).sum().item()
    res = {"loss": tot_loss/max(n,1), "mae_um": tot_mae/max(n,1)}
    res["corr_abs_um"] = corr_abs_sum/max(n,1) if hard is not None else 0.0
    res["sat_frac"]    = sat_sum/max(n,1) if hard is not None else 0.0
    return res

@torch.no_grad()
def evaluate(model, hard, loader, device) -> dict:
    model.eval()
    preds, targs, types = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        pred_norm, _ = _predict(model, hard, images, batch["set"], batch["image_id"])
        preds.append(predictions_to_um(pred_norm).cpu().numpy())
        targs.append(batch["wear_raw"].numpy())
        types.extend(list(batch["type"]))
    return compute_mae_v1(np.concatenate(preds), np.concatenate(targs), types)

def run_experiment(cfg: HardConfig, args, device, out_dir: Path) -> None:
    print(f"\n{'='*80}\n  STAGE 3 HARD: {cfg.name}  (backbone={cfg.backbone})")
    run_dir = out_dir / cfg.name; run_dir.mkdir(parents=True, exist_ok=True)

    ecfg = VisionExperimentConfig(
        name=cfg.name, backbone=cfg.backbone, set_range=cfg.set_range,
        image_size=cfg.image_size, epochs=cfg.epochs, lr=cfg.lr,
        normalisation=cfg.normalisation, augment=False,
        weight_decay=cfg.weight_decay, batch_size=cfg.batch_size,
        use_scheduler=cfg.use_scheduler)
    loaders = build_loaders(ecfg, args.data_dir, args.labels_csv,
                            args.sets_csv, args.num_workers)

    # ── Build the hard constraint (or None for the control) ───────────────────
    hard = None
    C_um = None
    if args.hard:
        C_um = args.C_um
        if args.auto_C:
            C_um = recommend_C(args.labels_csv, args.sets_csv, args.constants,
                               PINN_HARD_TRAIN_SETS, percentile=args.auto_C_pct)
            print(f"  [auto-C] {args.auto_C_pct:.0f}th pct of |wear-VB_taylor| on train "
                  f"=> C_um={C_um:.1f}")
        hard = HardTaylorConstraint(args.constants, args.sets_csv, C_um=C_um).to(device)
        print(f"  HardConstraint: {hard.extra_repr()}")
    else:
        print("  (no --hard: unconstrained control)")

    model = MATWIVisionModel(backbone=cfg.backbone, pretrained=not args.no_pretrained,
                             head_hidden_dim=args.head_hidden_dim,
                             dropout_backbone=args.dropout_backbone,
                             dropout_head=args.dropout_head).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, total_steps=cfg.epochs*len(loaders["train"]),
            pct_start=0.3, anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4)
    criterion = nn.MSELoss() if cfg.data_loss == "mse" else nn.L1Loss()

    history, best_state, best_val, best_ep = [], None, float("inf"), -1
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch_hard(model, hard, loaders["train"], optimizer,
                                  scheduler, criterion, device)
        val = evaluate(model, hard, loaders["val"], device)
        print(f"  [{epoch:02d}/{cfg.epochs}] loss={tr['loss']:.5f} "
              f"corr={tr['corr_abs_um']:.1f}µm sat={tr['sat_frac']:.2f} "
              f"train_mae={tr['mae_um']:.1f}µm | val {fmt_metrics_v1(val)} ({time.time()-t0:.1f}s)")
        history.append({"epoch": epoch, "train_loss": tr["loss"],
                        "train_mae_um": tr["mae_um"], "train_corr_abs_um": tr["corr_abs_um"],
                        "train_sat_frac": tr["sat_frac"],
                        "val_mae_overall_um": val["mae_overall_um"],
                        "val_mae_flank_wear_um": val["mae_flank_wear_um"],
                        "val_mae_adhesion_um": val["mae_adhesion_um"],
                        "val_mae_flank_wear+adhesion_um": val["mae_flank_wear+adhesion_um"]})
        if val["mae_overall_um"] < best_val:
            best_val, best_ep = val["mae_overall_um"], epoch
            best_state = deepcopy(model.state_dict())
            torch.save({"epoch": epoch, "model_state_dict": best_state,
                        "val_mae_um": best_val, "config": cfg.__dict__,
                        "hard": {"enabled": args.hard, "C_um": C_um}},
                       run_dir / "best_model.pt")
            print(f"    ✓ new best val_mae={best_val:.2f}µm")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  --- Final results: {cfg.name} (best epoch {best_ep}) ---")
    final = {}
    for split in [k for k in ("val", "test", "unseen") if k in loaders]:
        m = evaluate(model, hard, loaders[split], device)
        final[split] = m
        print(f"  [{split:7s}] {fmt_metrics_v1(m)}")
    with open(run_dir / "results.json", "w") as fh:
        json.dump({"experiment": cfg.name, "config": cfg.__dict__,
                   "hard": {"enabled": bool(args.hard), "C_um": C_um,
                            "anchor": "taylor_ramp_from_Vc"},
                   "best_epoch": best_ep, "best_val_mae_um": best_val,
                   "splits": final}, fh, indent=2)
