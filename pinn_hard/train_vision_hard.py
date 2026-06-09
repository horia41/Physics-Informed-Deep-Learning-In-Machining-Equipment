

from __future__ import annotations
import argparse, importlib.util, json, sys, time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent


def _load_module(name: str, candidates: list[Path]):
    for path in candidates:
        if path.exists():
            print(f"[_load_module] {name} <- {path}")
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"Could not find '{name}'. Searched:\n  "
                            + "\n  ".join(str(p) for p in candidates))

# Stage-1 vision building blocks (self-resolves its own model/dataset deps in
# vision-only/), and the local hard-constraint module.
_TV = _load_module("train_vision_base", [
    _HERE / "train_vision.py",
    _HERE / ".." / "vision-only" / "train_vision.py",
])
_TH = _load_module("taylor_hard", [_HERE / "taylor_hard.py"])

MATWIVisionModel  = _TV.MATWIVisionModel
build_loaders     = _TV.build_loaders
predictions_to_um = _TV.predictions_to_um
fmt_metrics       = _TV.fmt_metrics
set_seed          = _TV.set_seed
resolve_device    = _TV.resolve_device
WEAR_TYPES        = _TV.WEAR_TYPES
HardTaylorConstraint = _TH.HardTaylorConstraint
recommend_C          = _TH.recommend_C

# Stage-1 train split (CK45 1-13) — only needed for --auto-C on the train sets.
TRAIN_SETS_1_13 = [1, 2, 5, 7, 8, 10, 11]


@dataclass
class HardConfig:
    name:          str
    backbone:      str   = "efficientnetv2_s"
    set_range:     str   = "1-13"
    image_size:    tuple = (384, 384)
    epochs:        int   = 17
    lr:            float = 3e-4
    normalisation: str   = "dataset"
    weight_decay:  float = 1e-4
    batch_size:    int   = 32
    use_scheduler: bool  = True
    data_loss:     str   = "mse"


def compute_mae(pred_um: np.ndarray, target_um: np.ndarray, types: list[str]) -> dict:
    abs_err = np.abs(pred_um - target_um)
    out = {"mae_overall_um": float(abs_err.mean()) if len(abs_err) else float("nan"),
           "n_total": int(len(target_um))}
    for wt in WEAR_TYPES:
        mask = np.array([t == wt for t in types], dtype=bool)
        out[f"mae_{wt}_um"] = float(abs_err[mask].mean()) if mask.any() else float("nan")
        out[f"n_{wt}"] = int(mask.sum())
    return out


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
def evaluate_hard(model, hard, loader, device) -> dict:
    model.eval()
    preds, targs, types = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        pred_norm, _ = _predict(model, hard, images, batch["set"], batch["image_id"])
        preds.append(predictions_to_um(pred_norm).cpu().numpy())
        targs.append(batch["wear_raw"].numpy())
        types.extend(list(batch["type"]))
    return compute_mae(np.concatenate(preds), np.concatenate(targs), types)


def run(cfg: HardConfig, args, device, out_dir: Path) -> None:
    print(f"\n{'='*80}\n  STAGE 3 HARD: {cfg.name}  (backbone={cfg.backbone})")
    run_dir = out_dir / cfg.name; run_dir.mkdir(parents=True, exist_ok=True)

    ecfg = _TV.ExperimentConfig(
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
                               TRAIN_SETS_1_13, percentile=args.auto_C_pct)
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
        val = evaluate_hard(model, hard, loaders["val"], device)
        print(f"  [{epoch:02d}/{cfg.epochs}] loss={tr['loss']:.5f} "
              f"corr={tr['corr_abs_um']:.1f}µm sat={tr['sat_frac']:.2f} "
              f"train_mae={tr['mae_um']:.1f}µm | val {fmt_metrics(val)} ({time.time()-t0:.1f}s)")
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
        m = evaluate_hard(model, hard, loaders[split], device)
        final[split] = m
        print(f"  [{split:7s}] {fmt_metrics(m)}")
    with open(run_dir / "results.json", "w") as fh:
        json.dump({"experiment": cfg.name, "config": cfg.__dict__,
                   "hard": {"enabled": bool(args.hard), "C_um": C_um,
                            "anchor": "taylor_ramp_from_Vc"},
                   "best_epoch": best_ep, "best_val_mae_um": best_val,
                   "splits": final}, fh, indent=2)


def main():
    p = argparse.ArgumentParser(description="MATWI Stage 3: vision + HARD Taylor constraint.")
    p.add_argument("--data-dir",   type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--sets-csv",   type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--constants",  type=Path, required=True, help="taylor_constants.json")
    p.add_argument("--name",       type=str,  default="vision_hard")
    # hard-constraint knobs
    p.add_argument("--hard",       action="store_true", help="enable the band constraint (omit = control)")
    p.add_argument("--C-um",       type=float, default=150.0, help="band half-width µm (VB_taylor ± C)")
    p.add_argument("--auto-C",     action="store_true", help="set C from train |wear-VB_taylor| percentile")
    p.add_argument("--auto-C-pct", type=float, default=95.0)
    # training knobs (defaults = best vision baseline)
    p.add_argument("--backbone",   type=str, default="efficientnetv2_s")
    p.add_argument("--set-range",  type=str, default="1-13", choices=["1-13", "1-17"])
    p.add_argument("--data-loss",  type=str, default="mse", choices=["mse", "l1"])
    p.add_argument("--epochs",     type=int, default=17)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p.add_argument("--no-pretrained",    action="store_true")
    p.add_argument("--head-hidden-dim",  type=int,   default=256)
    p.add_argument("--dropout-backbone", type=float, default=0.3)
    p.add_argument("--dropout-head",     type=float, default=0.4)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    img = (384, 384) if "efficientnet" in args.backbone else (224, 224)
    cfg = HardConfig(name=args.name, backbone=args.backbone, set_range=args.set_range,
                     image_size=img, epochs=args.epochs, lr=args.lr,
                     batch_size=args.batch_size, data_loss=args.data_loss,
                     use_scheduler=("efficientnet" in args.backbone))
    print(f"Device {device} | seed {args.seed} | hard={args.hard} | out {args.output_dir.resolve()}")
    run(cfg, args, device, args.output_dir.resolve())


if __name__ == "__main__":
    main()
