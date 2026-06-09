

from __future__ import annotations
import argparse, importlib.util, json, math, sys, time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent

def _load_module(name: str, candidates: list[Path]):
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"Could not find '{name}'. Searched: "
                            + ", ".join(str(p) for p in candidates))

# Reuse Stage-1 training utilities + the physics loss (all by filename, like the original).
# train_vision.py is looked up in this dir first, then in the sibling vision-only/ dir
# (where the Stage-1 files live). Loaded from vision-only/, it self-resolves its own
# dataset/model deps there, so nothing needs copying into pinn/.
_TV  = _load_module("train_vision_base", [
    _HERE / "train_vision.py",
    _HERE / ".." / "vision-only" / "train_vision.py",
])
_TPL = _load_module("taylor_physics_loss", [
    _HERE / "taylor_physics_loss.py",
])

MATWIVisionModel   = _TV.MATWIVisionModel
build_loaders      = _TV.build_loaders
evaluate           = _TV.evaluate
predictions_to_um  = _TV.predictions_to_um
fmt_metrics        = _TV.fmt_metrics
set_seed           = _TV.set_seed
resolve_device     = _TV.resolve_device
WEAR_TYPES         = _TV.WEAR_TYPES
TaylorPhysicsLoss  = _TPL.TaylorPhysicsLoss


@dataclass
class Stage3Config:
    name:          str
    backbone:      str   = "efficientnetv2_s"
    set_range:     str   = "1-13"
    image_size:    tuple[int, int] = (384, 384)
    epochs:        int   = 17
    lr:            float = 3e-4
    normalisation: str   = "dataset"      # best vision baseline used dataset norm
    weight_decay:  float = 1e-4
    batch_size:    int   = 32
    use_scheduler: bool  = True
    data_loss:     str   = "mse"          # best vision config used MSE


# ── Training loop with the physics term ──────────────────────────────────────
def train_one_epoch_taylor(model, loader, optimizer, scheduler, data_criterion,
                           physics_loss, device, epoch0, log_every=0):
    """epoch0 is 0-indexed (for the lambda warm-up schedule)."""
    model.train()
    tot_loss = tot_data = tot_phys = tot_mae = 0.0; n = 0
    for step, batch in enumerate(loader, 1):
        images    = batch["image"].to(device, non_blocking=True)
        target    = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out  = model(images)
        d_loss = data_criterion(out["wear"], target)
        p_loss = physics_loss(out["wear"], batch["set"], batch["image_id"],
                              list(batch["material"]), epoch=epoch0)
        loss = d_loss + p_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bsz = images.size(0)
        pred_um = predictions_to_um(out["wear"].detach())
        tot_mae  += torch.abs(pred_um - target_um).mean().item() * bsz
        tot_loss += loss.item()  * bsz
        tot_data += d_loss.item()* bsz
        tot_phys += float(p_loss.item()) * bsz
        n += bsz
        if log_every and step % log_every == 0:
            print(f"    step {step:04d}/{len(loader)}  loss={loss.item():.5f} "
                  f"data={d_loss.item():.5f} phys={float(p_loss.item()):.6f}")
    return {"loss": tot_loss/max(n,1), "data_loss": tot_data/max(n,1),
            "phys_loss": tot_phys/max(n,1), "mae_um": tot_mae/max(n,1)}


def run(cfg: Stage3Config, args, device, out_dir: Path) -> dict:
    print(f"\n{'='*80}\n  STAGE 3: {cfg.name}  (backbone={cfg.backbone}, data_loss={cfg.data_loss})")
    print(f"  physics: lambda_max={args.lambda_max} warmup={args.warmup} "
          f"apply_to={args.apply_to} one_sided={args.one_sided} "
          f"taylor_slope={args.use_taylor_slope}\n{'='*80}")
    run_dir = out_dir / cfg.name; run_dir.mkdir(parents=True, exist_ok=True)

    # mirror train_vision's ExperimentConfig so build_loaders is happy
    ecfg = _TV.ExperimentConfig(
        name=cfg.name, backbone=cfg.backbone, set_range=cfg.set_range,
        image_size=cfg.image_size, epochs=cfg.epochs, lr=cfg.lr,
        normalisation=cfg.normalisation, augment=False,
        weight_decay=cfg.weight_decay, batch_size=cfg.batch_size,
        use_scheduler=cfg.use_scheduler)
    loaders = build_loaders(ecfg, args.data_dir, args.labels_csv,
                            args.sets_csv, args.num_workers)

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

    data_criterion = nn.MSELoss() if cfg.data_loss == "mse" else nn.L1Loss()
    physics_loss = TaylorPhysicsLoss(
        constants_path=args.constants, lambda_max=args.lambda_max,
        warmup_epochs=args.warmup, apply_to=args.apply_to,
        one_sided=args.one_sided, use_taylor_slope=args.use_taylor_slope)
    print(f"  PhysicsLoss: {physics_loss.extra_repr()}")
    eval_crit = nn.L1Loss()                        # eval loss is reported in MAE anyway

    history, best_state, best_val, best_ep = [], None, float("inf"), -1
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch_taylor(model, loaders["train"], optimizer, scheduler,
                                    data_criterion, physics_loss, device,
                                    epoch0=epoch-1, log_every=args.log_every)
        val = evaluate(model, loaders["val"], eval_crit, device)
        lr_now = (scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"])
        print(f"  [{epoch:02d}/{cfg.epochs}] data={tr['data_loss']:.5f} "
              f"phys={tr['phys_loss']:.6f} train_mae={tr['mae_um']:.1f}µm | "
              f"val {fmt_metrics(val)} ({time.time()-t0:.1f}s)")
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                        "val_mae_overall_um": val["mae_overall_um"],
                        "val_mae_flank_wear_um": val["mae_flank_wear_um"],
                        "val_mae_adhesion_um": val["mae_adhesion_um"],
                        "val_mae_flank_wear+adhesion_um": val["mae_flank_wear+adhesion_um"],
                        "lr": lr_now})
        if val["mae_overall_um"] < best_val:
            best_val, best_ep = val["mae_overall_um"], epoch
            best_state = deepcopy(model.state_dict())
            torch.save({"epoch": epoch, "model_state_dict": best_state,
                        "val_mae_um": best_val, "config": cfg.__dict__},
                       run_dir / "best_model.pt")
            print(f"    ✓ new best val_mae={best_val:.2f}µm")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  --- Final results: {cfg.name} (best epoch {best_ep}) ---")
    final = {}
    for split in [k for k in ("val", "test", "unseen") if k in loaders]:
        m = evaluate(model, loaders[split], eval_crit, device)
        final[split] = m
        print(f"  [{split:7s}] {fmt_metrics(m)}")
    with open(run_dir / "results.json", "w") as fh:
        json.dump({"experiment": cfg.name, "config": cfg.__dict__,
                   "physics": {"lambda_max": args.lambda_max, "warmup": args.warmup,
                               "apply_to": args.apply_to, "one_sided": args.one_sided,
                               "use_taylor_slope": args.use_taylor_slope},
                   "best_epoch": best_ep, "best_val_mae_um": best_val,
                   "splits": {s: {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
                              for s, m in final.items()}}, fh, indent=2)
    return {"name": cfg.name, "best_epoch": best_ep, "best_val_mae_um": best_val,
            "final_results": final}


def main():
    p = argparse.ArgumentParser(description="MATWI Stage 3: vision + Taylor physics loss.")
    p.add_argument("--data-dir",   type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--sets-csv",   type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--constants",  type=Path, required=True, help="taylor_constants.json from fit_taylor.py")
    p.add_argument("--name",       type=str,  default="vis_taylor")
    # physics knobs
    p.add_argument("--lambda-max", type=float, default=0.05, help="0 => vision-only control")
    p.add_argument("--warmup",     type=int,   default=4)
    p.add_argument("--apply-to",   type=str,   default="all", choices=["all", "rvs", "ck45"])
    p.add_argument("--one-sided",  action="store_true", help="penalise overprediction only (adhesion ceiling)")
    p.add_argument("--use-taylor-slope", action="store_true", help="use Taylor-coupled slope instead of per-set observed slope")
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
    p.add_argument("--log-every",        type=int,   default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    img = (384, 384) if "efficientnet" in args.backbone else (224, 224)
    cfg = Stage3Config(name=args.name, backbone=args.backbone, set_range=args.set_range,
                       image_size=img, epochs=args.epochs, lr=args.lr,
                       batch_size=args.batch_size, data_loss=args.data_loss,
                       use_scheduler=("efficientnet" in args.backbone))
    print(f"Device {device} | seed {args.seed} | out {args.output_dir.resolve()}")
    run(cfg, args, device, args.output_dir.resolve())


if __name__ == "__main__":
    main()