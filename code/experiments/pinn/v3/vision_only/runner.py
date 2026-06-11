from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy
import json
import time

from experiments.vision_only.config import VisionExperimentConfig
from experiments.vision_only.runner import build_loaders, evaluate
from experiments.pinn.v1.vision_only.config import Stage3Config
from experiments.pinn.v1.vision_only.runner import train_one_epoch_taylor
from model.vision_only.vision_model import MATWIVisionModel
from metrics.loss import TaylorPhysicsLossV3
from metrics.fmt import fmt_metrics_v1

def run_experiment(cfg: Stage3Config, args, device, out_dir: Path) -> dict:
    print(f"\n{'='*80}\n  STAGE 3: {cfg.name}  (backbone={cfg.backbone}, data_loss={cfg.data_loss})")
    print(f"  physics: lambda_max={args.lambda_max} warmup={args.warmup} "
          f"apply_to={args.apply_to} one_sided={args.one_sided} "
          f"taylor_slope={args.use_taylor_slope}\n{'='*80}")
    run_dir = out_dir / cfg.name; run_dir.mkdir(parents=True, exist_ok=True)

    # mirror train_vision's ExperimentConfig so build_loaders is happy
    ecfg = VisionExperimentConfig(
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
    physics_loss = TaylorPhysicsLossV3(
        constants_path=args.constants, lambda_max=max(args.lambda_max, 1e-9),
        warmup_epochs=args.warmup, apply_to=args.apply_to,
        one_sided=args.one_sided, use_taylor_slope=args.use_taylor_slope)
    if args.lambda_max <= 0.0:                     # lambda 0 == pure vision control
        physics_loss.lambda_max = 0.0
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
              f"val {fmt_metrics_v1(val)} ({time.time()-t0:.1f}s)")
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
        print(f"  [{split:7s}] {fmt_metrics_v1(m)}")
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

