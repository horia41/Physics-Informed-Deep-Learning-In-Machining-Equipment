from pathlib import Path
import torch
import torch.nn as nn
import time
from copy import deepcopy
import json
import numpy as np
import pandas as pd

from experiments.pinn.v1.multimodal.runner import train_one_epoch_taylor
from experiments.mutlimodal.runner import build_loaders_v1, evaluate
from model.vision_sensor.multimodal_v1 import MATWIMultimodalModelV1
from metrics.loss import TaylorPhysicsLossV2
from metrics.fmt import fmt_metrics_v1

def run_experiment(base_cfg, args, device, out_dir: Path, name: str) -> None:
    print(f"\n{'='*80}\n  STAGE 3 FUSION: {name}")
    print(f"  base={base_cfg.name} (fusion={base_cfg.fusion_mode}, "
          f"features={base_cfg.feature_set}, mdrop={base_cfg.modality_dropout_p})")
    print(f"  physics: lambda_max={args.lambda_max} warmup={args.warmup} "
          f"apply_to={args.apply_to} one_sided={args.one_sided} "
          f"taylor_slope={args.use_taylor_slope}\n{'='*80}")
    run_dir = out_dir / name; run_dir.mkdir(parents=True, exist_ok=True)

    # ── Data (reuses the multimodal loader + scaler fit) ──────────────────────
    loaders, scaler = build_loaders_v1(base_cfg, args.data_dir, args.labels_csv,
                                    args.sets_csv, args.num_workers)
    if scaler is not None:
        scaler.save(run_dir / "sensor_scaler.pkl")
    n_feat = loaders["train"].dataset.n_selected_features

    # ── Model (same construction as train_vision_sensor.run_experiment) ───────
    model_kwargs = dict(use_sensors=base_cfg.use_sensors,
                        pretrained=not args.no_pretrained,
                        dropout_backbone=base_cfg.dropout_backbone)
    if base_cfg.use_sensors:
        model_kwargs.update(fusion_mode=base_cfg.fusion_mode, n_sensor_features=n_feat,
                            sensor_encoder_dim=base_cfg.sensor_encoder_dim,
                            modality_dropout_p=base_cfg.modality_dropout_p)
    model = MATWIMultimodalModelV1(**model_kwargs).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_cfg.lr,
                                  weight_decay=base_cfg.weight_decay)   # fixed LR, no scheduler
    data_criterion = nn.MSELoss()
    physics_loss = TaylorPhysicsLossV2(
        constants_path=args.constants, lambda_max=max(args.lambda_max, 1e-9),
        warmup_epochs=args.warmup, apply_to=args.apply_to,
        one_sided=args.one_sided, use_taylor_slope=args.use_taylor_slope)
    if args.lambda_max <= 0.0:
        physics_loss.lambda_max = 0.0
    print(f"  PhysicsLoss: {physics_loss.extra_repr()}")

    history, best_state, best_val, best_ep = [], None, float("inf"), -1
    for epoch in range(1, base_cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch_taylor(model, loaders["train"], optimizer, data_criterion,
                                    physics_loss, device, base_cfg.use_sensors,
                                    epoch0=epoch-1, log_every=args.log_every)
        val = evaluate(model, loaders["val"], data_criterion, device, base_cfg.use_sensors)
        print(f"  [{epoch:02d}/{base_cfg.epochs}] data={tr['data_loss']:.5f} "
              f"phys={tr['phys_loss']:.6f} gate={tr['gate_mean']:.3f} "
              f"train_mae={tr['mae_um']:.1f}µm | val {fmt_metrics_v1(val)} ({time.time()-t0:.1f}s)")
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                        "val_mae_overall_um": val["mae_overall_um"],
                        "val_mae_flank_wear_um": val["mae_flank_wear_um"],
                        "val_mae_adhesion_um": val["mae_adhesion_um"],
                        "val_mae_flank_wear+adhesion_um": val["mae_flank_wear+adhesion_um"]})
        if val["mae_overall_um"] < best_val:
            best_val, best_ep = val["mae_overall_um"], epoch
            best_state = deepcopy(model.state_dict())
            torch.save({"epoch": epoch, "model_state_dict": best_state,
                        "val_mae_um": best_val, "base_config": base_cfg.__dict__},
                       run_dir / "best_model.pt")
            print(f"    ✓ new best val_mae={best_val:.2f}µm")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  --- Final results: {name} (best epoch {best_ep}) ---")
    final = {}
    for split in [k for k in ("val", "test") if k in loaders]:
        m = evaluate(model, loaders[split], data_criterion, device, base_cfg.use_sensors)
        final[split] = m
        print(f"  [{split:7s}] {fmt_metrics_v1(m)}")
    with open(run_dir / "results.json", "w") as fh:
        json.dump({"experiment": name, "base_config": base_cfg.__dict__,
                   "physics": {"lambda_max": args.lambda_max, "warmup": args.warmup,
                               "apply_to": args.apply_to, "one_sided": args.one_sided,
                               "use_taylor_slope": args.use_taylor_slope},
                   "best_epoch": best_ep, "best_val_mae_um": best_val,
                   "splits": {s: {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
                              for s, m in final.items()}}, fh, indent=2)
