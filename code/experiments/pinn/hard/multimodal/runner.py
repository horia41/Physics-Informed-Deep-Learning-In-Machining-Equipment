import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import time
import json

from constants.matwi_dataset_constants import PINN_HARD_TRAIN_SETS
from model.pinn.constraint.hard import HardTaylorConstraint, recommend_C
from model.vision_sensor.multimodal_v2 import MATWIMultimodalModelV2
from experiments.mutlimodal.runner import build_loaders_v2
from metrics.mae import compute_mae_v1
from metrics.fmt import fmt_metrics_v1
from utils.experiment import predictions_to_um

def _predict(model, hard, images, sensor, set_ids, image_ids):
    out = model(images, sensor)
    raw = out["wear"]
    gate = out.get("gate")
    if hard is None:
        pred = raw.squeeze(-1) if raw.dim() > 1 else raw
        return pred, None, gate
    pred_norm, corr_um, _ = hard(raw, set_ids, image_ids)
    return pred_norm, corr_um, gate


def train_one_epoch_hard(model, hard, loader, optimizer, criterion, device, use_sensors):
    model.train()
    tot_loss = tot_mae = 0.0; n = 0
    corr_abs_sum = 0.0; sat_sum = 0; gate_sum = 0.0; gate_n = 0
    for batch in loader:
        images    = batch["image"].to(device, non_blocking=True)
        target    = batch["wear"].to(device, non_blocking=True)
        target_um = batch["wear_raw"].to(device, non_blocking=True)
        sensor = (batch["sensor_features"].to(device, non_blocking=True) if use_sensors else None)

        optimizer.zero_grad(set_to_none=True)
        pred_norm, corr_um, gate = _predict(model, hard, images, sensor,
                                            batch["set"], batch["image_id"])
        loss = criterion(pred_norm, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bsz = images.size(0)
        tot_mae  += torch.abs(predictions_to_um(pred_norm.detach()) - target_um).mean().item() * bsz
        tot_loss += loss.item() * bsz; n += bsz
        if corr_um is not None and hard is not None:
            corr_abs_sum += corr_um.detach().abs().mean().item() * bsz
            sat_sum += (corr_um.detach().abs() > 0.95 * hard.C_um).sum().item()
        if gate is not None:
            gate_sum += gate.mean().item() * bsz; gate_n += bsz
    res = {"loss": tot_loss/max(n,1), "mae_um": tot_mae/max(n,1),
           "corr_abs_um": corr_abs_sum/max(n,1) if hard is not None else 0.0,
           "sat_frac": sat_sum/max(n,1) if hard is not None else 0.0,
           "gate_mean": (gate_sum/gate_n) if gate_n else float("nan")}
    return res


@torch.no_grad()
def evaluate_hard(model, hard, loader, device, use_sensors) -> dict:
    model.eval()
    preds, targs, types = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        sensor = (batch["sensor_features"].to(device, non_blocking=True) if use_sensors else None)
        pred_norm, _, _ = _predict(model, hard, images, sensor, batch["set"], batch["image_id"])
        preds.append(predictions_to_um(pred_norm).cpu().numpy())
        targs.append(batch["wear_raw"].numpy())
        types.extend(list(batch["type"]))
    return compute_mae_v1(np.concatenate(preds), np.concatenate(targs), types)


def run_experiment(base_cfg, args, device, out_dir: Path, name: str) -> None:
    print(f"\n{'='*80}\n  STAGE 3 FUSION HARD: {name}")
    print(f"  base={base_cfg.name} (fusion={base_cfg.fusion_mode}, "
          f"features={base_cfg.feature_set}, mdrop={base_cfg.modality_dropout_p})")
    run_dir = out_dir / name; run_dir.mkdir(parents=True, exist_ok=True)

    loaders, scaler = build_loaders_v2(base_cfg, args.data_dir, args.labels_csv,
                                    args.sets_csv, args.num_workers)
    if scaler is not None:
        scaler.save(run_dir / "sensor_scaler.pkl")
    n_feat = loaders["train"].dataset.n_selected_features

    hard = None; C_um = None
    if args.hard:
        C_um = args.C_um
        if args.auto_C:
            C_um = recommend_C(args.labels_csv, args.sets_csv, args.constants,
                               PINN_HARD_TRAIN_SETS, percentile=args.auto_C_pct)
            print(f"  [auto-C] => C_um={C_um:.1f}")
        hard = HardTaylorConstraint(args.constants, args.sets_csv, C_um=C_um).to(device)
        print(f"  HardConstraint: {hard.extra_repr()}")
    else:
        print("  (no --hard: unconstrained fusion control)")

    model_kwargs = dict(use_sensors=base_cfg.use_sensors,
                        pretrained=not args.no_pretrained,
                        dropout_backbone=base_cfg.dropout_backbone)
    if base_cfg.use_sensors:
        model_kwargs.update(fusion_mode=base_cfg.fusion_mode, n_sensor_features=n_feat,
                            sensor_encoder_dim=base_cfg.sensor_encoder_dim,
                            modality_dropout_p=base_cfg.modality_dropout_p)
    model = MATWIMultimodalModelV2(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_cfg.lr,
                                  weight_decay=base_cfg.weight_decay)   # fixed LR, no scheduler
    criterion = nn.MSELoss()

    history, best_state, best_val, best_ep = [], None, float("inf"), -1
    for epoch in range(1, base_cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch_hard(model, hard, loaders["train"], optimizer, criterion,
                                  device, base_cfg.use_sensors)
        val = evaluate_hard(model, hard, loaders["val"], device, base_cfg.use_sensors)
        print(f"  [{epoch:02d}/{base_cfg.epochs}] loss={tr['loss']:.5f} "
              f"corr={tr['corr_abs_um']:.1f}µm sat={tr['sat_frac']:.2f} gate={tr['gate_mean']:.3f} "
              f"train_mae={tr['mae_um']:.1f}µm | val {fmt_metrics_v1(val)} ({time.time()-t0:.1f}s)")
        history.append({"epoch": epoch, "train_loss": tr["loss"], "train_mae_um": tr["mae_um"],
                        "train_corr_abs_um": tr["corr_abs_um"], "train_sat_frac": tr["sat_frac"],
                        "train_gate_mean": tr["gate_mean"],
                        "val_mae_overall_um": val["mae_overall_um"],
                        "val_mae_flank_wear_um": val["mae_flank_wear_um"],
                        "val_mae_adhesion_um": val["mae_adhesion_um"],
                        "val_mae_flank_wear+adhesion_um": val["mae_flank_wear+adhesion_um"]})
        if val["mae_overall_um"] < best_val:
            best_val, best_ep = val["mae_overall_um"], epoch
            best_state = deepcopy(model.state_dict())
            torch.save({"epoch": epoch, "model_state_dict": best_state, "val_mae_um": best_val,
                        "base_config": base_cfg.__dict__, "hard": {"enabled": args.hard, "C_um": C_um}},
                       run_dir / "best_model.pt")
            print(f"    ✓ new best val_mae={best_val:.2f}µm")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  --- Final results: {name} (best epoch {best_ep}) ---")
    final = {}
    for split in [k for k in ("val", "test") if k in loaders]:
        m = evaluate_hard(model, hard, loaders[split], device, base_cfg.use_sensors)
        final[split] = m
        print(f"  [{split:7s}] {fmt_metrics_v1(m)}")
    with open(run_dir / "results.json", "w") as fh:
        json.dump({"experiment": name, "base_config": base_cfg.__dict__,
                   "hard": {"enabled": bool(args.hard), "C_um": C_um,
                            "anchor": "taylor_ramp_from_Vc"},
                   "best_epoch": best_ep, "best_val_mae_um": best_val,
                   "splits": final}, fh, indent=2)

