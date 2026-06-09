
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from copy import deepcopy
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

# Fusion base (task-3 grid incl. t3_gated_top25_md30) + hard-constraint module.
_TVS = _load_module("train_vision_sensor_base", [
    _HERE / "train_vision_sensorV2.py",
    _HERE / ".." / "vision-sensor" / "improve_attempt" / "train_vision_sensorV2.py",
])
_TH = _load_module("taylor_hard", [_HERE / "taylor_hard.py"])

if not hasattr(_TVS, "ALL_EXPERIMENTS"):
    raise SystemExit("ERROR: loaded sensor script has no ALL_EXPERIMENTS (need the "
                     "task-3 train_vision_sensorV2.py with t3_gated_top25_md30).")

MATWIMultimodalModel = _TVS.MATWIMultimodalModel
build_loaders        = _TVS.build_loaders
predictions_to_um    = _TVS.predictions_to_um
fmt_metrics          = _TVS.fmt_metrics
set_seed             = _TVS.set_seed
resolve_device       = _TVS.resolve_device
ALL_EXPERIMENTS      = _TVS.ALL_EXPERIMENTS
WEAR_TYPES           = _TVS.WEAR_TYPES
HardTaylorConstraint = _TH.HardTaylorConstraint
recommend_C          = _TH.recommend_C

TRAIN_SETS_1_13 = [1, 2, 5, 7, 8, 10, 11]


def _get_base_cfg(name: str):
    for cfg in ALL_EXPERIMENTS:
        if cfg.name == name:
            return cfg
    raise SystemExit(f"--base-exp '{name}' not found. Available: {[c.name for c in ALL_EXPERIMENTS]}")


def compute_mae(pred_um: np.ndarray, target_um: np.ndarray, types: list[str]) -> dict:
    abs_err = np.abs(pred_um - target_um)
    out = {"mae_overall_um": float(abs_err.mean()) if len(abs_err) else float("nan"),
           "n_total": int(len(target_um))}
    for wt in WEAR_TYPES:
        mask = np.array([t == wt for t in types], dtype=bool)
        out[f"mae_{wt}_um"] = float(abs_err[mask].mean()) if mask.any() else float("nan")
        out[f"n_{wt}"] = int(mask.sum())
    return out


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
    return compute_mae(np.concatenate(preds), np.concatenate(targs), types)


def run(base_cfg, args, device, out_dir: Path, name: str) -> None:
    print(f"\n{'='*80}\n  STAGE 3 FUSION HARD: {name}")
    print(f"  base={base_cfg.name} (fusion={base_cfg.fusion_mode}, "
          f"features={base_cfg.feature_set}, mdrop={base_cfg.modality_dropout_p})")
    run_dir = out_dir / name; run_dir.mkdir(parents=True, exist_ok=True)

    loaders, scaler = build_loaders(base_cfg, args.data_dir, args.labels_csv,
                                    args.sets_csv, args.num_workers)
    if scaler is not None:
        scaler.save(run_dir / "sensor_scaler.pkl")
    n_feat = loaders["train"].dataset.n_selected_features

    hard = None; C_um = None
    if args.hard:
        C_um = args.C_um
        if args.auto_C:
            C_um = recommend_C(args.labels_csv, args.sets_csv, args.constants,
                               TRAIN_SETS_1_13, percentile=args.auto_C_pct)
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
    model = MATWIMultimodalModel(**model_kwargs).to(device)
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
              f"train_mae={tr['mae_um']:.1f}µm | val {fmt_metrics(val)} ({time.time()-t0:.1f}s)")
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
        print(f"  [{split:7s}] {fmt_metrics(m)}")
    with open(run_dir / "results.json", "w") as fh:
        json.dump({"experiment": name, "base_config": base_cfg.__dict__,
                   "hard": {"enabled": bool(args.hard), "C_um": C_um,
                            "anchor": "taylor_ramp_from_Vc"},
                   "best_epoch": best_ep, "best_val_mae_um": best_val,
                   "splits": final}, fh, indent=2)


def main():
    p = argparse.ArgumentParser(description="MATWI Stage 3: gated fusion + HARD Taylor constraint.")
    p.add_argument("--data-dir",   type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--sets-csv",   type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--constants",  type=Path, required=True)
    p.add_argument("--name",       type=str,  default="fusion_hard")
    p.add_argument("--base-exp",   type=str,  default="t3_gated_top25_md30")
    p.add_argument("--epochs",     type=int,  default=0, help="override base epoch count (0=keep, =17)")
    # hard-constraint knobs
    p.add_argument("--hard",       action="store_true")
    p.add_argument("--C-um",       type=float, default=150.0)
    p.add_argument("--auto-C",     action="store_true")
    p.add_argument("--auto-C-pct", type=float, default=95.0)
    # misc
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    base_cfg = deepcopy(_get_base_cfg(args.base_exp))
    if args.epochs:
        base_cfg.epochs = args.epochs
    print(f"Device {device} | seed {args.seed} | epochs {base_cfg.epochs} | hard={args.hard} "
          f"| out {args.output_dir.resolve()}")
    run(base_cfg, args, device, args.output_dir.resolve(), args.name)


if __name__ == "__main__":
    main()
