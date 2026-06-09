

from __future__ import annotations
import argparse, importlib.util, json, sys, time
from copy import deepcopy
from pathlib import Path
from typing import Optional

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

# train_vision_sensor.py is looked up locally, then in the sibling vision-sensor/
# dir (and its improve_attempt/ subdir) where the Stage-2/task-3 files live.
# Loaded from there, it self-resolves its own model/dataset deps, so nothing
# needs copying into pinn/.
_TVS = _load_module("train_vision_sensor_base", [
    _HERE / "train_vision_sensor.py",
    _HERE / ".." / "vision-sensor" / "train_vision_sensor.py",
    _HERE / ".." / "vision-sensor" / "improve_attempt" / "train_vision_sensor.py",
])
_TPL = _load_module("taylor_physics_loss", [
    _HERE / "taylor_physics_loss.py",
])

MATWIMultimodalModel = _TVS.MATWIMultimodalModel
build_loaders        = _TVS.build_loaders
evaluate             = _TVS.evaluate
predictions_to_um    = _TVS.predictions_to_um
fmt_metrics          = _TVS.fmt_metrics
set_seed             = _TVS.set_seed
resolve_device       = _TVS.resolve_device
# The non-V2 fusion base exposes its grid as DEFAULT_EXPERIMENTS; the V2 base also
# defines ALL_EXPERIMENTS. Accept whichever exists so this trainer works against
# either base (mirrors the fix already applied in pinn/train_vision_sensor_taylor.py).
ALL_EXPERIMENTS      = getattr(_TVS, "DEFAULT_EXPERIMENTS",
                               getattr(_TVS, "ALL_EXPERIMENTS", []))
TaylorPhysicsLoss    = _TPL.TaylorPhysicsLoss


def _get_base_cfg(name: str):
    for cfg in ALL_EXPERIMENTS:
        if cfg.name == name:
            return cfg
    avail = [c.name for c in ALL_EXPERIMENTS]
    raise SystemExit(f"--base-exp '{name}' not found. Available: {avail}")


# ── Training loop with the physics term ──────────────────────────────────────
def train_one_epoch_taylor(model, loader, optimizer, data_criterion, physics_loss,
                           device, use_sensors, epoch0, log_every=0):
    """epoch0 is 0-indexed (for the lambda warm-up)."""
    model.train()
    tot = tot_d = tot_p = tot_mae = 0.0; n = 0
    gate_sum = 0.0; gate_n = 0
    for step, batch in enumerate(loader, 1):
        images    = batch["image"].to(device, non_blocking=True)
        target    = batch["wear"].to(device, non_blocking=True).unsqueeze(1)
        target_um = batch["wear_raw"].to(device, non_blocking=True)
        sensor = (batch["sensor_features"].to(device, non_blocking=True)
                  if use_sensors else None)

        optimizer.zero_grad(set_to_none=True)
        out = model(images, sensor)
        d_loss = data_criterion(out["wear"], target)
        p_loss = physics_loss(out["wear"], batch["set"], batch["image_id"],
                              list(batch["material"]), epoch=epoch0)
        loss = d_loss + p_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bsz = images.size(0)
        tot_mae += torch.abs(predictions_to_um(out["wear"].detach()) - target_um).mean().item() * bsz
        tot += loss.item() * bsz; tot_d += d_loss.item() * bsz
        tot_p += float(p_loss.item()) * bsz; n += bsz
        if out.get("gate") is not None:
            gate_sum += out["gate"].mean().item() * bsz; gate_n += bsz
        if log_every and step % log_every == 0:
            print(f"    step {step:04d}/{len(loader)}  data={d_loss.item():.5f} "
                  f"phys={float(p_loss.item()):.6f}")
    res = {"loss": tot/max(n,1), "data_loss": tot_d/max(n,1),
           "phys_loss": tot_p/max(n,1), "mae_um": tot_mae/max(n,1)}
    res["gate_mean"] = (gate_sum/gate_n) if gate_n else float("nan")
    return res


def run(base_cfg, args, device, out_dir: Path, name: str) -> None:
    print(f"\n{'='*80}\n  STAGE 3 FUSION: {name}")
    print(f"  base={base_cfg.name} (fusion={base_cfg.fusion_mode}, "
          f"features={base_cfg.feature_set}, mdrop={base_cfg.modality_dropout_p})")
    print(f"  physics: lambda_max={args.lambda_max} warmup={args.warmup} "
          f"apply_to={args.apply_to} one_sided={args.one_sided} "
          f"taylor_slope={args.use_taylor_slope}\n{'='*80}")
    run_dir = out_dir / name; run_dir.mkdir(parents=True, exist_ok=True)

    # ── Data (reuses the multimodal loader + scaler fit) ──────────────────────
    loaders, scaler = build_loaders(base_cfg, args.data_dir, args.labels_csv,
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
    model = MATWIMultimodalModel(**model_kwargs).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_cfg.lr,
                                  weight_decay=base_cfg.weight_decay)   # fixed LR, no scheduler
    data_criterion = nn.MSELoss()
    physics_loss = TaylorPhysicsLoss(
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
              f"train_mae={tr['mae_um']:.1f}µm | val {fmt_metrics(val)} ({time.time()-t0:.1f}s)")
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
        print(f"  [{split:7s}] {fmt_metrics(m)}")
    with open(run_dir / "results.json", "w") as fh:
        json.dump({"experiment": name, "base_config": base_cfg.__dict__,
                   "physics": {"lambda_max": args.lambda_max, "warmup": args.warmup,
                               "apply_to": args.apply_to, "one_sided": args.one_sided,
                               "use_taylor_slope": args.use_taylor_slope},
                   "best_epoch": best_ep, "best_val_mae_um": best_val,
                   "splits": {s: {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
                              for s, m in final.items()}}, fh, indent=2)


def main():
    p = argparse.ArgumentParser(description="MATWI Stage 3: gated fusion + Taylor physics loss.")
    p.add_argument("--data-dir",   type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--sets-csv",   type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--constants",  type=Path, required=True)
    p.add_argument("--name",       type=str,  default="t3gated_taylor")
    p.add_argument("--base-exp",   type=str,  default="t3_gated_top25_md30",
                   help="config name from train_vision_sensor.py to build on")
    # physics knobs
    p.add_argument("--lambda-max", type=float, default=0.05, help="0 => fusion control, no physics")
    p.add_argument("--warmup",     type=int,   default=4)
    p.add_argument("--apply-to",   type=str,   default="all", choices=["all", "rvs", "ck45"])
    p.add_argument("--one-sided",  action="store_true")
    p.add_argument("--use-taylor-slope", action="store_true")
    # misc
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--log-every",   type=int, default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    base_cfg = _get_base_cfg(args.base_exp)
    print(f"Device {device} | seed {args.seed} | out {args.output_dir.resolve()}")
    run(base_cfg, args, device, args.output_dir.resolve(), args.name)


if __name__ == "__main__":
    main()
