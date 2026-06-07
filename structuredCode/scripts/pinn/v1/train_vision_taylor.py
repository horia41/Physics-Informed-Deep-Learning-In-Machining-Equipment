"""
MATWI — Stage 3: Vision + Taylor Physics-Informed training
===========================================================
Fork of train_vision.py that adds the frozen-constant Taylor physics penalty.
It REUSES the Stage-1 building blocks (dataset, model, loaders, evaluate,
metrics) via the project's `_load_module` convention, so the only new code is
the physics wiring. Everything else (minimal setup: no aug, no oversampling,
simple head) is kept identical for a clean controlled comparison.

Run the physics experiment AND its matched vision-only control with the SAME
data loss so the only difference is the physics term:

    # 0. Fit constants offline first (login node — needs internet only for nothing here)
    python fit_taylor.py --labels-csv $D/labels.csv --sets-csv $D/sets.csv \
                         --out ./taylor_constants.json

    # 1. Vision-only control (lambda 0)
    python train_vision_taylor.py --data-dir $D --labels-csv $D/labels.csv \
        --sets-csv $D/sets.csv --output-dir ./runs/stage3 \
        --constants ./taylor_constants.json --name vision_only_ctrl --lambda-max 0.0

    # 2. Vision + Taylor (symmetric)
    python train_vision_taylor.py ... --name vis_taylor_l05 --lambda-max 0.05

    # 3. Vision + Taylor as an adhesion CEILING on RVS only (recommended variant)
    python train_vision_taylor.py ... --name vis_taylor_ceil_rvs \
        --lambda-max 0.05 --one-sided --apply-to rvs

Compare against the known references: vision-only full-664 = 19.0 µm,
vision-only multimodal-647 = 22.4 µm. The win condition is lower F+A / adhesion
MAE WITHOUT regressing flank-wear MAE.
"""

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

from experiments.utils import set_seed, resolve_device
from experiments.pinn.pinn_v1_vision_only_experiments import run_experiment_vision_only, Stage3Config

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
    run_experiment_vision_only(cfg, args, device, args.output_dir.resolve())


if __name__ == "__main__":
    main()