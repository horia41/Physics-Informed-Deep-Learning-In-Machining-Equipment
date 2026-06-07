"""
MATWI — Stage 3: Vision+Sensor (gated fusion) + Taylor physics loss
====================================================================
Fork of train_vision_sensor.py. REUSES its loaders / model / eval via the
project's _load_module convention and only adds the frozen-constant Taylor
physics penalty (taylor_physics_loss.py). The data loss, optimiser (fixed-LR
AdamW, no scheduler), and 647/300/247 multimodal splits are kept identical to
the Stage-2/task-3 setup so the only variable vs. the control is the physics.

Default base model = t3_gated_top25_md30 (gated fusion, top25 features,
modality dropout 0.3) — the only fusion config that beat the vision-only-647
control. Pass --base-exp to use a different one from train_vision_sensor.py.

Single seed (42) by design: this run answers "does physics help, yes/no?" for
the best fusion model, not "is a 1 µm gap significant?".

Bonus: logs the mean learned gate value each epoch (out["gate"]), so you can
see whether the gated model is actually using the sensors or has learned to
suppress them.

    python train_vision_sensor_taylor.py \
        --data-dir $D --labels-csv $D/labels.csv --sets-csv $D/sets.csv \
        --output-dir ./runs/stage3_fusion --constants ./taylor_constants.json \
        --name t3gated_taylor_sym --lambda-max 0.05 --apply-to all
"""

import argparse
from copy import deepcopy
from pathlib import Path
from experiments.pinn.pinn_v1_vision_sensor_experiments import run_experiment_vision_sensor
from experiments.default_experiments import DEFAULT_EXPERIMENTS, TASK3_EXPERIMENTS
from experiments.utils import set_seed, resolve_device

ALL_EXPERIMENTS = DEFAULT_EXPERIMENTS + TASK3_EXPERIMENTS

def _get_base_cfg(name: str):
    for cfg in ALL_EXPERIMENTS:
        if cfg.name == name:
            return cfg
    avail = [c.name for c in ALL_EXPERIMENTS]
    raise SystemExit(f"--base-exp '{name}' not found. Available: {avail}")

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
    p.add_argument("--epochs",     type=int,  default=0,
                   help="override the base config's epoch count (0 = keep base, =17)")
    # physics knobs
    p.add_argument("--lambda-max", type=float, default=0.05, help="0 => fusion control, no physics")
    p.add_argument("--warmup",     type=int,   default=4)
    p.add_argument("--apply-to",   type=str,   default="all", choices=["all", "rvs", "ck45"])
    p.add_argument("--one-sided",  action="store_true")
    p.add_argument("--use-taylor-slope", action="store_true")
    # air-cut gating: compute sensor features on the tool-engaged window only
    p.add_argument("--gate-aircuts", action="store_true",
                   help="remove tool-approach/retraction (air-cut) phases before "
                        "sensor feature extraction (uses relative band energy)")
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
    base_cfg = deepcopy(_get_base_cfg(args.base_exp))   # copy: don't mutate the grid
    if args.epochs:
        base_cfg.epochs = args.epochs
    if args.gate_aircuts:
        setattr(base_cfg, "gate_aircuts", True)
    print(f"Device {device} | seed {args.seed} | epochs {base_cfg.epochs} "
          f"| gate_aircuts={args.gate_aircuts} | out {args.output_dir.resolve()}")
    run_experiment_vision_sensor(base_cfg, args, device, args.output_dir.resolve(), args.name)


if __name__ == "__main__":
    main()