import argparse
from copy import deepcopy
from pathlib import Path
import sys

# Make top-level project modules (dataset, constants, etc.) importable
# when this script is executed via file path.
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiments.mutlimodal.config import MULTIMODAL_EXPERIMENTS_V2
from experiments.pinn.aircuts.v1.runner import run_experiment
from utils.experiment import set_seed, resolve_device

def _get_base_cfg(name: str):
    for cfg in MULTIMODAL_EXPERIMENTS_V2:
        if cfg.name == name:
            return cfg
    avail = [c.name for c in MULTIMODAL_EXPERIMENTS_V2]
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
    run_experiment(base_cfg, args, device, args.output_dir.resolve(), args.name)


if __name__ == "__main__":
    main()