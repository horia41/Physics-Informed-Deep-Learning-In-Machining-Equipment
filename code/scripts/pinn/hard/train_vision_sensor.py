import argparse
from pathlib import Path
from copy import deepcopy
import sys

# Make top-level project modules (dataset, constants, etc.) importable
# when this script is executed via file path.
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiments.pinn.hard.multimodal.runner import run_experiment
from experiments.mutlimodal.config import MULTIMODAL_EXPERIMENTS_V2
from utils.experiment import set_seed, resolve_device

def _get_base_cfg(name: str):
    for cfg in MULTIMODAL_EXPERIMENTS_V2:
        if cfg.name == name:
            return cfg
    raise SystemExit(f"--base-exp '{name}' not found. Available: {[c.name for c in MULTIMODAL_EXPERIMENTS_V2]}")


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
    run_experiment(base_cfg, args, device, args.output_dir.resolve(), args.name)


if __name__ == "__main__":
    main()
