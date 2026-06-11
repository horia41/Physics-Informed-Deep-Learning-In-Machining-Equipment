import argparse
from copy import deepcopy
from pathlib import Path

from utils.experiment import set_seed, resolve_device
from experiments.mutlimodal.config import MULTIMODAL_EXPERIMENTS_V1
from experiments.pinn.v1.multimodal.runner import run_experiment

def _get_base_cfg(name: str):
    for cfg in MULTIMODAL_EXPERIMENTS_V1:
        if cfg.name == name:
            return cfg
    avail = [c.name for c in MULTIMODAL_EXPERIMENTS_V1]
    raise SystemExit(f"--base-exp '{name}' not found. Available: {avail}")

def main():
    p = argparse.ArgumentParser(description="MATWI Stage 3: fusion + Taylor physics loss.")
    p.add_argument("--data-dir",   type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--sets-csv",   type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--constants",  type=Path, required=True)
    p.add_argument("--name",       type=str,  default="t3fusion_taylor")
    p.add_argument("--base-exp",   type=str,  default="intermediate_top25",
                   help="config name from train_vision_sensor.py DEFAULT_EXPERIMENTS "
                        "to build on (e.g. intermediate_top25, early_top25)")
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
    # Apply the air-cut flag onto the chosen base config (ExperimentConfig now
    # carries gate_aircuts; older configs without it still work via setattr).
    if args.gate_aircuts:
        setattr(base_cfg, "gate_aircuts", True)
    print(f"Device {device} | seed {args.seed} | gate_aircuts={args.gate_aircuts} "
          f"| out {args.output_dir.resolve()}")
    run_experiment(base_cfg, args, device, args.output_dir.resolve(), args.name)


if __name__ == "__main__":
    main()
