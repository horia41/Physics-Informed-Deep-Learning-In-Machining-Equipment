import argparse
from pathlib import Path
import sys

# Make top-level project modules (dataset, constants, etc.) importable
# when this script is executed via file path.
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiments.pinn.v3.vision_only.runner import run_experiment
from experiments.pinn.v1.vision_only.config import Stage3Config
from utils.experiment import set_seed, resolve_device

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
    run_experiment(cfg, args, device, args.output_dir.resolve())


if __name__ == "__main__":
    main()