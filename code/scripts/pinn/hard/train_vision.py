import argparse
from pathlib import Path

from experiments.pinn.hard.vision_only.config import HardConfig
from experiments.pinn.hard.vision_only.runner import run_experiment
from utils.experiment import set_seed, resolve_device

def main():
    p = argparse.ArgumentParser(description="MATWI Stage 3: vision + HARD Taylor constraint.")
    p.add_argument("--data-dir",   type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--sets-csv",   type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--constants",  type=Path, required=True, help="taylor_constants.json")
    p.add_argument("--name",       type=str,  default="vision_hard")
    # hard-constraint knobs
    p.add_argument("--hard",       action="store_true", help="enable the band constraint (omit = control)")
    p.add_argument("--C-um",       type=float, default=150.0, help="band half-width µm (VB_taylor ± C)")
    p.add_argument("--auto-C",     action="store_true", help="set C from train |wear-VB_taylor| percentile")
    p.add_argument("--auto-C-pct", type=float, default=95.0)
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
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    img = (384, 384) if "efficientnet" in args.backbone else (224, 224)
    cfg = HardConfig(name=args.name, backbone=args.backbone, set_range=args.set_range,
                     image_size=img, epochs=args.epochs, lr=args.lr,
                     batch_size=args.batch_size, data_loss=args.data_loss,
                     use_scheduler=("efficientnet" in args.backbone))
    print(f"Device {device} | seed {args.seed} | hard={args.hard} | out {args.output_dir.resolve()}")
    run_experiment(cfg, args, device, args.output_dir.resolve())


if __name__ == "__main__":
    main()
