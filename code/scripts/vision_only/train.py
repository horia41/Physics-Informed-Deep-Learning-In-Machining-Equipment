from pathlib import Path
import argparse
import torch
import sys
import pandas as pd

from experiments.vision_only.config import DEFAULT_VISION_EXPERIMENTS
from experiments.vision_only.runner import run_experiment
from utils.experiment import set_seed

# ── Comparison summary ────────────────────────────────────────────────────────

def write_comparison(results: list[dict], output_dir: Path) -> None:
    rows = []
    for res in results:
        cfg = res["config"]
        for split, m in res["final_results"].items():
            rows.append({
                "experiment":                  res["name"],
                "backbone":                    cfg["backbone"],
                "head_type":                   cfg["head_type"],
                "normalisation":               cfg["normalisation"],
                "loss":                        cfg["loss"],
                "scheduler":                   cfg["use_scheduler"],
                "eval_split":                  split,
                "best_epoch":                  res["best_epoch"],
                "best_val_mae_um":             res["best_val_mae_um"],
                "mae_overall_um":              m["mae_overall_um"],
                "mae_flank_wear_um":           m["mae_flank_wear_um"],
                "mae_adhesion_um":             m["mae_adhesion_um"],
                "mae_flank_wear+adhesion_um":  m["mae_flank_wear+adhesion_um"],
                "n_total":                     m["n_total"],
            })

    # Paper reference
    rows.append({
        "experiment": "paper_baseline", "backbone": "resnet50",
        "head_type": "simple", "normalisation": "?", "loss": "?",
        "scheduler": False, "eval_split": "test", "best_epoch": "—",
        "best_val_mae_um": "—", "mae_overall_um": 30.0,
        "mae_flank_wear_um": 14.0, "mae_adhesion_um": 39.0,
        "mae_flank_wear+adhesion_um": 91.0, "n_total": "—",
    })

    df = pd.DataFrame(rows)
    path = output_dir / "comparison_summary.csv"
    df.to_csv(path, index=False)

    print(f"\n{'='*90}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*90}")
    # Show test results only for a clean overview
    test_df = df[df["eval_split"] == "test"]
    print(test_df.to_string(index=False))
    print(f"\n  Full results (all splits) → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MATWI vision baseline ablation: head/norm/loss grid."
    )
    p.add_argument("--data-dir",    type=Path, required=True)
    p.add_argument("--labels-csv",  type=Path, required=True)
    p.add_argument("--sets-csv",    type=Path, required=True)
    p.add_argument("--output-dir",  type=Path, required=True)

    p.add_argument("--only", type=str, default=None,
                   help="Run only one experiment by name.")
    p.add_argument("--list-experiments", action="store_true",
                   help="Print available experiment names and exit.")
    p.add_argument("--require-sensors", action="store_true",
                   help="Restrict to samples that also have sensor data (the "
                        "647-sample multimodal subset). Default off = full 664.")

    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--device",           type=str,   default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-pretrained",    action="store_true")
    p.add_argument("--head-hidden-dim",  type=int,   default=256)
    p.add_argument("--dropout-backbone", type=float, default=0.3)
    p.add_argument("--dropout-head",     type=float, default=0.4)
    p.add_argument("--log-every",        type=int,   default=0)
    return p


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def main() -> None:
    args = build_parser().parse_args()

    if args.list_experiments:
        print("Available experiments:")
        for cfg in DEFAULT_VISION_EXPERIMENTS:
            print(f"  {cfg.name:40s}  backbone={cfg.backbone}  head={cfg.head_type}  "
                  f"norm={cfg.normalisation}  loss={cfg.loss}  sched={cfg.use_scheduler}")
        sys.exit(0)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    experiments = []
    for cfg in DEFAULT_VISION_EXPERIMENTS:
        if args.only and cfg.name != args.only:
            continue
        experiments.append(cfg)

    if not experiments:
        available = [c.name for c in DEFAULT_VISION_EXPERIMENTS]
        print(f"No experiment matched --only='{args.only}'.")
        print(f"Available: {available}")
        sys.exit(1)

    print(f"Device       : {device}")
    print(f"Experiments  : {[c.name for c in experiments]}")
    print(f"Output dir   : {output_dir}")

    all_results = []
    for cfg in experiments:
        set_seed(args.seed)
        result = run_experiment(cfg, args, device, output_dir)
        all_results.append(result)

    write_comparison(all_results, output_dir)


if __name__ == "__main__":
    main()