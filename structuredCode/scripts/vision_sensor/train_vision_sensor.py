
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import torch

from experiments.vision_sensor_experiment import run_experiment
from experiments.default_experiments import DEFAULT_EXPERIMENTS, AIRCUT_EXPERIMENTS
from experiments.utils import set_seed, resolve_device

ALL_EXPERIMENTS = DEFAULT_EXPERIMENTS + AIRCUT_EXPERIMENTS

def write_comparison(results: list[dict], output_dir: Path) -> None:
    rows = []
    for res in results:
        cfg = res["config"]
        for split, m in res["final_results"].items():
            rows.append({
                "experiment":                  res["name"],
                "fusion_mode":                 cfg["fusion_mode"],
                "feature_set":                 cfg["feature_set"],
                "use_sensors":                 cfg["use_sensors"],
                "eval_split":                  split,
                "best_epoch":                  res["best_epoch"],
                "best_val_mae_um":             res["best_val_mae_um"],
                "mae_overall_um":              m["mae_overall_um"],
                "mae_flank_wear_um":           m["mae_flank_wear_um"],
                "mae_adhesion_um":             m["mae_adhesion_um"],
                "mae_flank_wear+adhesion_um":  m["mae_flank_wear+adhesion_um"],
                "n_total":                     m["n_total"],
            })

    # Reference rows
    rows.append({
        "experiment": "vision_only_664 (reference)", "fusion_mode": "none",
        "feature_set": "none", "use_sensors": False, "eval_split": "test",
        "best_epoch": 14, "best_val_mae_um": "—",
        "mae_overall_um": 19.0, "mae_flank_wear_um": 16.6,
        "mae_adhesion_um": 37.2, "mae_flank_wear+adhesion_um": 23.2,
        "n_total": 254,
    })
    rows.append({
        "experiment": "paper_baseline", "fusion_mode": "none",
        "feature_set": "none", "use_sensors": False, "eval_split": "test",
        "best_epoch": "—", "best_val_mae_um": "—",
        "mae_overall_um": 30.0, "mae_flank_wear_um": 14.0,
        "mae_adhesion_um": 39.0, "mae_flank_wear+adhesion_um": 91.0,
        "n_total": 254,
    })

    df = pd.DataFrame(rows)
    path = output_dir / "comparison_summary.csv"
    df.to_csv(path, index=False)

    print(f"\n{'='*100}")
    print("  SENSOR FUSION ABLATION — TEST SET RESULTS (sorted by overall MAE)")
    print(f"{'='*100}")
    test_df = df[df["eval_split"] == "test"].copy()
    test_df = test_df.sort_values("mae_overall_um")
    cols = ["experiment", "fusion_mode", "feature_set",
            "mae_overall_um", "mae_flank_wear_um", "mae_adhesion_um",
            "mae_flank_wear+adhesion_um", "best_epoch"]
    print(test_df[cols].to_string(index=False))
    print(f"\n  Full results (all splits) → {path}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MATWI sensor fusion ablation: "
                    "3 fusion modes × 3 feature sets + vision-only control."
    )
    p.add_argument("--data-dir",    type=Path, required=True)
    p.add_argument("--labels-csv",  type=Path, required=True)
    p.add_argument("--sets-csv",    type=Path, required=True)
    p.add_argument("--output-dir",  type=Path, required=True)

    p.add_argument("--only", type=str, default=None,
                   help="Run only one experiment by name.")
    p.add_argument("--list-experiments", action="store_true",
                   help="Print available experiment names and exit.")

    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--device",           type=str,   default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-pretrained",    action="store_true")
    p.add_argument("--log-every",        type=int,   default=0)
    return p

def main() -> None:
    args = build_parser().parse_args()

    if args.list_experiments:
        print("Available experiments:")
        for cfg in ALL_EXPERIMENTS:
            print(f"  {cfg.name:25s}  fusion={cfg.fusion_mode:15s}  "
                  f"features={cfg.feature_set:6s}  sensors={cfg.use_sensors}")
        sys.exit(0)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    experiments = []
    for cfg in ALL_EXPERIMENTS:
        if args.only and cfg.name != args.only:
            continue
        experiments.append(cfg)

    if not experiments:
        available = [c.name for c in ALL_EXPERIMENTS]
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