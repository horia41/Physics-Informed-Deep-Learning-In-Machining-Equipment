#!/usr/bin/env python3


import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for results_file in sorted(args.output_dir.rglob("results.json")):
        with open(results_file) as f:
            data = json.load(f)

        cfg = data.get("config", {})
        for split, m in data.get("splits", {}).items():
            rows.append({
                "experiment":                  data["experiment"],
                "backbone":                    cfg.get("backbone", "?"),
                "head_type":                   cfg.get("head_type", "?"),
                "normalisation":               cfg.get("normalisation", "?"),
                "loss":                        cfg.get("loss", "?"),
                "scheduler":                   cfg.get("use_scheduler", "?"),
                "eval_split":                  split,
                "best_epoch":                  data["best_epoch"],
                "best_val_mae_um":             data["best_val_mae_um"],
                "mae_overall_um":              m["mae_overall_um"],
                "mae_flank_wear_um":           m["mae_flank_wear_um"],
                "mae_adhesion_um":             m["mae_adhesion_um"],
                "mae_flank_wear+adhesion_um":  m["mae_flank_wear+adhesion_um"],
                "n_total":                     m["n_total"],
            })

    # Add paper reference
    rows.append({
        "experiment": "paper_baseline", "backbone": "resnet50",
        "head_type": "simple", "normalisation": "?", "loss": "?",
        "scheduler": False, "eval_split": "test", "best_epoch": "—",
        "best_val_mae_um": "—", "mae_overall_um": 30.0,
        "mae_flank_wear_um": 14.0, "mae_adhesion_um": 39.0,
        "mae_flank_wear+adhesion_um": 91.0, "n_total": "—",
    })

    df = pd.DataFrame(rows)

    # Save full results
    full_path = args.output_dir / "comparison_summary.csv"
    df.to_csv(full_path, index=False)

    # Print test-set overview
    test_df = df[df["eval_split"] == "test"].copy()
    test_df = test_df.sort_values("mae_overall_um")

    print("=" * 100)
    print("  VISION ABLATION — TEST SET RESULTS (sorted by overall MAE)")
    print("=" * 100)
    cols = ["experiment", "head_type", "normalisation", "loss",
            "mae_overall_um", "mae_flank_wear_um", "mae_adhesion_um",
            "mae_flank_wear+adhesion_um", "best_epoch"]
    print(test_df[cols].to_string(index=False))
    print(f"\nFull results → {full_path}")


if __name__ == "__main__":
    main()