"""
Gather results from all sensor fusion ablation experiments into a comparison table.
Run this after all experiments have finished.

Usage:
    python gather_results_sensor.py --output-dir ./runs/sensor_ablation
"""

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
                "fusion_mode":                 cfg.get("fusion_mode", "?"),
                "feature_set":                 cfg.get("feature_set", "?"),
                "use_sensors":                 cfg.get("use_sensors", "?"),
                "eval_split":                  split,
                "best_epoch":                  data["best_epoch"],
                "best_val_mae_um":             data["best_val_mae_um"],
                "mae_overall_um":              m["mae_overall_um"],
                "mae_flank_wear_um":           m["mae_flank_wear_um"],
                "mae_adhesion_um":             m["mae_adhesion_um"],
                "mae_flank_wear+adhesion_um":  m["mae_flank_wear+adhesion_um"],
                "n_total":                     m["n_total"],
            })

    # Reference rows
    rows.append({
        "experiment": "vision_only_664 (reference)", "fusion_mode": "none",
        "feature_set": "none", "use_sensors": False,
        "eval_split": "test", "best_epoch": 14, "best_val_mae_um": "—",
        "mae_overall_um": 19.0, "mae_flank_wear_um": 16.6,
        "mae_adhesion_um": 37.2, "mae_flank_wear+adhesion_um": 23.2,
        "n_total": 254,
    })
    rows.append({
        "experiment": "paper_baseline", "fusion_mode": "none",
        "feature_set": "none", "use_sensors": False,
        "eval_split": "test", "best_epoch": "—", "best_val_mae_um": "—",
        "mae_overall_um": 30.0, "mae_flank_wear_um": 14.0,
        "mae_adhesion_um": 39.0, "mae_flank_wear+adhesion_um": 91.0,
        "n_total": 254,
    })

    df = pd.DataFrame(rows)

    # Save full results
    full_path = args.output_dir / "comparison_summary.csv"
    df.to_csv(full_path, index=False)

    # Print test-set overview
    test_df = df[df["eval_split"] == "test"].copy()
    test_df = test_df.sort_values("mae_overall_um")

    print("=" * 120)
    print("  SENSOR FUSION ABLATION — TEST SET RESULTS (sorted by overall MAE)")
    print("=" * 120)
    cols = ["experiment", "fusion_mode", "feature_set",
            "mae_overall_um", "mae_flank_wear_um", "mae_adhesion_um",
            "mae_flank_wear+adhesion_um", "best_epoch"]
    print(test_df[cols].to_string(index=False))
    print(f"\nFull results → {full_path}")

    # ── Quick analysis ────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("  QUICK ANALYSIS")
    print("=" * 120)

    # Exclude reference rows for analysis
    actual = test_df[~test_df["experiment"].str.contains("reference|paper", na=False)].copy()
    if len(actual) == 0:
        print("  No experiment results found yet.")
        return

    # Best overall
    best = actual.loc[actual["mae_overall_um"].idxmin()]
    print(f"\n  Best overall: {best['experiment']} at {best['mae_overall_um']:.1f} µm")

    # Vision-only control
    control = actual[actual["fusion_mode"] == "none"]
    if len(control):
        ctrl_mae = control.iloc[0]["mae_overall_um"]
        print(f"  Vision-only control (647 samples): {ctrl_mae:.1f} µm")
        print(f"  Vision-only reference (664 samples): 19.0 µm")

        # Did any sensor experiment beat the control?
        sensor_exps = actual[actual["use_sensors"] == True]
        if len(sensor_exps):
            best_sensor = sensor_exps.loc[sensor_exps["mae_overall_um"].idxmin()]
            delta = best_sensor["mae_overall_um"] - ctrl_mae
            direction = "better" if delta < 0 else "worse"
            print(f"\n  Best sensor experiment: {best_sensor['experiment']} "
                  f"at {best_sensor['mae_overall_um']:.1f} µm "
                  f"({abs(delta):.1f} µm {direction} than control)")

    # Per fusion mode summary
    print("\n  Per fusion mode (best of each):")
    for mode in ["early", "intermediate", "late"]:
        subset = actual[actual["fusion_mode"] == mode]
        if len(subset):
            best_row = subset.loc[subset["mae_overall_um"].idxmin()]
            print(f"    {mode:15s}: {best_row['experiment']:25s} "
                  f"overall={best_row['mae_overall_um']:.1f} µm  "
                  f"f+a={best_row['mae_flank_wear+adhesion_um']:.1f} µm")

    # Per feature set summary
    print("\n  Per feature set (best of each):")
    for fs in ["raw25", "top25", "all40"]:
        subset = actual[actual["feature_set"] == fs]
        if len(subset):
            best_row = subset.loc[subset["mae_overall_um"].idxmin()]
            print(f"    {fs:6s}: {best_row['experiment']:25s} "
                  f"overall={best_row['mae_overall_um']:.1f} µm  "
                  f"f+a={best_row['mae_flank_wear+adhesion_um']:.1f} µm")


if __name__ == "__main__":
    main()