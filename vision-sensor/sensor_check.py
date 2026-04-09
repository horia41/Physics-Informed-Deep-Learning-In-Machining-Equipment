"""
MATWI — Sensor Feature Sanity Check
=====================================
Answers: do the 40 engineered sensor features carry predictive signal for wear?

Approach:
  1. Extract 40 sensor features for all training samples
  2. Fit Ridge regression (sensor features → wear) on training set
  3. Evaluate MAE on val and test, broken down by wear type
  4. Also report feature importances (which channels/features matter most)

If Ridge MAE is:
  - <50 µm  → strong signal, fusion should help
  - 50-80 µm → moderate signal, fusion might help with good architecture
  - >80 µm  → weak signal, need better features or fusion won't help

This takes ~5 minutes to run (dominated by sensor CSV loading).

Usage:
    python sensor_sanity_check.py \
        --data-dir   ./data/matwi \
        --labels-csv ./data/matwi/labels.csv \
        --sets-csv   ./data/matwi/sets.csv
"""

import argparse
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# ── Load dataset module ───────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent


def _load_module(name: str, candidates: list[Path]):
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(name, path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
                return mod
    raise FileNotFoundError(
        f"Could not find '{name}'. Searched:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


_DS = _load_module("dataset_module", [
    _HERE / "DatasetClass_VisionSensors.py",
])

MATWIMultimodalDataset = _DS.MATWIMultimodalDataset
SensorScaler           = _DS.SensorScaler
extract_sensor_features = _DS.extract_sensor_features
SENSOR_CHANNELS        = _DS.SENSOR_CHANNELS
N_SENSOR_FEATURES      = _DS.N_SENSOR_FEATURES

WEAR_TYPES = ["flank_wear", "adhesion", "flank_wear+adhesion"]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_for_split(ds: MATWIMultimodalDataset, verbose: bool = True):
    """Extract raw sensor features + wear labels for all samples in a dataset."""
    features = []
    wear_values = []
    wear_types = []

    t0 = time.time()
    for i in range(len(ds)):
        row = ds.df.iloc[i]
        sensor_path = ds.data_dir / str(row["SensorFile"])
        feat = extract_sensor_features(sensor_path)
        features.append(feat)
        wear_values.append(float(row["wear"]))
        wear_types.append(str(row["type"]))

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(ds)}  ({elapsed:.1f}s)")

    X = np.stack(features, axis=0)    # (N, 40)
    y = np.array(wear_values)          # (N,) in µm
    elapsed = time.time() - t0
    if verbose:
        print(f"    Done: {len(ds)} samples in {elapsed:.1f}s")
    return X, y, wear_types


def compute_mae_by_type(y_true, y_pred, types):
    """Compute overall and per-wear-type MAE."""
    abs_err = np.abs(y_true - y_pred)
    results = {"overall": float(abs_err.mean()), "n": len(y_true)}
    for wt in WEAR_TYPES:
        mask = np.array([t == wt for t in types])
        if mask.any():
            results[wt] = float(abs_err[mask].mean())
            results[f"n_{wt}"] = int(mask.sum())
        else:
            results[wt] = float("nan")
            results[f"n_{wt}"] = 0
    return results


def print_results(label, results):
    print(f"  [{label:6s}]  "
          f"overall={results['overall']:.1f}µm  "
          f"flank={results.get('flank_wear', float('nan')):.1f}  "
          f"adh={results.get('adhesion', float('nan')):.1f}  "
          f"f+a={results.get('flank_wear+adhesion', float('nan')):.1f}  "
          f"(n={results['n']})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sensor feature sanity check")
    parser.add_argument("--data-dir",    type=Path, required=True)
    parser.add_argument("--labels-csv",  type=Path, required=True)
    parser.add_argument("--sets-csv",    type=Path, required=True)
    parser.add_argument("--set-range",   type=str,  default="1-13")
    parser.add_argument("--alpha",       type=float, default=1.0,
                        help="Ridge regularisation strength")
    args = parser.parse_args()

    common = dict(
        data_dir         = args.data_dir,
        labels_csv       = args.labels_csv,
        sets_csv         = args.sets_csv,
        set_range        = args.set_range,
        normalisation    = "dataset",
        augment          = False,
        wear_cap         = 450.0,
        impute_zero_wear = False,
        image_size       = (384, 384),   # won't load images, but needed for init
        sensor_scaler    = None,
    )

    print("=" * 70)
    print("  SENSOR FEATURE SANITY CHECK")
    print("  Can 40 sensor features predict wear without images?")
    print("=" * 70)

    # ── Extract features ──────────────────────────────────────────────────────
    print("\n  Extracting training features...")
    train_ds = MATWIMultimodalDataset(**common, split="train")
    X_train, y_train, types_train = extract_features_for_split(train_ds)

    print("\n  Extracting val features...")
    val_ds = MATWIMultimodalDataset(**common, split="val")
    X_val, y_val, types_val = extract_features_for_split(val_ds)

    print("\n  Extracting test features...")
    test_ds = MATWIMultimodalDataset(**common, split="test")
    X_test, y_test, types_test = extract_features_for_split(test_ds)

    # ── Standardise (fit on train only) ───────────────────────────────────────
    scaler = SensorScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    print(f"\n  Feature matrix shapes: train={X_train_s.shape}, "
          f"val={X_val_s.shape}, test={X_test_s.shape}")

    # ── Baseline: predict mean ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  BASELINE: predict training mean for everything")
    print("─" * 70)
    mean_pred = np.full_like(y_val, y_train.mean())
    mean_results_val = compute_mae_by_type(y_val, mean_pred, types_val)
    print_results("val", mean_results_val)
    mean_pred_test = np.full_like(y_test, y_train.mean())
    mean_results_test = compute_mae_by_type(y_test, mean_pred_test, types_test)
    print_results("test", mean_results_test)

    # ── Ridge regression ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"  RIDGE REGRESSION (alpha={args.alpha}) on 40 sensor features")
    print("─" * 70)

    ridge = Ridge(alpha=args.alpha)
    ridge.fit(X_train_s, y_train)

    pred_train = ridge.predict(X_train_s).clip(0, 450)
    pred_val   = ridge.predict(X_val_s).clip(0, 450)
    pred_test  = ridge.predict(X_test_s).clip(0, 450)

    train_results = compute_mae_by_type(y_train, pred_train, types_train)
    val_results   = compute_mae_by_type(y_val, pred_val, types_val)
    test_results  = compute_mae_by_type(y_test, pred_test, types_test)

    print_results("train", train_results)
    print_results("val", val_results)
    print_results("test", test_results)

    # ── Feature importance (absolute Ridge coefficients) ──────────────────────
    print("\n" + "─" * 70)
    print("  FEATURE IMPORTANCE (|Ridge coefficient| after standardisation)")
    print("─" * 70)

    feature_names = []
    for ch in SENSOR_CHANNELS:
        for feat in ["mean", "std", "rms", "p2p", "kurtosis",
                     "dom_freq", "low_energy", "mid_energy"]:
            feature_names.append(f"{ch}_{feat}")

    coef_abs = np.abs(ridge.coef_)
    sorted_idx = np.argsort(coef_abs)[::-1]

    print(f"\n  Top 15 features:")
    for rank, idx in enumerate(sorted_idx[:15], 1):
        print(f"    {rank:2d}. {feature_names[idx]:25s}  |coef|={coef_abs[idx]:.4f}")

    print(f"\n  Bottom 5 features (least useful):")
    for rank, idx in enumerate(sorted_idx[-5:], 1):
        print(f"    {rank:2d}. {feature_names[idx]:25s}  |coef|={coef_abs[idx]:.6f}")

    # ── Per-channel summary ───────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PER-CHANNEL IMPORTANCE (sum of |coef| across 8 features)")
    print("─" * 70)

    channel_importance = {}
    for i, ch in enumerate(SENSOR_CHANNELS):
        ch_coefs = coef_abs[i*8 : (i+1)*8]
        channel_importance[ch] = float(ch_coefs.sum())

    for ch, imp in sorted(channel_importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp / max(channel_importance.values()) * 30)
        print(f"    {ch:12s}  {bar}  ({imp:.4f})")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    test_mae = test_results["overall"]
    mean_mae = mean_results_test["overall"]
    improvement = (1 - test_mae / mean_mae) * 100

    print(f"  VERDICT:")
    print(f"    Predict-mean baseline : {mean_mae:.1f} µm")
    print(f"    Ridge on 40 features  : {test_mae:.1f} µm")
    print(f"    Improvement           : {improvement:.1f}%")
    print()

    if test_mae < 50:
        print("    → STRONG signal. Sensor features carry meaningful wear information.")
        print("      Multimodal fusion should improve over vision-only.")
    elif test_mae < 80:
        print("    → MODERATE signal. Some predictive value in sensor features.")
        print("      Fusion might help with a good architecture, but gains may be small.")
    else:
        print("    → WEAK signal. Sensor features alone are poor predictors.")
        print("      Need better feature engineering before fusion will help.")

    # ── Reference comparison ──────────────────────────────────────────────────
    print(f"\n  For reference:")
    print(f"    Our best vision-only  : 19.0 µm (efficientnetv2_dataset_MSE)")
    print(f"    Paper regression      : 30.0 µm")
    print(f"    Sensor-only Ridge     : {test_mae:.1f} µm")
    print("=" * 70)


if __name__ == "__main__":
    main()
