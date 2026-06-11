import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge

from dataset.multimodal import MATWIMultimodalDatasetV1
from dataset.features import extract_features_for_split
from dataset.transform.sensor_scaler import SensorScaler
from constants.matwi_dataset_constants import SENSOR_CHANNELS
from metrics.mae import compute_mae_v2

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
    train_ds = MATWIMultimodalDatasetV1(**common, split="train")
    X_train, y_train, types_train = extract_features_for_split(train_ds)

    print("\n  Extracting val features...")
    val_ds = MATWIMultimodalDatasetV1(**common, split="val")
    X_val, y_val, types_val = extract_features_for_split(val_ds)

    print("\n  Extracting test features...")
    test_ds = MATWIMultimodalDatasetV1(**common, split="test")
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
    mean_results_val = compute_mae_v2(y_val, mean_pred, types_val)
    print_results("val", mean_results_val)
    mean_pred_test = np.full_like(y_test, y_train.mean())
    mean_results_test = compute_mae_v2(y_test, mean_pred_test, types_test)
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

    train_results = compute_mae_v2(y_train, pred_train, types_train)
    val_results   = compute_mae_v2(y_val, pred_val, types_val)
    test_results  = compute_mae_v2(y_test, pred_test, types_test)

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