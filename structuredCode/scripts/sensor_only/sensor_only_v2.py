"""
MATWI — Sensor-Only Baseline v2
==================================
A better sensor-only wear-prediction baseline than the v1 Ridge-on-40-features
attempt (which got 56 µm test MAE — worse than predict-mean at 40.9 µm).

What this script implements:

  Tier 1 — Richer feature set (~85 features) that addresses the v1 failures:
    • Relative band energy + spectral entropy + spectral centroid
      (gain- and length-invariant — fixes v1's set-identity leakage via
      absolute low/mid band energy)
    • Within-pass windowed RMS → trend (mean / std / slope / intercept)
      across N=10 segments — captures wear-driven temporal structure that
      whole-pass globals destroy
    • Robust spread (IQR, p95-p05) and frac-above-3σ instead of v1's
      outlier-dominated p2p and kurtosis
    • Cross-channel physical features: force resultant √(Fx²+Fy²+Fz²)
      (windowed stats), Fx/Fy ratio, per-channel coefficient of variation
    • Drops v1's whole-pass dom_freq (meaningless over 99k samples)

  Tier 2 — Per-set delta features that handle train→test distribution shift:
    • Use the K=3 lowest-SensorID passes in each set as that set's
      "unworn baseline reference" (legitimate — uses recording order,
      not labels)
    • All features become deltas from this per-set baseline → model sees
      wear-relative deviations instead of absolute setup-dependent levels
    • This is the single biggest leverage point for cross-material transfer

  Tier 3 — LightGBM with monotone bias + LOSO CV evaluation
    • LightGBM handles tabular nonlinearities and is robust to leftover
      scale differences Ridge can't reconcile
    • Leave-one-set-out CV on train measures honest cross-set generalisation
      (separates "feature quality" from "train→test domain shift")
    • Falls back to a Ridge-only run if lightgbm is not installed

Comparison matrix:
   1. predict_mean_train  — reference (~40.9 µm test)
   2. ridge_v1_absolute   — reproduces the existing 56 µm baseline
   3. ridge_v2_absolute   — Tier 1 alone (does new feature engineering help?)
   4. ridge_v2_delta      — Tier 1 + Tier 2 (does per-set delta close the gap?)
   5. lgbm_v2_absolute    — Tier 1 + Tier 3
   6. lgbm_v2_delta       — Tier 1 + Tier 2 + Tier 3  (the full proposed method)

All methods are also run under leave-one-set-out CV on the training sets so
we get an honest cross-setup generalisation estimate, not just train→test MAE.

Output (default --output-dir ../runs/sensor_only_v2/):
    features_cache.npz   — extracted v1 + v2 features per sample (reusable)
    results.json         — full results, per method, per split
    results.csv          — flat table for quick reading
    loso_results.csv     — leave-one-set-out CV per method
    feature_importance.csv — top features for the best method

Usage:
    python sensor_only_v2.py \
        --data-dir   ../dataset/matwi \
        --labels-csv ../dataset/matwi/labels.csv \
        --sets-csv   ../dataset/matwi/sets.csv \
        --output-dir ../runs/sensor_only_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# Force UTF-8 stdout/stderr so the box-drawing (─) and µ characters in the
# progress/summary output don't crash on Windows when piped or redirected
# (cp1252 default raises UnicodeEncodeError under `... | tail`, `> log.txt`,
# or SLURM log files). No-op on POSIX / already-UTF-8 streams.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Optional: LightGBM (soft dependency).
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

from model.sensor_only.lgbm import fit_predict_lgbm
from model.sensor_only.ridge import fit_predict_ridge
from dataset.matwi.sensor_features import compute_set_baselines, apply_delta, split_indices, extract_all_features, extract_v3_features, extract_v4_features, get_v2_feature_names, build_cutting_features
from metrics.mae import compute_mae_by_type
from metrics.fmt import fmt_metrics_v2
from dataset.matwi.dataset_loader import load_labels_v2, load_set_params
from constants import WEAR_CAP, CUTTING_FEATURE_NAMES

# ═════════════════════════════════════════════════════════════════════════════
# Leave-one-set-out CV
# ═════════════════════════════════════════════════════════════════════════════

def loso_cv(
    X:          np.ndarray,
    y:          np.ndarray,
    types:      np.ndarray,
    sets:       np.ndarray,
    sensor_ids: np.ndarray,
    use_delta:  bool,
    model_kind: str,     # "ridge" or "lgbm"
    seed:       int = 42,
) -> dict:
    """
    Leave-one-set-out on the provided sample matrix (typically training sets).

    Per fold:
      - Hold out set s
      - If use_delta: recompute baselines using only the remaining sets that
        appear in the in-fold data + the held-out set itself (so the
        held-out set still has its own setup baseline; no label leakage)
      - Train on remaining samples, predict on the held-out set
    """
    unique_sets = np.unique(sets)
    fold_results = []
    fold_preds   = np.full_like(y, fill_value=np.nan, dtype=np.float64)

    for s in unique_sets:
        ho   = sets == s
        keep = ~ho
        if not ho.any() or not keep.any():
            continue

        X_tr_raw, X_te_raw = X[keep], X[ho]
        y_tr,    y_te      = y[keep], y[ho]

        if use_delta:
            # Baselines on training portion + held-out set
            # (held-out uses its own SensorID, never its labels)
            base_tr = compute_set_baselines(X_tr_raw, sets[keep], sensor_ids[keep])
            base_te = compute_set_baselines(X_te_raw, sets[ho],   sensor_ids[ho])
            X_tr    = apply_delta(X_tr_raw, sets[keep], base_tr)
            X_te    = apply_delta(X_te_raw, sets[ho],   base_te)
        else:
            X_tr, X_te = X_tr_raw, X_te_raw

        if model_kind == "ridge":
            pred, _, _ = fit_predict_ridge(X_tr, y_tr, X_te)
        elif model_kind == "lgbm":
            # fit_predict_lgbm now carves its own in-distribution ES slice.
            pred, _ = fit_predict_lgbm(X_tr, y_tr, X_te, seed=seed + int(s))
        else:
            raise ValueError(f"Unknown model_kind={model_kind}")

        fold_preds[ho] = pred
        m = compute_mae_by_type(y_te, pred, types[ho])
        m["set"] = int(s)
        fold_results.append(m)

    # Aggregate (sample-weighted) across folds — the per-fold "overall" is
    # already a per-sample average within a fold, so pool predictions.
    valid = ~np.isnan(fold_preds)
    overall = compute_mae_by_type(y[valid], fold_preds[valid].astype(np.float32), types[valid])
    return {"per_fold": fold_results, "pooled": overall}


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MATWI sensor-only baseline v2")
    parser.add_argument("--data-dir",     type=Path, required=True)
    parser.add_argument("--labels-csv",   type=Path, required=True)
    parser.add_argument("--sets-csv",     type=Path, required=False, default=None,
                        help="Path to sets.csv. Defaults to <data-dir>/sets.csv. "
                             "Used to build cutting-parameter features (Vc, fz, ...).")
    parser.add_argument("--output-dir",   type=Path, required=True)
    parser.add_argument("--set-range",    type=str,  default="1-13",
                        choices=["1-13", "1-17"])
    parser.add_argument("--require-image", action="store_true", default=True,
                        help="Match the existing baseline (samples with both modalities). "
                             "Default True for comparability.")
    parser.add_argument("--all-sensor-samples", action="store_true",
                        help="Drop the require-image filter — use all sensor samples.")
    parser.add_argument("--force-extract", action="store_true",
                        help="Re-extract features even if cache exists.")
    parser.add_argument("--no-cutting-params", action="store_true",
                        help="Skip cutting-parameter feature methods (Vc, fz, ...). "
                             "By default, methods using cutting params are included.")
    parser.add_argument("--ridge-alpha",  type=float, default=1.0)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    if args.sets_csv is None:
        args.sets_csv = args.data_dir / "sets.csv"

    require_image = args.require_image and not args.all_sensor_samples
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  SENSOR-ONLY BASELINE v2")
    print(f"  set_range={args.set_range}  require_image={require_image}  "
          f"lgbm_available={_HAS_LGBM}")
    print("=" * 78)

    # ── Load labels and split ─────────────────────────────────────────────────
    df = load_labels_v2(args.data_dir, args.labels_csv,
                     set_range=args.set_range, require_image=require_image)
    idx = split_indices(df)
    print(f"\nSplit sizes: " + "  ".join(f"{k}={len(v)}" for k, v in idx.items()))

    types       = df["type"].values
    sets        = df["Set"].astype(int).values
    sensor_ids  = df["SensorID"].values
    y           = df["wear"].astype(np.float32).values

    # ── Extract features (cached) ─────────────────────────────────────────────
    cache_path = args.output_dir / f"features_cache_{args.set_range.replace('-','_')}.npz"
    X_v1, X_v2 = extract_all_features(df, args.data_dir,
                                      cache_npz=cache_path,
                                      force=args.force_extract)
    print(f"  v1 features: {X_v1.shape}   v2 features: {X_v2.shape}")

    # ── v3 features: same as v2 but computed only on in-cut samples ──────────
    v3_cache = args.output_dir / f"features_v3_cache_{args.set_range.replace('-','_')}.npz"
    X_v3, in_cut_frac = extract_v3_features(df, args.data_dir,
                                            cache_npz=v3_cache,
                                            force=args.force_extract)
    print(f"  v3 features: {X_v3.shape}   "
          f"in-cut keep fraction: mean={in_cut_frac.mean():.1%}  "
          f"p10={np.percentile(in_cut_frac,10):.1%}  "
          f"p50={np.percentile(in_cut_frac,50):.1%}  "
          f"p90={np.percentile(in_cut_frac,90):.1%}")
    # Per-set breakdown so we can see which sets had the most air cut
    by_set = pd.DataFrame({"Set": sets, "in_cut_frac": in_cut_frac})
    set_summary = by_set.groupby("Set")["in_cut_frac"].agg(["mean", "min", "max"])
    print("  in-cut fraction by set (sample mean / min / max):")
    for s, row in set_summary.iterrows():
        print(f"    Set {int(s):>2}:  mean={row['mean']:.1%}  "
              f"min={row['min']:.1%}  max={row['max']:.1%}")

    # ── v4 features: per-cutting-pass extraction, segment-averaged ───────────
    v4_cache = args.output_dir / f"features_v4_cache_{args.set_range.replace('-','_')}.npz"
    X_v4, n_segments = extract_v4_features(df, args.data_dir,
                                           cache_npz=v4_cache,
                                           force=args.force_extract)
    print(f"  v4 features: {X_v4.shape}   "
          f"segments/CSV: mean={n_segments.mean():.1f}  "
          f"min={int(n_segments.min())}  max={int(n_segments.max())}")

    # ── Precompute deltas (using each set's own first K passes) ──────────────
    base_v1 = compute_set_baselines(X_v1, sets, sensor_ids)
    base_v2 = compute_set_baselines(X_v2, sets, sensor_ids)
    base_v3 = compute_set_baselines(X_v3, sets, sensor_ids)
    base_v4 = compute_set_baselines(X_v4, sets, sensor_ids)
    X_v1_d  = apply_delta(X_v1, sets, base_v1)
    X_v2_d  = apply_delta(X_v2, sets, base_v2)
    X_v3_d  = apply_delta(X_v3, sets, base_v3)
    X_v4_d  = apply_delta(X_v4, sets, base_v4)

    # ── Cutting parameters (per-set Vc, fz, Vf, Ae, Ap, z, material) ─────────
    use_cutting = (not args.no_cutting_params) and args.sets_csv.exists()
    if use_cutting:
        set_params = load_set_params(args.sets_csv)
        X_cut      = build_cutting_features(sets, set_params)
        # Diagnostic: report which active sets have NaN values
        active = sorted({int(s) for s in sets})
        unknown = [s for s in active if any(np.isnan(list(set_params.get(s, {}).values())))]
        print(f"  cutting params: {X_cut.shape[1]} features  "
              f"(sets with NaN: {unknown if unknown else 'none'})")
    else:
        X_cut = None
        if args.no_cutting_params:
            print("  cutting params: disabled via --no-cutting-params")
        else:
            print(f"  cutting params: sets.csv not found at {args.sets_csv} — skipping")

    v1_names  = [f"v1_{i}"  for i in range(40)]
    v1d_names = [f"v1d_{i}" for i in range(40)]
    v2_names  = get_v2_feature_names()
    cut_names = list(CUTTING_FEATURE_NAMES)

    # ── Method matrix ─────────────────────────────────────────────────────────
    # X      = features used for training the model
    # X_raw  = pre-delta features for LOSO to recompute baselines per fold
    @dataclass
    class Method:
        name:       str
        X:          np.ndarray
        X_raw:      np.ndarray
        kind:       str          # "ridge" or "lgbm"
        feat_names: list[str]
        use_delta:  bool

    methods: list[Method] = [
        Method("ridge_v1_absolute", X_v1,   X_v1, "ridge", v1_names,  False),
        Method("ridge_v1_delta",    X_v1_d, X_v1, "ridge", v1d_names, True),
        Method("ridge_v2_absolute", X_v2,   X_v2, "ridge", v2_names,  False),
        Method("ridge_v2_delta",    X_v2_d, X_v2, "ridge", v2_names,  True),
        Method("ridge_v3_absolute", X_v3,   X_v3, "ridge", v2_names,  False),
        Method("ridge_v3_delta",    X_v3_d, X_v3, "ridge", v2_names,  True),
        Method("ridge_v4_absolute", X_v4,   X_v4, "ridge", v2_names,  False),
        Method("ridge_v4_delta",    X_v4_d, X_v4, "ridge", v2_names,  True),
    ]
    if _HAS_LGBM:
        methods += [
            Method("lgbm_v2_absolute", X_v2,   X_v2, "lgbm", v2_names, False),
            Method("lgbm_v2_delta",    X_v2_d, X_v2, "lgbm", v2_names, True),
            Method("lgbm_v3_absolute", X_v3,   X_v3, "lgbm", v2_names, False),
            Method("lgbm_v3_delta",    X_v3_d, X_v3, "lgbm", v2_names, True),
            Method("lgbm_v4_absolute", X_v4,   X_v4, "lgbm", v2_names, False),
            Method("lgbm_v4_delta",    X_v4_d, X_v4, "lgbm", v2_names, True),
        ]
        if use_cutting:
            X_v2_cut   = np.hstack([X_v2, X_cut]).astype(np.float32)
            X_v1v2     = np.hstack([X_v1, X_v2]).astype(np.float32)
            X_v1v2_cut = np.hstack([X_v1, X_v2, X_cut]).astype(np.float32)
            X_v3_cut   = np.hstack([X_v3, X_cut]).astype(np.float32)
            X_v2v3     = np.hstack([X_v2, X_v3]).astype(np.float32)
            X_v4_cut   = np.hstack([X_v4, X_cut]).astype(np.float32)
            methods += [
                Method("lgbm_v2cut_absolute",   X_v2_cut,   X_v2_cut,
                       "lgbm", v2_names + cut_names, False),
                Method("lgbm_v1v2_absolute",    X_v1v2,     X_v1v2,
                       "lgbm", v1_names + v2_names, False),
                Method("lgbm_v1v2cut_absolute", X_v1v2_cut, X_v1v2_cut,
                       "lgbm", v1_names + v2_names + cut_names, False),
                Method("lgbm_v4cut_absolute",   X_v4_cut,   X_v4_cut,
                       "lgbm", v2_names + cut_names, False),
                Method("lgbm_v3cut_absolute",   X_v3_cut,   X_v3_cut,
                       "lgbm", v2_names + cut_names, False),
                Method("lgbm_v2v3_absolute",    X_v2v3,     X_v2v3,
                       "lgbm", [f"v2_{n}" for n in v2_names] +
                                [f"v3_{n}" for n in v2_names], False),
            ]
    else:
        print("\n[warn] lightgbm not installed — skipping LGBM methods. "
              "`pip install lightgbm` to enable.")

    tr, va, te = idx["train"], idx["val"], idx["test"]

    # ── Baseline: predict training mean ───────────────────────────────────────
    print("\n" + "─" * 78)
    print("  BASELINE — predict mean(y_train)")
    print("─" * 78)
    y_mean = float(y[tr].mean())
    baselines_result = {}
    for name, sel in [("val", va), ("test", te)]:
        m = compute_mae_by_type(y[sel], np.full(len(sel), y_mean, dtype=np.float32), types[sel])
        baselines_result[name] = m
        print(fmt_metrics_v2(f"predict_mean / {name}", m))

    # ── Train/eval all methods on the official split ─────────────────────────
    print("\n" + "─" * 78)
    print("  OFFICIAL SPLIT — train → val/test")
    print("─" * 78)

    all_results: dict[str, dict] = {
        "predict_mean_train": {"val": baselines_result["val"],
                                "test": baselines_result["test"],
                                "params": {"y_mean": y_mean}},
    }
    best_test_method = None
    best_test_mae   = float("inf")

    for meth in methods:
        X_train = meth.X[tr]
        X_val   = meth.X[va]
        X_test  = meth.X[te]
        y_train = y[tr]
        y_val   = y[va]

        if meth.kind == "ridge":
            # Fit once, predict all splits (Ridge has a closed-form fit).
            _, m_r, scaler = fit_predict_ridge(X_train, y_train, X_train,
                                               alpha=args.ridge_alpha)
            col_means = np.nanmean(X_train, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            def _ridge_pred(Xe):
                Xi = np.where(np.isnan(Xe), col_means, Xe)
                return m_r.predict(scaler.transform(Xi)).clip(0.0, WEAR_CAP)
            pred_tr_self = _ridge_pred(X_train)
            pred_val     = _ridge_pred(X_val)
            pred_test    = _ridge_pred(X_test)
            coefs = m_r.coef_
        else:
            # Train LGBM ONCE (in-distribution early stopping), predict all splits.
            pred_test, m_lgb = fit_predict_lgbm(X_train, y_train, X_test, seed=args.seed)
            pred_val     = m_lgb.predict(X_val,   num_iteration=m_lgb.best_iteration).clip(0.0, WEAR_CAP)
            pred_tr_self = m_lgb.predict(X_train, num_iteration=m_lgb.best_iteration).clip(0.0, WEAR_CAP)
            coefs = m_lgb.feature_importance(importance_type="gain")

        m_train = compute_mae_by_type(y_train, pred_tr_self, types[tr])
        m_val   = compute_mae_by_type(y_val,   pred_val,     types[va])
        m_test  = compute_mae_by_type(y[te],   pred_test,    types[te])

        all_results[meth.name] = {"train": m_train, "val": m_val, "test": m_test,
                                  "feature_importance": coefs.tolist(),
                                  "feature_names": meth.feat_names}

        print()
        print(fmt_metrics_v2(f"{meth.name} / train", m_train))
        print(fmt_metrics_v2(f"{meth.name} / val",   m_val))
        print(fmt_metrics_v2(f"{meth.name} / test",  m_test))

        if m_test["overall"] < best_test_mae:
            best_test_mae   = m_test["overall"]
            best_test_method = meth

    # ── Leave-one-set-out CV on train ────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  LEAVE-ONE-SET-OUT CV ON TRAIN  (honest cross-setup generalisation)")
    print("─" * 78)
    loso_rows = []
    for meth in methods:
        # Pass raw (pre-delta) features so loso_cv can rebuild baselines per fold
        loso = loso_cv(
            X            = meth.X_raw[tr],
            y            = y[tr],
            types        = types[tr],
            sets         = sets[tr],
            sensor_ids   = sensor_ids[tr],
            use_delta    = meth.use_delta,
            model_kind   = meth.kind,
            seed         = args.seed,
        )
        print(fmt_metrics_v2(f"{meth.name} / LOSO-pooled", loso["pooled"]))
        # Per-fold details
        for fold in loso["per_fold"]:
            loso_rows.append({
                "method":              meth.name,
                "heldout_set":         fold["set"],
                "mae_overall_um":      fold["overall"],
                "mae_flank_wear_um":   fold.get("flank_wear", float("nan")),
                "mae_adhesion_um":     fold.get("adhesion", float("nan")),
                "mae_flank_wear+adh":  fold.get("flank_wear+adhesion", float("nan")),
                "n":                   fold["n"],
            })
        all_results[meth.name]["loso_pooled"] = loso["pooled"]
        all_results[meth.name]["loso_folds"]  = loso["per_fold"]

    # ── Feature importance for best method ───────────────────────────────────
    if best_test_method is not None:
        print("\n" + "─" * 78)
        print(f"  TOP 20 FEATURES — best method: {best_test_method.name} "
              f"(test={best_test_mae:.1f}µm)")
        print("─" * 78)
        coefs = np.array(all_results[best_test_method.name]["feature_importance"])
        names = best_test_method.feat_names
        order = np.argsort(np.abs(coefs))[::-1]
        rows  = []
        for rank, i in enumerate(order[:20], 1):
            print(f"    {rank:2d}. {names[i]:32s}  {abs(coefs[i]):.4f}")
            rows.append({"rank": rank, "feature": names[i],
                         "importance_abs": float(abs(coefs[i]))})
        pd.DataFrame(rows).to_csv(args.output_dir / "feature_importance.csv", index=False)

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(args.output_dir / "results.json", "w") as fh:
        json.dump({"y_mean_train": y_mean,
                   "lgbm_available": _HAS_LGBM,
                   "split_sizes":    {k: int(len(v)) for k, v in idx.items()},
                   "results":        all_results},
                  fh, indent=2, default=float)

    flat_rows = []
    for meth_name, res in all_results.items():
        for split in ("train", "val", "test"):
            if split not in res:
                continue
            m = res[split]
            flat_rows.append({
                "method":             meth_name,
                "split":              split,
                "mae_overall_um":     m["overall"],
                "mae_flank_wear_um":  m.get("flank_wear", float("nan")),
                "mae_adhesion_um":    m.get("adhesion", float("nan")),
                "mae_f_plus_a_um":    m.get("flank_wear+adhesion", float("nan")),
                "n":                  m["n"],
            })
        if "loso_pooled" in res:
            m = res["loso_pooled"]
            flat_rows.append({
                "method":             meth_name,
                "split":              "loso_pooled",
                "mae_overall_um":     m["overall"],
                "mae_flank_wear_um":  m.get("flank_wear", float("nan")),
                "mae_adhesion_um":    m.get("adhesion", float("nan")),
                "mae_f_plus_a_um":    m.get("flank_wear+adhesion", float("nan")),
                "n":                  m["n"],
            })
    pd.DataFrame(flat_rows).to_csv(args.output_dir / "results.csv", index=False)
    pd.DataFrame(loso_rows).to_csv(args.output_dir / "loso_results.csv", index=False)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  SUMMARY — test set MAE (lower = better)")
    print("=" * 78)
    test_df = (pd.DataFrame(flat_rows)
                 .query("split == 'test'")
                 .sort_values("mae_overall_um")
                 .reset_index(drop=True))
    print(test_df.to_string(index=False))

    print(f"\n  Predict-mean reference : {baselines_result['test']['overall']:.1f} µm")
    print(f"  Best method            : {test_df.iloc[0]['method']} "
          f"@ {test_df.iloc[0]['mae_overall_um']:.1f} µm")
    print(f"\n  Output dir: {args.output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
