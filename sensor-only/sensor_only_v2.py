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
import importlib.util
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Optional: LightGBM (soft dependency).
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# ── Load v1 feature extractor for direct comparison ──────────────────────────

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


# DatasetClass_VisionSensors.py lives in the sibling vision-sensor/ folder
# (it is the shared multimodal dataset class). Search there first, but accept
# a local copy too in case someone vendors it for portability.
_DS = _load_module("dataset_module", [
    _HERE.parent / "vision-sensor" / "DatasetClass_VisionSensors.py",
    _HERE / "DatasetClass_VisionSensors.py",
])
extract_features_v1 = _DS.extract_sensor_features      # 40 hand features
SENSOR_CHANNELS     = _DS.SENSOR_CHANNELS              # acc, acoustic, Fx, Fy, Fz
SENSOR_FS           = _DS.SENSOR_FS                    # ≈ 1666.67 Hz
WEAR_DIVISOR        = _DS.WEAR_DIVISOR


# ── Constants ────────────────────────────────────────────────────────────────

SPLIT_SETS = {
    "train":  [1, 2, 5, 7, 8, 10, 11],
    "val":    [3, 6, 12],
    "test":   [4, 9, 13],
    "unseen": [14, 15, 16, 17],
}

WEAR_TYPES = ["flank_wear", "adhesion", "flank_wear+adhesion"]
WEAR_CAP   = 450.0

# Number of windows used for within-pass segmented features
N_WINDOWS  = 10

# Number of lowest-SensorID passes used to define the per-set baseline
BASELINE_K = 3

# Type-label canonicalisation (matches DatasetClass_VisionSensors v2)
TYPE_MAP = {
    "adhesive":                 "adhesion",
    "adhesive_wear":            "adhesion",
    "flank":                    "flank_wear",
    "flank_wear_and_adhesion":  "flank_wear+adhesion",
    "flank_wear_&_adhesion":    "flank_wear+adhesion",
    "flank_wear+adhesive_wear": "flank_wear+adhesion",
    "combination":              "flank_wear+adhesion",
}

# Cutting-parameter feature columns (from sets.csv).
# LGBM consumes NaN natively (split routes missing to a side); Ridge gets
# train-mean imputation in fit_predict_ridge.
CUTTING_FEATURE_NAMES = ["Vc", "n", "fz", "Vf", "Ae", "Ap", "z", "material_RVS"]


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1 — v2 feature extractor
# ═════════════════════════════════════════════════════════════════════════════

def _windowed_rms_trend(signal: np.ndarray, n_windows: int) -> tuple[float, float, float, float]:
    """
    Split `signal` into `n_windows` contiguous segments, compute RMS per
    segment, then fit a line (RMS vs segment-index normalised to [0,1]).

    Returns: (mean_rms, std_rms, slope, intercept).

    Captures within-pass temporal structure that whole-pass globals destroy.
    Slope/intercept describe how the channel's energy evolves during the cut.
    """
    n = len(signal)
    if n < n_windows:
        rms = float(np.sqrt(np.mean(signal ** 2))) if n else 0.0
        return rms, 0.0, 0.0, rms

    # Trim to a multiple of n_windows then reshape — much faster than np.split
    trim = n - (n % n_windows)
    seg  = signal[:trim].reshape(n_windows, -1)
    rms_per_seg = np.sqrt(np.mean(seg ** 2, axis=1))      # (n_windows,)

    x_norm = np.linspace(0.0, 1.0, n_windows)
    if np.std(rms_per_seg) < 1e-12:
        slope, intercept = 0.0, float(rms_per_seg.mean())
    else:
        slope, intercept = np.polyfit(x_norm, rms_per_seg, 1)
        slope, intercept = float(slope), float(intercept)

    return (
        float(rms_per_seg.mean()),
        float(rms_per_seg.std()),
        slope,
        intercept,
    )


def _spectral_features(signal: np.ndarray, fs: float) -> tuple[float, float, float, float, float]:
    """
    Returns 5 dimensionless spectral descriptors:
        rel_e_band0, rel_e_band1, rel_e_band2,
        spectral_entropy,
        spectral_centroid_hz

    Bands split the [0, Nyquist] range into thirds (≈ 0–278, 278–556,
    556–833 Hz at fs≈1666 Hz). Relative energy = band_power / total_power,
    which is gain- and length-invariant — the key v1 failure mode is gone.
    """
    n = len(signal)
    if n < 4:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    signal_zm = signal - signal.mean()
    freqs     = np.fft.rfftfreq(n, d=1.0 / fs)
    power     = np.abs(np.fft.rfft(signal_zm)) ** 2

    total = float(power.sum())
    if total < 1e-12:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    nyq = fs / 2.0
    b1, b2 = nyq / 3.0, 2.0 * nyq / 3.0
    m0 = freqs <= b1
    m1 = (freqs > b1) & (freqs <= b2)
    m2 = freqs > b2

    rel0 = float(power[m0].sum() / total)
    rel1 = float(power[m1].sum() / total)
    rel2 = float(power[m2].sum() / total)

    # Spectral entropy (Shannon) — peaked spectra → low entropy, broadband → high
    p_norm  = power / total
    nz      = p_norm > 1e-12
    entropy = float(-np.sum(p_norm[nz] * np.log(p_norm[nz])))

    # Spectral centroid (Hz) — frequency-weighted mean of power
    centroid = float(np.sum(freqs * power) / total)

    return rel0, rel1, rel2, entropy, centroid


def _channel_features(signal: np.ndarray, fs: float) -> list[float]:
    """
    15 features per channel:
        Basic stats (3):   mean, std, rms
        Robust spread (2): IQR (p75-p25), p95_minus_p05
        Robust shape (1):  frac_above_3sigma
        Windowed (4):      mean_rms, std_rms, slope_rms, intercept_rms
        Spectral (5):      rel_e_band0, rel_e_band1, rel_e_band2,
                            spectral_entropy, spectral_centroid
    """
    mean_v = float(signal.mean())
    std_v  = float(signal.std())
    rms_v  = float(np.sqrt(np.mean(signal ** 2)))

    p05, p25, p75, p95 = np.percentile(signal, [5, 25, 75, 95])
    iqr        = float(p75 - p25)
    p95_p05    = float(p95 - p05)
    frac_3sig  = float(np.mean(np.abs(signal - mean_v) > 3.0 * (std_v + 1e-12)))

    mean_rms, std_rms, slope_rms, int_rms = _windowed_rms_trend(signal, N_WINDOWS)
    rel0, rel1, rel2, entropy, centroid   = _spectral_features(signal, fs)

    return [
        mean_v, std_v, rms_v,
        iqr, p95_p05, frac_3sig,
        mean_rms, std_rms, slope_rms, int_rms,
        rel0, rel1, rel2, entropy, centroid,
    ]


_CHANNEL_FEATURE_NAMES = [
    "mean", "std", "rms",
    "iqr", "p95mp05", "frac3sig",
    "win_mean_rms", "win_std_rms", "win_slope_rms", "win_intercept_rms",
    "rel_e_b0", "rel_e_b1", "rel_e_b2", "spec_entropy", "spec_centroid_hz",
]


def extract_features_v2(sensor_path: Path) -> np.ndarray:
    """
    Extract the full v2 feature vector for one sensor CSV.

    Layout (returned in this order):
        per channel × 15 features = 75   (acc, acoustic, Fx, Fy, Fz)
        Force resultant block (6):       F_res mean, std, win_mean_rms,
                                          win_std_rms, win_slope_rms,
                                          win_intercept_rms
        Fx/Fy ratio (1):                  mean |Fx|/(|Fy|+ε)
        Force CV (3):                     std/(|mean|+ε) for Fx, Fy, Fz
                                                                     ─────
                                                              total: 85

    Returns zeros on load failure (with a warning).
    """
    try:
        df = pd.read_csv(
            sensor_path, header=None, usecols=[0, 1, 2, 3, 4],
            dtype=np.float32, low_memory=False,
        )
        df.columns = SENSOR_CHANNELS
    except Exception as e:
        warnings.warn(f"Could not load sensor file {sensor_path}: {e}")
        return np.zeros(get_v2_feature_dim(), dtype=np.float32)

    feats: list[float] = []

    # ── Per-channel features ───────────────────────────────────────────────
    for ch in SENSOR_CHANNELS:
        sig = df[ch].values.astype(np.float64)
        feats.extend(_channel_features(sig, SENSOR_FS))

    # ── Cross-channel: force resultant ─────────────────────────────────────
    Fx = df["Fx"].values.astype(np.float64)
    Fy = df["Fy"].values.astype(np.float64)
    Fz = df["Fz"].values.astype(np.float64)
    F_res = np.sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2)

    fres_mean = float(F_res.mean())
    fres_std  = float(F_res.std())
    fres_mr, fres_sr, fres_sl, fres_ic = _windowed_rms_trend(F_res, N_WINDOWS)
    feats.extend([fres_mean, fres_std, fres_mr, fres_sr, fres_sl, fres_ic])

    # ── Fx/Fy ratio ────────────────────────────────────────────────────────
    fxfy = float(np.mean(np.abs(Fx) / (np.abs(Fy) + 1e-6)))
    feats.append(fxfy)

    # ── Force CV (std/|mean|) per axis ─────────────────────────────────────
    for F in (Fx, Fy, Fz):
        feats.append(float(F.std() / (abs(F.mean()) + 1e-6)))

    return np.array(feats, dtype=np.float32)


def get_v2_feature_names() -> list[str]:
    names = []
    for ch in SENSOR_CHANNELS:
        for f in _CHANNEL_FEATURE_NAMES:
            names.append(f"{ch}_{f}")
    names.extend([
        "Fres_mean", "Fres_std",
        "Fres_win_mean_rms", "Fres_win_std_rms",
        "Fres_win_slope_rms", "Fres_win_intercept_rms",
    ])
    names.append("FxFy_ratio_mean")
    names.extend(["Fx_cv", "Fy_cv", "Fz_cv"])
    return names


def get_v2_feature_dim() -> int:
    return len(SENSOR_CHANNELS) * len(_CHANNEL_FEATURE_NAMES) + 6 + 1 + 3


# ═════════════════════════════════════════════════════════════════════════════
# Tier 2 — Per-set delta (wear-relative features)
# ═════════════════════════════════════════════════════════════════════════════

def compute_set_baselines(
    features:   np.ndarray,
    sets:       np.ndarray,
    sensor_ids: np.ndarray,
    k:          int = BASELINE_K,
) -> dict[int, np.ndarray]:
    """
    For each set, compute the mean feature vector of its K lowest-SensorID
    passes. These early passes correspond to a fresh/unworn tool — they
    define that setup's reference signature.

    Critically: this uses recording order (SensorID), NOT wear labels, so
    it is valid to compute on val/test sets without leaking labels.

    Returns: {set_id: baseline_vector}
    """
    baselines: dict[int, np.ndarray] = {}
    for s in np.unique(sets):
        mask = sets == s
        if not mask.any():
            continue
        ids_s   = sensor_ids[mask]
        feats_s = features[mask]
        order   = np.argsort(ids_s)
        take    = order[: min(k, len(order))]
        baselines[int(s)] = feats_s[take].mean(axis=0)
    return baselines


def apply_delta(
    features:  np.ndarray,
    sets:      np.ndarray,
    baselines: dict[int, np.ndarray],
) -> np.ndarray:
    """Subtract each sample's per-set baseline vector."""
    out = features.copy()
    for i in range(len(out)):
        s = int(sets[i])
        if s in baselines:
            out[i] -= baselines[s]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Data loading (sensor-only, no image dependency)
# ═════════════════════════════════════════════════════════════════════════════

def load_labels(
    data_dir:      Path,
    labels_csv:    Path,
    set_range:     str = "1-13",
    require_image: bool = True,
) -> pd.DataFrame:
    """
    Load + normalise labels.csv. Mirrors the filtering in
    MATWIMultimodalDataset so this script is directly comparable to the
    existing 56 µm Ridge baseline (which required both modalities).

    require_image=False → use ALL samples with sensor data, even when image
                          is missing (gives a few extra training samples).
    """
    df = pd.read_csv(labels_csv)
    df.columns = df.columns.str.strip()

    for col in ["ImageFile", "SensorFile"]:
        if col in df.columns:
            df[col] = df[col].str.replace(r"^MATWI[\\/]", "", regex=True)
            df[col] = df[col].str.replace(r"^(Set\d+)[\\/]", r"\1/\1/", regex=True)

    df["type"] = (df["type"].astype(str).str.strip().str.lower().str.replace(" ", "_"))
    df["type"] = df["type"].replace(TYPE_MAP)

    df["wear"]     = pd.to_numeric(df["wear"],     errors="coerce")
    df["Set"]      = pd.to_numeric(df["Set"],      errors="coerce").astype("Int64")
    df["SensorID"] = pd.to_numeric(df["SensorID"], errors="coerce")

    has_sensor = df["SensorName"].notna() & df["SensorID"].notna()
    if require_image:
        has_image = df["ImageName"].notna() & df["ImageID"].notna()
        df = df[has_image & has_sensor].copy()
    else:
        df = df[has_sensor].copy()

    df = df[df["wear"].notna()].copy()
    df["wear"] = df["wear"].clip(upper=WEAR_CAP)

    if set_range == "1-13":
        active = list(range(1, 14))
    elif set_range == "1-17":
        active = list(range(1, 18))
    else:
        raise ValueError(f"set_range must be '1-13' or '1-17', got '{set_range}'")
    df = df[df["Set"].isin(active)].copy().reset_index(drop=True)

    df["material"] = df["Set"].apply(lambda s: "CK45" if s <= 11 else "RVS_304")
    return df


def split_indices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build {split_name: row-index array} from SPLIT_SETS."""
    out = {}
    for split, sets in SPLIT_SETS.items():
        mask = df["Set"].isin(sets).values
        out[split] = np.nonzero(mask)[0]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Cutting parameters (sets.csv)
# ═════════════════════════════════════════════════════════════════════════════

def _parse_param(x) -> float:
    """Parse one cell of sets.csv. Handles '?', 'a/b' (variable-rate, takes mean)."""
    if pd.isna(x):
        return float("nan")
    s = str(x).strip()
    if s in ("", "?"):
        return float("nan")
    if "/" in s:
        parts = []
        for p in s.split("/"):
            try:
                parts.append(float(p))
            except ValueError:
                pass
        return float(np.mean(parts)) if parts else float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_set_params(sets_csv: Path) -> dict[int, dict]:
    """
    Load cutting parameters per set from sets.csv.

    Each row's first column is "Set N". Columns include Vc (cutting speed,
    m/min), n (spindle rpm), fz (feed per tooth, mm), Vf (feed rate, mm/min),
    Ae (radial engagement), Ap (axial depth), material, Coating, z (tooth count).

    Returns: {set_id: {param: value or nan}}
    """
    df = pd.read_csv(sets_csv)
    df = df.rename(columns={df.columns[0]: "set_name"})
    df["Set"] = df["set_name"].str.extract(r"(\d+)").astype(int)

    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        s = int(row["Set"])
        mat = str(row.get("material", "")).strip().upper()
        out[s] = {
            "Vc": _parse_param(row.get("Vc")),
            "n":  _parse_param(row.get("n")),
            "fz": _parse_param(row.get("fz")),
            "Vf": _parse_param(row.get("Vf")),
            "Ae": _parse_param(row.get("Ae")),
            "Ap": _parse_param(row.get("Ap")),
            "z":  _parse_param(row.get("z")),
            "material_RVS": 1.0 if "RVS" in mat else 0.0,
        }
    return out


def build_cutting_features(sets: np.ndarray, set_params: dict[int, dict]) -> np.ndarray:
    """Per-sample (N, 8) cutting feature matrix. NaN for unknown values."""
    out = np.full((len(sets), len(CUTTING_FEATURE_NAMES)), np.nan, dtype=np.float32)
    for i, s in enumerate(sets):
        p = set_params.get(int(s))
        if p is None:
            continue
        for j, c in enumerate(CUTTING_FEATURE_NAMES):
            out[i, j] = p[c]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Feature extraction + cache
# ═════════════════════════════════════════════════════════════════════════════

def extract_all_features(
    df:        pd.DataFrame,
    data_dir:  Path,
    cache_npz: Optional[Path] = None,
    force:     bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract v1 (40-dim) and v2 (~85-dim) features for every row of df.

    If cache_npz exists and force=False, load from cache.

    Returns: (X_v1, X_v2), both shape (N, D).
    """
    if cache_npz is not None and cache_npz.exists() and not force:
        print(f"\n[cache] Loading features from {cache_npz}")
        npz = np.load(cache_npz)
        if npz["X_v1"].shape[0] == len(df):
            return npz["X_v1"], npz["X_v2"]
        print(f"[cache] Sample count mismatch ({npz['X_v1'].shape[0]} vs "
              f"{len(df)}) — re-extracting.")

    n = len(df)
    print(f"\n[extract] Building features for {n} samples...")
    t0 = time.time()

    X_v1 = np.zeros((n, 40),                  dtype=np.float32)
    X_v2 = np.zeros((n, get_v2_feature_dim()), dtype=np.float32)

    for i in range(n):
        path = data_dir / str(df.iloc[i]["SensorFile"])
        X_v1[i] = extract_features_v1(path)
        X_v2[i] = extract_features_v2(path)
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 1e-6)
            print(f"  {i+1}/{n}  ({elapsed:.1f}s, {rate:.1f}/s)")

    if cache_npz is not None:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_npz, X_v1=X_v1, X_v2=X_v2)
        print(f"[cache] Saved features to {cache_npz}")

    return X_v1, X_v2


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═════════════════════════════════════════════════════════════════════════════

def mae_by_type(y_true: np.ndarray, y_pred: np.ndarray, types: np.ndarray) -> dict:
    err = np.abs(y_true - y_pred)
    out = {"overall": float(err.mean()), "n": int(len(y_true))}
    for wt in WEAR_TYPES:
        mask = (types == wt)
        out[wt]      = float(err[mask].mean()) if mask.any() else float("nan")
        out[f"n_{wt}"] = int(mask.sum())
    return out


def fmt_row(label: str, m: dict) -> str:
    return (f"  [{label:24s}]  overall={m['overall']:5.1f}µm  "
            f"flank={m.get('flank_wear', float('nan')):5.1f}  "
            f"adh={m.get('adhesion', float('nan')):5.1f}  "
            f"f+a={m.get('flank_wear+adhesion', float('nan')):5.1f}  "
            f"(n={m['n']})")


# ═════════════════════════════════════════════════════════════════════════════
# Models
# ═════════════════════════════════════════════════════════════════════════════

def fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval:  np.ndarray,
    alpha:   float = 1.0,
) -> tuple[np.ndarray, Ridge, StandardScaler]:
    """Standardise → Ridge → predict; clipped to [0, WEAR_CAP].

    NaN-safe: any NaN columns (e.g. cutting parameters for Set 1) are
    replaced with the per-column training mean before standardisation.
    """
    # Impute NaN per column with train mean (fall back to 0 if all-NaN)
    col_means = np.nanmean(X_train, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_train_i = np.where(np.isnan(X_train), col_means, X_train)
    X_eval_i  = np.where(np.isnan(X_eval),  col_means, X_eval)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train_i)
    X_ev   = scaler.transform(X_eval_i)

    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_train)
    pred = model.predict(X_ev).clip(0.0, WEAR_CAP)
    return pred, model, scaler


def fit_predict_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_eval:  np.ndarray,
    seed:    int = 42,
) -> tuple[np.ndarray, "lgb.Booster"]:
    """
    LightGBM regression with early stopping on val MAE.
    Uses small-data-friendly hyperparams (shallow trees, low LR, regularised).
    """
    params = dict(
        objective         = "regression_l1",
        metric            = "mae",
        learning_rate     = 0.03,
        num_leaves        = 31,           # was 15 — more capacity (max_depth caps it)
        max_depth         = 5,            # was 4
        min_data_in_leaf  = 10,
        feature_fraction  = 0.8,
        bagging_fraction  = 0.8,
        bagging_freq      = 1,
        lambda_l2         = 1.0,
        verbose           = -1,
        seed              = seed,
    )
    dtr  = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val,   label=y_val, reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round   = 3000,         # was 2000
        valid_sets        = [dval],
        callbacks         = [lgb.early_stopping(150, verbose=False),   # was 50
                             lgb.log_evaluation(0)],
    )
    pred = model.predict(X_eval, num_iteration=model.best_iteration)
    return pred.clip(0.0, WEAR_CAP), model


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
            # Carve a tiny val split inside the training fold for early stop
            rng    = np.random.RandomState(seed + int(s))
            order  = rng.permutation(len(y_tr))
            n_val  = max(20, int(0.1 * len(y_tr)))
            v_idx, t_idx = order[:n_val], order[n_val:]
            pred, _ = fit_predict_lgbm(
                X_tr[t_idx], y_tr[t_idx],
                X_tr[v_idx], y_tr[v_idx],
                X_te,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown model_kind={model_kind}")

        fold_preds[ho] = pred
        m = mae_by_type(y_te, pred, types[ho])
        m["set"] = int(s)
        fold_results.append(m)

    # Aggregate (sample-weighted) across folds — the per-fold "overall" is
    # already a per-sample average within a fold, so pool predictions.
    valid = ~np.isnan(fold_preds)
    overall = mae_by_type(y[valid], fold_preds[valid].astype(np.float32), types[valid])
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
    df = load_labels(args.data_dir, args.labels_csv,
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

    # ── Precompute deltas (using each set's own first K passes) ──────────────
    base_v1 = compute_set_baselines(X_v1, sets, sensor_ids)
    base_v2 = compute_set_baselines(X_v2, sets, sensor_ids)
    X_v1_d  = apply_delta(X_v1, sets, base_v1)
    X_v2_d  = apply_delta(X_v2, sets, base_v2)

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
    ]
    if _HAS_LGBM:
        methods += [
            Method("lgbm_v2_absolute", X_v2,   X_v2, "lgbm", v2_names, False),
            Method("lgbm_v2_delta",    X_v2_d, X_v2, "lgbm", v2_names, True),
        ]
        if use_cutting:
            X_v2_cut   = np.hstack([X_v2, X_cut]).astype(np.float32)
            X_v1v2     = np.hstack([X_v1, X_v2]).astype(np.float32)
            X_v1v2_cut = np.hstack([X_v1, X_v2, X_cut]).astype(np.float32)
            methods += [
                Method("lgbm_v2cut_absolute",   X_v2_cut,   X_v2_cut,
                       "lgbm", v2_names + cut_names, False),
                Method("lgbm_v1v2_absolute",    X_v1v2,     X_v1v2,
                       "lgbm", v1_names + v2_names, False),
                Method("lgbm_v1v2cut_absolute", X_v1v2_cut, X_v1v2_cut,
                       "lgbm", v1_names + v2_names + cut_names, False),
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
        m = mae_by_type(y[sel], np.full(len(sel), y_mean, dtype=np.float32), types[sel])
        baselines_result[name] = m
        print(fmt_row(f"predict_mean / {name}", m))

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
            pred_train, model, scaler = fit_predict_ridge(
                X_train, y_train, X_train, alpha=args.ridge_alpha,
            )
            _, _, _ = scaler, model, pred_train  # silence linter
            pred_val,  _, _ = fit_predict_ridge(X_train, y_train, X_val,  alpha=args.ridge_alpha)
            pred_test, m_r, sc = fit_predict_ridge(X_train, y_train, X_test, alpha=args.ridge_alpha)
            coefs = m_r.coef_
        else:
            pred_val,  _      = fit_predict_lgbm(X_train, y_train, X_val, y_val, X_val,  seed=args.seed)
            pred_test, m_lgb  = fit_predict_lgbm(X_train, y_train, X_val, y_val, X_test, seed=args.seed)
            coefs = m_lgb.feature_importance(importance_type="gain")

        # also predict on train for sanity (gap diagnosis)
        if meth.kind == "ridge":
            pred_tr_self, _, _ = fit_predict_ridge(X_train, y_train, X_train,
                                                   alpha=args.ridge_alpha)
        else:
            pred_tr_self, _ = fit_predict_lgbm(X_train, y_train, X_val, y_val, X_train,
                                               seed=args.seed)

        m_train = mae_by_type(y_train, pred_tr_self, types[tr])
        m_val   = mae_by_type(y_val,   pred_val,     types[va])
        m_test  = mae_by_type(y[te],   pred_test,    types[te])

        all_results[meth.name] = {"train": m_train, "val": m_val, "test": m_test,
                                  "feature_importance": coefs.tolist(),
                                  "feature_names": meth.feat_names}

        print()
        print(fmt_row(f"{meth.name} / train", m_train))
        print(fmt_row(f"{meth.name} / val",   m_val))
        print(fmt_row(f"{meth.name} / test",  m_test))

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
        print(fmt_row(f"{meth.name} / LOSO-pooled", loso["pooled"]))
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
