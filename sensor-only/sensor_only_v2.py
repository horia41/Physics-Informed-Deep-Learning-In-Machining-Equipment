

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
import pywt
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
# Tier 4 — In-cut filtering (v3 features)
# ═════════════════════════════════════════════════════════════════════════════
#
# Each ~50–60 sec sensor recording starts with the tool approaching the
# workpiece (no contact) and ends with it retracting (no contact). The
# middle 55–70% is actual cutting. Empirically across sets 2/5/7/8/11/13
# the "air-cut" fraction is 28–45% — Set 13 (RVS304) is worst at ~45%.
#
# Computing features over the full CSV dilutes magnitude features by the
# air-cut fraction and corrupts spectral / windowed-trend features (the
# first and last segments are pure silence). Filtering to in-cut samples
# before extraction sharpens every feature the LGBM model relies on.
#
# Detection: db4 level-4 Discrete Wavelet Transform on the vibration (acc) and
# acoustic channels (ported from the previous group's RemoveAircuts notebook).
# The detail-coefficient energy captures the high-frequency transients of tool
# engagement; it is z-scored against a baseline from the first WAVELET_BASELINE_N
# detail coeffs (the recording starts in air). The cutting window runs from the
# first to the last "active" sample (z > threshold), intersected across the two
# channels and expanded 5% outward.

WAVELET_NAME       = "db4"
WAVELET_LEVEL      = 4
WAVELET_Z_THRESH   = 10.0     # detail-energy z-score for "active" (their FIXED_SENSITIVITY)
WAVELET_BASELINE_N = 1000     # first N detail coeffs assumed to be air (approach)
WAVELET_MIN_ACTIVE = 1000     # < this many active samples -> keep the full signal
WAVELET_TRIM       = -0.05    # negative => expand window outward 5% each side (their FIXED_TRIM)


def _wavelet_edges(sig: np.ndarray, z_thresh: float = WAVELET_Z_THRESH) -> tuple[int, int]:
    """Single-channel cutting window via db4-L4 detail-energy z-scores
    (their get_wavelet_indices). Falls back to (0, len) on weak detection."""
    sig = np.array(sig, dtype=np.float64)   # copy: pywt needs a writable buffer
    n = len(sig)
    if n < 2:
        return 0, n
    coeffs = pywt.wavedec(sig, WAVELET_NAME, level=WAVELET_LEVEL)
    detail_energy = np.abs(coeffs[-1])
    if detail_energy.size == 0:
        return 0, n
    base = detail_energy[:WAVELET_BASELINE_N]
    mu  = float(np.mean(base))
    std = float(np.std(base)) + 1e-9
    z = (detail_energy - mu) / std
    upscale = n // len(z) if len(z) else 1
    z_full = np.repeat(z, upscale + 1)[:n]
    active = np.where(z_full > z_thresh)[0]
    if len(active) < WAVELET_MIN_ACTIVE:
        return 0, n
    g_start, g_end = int(active[0]), int(active[-1])
    duration = g_end - g_start
    s = int(np.clip(g_start + duration * WAVELET_TRIM, 0, n - 1))
    e = int(np.clip(g_end   - duration * WAVELET_TRIM, 0, n - 1))
    return s, e


def wavelet_cut_window(df: pd.DataFrame) -> tuple[int, int]:
    """Cutting window [start, end) = intersection of vibration (acc) and
    acoustic wavelet edges. Falls back to the full signal on weak detection."""
    n = len(df)
    if n == 0:
        return 0, 0
    v_s, v_e = _wavelet_edges(df["acc"].astype(np.float64).to_numpy())
    a_s, a_e = _wavelet_edges(df["acoustic"].astype(np.float64).to_numpy())
    s, e = max(v_s, a_s), min(v_e, a_e)
    if e <= s:
        return 0, n
    return s, e


def extract_features_v3(sensor_path: Path) -> tuple[np.ndarray, float]:
    """
    Same 85 features as v2, but computed over the wavelet-detected cutting
    window only (air cuts removed). Returns (features, keep_fraction) where
    keep_fraction is the fraction of the recording retained. Zeros on failure.
    """
    dim = get_v2_feature_dim()
    try:
        df = pd.read_csv(
            sensor_path, header=None, usecols=[0, 1, 2, 3, 4],
            dtype=np.float32, low_memory=False,
        )
        df.columns = SENSOR_CHANNELS
    except Exception as e:
        warnings.warn(f"Could not load sensor file {sensor_path}: {e}")
        return np.zeros(dim, dtype=np.float32), 0.0

    n = len(df)
    s, e = wavelet_cut_window(df)
    df_cut = df.iloc[s:e].reset_index(drop=True)
    if len(df_cut) < 4:
        df_cut, s, e = df, 0, n
    keep_frac = float((e - s) / n) if n else 0.0

    feats: list[float] = []
    for ch in SENSOR_CHANNELS:
        sig = df_cut[ch].values.astype(np.float64)
        feats.extend(_channel_features(sig, SENSOR_FS))

    Fx = df_cut["Fx"].values.astype(np.float64)
    Fy = df_cut["Fy"].values.astype(np.float64)
    Fz = df_cut["Fz"].values.astype(np.float64)
    F_res = np.sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2)
    fres_mean = float(F_res.mean())
    fres_std  = float(F_res.std())
    fres_mr, fres_sr, fres_sl, fres_ic = _windowed_rms_trend(F_res, N_WINDOWS)
    feats.extend([fres_mean, fres_std, fres_mr, fres_sr, fres_sl, fres_ic])

    fxfy = float(np.mean(np.abs(Fx) / (np.abs(Fy) + 1e-6)))
    feats.append(fxfy)
    for F in (Fx, Fy, Fz):
        feats.append(float(F.std() / (abs(F.mean()) + 1e-6)))

    return np.array(feats, dtype=np.float32), keep_frac


# ═════════════════════════════════════════════════════════════════════════════
# Tier 5 — Per-cutting-pass features (v4)
# ═════════════════════════════════════════════════════════════════════════════
#
# Under the single-window wavelet detector (their method) there is exactly ONE
# contiguous cutting window per recording, so the old per-segment averaging
# degenerates to "features on that one window" — i.e. v4 is identical to v3. v4
# is kept as an alias below so the existing method matrix / result columns and
# downstream scripts keep working unchanged.


def find_cut_segments(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Compatibility helper: the wavelet detector yields a single cutting
    window, so this returns a one-element list [(start, end)]."""
    return [wavelet_cut_window(df)]


def extract_features_v4(sensor_path: Path) -> tuple[np.ndarray, int]:
    """Under the single-window wavelet method, v4 == v3 (one cutting window).
    Kept as an alias for backward-compatible result columns. Returns (feats, 1)."""
    feats, _ = extract_features_v3(sensor_path)
    return feats, 1


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


def extract_v3_features(
    df:        pd.DataFrame,
    data_dir:  Path,
    cache_npz: Optional[Path] = None,
    force:     bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract v3 (in-cut-filtered) features for every row of df.

    Returns (X_v3, in_cut_frac):
        X_v3         shape (N, get_v2_feature_dim())
        in_cut_frac  shape (N,) — fraction of each CSV kept after filtering
    """
    dim = get_v2_feature_dim()
    if cache_npz is not None and cache_npz.exists() and not force:
        print(f"\n[cache] Loading v3 features from {cache_npz}")
        npz = np.load(cache_npz)
        if npz["X_v3"].shape[0] == len(df):
            return npz["X_v3"], npz["in_cut_frac"]
        print(f"[cache] v3 sample count mismatch ({npz['X_v3'].shape[0]} vs "
              f"{len(df)}) — re-extracting.")

    n = len(df)
    print(f"\n[extract-v3] Building in-cut-filtered features for {n} samples...")
    t0 = time.time()

    X_v3 = np.zeros((n, dim),  dtype=np.float32)
    frac = np.zeros(n,         dtype=np.float32)
    for i in range(n):
        path = data_dir / str(df.iloc[i]["SensorFile"])
        X_v3[i], frac[i] = extract_features_v3(path)
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 1e-6)
            print(f"  v3 {i+1}/{n}  ({elapsed:.1f}s, {rate:.1f}/s)")

    if cache_npz is not None:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_npz, X_v3=X_v3, in_cut_frac=frac)
        print(f"[cache] Saved v3 features to {cache_npz}")

    return X_v3, frac


def extract_v4_features(
    df:        pd.DataFrame,
    data_dir:  Path,
    cache_npz: Optional[Path] = None,
    force:     bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract v4 (per-cutting-pass, segment-averaged) features for every row.

    Returns (X_v4, n_segments):
        X_v4        shape (N, get_v2_feature_dim())
        n_segments  shape (N,) — number of cutting passes detected per CSV
    """
    dim = get_v2_feature_dim()
    if cache_npz is not None and cache_npz.exists() and not force:
        print(f"\n[cache] Loading v4 features from {cache_npz}")
        npz = np.load(cache_npz)
        if npz["X_v4"].shape[0] == len(df):
            return npz["X_v4"], npz["n_segments"]
        print(f"[cache] v4 sample count mismatch ({npz['X_v4'].shape[0]} vs "
              f"{len(df)}) — re-extracting.")

    n = len(df)
    print(f"\n[extract-v4] Building per-cutting-pass features for {n} samples...")
    t0 = time.time()

    X_v4 = np.zeros((n, dim), dtype=np.float32)
    nseg = np.zeros(n,        dtype=np.int32)
    for i in range(n):
        path = data_dir / str(df.iloc[i]["SensorFile"])
        X_v4[i], nseg[i] = extract_features_v4(path)
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 1e-6)
            print(f"  v4 {i+1}/{n}  ({elapsed:.1f}s, {rate:.1f}/s)")

    if cache_npz is not None:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_npz, X_v4=X_v4, n_segments=nseg)
        print(f"[cache] Saved v4 features to {cache_npz}")

    return X_v4, nseg


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
    X_eval:  np.ndarray,
    seed:    int = 42,
    es_frac: float = 0.15,
) -> tuple[np.ndarray, "lgb.Booster"]:
    """
    LightGBM regression with early stopping on an IN-DISTRIBUTION slice of the
    training data. Uses small-data-friendly hyperparams.

    NOTE (bug fix): the previous version early-stopped on the passed val set
    (sets 3/6/12), which is out-of-distribution (RVS 304) relative to the
    CK45-heavy train. Val MAE plateaued instantly, so the model stopped after
    2-7 boosting rounds and never trained — drowning out any feature-quality
    differences. We now carve a random `es_frac` slice from the training data
    itself for early stopping, so the stopping signal is in-distribution and
    the model actually fits.
    """
    params = dict(
        objective         = "regression_l1",
        metric            = "mae",
        learning_rate     = 0.03,
        num_leaves        = 31,
        max_depth         = 5,
        min_data_in_leaf  = 10,
        feature_fraction  = 0.8,
        bagging_fraction  = 0.8,
        bagging_freq      = 1,
        lambda_l2         = 1.0,
        verbose           = -1,
        seed              = seed,
    )

    # Carve an in-distribution early-stopping slice from train.
    n      = len(y_train)
    rng    = np.random.RandomState(seed)
    order  = rng.permutation(n)
    n_es   = max(30, int(es_frac * n))
    es_idx = order[:n_es]
    fit_idx = order[n_es:]

    dtr  = lgb.Dataset(X_train[fit_idx], label=y_train[fit_idx])
    dval = lgb.Dataset(X_train[es_idx],  label=y_train[es_idx], reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round   = 3000,
        valid_sets        = [dval],
        callbacks         = [lgb.early_stopping(150, verbose=False),
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
            # fit_predict_lgbm now carves its own in-distribution ES slice.
            pred, _ = fit_predict_lgbm(X_tr, y_tr, X_te, seed=seed + int(s))
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
