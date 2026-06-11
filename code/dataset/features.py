import warnings
from pathlib import Path
import numpy as np
import time
import pandas as pd
from scipy import stats as scipy_stats
import pywt
from typing import Optional

from constants.matwi_dataset_constants import (
    WAVELET_BASELINE_N,
    WAVELET_LEVEL,
    WAVELET_MIN_ACTIVE,
    WAVELET_NAME,
    WAVELET_TRIM,
    WAVELET_Z_THRESH,
    SENSOR_CHANNELS,
    SENSOR_FS,
    N_SENSOR_FEATURES,
    BASELINE_K,
    N_WINDOWS,
    CUTTING_FEATURE_NAMES
)

from dataset.multimodal import MATWIMultimodalDatasetV1

# ══════════════════════════════════════════════════════════════════════════════
# Sensor feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def _wavelet_edges(sig: np.ndarray, z_thresh: float = WAVELET_Z_THRESH) -> tuple[int, int]:
    """
    Single-channel cutting window [start, end] via db4 level-4 detail-energy
    z-scores (their get_wavelet_indices). The detail-coefficient energy captures
    high-frequency engagement transients; the baseline mean/std come from the
    first WAVELET_BASELINE_N detail coeffs (the recording starts in air). The
    window is then expanded outward by |WAVELET_TRIM| so cut edges aren't clipped.
    Falls back to (0, len) when fewer than WAVELET_MIN_ACTIVE samples are active.
    """
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
    """
    Cutting window [start, end) = intersection of the vibration (acc) and
    acoustic wavelet edges (their dual-detection: max of starts, min of ends).
    Falls back to the full signal when detection fails.
    """
    n = len(df)
    if n == 0:
        return 0, 0
    v_s, v_e = _wavelet_edges(df["acc"].astype(np.float64).to_numpy())
    a_s, a_e = _wavelet_edges(df["acoustic"].astype(np.float64).to_numpy())
    s, e = max(v_s, a_s), min(v_e, a_e)
    if e <= s:
        return 0, n
    return s, e

def find_cut_segments(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Compatibility helper. The wavelet detector yields a single contiguous
    cutting window, so this returns a one-element list [(start, end)]."""
    return [wavelet_cut_window(df)]

def _extract_features_from_df(df: pd.DataFrame) -> np.ndarray:
    """Extract the fixed 40-feature vector from an already-selected signal."""
    features = []
    n = len(df)
    if n == 0:
        return np.zeros(N_SENSOR_FEATURES, dtype=np.float32)

    # Pre-compute FFT frequency bins (shared across channels). Recomputed from
    # the (possibly gated) length n so the Hz axis is always correct.
    freqs = np.fft.rfftfreq(n, d=1.0 / SENSOR_FS)

    for ch in SENSOR_CHANNELS:
        signal = df[ch].values.astype(np.float64)

        # ── Time domain features ──
        mean_val = float(np.mean(signal))
        std_val  = float(np.std(signal))
        rms_val  = float(np.sqrt(np.mean(signal ** 2)))
        p2p_val  = float(np.max(signal) - np.min(signal))
        kurt_val = float(scipy_stats.kurtosis(signal, fisher=True))
        if not np.isfinite(kurt_val):
            kurt_val = 0.0

        # ── Frequency domain features ──
        # Remove DC offset before FFT so 0 Hz bin is ~0
        signal_zero_mean = signal - mean_val
        fft_mag   = np.abs(np.fft.rfft(signal_zero_mean))
        fft_power = fft_mag ** 2
        total_power = float(np.sum(fft_power))

        # Dominant frequency: bin with highest magnitude (excluding DC)
        fft_mag_no_dc = fft_mag[1:]
        if len(fft_mag_no_dc) == 0 or np.all(fft_mag_no_dc == 0):
            dom_freq = 0.0
        else:
            dom_freq = float(freqs[np.argmax(fft_mag_no_dc) + 1])

        # Band energy — RELATIVE (fraction of total power), length-invariant.
        low_mask = freqs <= 200
        mid_mask = (freqs > 200) & (freqs <= 800)
        if total_power > 1e-12:
            low_energy = float(np.sum(fft_power[low_mask]) / total_power)
            mid_energy = float(np.sum(fft_power[mid_mask]) / total_power)
        else:
            low_energy = 0.0
            mid_energy = 0.0

        features.extend([mean_val, std_val, rms_val, p2p_val, kurt_val,
                         dom_freq, low_energy, mid_energy])

    return np.array(features, dtype=np.float32)

def extract_sensor_features_v1(sensor_path: Path,
                            gate_aircuts: bool = False) -> np.ndarray:
    """
    Load one sensor CSV file and extract a fixed-length feature vector.

    Sensor CSV format (no header):
        col 0: accelerometer
        col 1: acoustic emission
        col 2: Force X
        col 3: Force Y
        col 4: Force Z
        col 5: datetime (ignored here)

    Features extracted per channel (8 per channel × 5 channels = 40 total):
        Time domain (5):
            mean, std, rms, peak-to-peak (p2p), kurtosis
        Frequency domain (3):
            dominant frequency (Hz), low-band energy fraction (0–200 Hz),
            mid-band energy fraction (200–800 Hz)

    Frequency note: low/mid band energy are RELATIVE (band power / total power),
    not absolute sums. Absolute energy scales with the signal length, which is
    already variable across raw recordings (78k–99k samples) and becomes more
    variable under air-cut gating — corrupting any length-dependent feature.
    Relative band energy is length- and gain-invariant, so the same feature is
    valid whether or not gating is applied. (FFT frequencies themselves are
    NOT distorted by length: rfftfreq(n) recomputes the correct Hz axis, so a
    200 Hz component stays at 200 Hz regardless of n.)

    Parameters
    ----------
    gate_aircuts : bool
        If True, crop the signal to the wavelet-detected cutting window
        (`wavelet_cut_window`) before extracting features — the tool-approach /
        tool-retraction (air-cut) phases are removed. The 40-feature layout is
        unchanged, so downstream FEATURE_SETS and the model are unaffected.

    Returns
    -------
    np.ndarray of shape (40,), dtype float32.
    Returns zeros if the file cannot be loaded (with a warning).
    """
    try:
        df = pd.read_csv(
            sensor_path,
            header=None,
            usecols=[0, 1, 2, 3, 4],
            dtype=np.float32,
            low_memory=False,
        )
        df.columns = SENSOR_CHANNELS
    except Exception as e:
        warnings.warn(f"Could not load sensor file {sensor_path}: {e}")
        return np.zeros(N_SENSOR_FEATURES, dtype=np.float32)

    if gate_aircuts:
        start, end = wavelet_cut_window(df)
        df_cut = df.iloc[start:end].reset_index(drop=True)
        if len(df_cut) >= 4:
            return _extract_features_from_df(df_cut)

    return _extract_features_from_df(df)

def extract_sensor_features_v2(sensor_path: Path) -> np.ndarray:
    """
    Load one sensor CSV file and extract a fixed-length feature vector.

    Sensor CSV format (no header):
        col 0: accelerometer
        col 1: acoustic emission
        col 2: Force X
        col 3: Force Y
        col 4: Force Z
        col 5: datetime (ignored here)

    Features extracted per channel (8 per channel × 5 channels = 40 total):
        Time domain (5):
            mean, std, rms, peak-to-peak (p2p), kurtosis
        Frequency domain (3):
            dominant frequency (Hz), low-band energy (0–200 Hz),
            mid-band energy (200–800 Hz)

    Returns
    -------
    np.ndarray of shape (40,), dtype float32.
    Returns zeros if the file cannot be loaded (with a warning).
    """
    try:
        df = pd.read_csv(
            sensor_path,
            header=None,
            usecols=[0, 1, 2, 3, 4],
            dtype=np.float32,
            low_memory=False,
        )
        df.columns = SENSOR_CHANNELS
    except Exception as e:
        warnings.warn(f"Could not load sensor file {sensor_path}: {e}")
        return np.zeros(N_SENSOR_FEATURES, dtype=np.float32)

    features = []
    n = len(df)

    # Pre-compute FFT frequency bins (shared across channels)
    freqs = np.fft.rfftfreq(n, d=1.0 / SENSOR_FS)

    for ch in SENSOR_CHANNELS:
        signal = df[ch].values.astype(np.float64)

        # ── Time domain features ──
        mean_val = float(np.mean(signal))
        std_val  = float(np.std(signal))
        rms_val  = float(np.sqrt(np.mean(signal ** 2)))
        p2p_val  = float(np.max(signal) - np.min(signal))
        kurt_val = float(scipy_stats.kurtosis(signal, fisher=True))
        if not np.isfinite(kurt_val):
            kurt_val = 0.0

        # ── Frequency domain features ──
        # Remove DC offset before FFT so 0 Hz bin is ~0
        signal_zero_mean = signal - mean_val
        fft_mag   = np.abs(np.fft.rfft(signal_zero_mean))
        fft_power = fft_mag ** 2

        # Dominant frequency: bin with highest magnitude (excluding DC)
        fft_mag_no_dc = fft_mag[1:]
        if len(fft_mag_no_dc) == 0 or np.all(fft_mag_no_dc == 0):
            dom_freq = 0.0
        else:
            dom_freq = float(freqs[np.argmax(fft_mag_no_dc) + 1])

        # Band energy
        low_mask = freqs <= 200
        mid_mask = (freqs > 200) & (freqs <= 800)
        low_energy = float(np.sum(fft_power[low_mask]))
        mid_energy = float(np.sum(fft_power[mid_mask]))

        features.extend([mean_val, std_val, rms_val, p2p_val, kurt_val,
                         dom_freq, low_energy, mid_energy])

    return np.array(features, dtype=np.float32)

def extract_features_for_split(ds: MATWIMultimodalDatasetV1, verbose: bool = True):
    """Extract raw sensor features + wear labels for all samples in a dataset."""
    features = []
    wear_values = []
    wear_types = []

    t0 = time.time()
    for i in range(len(ds)):
        row = ds.df.iloc[i]
        sensor_path = ds.data_dir / str(row["SensorFile"])
        feat = extract_sensor_features_v1(sensor_path)
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

def extract_features_v4(sensor_path: Path) -> tuple[np.ndarray, int]:
    """Under the single-window wavelet method, v4 == v3 (one cutting window).
    Kept as an alias for backward-compatible result columns. Returns (feats, 1)."""
    feats, _ = extract_features_v3(sensor_path)
    return feats, 1

# ═════════════════════════════════════════════════════════════════════════════
# Cutting parameters (sets.csv)
# ═════════════════════════════════════════════════════════════════════════════

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
        X_v1[i] = extract_sensor_features_v1(path)
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

