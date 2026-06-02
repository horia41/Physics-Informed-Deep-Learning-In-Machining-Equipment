"""
MATWI — Vision + Sensor Dataset Class (v2)
============================================
Handles image loading, cropping, normalisation, sensor feature extraction,
sensor feature standardisation, and feature set selection for Stage 2+.

Changes from v1:
  - Added feature_set parameter ("all40", "top25", "raw25") for controlled
    sensor feature ablation
  - Fixed type_map: added "flank_wear+adhesive_wear" → "flank_wear+adhesion"
  - Added FEATURE_SETS constant with precomputed index arrays
  - Added n_selected_features property

Sensor feature engineering (per milling pass CSV):
  For each of the 5 channels (acc, acoustic, Fx, Fy, Fz):
    Time domain (5):  mean, std, rms, peak-to-peak, kurtosis
    Freq domain (3):  dominant FFT frequency, low-band energy, mid-band energy
  Total: 40 features per sample

Feature sets:
  "all40"  — all 40 features (time + frequency domain)
  "raw25"  — 25 time-domain features only (no FFT engineering)
  "top25"  — top 25 features by Ridge regression importance (mix of time + freq)

Sensor standardisation:
  Features are standardised (zero mean, unit std) computed on TRAINING SET ONLY.
  Scaler fits on all 40 features; subset selection happens at return time.

Usage:
    from DatasetClass_VisionSensors import MATWIMultimodalDataset, FEATURE_SETS

    train_ds = MATWIMultimodalDataset(
        data_dir        = "./data/matwi",
        labels_csv      = "./data/matwi/labels.csv",
        sets_csv        = "./data/matwi/sets.csv",
        split           = "train",
        set_range       = "1-13",
        normalisation   = "dataset",
        augment         = False,
        wear_cap        = 450.0,
        feature_set     = "all40",
        sensor_scaler   = None,
    )
    scaler = train_ds.fit_sensor_scaler()

    val_ds = MATWIMultimodalDataset(..., split="val", sensor_scaler=scaler)
"""

import warnings
import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as scipy_stats

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T

warnings.filterwarnings("ignore")


# ── Reuse constants from vision class ─────────────────────────────────────────

SPLIT_SETS = {
    "train":  [1, 2, 5, 7, 8, 10, 11],
    "val":    [3, 6, 12],
    "test":   [4, 9, 13],
    "unseen": [14, 15, 16, 17],
}

NORM_STATS = {
    "1-13": {
        "mean": [0.40560478076171874, 0.39624402888997395, 0.5083500896809896],
        "std":  [0.19804957525771552, 0.1920124143538162,  0.2173112347091954],
    },
    "1-17": {
        "mean": [0.42896144165039063, 0.4173878169759115,  0.5292330250651042],
        "std":  [0.20020134730149317, 0.19364826735868443, 0.21420715969306942],
    },
    "imagenet": {
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
    },
}

SET_CROPS = {
    1:  (2470, 1000, 3070, 1400),
    2:  (1150, 670,  1750, 1070),
    3:  (1150, 670,  1750, 1070),
    4:  (1150, 670,  1750, 1070),
    5:  (1150, 670,  1750, 1070),
    6:  (1150, 670,  1750, 1070),
    7:  (1150, 670,  1750, 1070),
    8:  (1150, 670,  1750, 1070),
    9:  (1150, 670,  1750, 1070),
    10: (1150, 670,  1750, 1070),
    11: (1150, 670,  1750, 1070),
    12: (1150, 670,  1750, 1070),
    13: (1150, 670,  1750, 1070),
    14: (1790, 2030, 2390, 2430),
    15: (1790, 2030, 2390, 2430),
    16: (1790, 2045, 2390, 2445),
    17: (1790, 2045, 2390, 2445),
}

ADHESION_TYPES = {"adhesion", "flank_wear+adhesion"}
WEAR_DIVISOR   = 1000.0

# Sensor channel names — order matches CSV columns (no header in sensor files)
SENSOR_CHANNELS = ["acc", "acoustic", "Fx", "Fy", "Fz"]

# Sampling rate: 1 sample every 0.6 ms → 1666.67 Hz
SENSOR_FS = 1000.0 / 0.6  # ≈ 1666.67 Hz

# Total number of sensor features extracted per sample
N_SENSOR_FEATURES = len(SENSOR_CHANNELS) * 8  # 5 channels × 8 features = 40


# ══════════════════════════════════════════════════════════════════════════════
# Feature set definitions
# ══════════════════════════════════════════════════════════════════════════════
#
# Extraction order per channel (8 features):
#   [mean, std, rms, p2p, kurtosis, dom_freq, low_energy, mid_energy]
#
# Channel layout (5 channels × 8 = 40 features):
#   acc:      indices  0– 7   (mean=0,  std=1,  rms=2,  p2p=3,  kurt=4,  dom=5,  low=6,  mid=7)
#   acoustic: indices  8–15   (mean=8,  std=9,  rms=10, p2p=11, kurt=12, dom=13, low=14, mid=15)
#   Fx:       indices 16–23   (mean=16, std=17, rms=18, p2p=19, kurt=20, dom=21, low=22, mid=23)
#   Fy:       indices 24–31   (mean=24, std=25, rms=26, p2p=27, kurt=28, dom=29, low=30, mid=31)
#   Fz:       indices 32–39   (mean=32, std=33, rms=34, p2p=35, kurt=36, dom=37, low=38, mid=39)
#

FEATURE_SETS = {
    # All 40 engineered features (time + frequency domain)
    "all40": list(range(40)),

    # 25 time-domain features only — no FFT/spectral engineering
    # First 5 features of each 8-feature block (mean, std, rms, p2p, kurtosis)
    "raw25": [
        0,  1,  2,  3,  4,     # acc:      mean, std, rms, p2p, kurtosis
        8,  9,  10, 11, 12,    # acoustic: mean, std, rms, p2p, kurtosis
        16, 17, 18, 19, 20,    # Fx:       mean, std, rms, p2p, kurtosis
        24, 25, 26, 27, 28,    # Fy:       mean, std, rms, p2p, kurtosis
        32, 33, 34, 35, 36,    # Fz:       mean, std, rms, p2p, kurtosis
    ],

    # Top 25 features by Ridge regression |coefficient| from sensor sanity check.
    # Drops all 5 dom_freq features (consistently weakest), all 5 kurtosis
    # features, and 5 additional weak features: acc_mean, acc_low_energy,
    # Fz_mean, Fz_p2p, acoustic_mean.
    # This is a mix of time-domain and frequency-domain features.
    "top25": [
        1,  2,  3,  7,             # acc:      std, rms, p2p, mid_energy
        9,  10, 11, 14, 15,        # acoustic: std, rms, p2p, low_energy, mid_energy
        16, 17, 18, 19, 22, 23,    # Fx:       mean, std, rms, p2p, low_energy, mid_energy
        24, 25, 26, 27, 30, 31,    # Fy:       mean, std, rms, p2p, low_energy, mid_energy
        33, 34, 38, 39,            # Fz:       std, rms, low_energy, mid_energy
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# Sensor feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def extract_sensor_features(sensor_path: Path) -> np.ndarray:
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


# ══════════════════════════════════════════════════════════════════════════════
# Sensor scaler
# ══════════════════════════════════════════════════════════════════════════════

class SensorScaler:
    """
    Simple zero-mean, unit-std standardiser for sensor feature vectors.
    Fit on training data only, applied to all splits.

    Always fits on all 40 features. Feature subset selection (raw25, top25)
    happens downstream — since standardisation is per-feature, selecting
    25 columns from a 40-feature scaler gives identical results to fitting
    a 25-feature scaler on those 25 columns.
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, features: np.ndarray) -> "SensorScaler":
        """features: (N, 40) array of raw sensor features from training set."""
        self.mean_ = features.mean(axis=0)
        self.std_  = features.std(axis=0)
        # Avoid division by zero for constant features
        self.mean_ = np.nan_to_num(self.mean_, nan=0.0)
        self.std_  = np.nan_to_num(self.std_, nan=1.0)
        self.std_[self.std_ < 1e-8] = 1.0
        self.fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("SensorScaler has not been fitted. Call fit() first.")
        return ((features - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_}, f)
        print(f"[SensorScaler] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SensorScaler":
        with open(path, "rb") as f:
            data = pickle.load(f)
        scaler        = cls()
        scaler.mean_  = data["mean"]
        scaler.std_   = data["std"]
        scaler.fitted = True
        print(f"[SensorScaler] Loaded from {path}")
        return scaler

    def __repr__(self) -> str:
        if not self.fitted:
            return "SensorScaler(not fitted)"
        return (f"SensorScaler(fitted, n_features={len(self.mean_)}, "
                f"mean_range=[{self.mean_.min():.3f}, {self.mean_.max():.3f}])")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset class
# ══════════════════════════════════════════════════════════════════════════════

class MATWIMultimodalDataset(Dataset):
    """
    PyTorch Dataset for MATWI — vision + sensor mode (Stage 2+).

    Only samples with BOTH image and sensor data available are included.
    This gives 647 train / 300 val / 247 test for sets 1-13.

    Parameters
    ----------
    feature_set : "all40" | "top25" | "raw25"
        Which sensor features to return.
        - "all40": full 40-feature vector (time + frequency domain)
        - "top25": top 25 features by Ridge importance
        - "raw25": 25 time-domain features only (no FFT engineering)
        Feature extraction always produces all 40; selection happens at
        return time. Scaler is fit on all 40.

    sensor_scaler : SensorScaler or None
        Pre-fitted SensorScaler to standardise sensor features.
        - For training split: pass None, then call fit_sensor_scaler()
        - For val/test splits: pass the scaler fitted on training data.

    All other parameters identical to MATWIVisionDataset.
    """

    def __init__(
        self,
        data_dir:           str | Path,
        labels_csv:         str | Path,
        sets_csv:           str | Path,
        split:              Literal["train", "val", "test", "unseen", "all"] = "train",
        set_range:          Literal["1-13", "1-17"] = "1-13",
        normalisation:      Literal["dataset", "imagenet"] = "dataset",
        augment:            bool = False,
        augment_strategy:   Literal["uniform", "oversample_adhesion"] = "uniform",
        wear_cap:           Optional[float] = 450.0,
        impute_zero_wear:   bool = False,
        image_size:         tuple[int, int] = (384, 384),
        feature_set:        Literal["all40", "top25", "raw25"] = "all40",
        sensor_scaler:      Optional[SensorScaler] = None,
    ):
        super().__init__()

        self.data_dir         = Path(data_dir)
        self.split            = split
        self.set_range        = set_range
        self.normalisation    = normalisation
        self.augment          = augment and (split == "train")
        self.augment_strategy = augment_strategy
        self.wear_cap         = wear_cap
        self.impute_zero_wear = impute_zero_wear
        self.image_size       = image_size
        self.feature_set      = feature_set
        self.sensor_scaler    = sensor_scaler

        # Validate feature_set
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"feature_set must be one of {list(FEATURE_SETS.keys())}, "
                f"got '{feature_set}'"
            )
        self._feature_indices = np.array(FEATURE_SETS[feature_set])

        if sensor_scaler is None and split not in ("train", "all"):
            warnings.warn(
                f"[MATWIMultimodalDataset] sensor_scaler is None for split='{split}'. "
                f"Sensor features will NOT be standardised. "
                f"Pass the scaler fitted on the training set.",
                UserWarning,
            )

        # Determine active sets
        if set_range == "1-13":
            self.active_sets = list(range(1, 14))
        elif set_range == "1-17":
            self.active_sets = list(range(1, 18))
        else:
            raise ValueError(f"set_range must be '1-13' or '1-17', got '{set_range}'")

        self.df = self._load_and_filter(labels_csv)

        # Build image transforms
        norm_stats         = self._get_norm_stats()
        self.transform     = self._build_transform(norm_stats, augment=self.augment)
        self.transform_val = self._build_transform(norm_stats, augment=False)

        # Sensor feature cache: stores full 40-dim standardised features
        # Feature subset selection happens in __getitem__
        self._sensor_cache: dict[int, np.ndarray] = {}

        print(f"[MATWIMultimodalDataset] split={split} | set_range={set_range} | "
              f"norm={normalisation} | augment={self.augment} | "
              f"n_samples={len(self.df)} | "
              f"wear_cap={wear_cap}µm | feature_set={feature_set} "
              f"({self.n_selected_features} features) | "
              f"scaler={'fitted' if sensor_scaler else 'None'}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_selected_features(self) -> int:
        """Number of sensor features returned by __getitem__."""
        return len(self._feature_indices)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_and_filter(self, labels_csv: str | Path) -> pd.DataFrame:
        df = pd.read_csv(labels_csv)
        df.columns = df.columns.str.strip()

        # Fix file paths
        for col in ["ImageFile", "SensorFile"]:
            if col in df.columns:
                df[col] = df[col].str.replace(r"^MATWI[\\/]", "", regex=True)
                df[col] = df[col].str.replace(r"^(Set\d+)[\\/]", r"\1/\1/", regex=True)

        # Normalise wear type labels
        df["type"] = (df["type"].astype(str)
                                .str.strip()
                                .str.lower()
                                .str.replace(" ", "_"))
        type_map = {
            "adhesive":                  "adhesion",
            "adhesive_wear":             "adhesion",
            "flank":                     "flank_wear",
            "flank_wear_and_adhesion":   "flank_wear+adhesion",
            "flank_wear_&_adhesion":     "flank_wear+adhesion",
            "flank_wear+adhesive_wear":  "flank_wear+adhesion",   # ← FIXED: was missing in v1
            "combination":               "flank_wear+adhesion",
        }
        df["type"] = df["type"].replace(type_map)

        # Numeric coercions
        df["wear"]     = pd.to_numeric(df["wear"],     errors="coerce")
        df["Set"]      = pd.to_numeric(df["Set"],      errors="coerce").astype("Int64")
        df["ImageID"]  = pd.to_numeric(df["ImageID"],  errors="coerce")
        df["SensorID"] = pd.to_numeric(df["SensorID"], errors="coerce")

        # ── Keep only rows with BOTH modalities ──
        has_image  = df["ImageName"].notna()  & df["ImageID"].notna()
        has_sensor = df["SensorName"].notna() & df["SensorID"].notna()
        df = df[has_image & has_sensor].copy()

        # Handle missing wear
        if self.impute_zero_wear:
            df["wear"] = df["wear"].fillna(0.0)
        else:
            df = df[df["wear"].notna()].copy()

        # Cap wear
        if self.wear_cap is not None:
            df["wear"] = df["wear"].clip(upper=self.wear_cap)

        # Normalise wear
        df["wear_norm"] = df["wear"] / WEAR_DIVISOR

        # Material column
        df["material"] = df["Set"].apply(
            lambda s: "CK45" if s <= 11 else "RVS 304"
        )

        # Apply split filter
        if self.split != "all":
            split_sets = SPLIT_SETS.get(self.split, [])
            if self.set_range == "1-17" and self.split == "train":
                split_sets = SPLIT_SETS["train"] + SPLIT_SETS["unseen"]
            df = df[df["Set"].isin(split_sets)].copy()
        else:
            df = df[df["Set"].isin(self.active_sets)].copy()

        df = df.reset_index(drop=True)
        return df

    # ── Normalisation & transforms ────────────────────────────────────────────

    def _get_norm_stats(self) -> dict:
        if self.normalisation == "imagenet":
            return NORM_STATS["imagenet"]
        return NORM_STATS[self.set_range]

    def _build_transform(self, norm_stats: dict, augment: bool) -> T.Compose:
        steps = [T.Resize(self.image_size)]
        if augment:
            steps += [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ]
        steps += [
            T.ToTensor(),
            T.Normalize(mean=norm_stats["mean"], std=norm_stats["std"]),
        ]
        return T.Compose(steps)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self, idx: int) -> torch.Tensor:
        row      = self.df.iloc[idx]
        set_num  = int(row["Set"])
        img_path = self.data_dir / str(row["ImageFile"])

        img  = Image.open(img_path).convert("RGB")
        crop = SET_CROPS.get(set_num)
        if crop is not None:
            img = img.crop(crop)

        transform = self.transform if self.augment else self.transform_val
        return transform(img)

    # ── Sensor feature loading ────────────────────────────────────────────────

    def _load_sensor_features(self, idx: int) -> np.ndarray:
        """
        Load sensor file, extract 40 features, apply standardisation if
        scaler is available. Results are cached in memory after first load.

        Returns the full 40-dim standardised vector. Feature subset selection
        happens in __getitem__.
        """
        if idx in self._sensor_cache:
            return self._sensor_cache[idx]

        row         = self.df.iloc[idx]
        sensor_path = self.data_dir / str(row["SensorFile"])

        raw_features = extract_sensor_features(sensor_path)

        if self.sensor_scaler is not None:
            features = self.sensor_scaler.transform(raw_features.reshape(1, -1))[0]
        else:
            features = raw_features

        self._sensor_cache[idx] = features
        return features

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image           = self._load_image(idx)
        full_features   = self._load_sensor_features(idx)         # (40,)
        sensor_features = torch.tensor(
            full_features[self._feature_indices], dtype=torch.float32
        )                                                          # (n_selected,)
        wear     = torch.tensor(row["wear_norm"], dtype=torch.float32)
        wear_raw = torch.tensor(row["wear"],      dtype=torch.float32)

        return {
            "image":           image,            # (3, H, W) float32
            "sensor_features": sensor_features,  # (n_selected,) float32
            "wear":            wear,              # scalar, normalised [0,1]
            "wear_raw":        wear_raw,          # scalar, µm
            "set":             int(row["Set"]),
            "image_id":        int(row["ImageID"]) if pd.notna(row["ImageID"]) else -1,
            "type":            str(row["type"]),
            "material":        str(row["material"]),
        }

    # ── Sensor scaler fitting ─────────────────────────────────────────────────

    def fit_sensor_scaler(
        self,
        save_path: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> SensorScaler:
        """
        Extract sensor features for all training samples, fit a SensorScaler,
        attach it to this dataset, and optionally save it to disk.

        Always fits on all 40 features regardless of feature_set setting.
        Feature subset selection happens at __getitem__ return time.

        Call this ONLY on the training split. Then pass the returned scaler
        to val/test dataset constructors via sensor_scaler=scaler.
        """
        if self.split not in ("train", "all"):
            warnings.warn(
                "fit_sensor_scaler() called on a non-training split. "
                "This will leak validation/test statistics into the scaler.",
                UserWarning,
            )

        if verbose:
            print(f"[fit_sensor_scaler] Extracting features for "
                  f"{len(self.df)} samples...")

        all_features = []
        for i in range(len(self.df)):
            row         = self.df.iloc[i]
            sensor_path = self.data_dir / str(row["SensorFile"])
            features    = extract_sensor_features(sensor_path)
            all_features.append(features)
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(self.df)}")

        all_features = np.stack(all_features, axis=0)   # (N, 40)

        scaler = SensorScaler()
        scaler.fit(all_features)

        if verbose:
            scaled = scaler.transform(all_features)
            print(f"[fit_sensor_scaler] Done. Feature stats after standardisation:")
            print(f"  mean range : [{scaled.mean(axis=0).min():.4f}, "
                  f"{scaled.mean(axis=0).max():.4f}]  (should be ~0)")
            print(f"  std  range : [{scaled.std(axis=0).min():.4f}, "
                  f"{scaled.std(axis=0).max():.4f}]  (should be ~1)")

        # Populate cache with standardised features (full 40-dim)
        for i, features in enumerate(all_features):
            self._sensor_cache[i] = scaler.transform(features.reshape(1, -1))[0]

        self.sensor_scaler = scaler

        if save_path is not None:
            scaler.save(save_path)

        return scaler

    # ── Utility methods ───────────────────────────────────────────────────────

    def get_sampler(self) -> WeightedRandomSampler:
        weights = []
        for _, row in self.df.iterrows():
            weights.append(3.0 if row["type"] in ADHESION_TYPES else 1.0)
        weights = torch.tensor(weights, dtype=torch.float32)
        return WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

    def get_wear_stats(self) -> dict:
        wear = self.df["wear"]
        return {
            "n":      len(wear),
            "mean":   float(wear.mean()),
            "std":    float(wear.std()),
            "min":    float(wear.min()),
            "max":    float(wear.max()),
            "median": float(wear.median()),
        }

    def get_type_counts(self) -> dict:
        return self.df["type"].value_counts().to_dict()

    def get_feature_names(self) -> list[str]:
        """Returns feature names for the selected feature_set."""
        all_names = []
        for ch in SENSOR_CHANNELS:
            for feat in ["mean", "std", "rms", "p2p", "kurtosis",
                         "dom_freq", "low_energy", "mid_energy"]:
                all_names.append(f"{ch}_{feat}")
        return [all_names[i] for i in self._feature_indices]

    def __repr__(self) -> str:
        stats = self.get_wear_stats()
        return (
            f"MATWIMultimodalDataset("
            f"split={self.split}, set_range={self.set_range}, "
            f"n={stats['n']}, "
            f"wear_mean={stats['mean']:.1f}µm, "
            f"wear_range=[{stats['min']:.0f}, {stats['max']:.0f}]µm, "
            f"norm={self.normalisation}, augment={self.augment}, "
            f"feature_set={self.feature_set} ({self.n_selected_features}), "
            f"scaler={'fitted' if self.sensor_scaler else 'None'})"
        )


# ── Convenience factory function ──────────────────────────────────────────────

def build_multimodal_dataloaders(
    data_dir:         str | Path,
    labels_csv:       str | Path,
    sets_csv:         str | Path,
    set_range:        Literal["1-13", "1-17"] = "1-13",
    normalisation:    Literal["dataset", "imagenet"] = "dataset",
    augment_train:    bool = False,
    augment_strategy: Literal["uniform", "oversample_adhesion"] = "uniform",
    wear_cap:         Optional[float] = 450.0,
    impute_zero_wear: bool = False,
    image_size:       tuple[int, int] = (384, 384),
    feature_set:      Literal["all40", "top25", "raw25"] = "all40",
    batch_size:       int = 32,
    num_workers:      int = 4,
    scaler_save_path: Optional[str | Path] = None,
) -> tuple[dict, SensorScaler]:
    """
    Builds train, val, and test DataLoaders in one call.
    Fits the sensor scaler on training data and applies it to val/test.

    Returns
    -------
    loaders : dict with keys "train", "val", "test" (and "unseen" if 1-17)
    scaler  : fitted SensorScaler
    """
    from torch.utils.data import DataLoader

    common = dict(
        data_dir         = data_dir,
        labels_csv       = labels_csv,
        sets_csv         = sets_csv,
        set_range        = set_range,
        normalisation    = normalisation,
        wear_cap         = wear_cap,
        impute_zero_wear = impute_zero_wear,
        image_size       = image_size,
        feature_set      = feature_set,
    )

    # Build training dataset and fit scaler
    train_ds = MATWIMultimodalDataset(
        **common,
        split            = "train",
        augment          = augment_train,
        augment_strategy = augment_strategy,
        sensor_scaler    = None,
    )
    print("\n[build_multimodal_dataloaders] Fitting sensor scaler on training set...")
    scaler = train_ds.fit_sensor_scaler(save_path=scaler_save_path)

    splits_to_build = ["val", "test"]
    if set_range == "1-17":
        splits_to_build.append("unseen")

    loaders = {}

    # Training loader
    if augment_train and augment_strategy == "oversample_adhesion":
        sampler = train_ds.get_sampler()
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loaders["train"] = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        sampler     = sampler,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )

    # Val / test / unseen loaders
    for split in splits_to_build:
        ds = MATWIMultimodalDataset(
            **common,
            split         = split,
            augment       = False,
            sensor_scaler = scaler,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

    return loaders, scaler


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    from torch.utils.data import DataLoader

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "matwi")

    labels_csv = Path(DATA_DIR) / "labels.csv"
    sets_csv   = Path(DATA_DIR) / "sets.csv"

    print("\n=== Sanity check: MATWIMultimodalDataset v2 ===\n")

    for fs in ["all40", "top25", "raw25"]:
        print(f"\n--- feature_set={fs} ---")
        ds = MATWIMultimodalDataset(
            data_dir       = DATA_DIR,
            labels_csv     = labels_csv,
            sets_csv       = sets_csv,
            split          = "train",
            set_range      = "1-13",
            normalisation  = "dataset",
            augment        = False,
            wear_cap       = 450.0,
            feature_set    = fs,
            sensor_scaler  = None,
        )
        scaler = ds.fit_sensor_scaler()
        print(repr(ds))
        print(f"  n_selected_features: {ds.n_selected_features}")
        print(f"  Feature names (first 5): {ds.get_feature_names()[:5]}")
        print(f"  Type counts: {ds.get_type_counts()}")

        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch  = next(iter(loader))
        print(f"  image shape         : {batch['image'].shape}")
        print(f"  sensor_features     : {batch['sensor_features'].shape}")
        print(f"  wear (norm) sample  : {batch['wear']}")