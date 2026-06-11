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
from pathlib import Path
from typing import Literal, Optional
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T
from dataset.transform.sensor_scaler import SensorScaler
from dataset.features import extract_sensor_features_v1, extract_sensor_features_v2
from constants.matwi_dataset_constants import (
    FEATURE_SETS, 
    SPLIT_SETS, 
    NORM_STATS, 
    SENSOR_CHANNELS, 
    SET_CROPS, 
    WEAR_DIVISOR, 
    ADHESION_TYPES
)

class MATWIMultimodalDatasetV1(Dataset):
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
        gate_aircuts:       bool = False,
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
        self.gate_aircuts     = gate_aircuts

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
              f"gate_aircuts={self.gate_aircuts} | "
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

        raw_features = extract_sensor_features_v1(sensor_path,
                                               gate_aircuts=self.gate_aircuts)

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
            features    = extract_sensor_features_v1(sensor_path,
                                                  gate_aircuts=self.gate_aircuts)
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

class MATWIMultimodalDatasetV2(Dataset):
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

        raw_features = extract_sensor_features_v2(sensor_path)

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
            features    = extract_sensor_features_v2(sensor_path)
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


def build_multimodal_dataloaders_v1(
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
    gate_aircuts:     bool = False,
) -> tuple[dict, SensorScaler]:
    """
    Builds train, val, and test DataLoaders in one call.
    Fits the sensor scaler on training data and applies it to val/test.

    gate_aircuts : bool
        If True, sensor features are computed on the in-cut (tool-engaged)
        portion of each recording only. The scaler is fit on gated features.

    Returns
    -------
    loaders : dict with keys "train", "val", "test" (and "unseen" if 1-17)
    scaler  : fitted SensorScaler
    """

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
        gate_aircuts     = gate_aircuts,
    )

    # Build training dataset and fit scaler
    train_ds = MATWIMultimodalDatasetV1(
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
        ds = MATWIMultimodalDatasetV1(
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

def build_multimodal_dataloaders_v2(
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
    train_ds = MATWIMultimodalDatasetV2(
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
        ds = MATWIMultimodalDatasetV2(
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
