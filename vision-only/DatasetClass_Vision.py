"""
MATWI — Vision-Only Dataset Class
===================================
Handles image loading, cropping, normalisation, and augmentation
for Stage 1 (EfficientNetV2 vision baseline).

Supports:
  - Sets 1–13 (paper split) or Sets 1–17 (extended)
  - Dataset-specific normalisation stats (precomputed per set range)
  - ImageNet normalisation (for pretrained weight compatibility)
  - Optional augmentation (safe augmentations only, physically motivated)
  - Optional adhesion oversampling
  - wear_cap: cap wear values at 450 µm (paper's stated upper bound)
  - impute_zero_wear: treat missing wear as 0.0 (new tool) or drop
  - Wear normalised to [0, 1] by dividing by 1000 (matches paper)

Usage:
    from DatasetClass_Vision import MATWIVisionDataset

    train_ds = MATWIVisionDataset(
        data_dir       = "./data/matwi",
        labels_csv     = "./data/matwi/labels.csv",
        sets_csv       = "./data/matwi/sets.csv",
        split          = "train",
        set_range      = "1-13",
        normalisation  = "dataset",
        augment        = False,
        wear_cap       = 450,
        impute_zero_wear = False,
    )
"""

import os
import warnings
from pathlib import Path
from typing import Literal, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Silence only the noisy third-party deprecation chatter (pandas/torchvision),
# NOT this module's own UserWarnings (e.g. missing/leaky sensor scaler) which
# are meant to be seen.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Official paper train/val/test split ───────────────────────────────────────
SPLIT_SETS = {
    "train":  [1, 2, 5, 7, 8, 10, 11],
    "val":    [3, 6, 12],
    "test":   [4, 9, 13],
    "unseen": [14, 15, 16, 17],
}

# ── Precomputed normalisation stats ──────────────────────────────────────────
# Computed over cropped images from the respective set ranges.
# Format: RGB mean and std in [0, 1] range.
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

# ── Crop coordinates per set (left, top, right, bottom) ──────────────────────
# From Table 2 of the MATWI paper. Applied to full 5496×3672 images.
# Produces 600×400 pixel crops as stated in the paper.
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

# Wear types containing adhesion — used for oversampling
ADHESION_TYPES = {"adhesion", "flank_wear+adhesion"}

# Wear normalisation divisor (matches paper: wear in mm = wear_µm / 1000)
WEAR_DIVISOR = 1000.0


class MATWIVisionDataset(Dataset):
    """
    PyTorch Dataset for the MATWI tool wear dataset — vision-only mode.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing SetX/SetX/images/ folders.
    labels_csv : str or Path
        Path to labels.csv.
    sets_csv : str or Path
        Path to sets.csv (used for crop coordinates).
    split : "train" | "val" | "test" | "unseen" | "all"
        Which split to load. "all" returns the full dataset regardless of set.
    set_range : "1-13" | "1-17"
        Which set range to use. Controls which normalisation stats are applied
        when normalisation="dataset", and which sets are considered for splits.
        "1-13" replicates the paper's experimental setup.
        "1-17" includes the unseen RVS304 sets.
    normalisation : "dataset" | "imagenet"
        "dataset" uses precomputed stats for the selected set_range.
        "imagenet" uses standard ImageNet stats (for pretrained weight init).
    augment : bool
        Whether to apply data augmentation. Only applied to training split.
        Physically motivated augmentations only (see notes below).
    augment_strategy : "uniform" | "oversample_adhesion"
        "uniform"            — augment all training samples equally.
        "oversample_adhesion"— adhesion and flank_wear+adhesion samples are
                               repeated to balance with flank_wear. Use with
                               WeightedRandomSampler (see get_sampler()).
    wear_cap : float or None
        Cap wear values at this value in µm before normalisation.
        Default 450 matches the paper's stated valid wear range.
        Set to None to disable.
    impute_zero_wear : bool
        If True, rows with missing wear labels are imputed with wear=0
        (physically: new tool reference images).
        If False (default), rows with missing wear are dropped.
    image_size : tuple of int
        (height, width) to resize cropped images to before feeding the model.
        Default (224, 224) matches EfficientNetV2 input size.
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
        require_sensors:    bool = False,
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
        # When True, restrict to samples that ALSO have sensor data (SensorName
        # & SensorID present) — reproduces the 647-sample multimodal subset used
        # by the vision+sensor fusion experiments, so a vision-only run on this
        # population is directly comparable to fusion. False = full 664 set.
        self.require_sensors  = require_sensors

        # Determine which sets are active for this set_range
        if set_range == "1-13":
            self.active_sets = list(range(1, 14))
        elif set_range == "1-17":
            self.active_sets = list(range(1, 18))
        else:
            raise ValueError(f"set_range must be '1-13' or '1-17', got '{set_range}'")

        # Load and filter the dataframe
        self.df = self._load_and_filter(labels_csv)

        # Build transforms
        norm_stats         = self._get_norm_stats()
        self.transform     = self._build_transform(norm_stats, augment=self.augment)
        self.transform_val = self._build_transform(norm_stats, augment=False)

        print(f"[MATWIVisionDataset] split={split} | set_range={set_range} | "
              f"norm={normalisation} | augment={self.augment} | "
              f"n_samples={len(self.df)} | require_sensors={self.require_sensors} | "
              f"wear_cap={wear_cap}µm | impute_zero={impute_zero_wear}")

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_and_filter(self, labels_csv: str | Path) -> pd.DataFrame:
        df = pd.read_csv(labels_csv)
        df.columns = df.columns.str.strip()

        # Fix file paths: MATWI/SetX/... → SetX/SetX/...
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
            "adhesive":                 "adhesion",
            "adhesive_wear":            "adhesion",
            "flank":                    "flank_wear",
            "flank_wear_and_adhesion":  "flank_wear+adhesion",
            "flank_wear_&_adhesion":    "flank_wear+adhesion",
            "combination":              "flank_wear+adhesion",
        }
        df["type"] = df["type"].replace(type_map)

        # Numeric coercions
        df["wear"]    = pd.to_numeric(df["wear"],    errors="coerce")
        df["Set"]     = pd.to_numeric(df["Set"],     errors="coerce").astype("Int64")
        df["ImageID"] = pd.to_numeric(df["ImageID"], errors="coerce")

        # Require an image
        df = df[df["ImageName"].notna() & df["ImageID"].notna()].copy()

        # Optionally require sensor data too → the 647-sample multimodal subset.
        # Uses the EXACT filter from DatasetClass_VisionSensors._load_and_filter
        # so the populations match the fusion experiments sample-for-sample.
        if self.require_sensors:
            if {"SensorName", "SensorID"}.issubset(df.columns):
                df = df[df["SensorName"].notna() & df["SensorID"].notna()].copy()
            else:
                raise KeyError(
                    "require_sensors=True but labels.csv has no SensorName/"
                    "SensorID columns — cannot build the 647 multimodal subset."
                )

        # Handle missing wear
        if self.impute_zero_wear:
            df["wear"] = df["wear"].fillna(0.0)
        else:
            df = df[df["wear"].notna()].copy()

        # Cap wear
        if self.wear_cap is not None:
            df["wear"] = df["wear"].clip(upper=self.wear_cap)

        # Normalise wear to [0, 1]
        df["wear_norm"] = df["wear"] / WEAR_DIVISOR

        # Material column
        df["material"] = df["Set"].apply(
            lambda s: "CK45" if s <= 11 else "RVS 304"
        )

        # Apply split filter.
        # NOTE: for a named split the split's set list fully determines the
        # sets, so we must NOT pre-restrict to active_sets first — doing so
        # silently emptied the "unseen" split (sets 14-17) whenever
        # set_range="1-13", making the held-out RVS 304 generalisation test
        # impossible. active_sets is only used for split="all".
        if self.split != "all":
            split_sets = SPLIT_SETS.get(self.split, [])
            # For "1-17" range, train split includes sets 14–17 as additional training data
            if self.set_range == "1-17" and self.split == "train":
                split_sets = SPLIT_SETS["train"] + SPLIT_SETS["unseen"]
            df = df[df["Set"].isin(split_sets)].copy()
        else:
            # "all" → every set in the active range
            df = df[df["Set"].isin(self.active_sets)].copy()

        df = df.reset_index(drop=True)
        return df

    # ── Normalisation stats ───────────────────────────────────────────────────

    def _get_norm_stats(self) -> dict:
        if self.normalisation == "imagenet":
            return NORM_STATS["imagenet"]
        # dataset-specific: use stats for the active set range
        return NORM_STATS[self.set_range]

    # ── Transforms ───────────────────────────────────────────────────────────

    def _build_transform(self, norm_stats: dict, augment: bool) -> T.Compose:
        """
        Physically motivated augmentations only:
          - Horizontal flip: cutting edge is symmetric, safe.
          - Small rotation (±5°): minor camera alignment variation.
          - Brightness/contrast jitter: lighting variation across sets.
          - Gaussian blur (mild): slight focus variation.

        NOT applied:
          - Vertical flip: wear is always at the bottom edge — invalid if flipped.
          - Hue/saturation jitter: colour of adhesion deposits is diagnostically meaningful.
          - Large crops / aggressive zoom: wear zone is small, could be cropped out.
          - Elastic / perspective transforms: wear geometry is the signal.
        """
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

        img = Image.open(img_path).convert("RGB")

        # Apply crop from paper's Table 2
        crop = SET_CROPS.get(set_num)
        if crop is not None:
            img = img.crop(crop)   # (left, top, right, bottom)

        # Apply transform
        transform = self.transform if self.augment else self.transform_val
        return transform(img)

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image    = self._load_image(idx)
        wear     = torch.tensor(row["wear_norm"], dtype=torch.float32)
        wear_raw = torch.tensor(row["wear"],      dtype=torch.float32)  # µm, for logging

        return {
            "image":    image,        # (3, H, W) float32 tensor
            "wear":     wear,         # scalar float32, normalised [0, 1]
            "wear_raw": wear_raw,     # scalar float32, in µm (for MAE reporting)
            "set":      int(row["Set"]),
            "image_id": int(row["ImageID"]) if pd.notna(row["ImageID"]) else -1,
            "type":     str(row["type"]),
            "material": str(row["material"]),
        }

    # ── Utility methods ───────────────────────────────────────────────────────

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Returns a WeightedRandomSampler that oversamples adhesion and
        flank_wear+adhesion rows to compensate for class imbalance.
        Only meaningful when augment_strategy="oversample_adhesion".

        Use this sampler with DataLoader instead of shuffle=True:
            DataLoader(dataset, sampler=dataset.get_sampler(), ...)
        """
        weights = []
        for _, row in self.df.iterrows():
            if row["type"] in ADHESION_TYPES:
                weights.append(3.0)   # oversample adhesion 3×
            else:
                weights.append(1.0)
        weights = torch.tensor(weights, dtype=torch.float32)
        return WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

    def get_wear_stats(self) -> dict:
        """Returns wear statistics (µm) for this split — useful for logging."""
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
        """Returns wear type counts for this split."""
        return self.df["type"].value_counts().to_dict()

    def __repr__(self) -> str:
        stats = self.get_wear_stats()
        return (
            f"MATWIVisionDataset("
            f"split={self.split}, set_range={self.set_range}, "
            f"n={stats['n']}, "
            f"wear_mean={stats['mean']:.1f}µm, "
            f"wear_range=[{stats['min']:.0f}, {stats['max']:.0f}]µm, "
            f"norm={self.normalisation}, augment={self.augment})"
        )


# ── Convenience factory function ──────────────────────────────────────────────

def build_vision_dataloaders(
    data_dir:         str | Path,
    labels_csv:       str | Path,
    sets_csv:         str | Path,
    set_range:        Literal["1-13", "1-17"] = "1-13",
    normalisation:    Literal["dataset", "imagenet"] = "dataset",
    augment_train:    bool = False,
    augment_strategy: Literal["uniform", "oversample_adhesion"] = "uniform",
    wear_cap:         Optional[float] = 450.0,
    impute_zero_wear: bool = False,
    image_size:       tuple[int, int] = (384, 384),   # matches the class default
    batch_size:       int = 32,
    num_workers:      int = 4,
) -> dict:
    """
    Builds train, val, and test DataLoaders in one call.

    Returns a dict with keys "train", "val", "test" (and "unseen" if set_range="1-17").

    Example:
        loaders = build_vision_dataloaders(
            data_dir    = "./data/matwi",
            labels_csv  = "./data/matwi/labels.csv",
            sets_csv    = "./data/matwi/sets.csv",
            set_range   = "1-13",
            augment_train = False,
        )
        for batch in loaders["train"]:
            images = batch["image"]   # (B, 3, 224, 224)
            wear   = batch["wear"]    # (B,) normalised
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
    )

    splits = ["train", "val", "test"]
    if set_range == "1-17":
        splits.append("unseen")

    loaders = {}
    for split in splits:
        is_train = (split == "train")
        ds = MATWIVisionDataset(
            **common,
            split            = split,
            augment          = augment_train if is_train else False,
            augment_strategy = augment_strategy,
        )

        if is_train and augment_strategy == "oversample_adhesion":
            sampler    = ds.get_sampler()
            shuffle    = False
        else:
            sampler    = None
            shuffle    = is_train

        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            sampler     = sampler,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = is_train,
        )

    return loaders


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "matwi")

    labels_csv = Path(DATA_DIR) / "labels.csv"
    sets_csv   = Path(DATA_DIR) / "sets.csv"

    print("\n=== Sanity check: MATWIVisionDataset ===\n")

    for set_range in ["1-13", "1-17"]:
        for split in ["train", "val", "test"]:
            ds = MATWIVisionDataset(
                data_dir       = DATA_DIR,
                labels_csv     = labels_csv,
                sets_csv       = sets_csv,
                split          = split,
                set_range      = set_range,
                normalisation  = "dataset",
                augment        = False,
                wear_cap       = 450.0,
                impute_zero_wear = False,
            )
            print(repr(ds))
            print(f"  Type counts: {ds.get_type_counts()}")

            # Load one batch
            from torch.utils.data import DataLoader
            loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
            batch  = next(iter(loader))
            print(f"  Image tensor shape : {batch['image'].shape}")
            print(f"  Wear (norm) sample : {batch['wear']}")
            print(f"  Wear (µm) sample   : {batch['wear_raw']}")
            print()