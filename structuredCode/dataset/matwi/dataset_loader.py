import sys
from pathlib import Path
import pandas as pd
from typing import Literal, Optional
from torch.utils.data import DataLoader
from dataset_vision import MATWIVisionDataset

from constants import TYPE_MAP, SPLIT_SETS, WEAR_CAP

def get_split_label(set_num: int) -> str:
    for label, sets in SPLIT_SETS.items():
        if set_num in sets:
            return label
    return "unseen"

def resolve_data_dir(data_dir: Path, filename: str = "labels.csv") -> Path:
    candidates = [
        data_dir,
        data_dir / "matwi",
        data_dir.parent,
        data_dir.parent / "matwi",
        Path.cwd() / "matwi",
        Path.cwd() / "data" / "matwi",
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / filename).exists():
            if candidate != data_dir:
                print(f" '{filename}' not found at '{data_dir}', "
                      f"using '{candidate}' instead.")
            return candidate
    sys.exit(
        f"  Could not find '{filename}' anywhere near '{data_dir}'.\n"
        f"  Tried: {[str(c) for c in candidates]}\n"
        f"  Make sure you have run download_data.py first."
    )

def load_labels(data_dir: Path) -> pd.DataFrame:
    data_dir = resolve_data_dir(data_dir, "labels.csv")
    path = data_dir / "labels.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    for col in ["ImageFile", "SensorFile"]:
        if col in df.columns:
            df[col] = df[col].str.replace(r"^MATWI[\\/]", "", regex=True)
            df[col] = df[col].str.replace(r"^(Set\d+)[\\/]", r"\1/\1/", regex=True)

    # prints for debugging
    # print(df["SensorFile"].dropna().iloc[0])
    # print(df["ImageFile"].dropna().iloc[0])

    # Normalise wear-type labels to consistent lowercase+underscore form
    df["type"] = (df["type"].astype(str)
                            .str.strip()
                            .str.lower()
                            .str.replace(" ", "_")
                            .str.replace("+", "+", regex=False))

    # Map any variant spellings
    df["type"] = df["type"].replace(TYPE_MAP)

    # Ensure numeric
    df["wear"]     = pd.to_numeric(df["wear"],     errors="coerce")
    df["Set"]      = pd.to_numeric(df["Set"],      errors="coerce").astype("Int64")
    df["ImageID"]  = pd.to_numeric(df["ImageID"],  errors="coerce")
    df["SensorID"] = pd.to_numeric(df["SensorID"], errors="coerce")

    # Datetimes
    for col in ["ImageDateTime", "SensorDateTime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Derived helpers
    df["has_image"]  = df["ImageName"].notna()  & df["ImageID"].notna()
    df["has_sensor"] = df["SensorName"].notna() & df["SensorID"].notna()
    df["both"]       = df["has_image"] & df["has_sensor"]
    df["material"]   = df["Set"].apply(
        lambda s: "CK45" if s <= 11 else "RVS 304"
    )
    df["split"] = df["Set"].apply(get_split_label)

    return df

def load_labels_v2(
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

def load_sets(data_dir: Path) -> pd.DataFrame:
    data_dir = resolve_data_dir(data_dir, "sets.csv")
    path = data_dir / "sets.csv"
    if not path.exists():
        print("sets.csv not found — cutting parameter analysis skipped.")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.str.strip()
    # Coerce numeric, keeping '?' as NaN
    for col in ["Vc", "n", "fz", "Vf", "Ae", "Ap", "z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def load_one_sensor_file(path: Path) -> pd.DataFrame | None:
    """
    Load a single sensor CSV. Per the README:
    columns in order: Acc, Acoustic, Fx, Fy, Fz, datetime
    No header row in the file.
    """
    col_names = ["acc", "acoustic", "Fx", "Fy", "Fz", "datetime"]
    try:
        sensor_df = pd.read_csv(path, header=None, names=col_names,
                                parse_dates=["datetime"], low_memory=False)
        return sensor_df
    except Exception as e:
        print(f"   Could not load {path.name}: {e}")
        return None

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


