from torch.utils.data import DataLoader
from pathlib import Path
import os
from dataset.multimodal import MATWIMultimodalDatasetV1

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "matwi")

    labels_csv = Path(DATA_DIR) / "labels.csv"
    sets_csv   = Path(DATA_DIR) / "sets.csv"

    print("\n=== Sanity check: MATWIMultimodalDataset v2 ===\n")

    for fs in ["all40", "top25", "raw25"]:
        print(f"\n--- feature_set={fs} ---")
        ds = MATWIMultimodalDatasetV1(
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
