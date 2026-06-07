import os
from pathlib import Path
from torch.utils.data import DataLoader
from dataset.matwi.dataset_vision import MATWIVisionDataset
from structuredCode.constants import MATWI_DATA_PATH

if __name__ == "__main__":
    labels_csv = Path(MATWI_DATA_PATH) / "labels.csv"
    sets_csv   = Path(MATWI_DATA_PATH) / "sets.csv"

    print("\n=== Sanity check: MATWIVisionDataset ===\n")

    for set_range in ["1-13", "1-17"]:
        for split in ["train", "val", "test"]:
            ds = MATWIVisionDataset(
                data_dir       = MATWI_DATA_PATH,
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
            loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
            batch  = next(iter(loader))
            print(f"  Image tensor shape : {batch['image'].shape}")
            print(f"  Wear (norm) sample : {batch['wear']}")
            print(f"  Wear (µm) sample   : {batch['wear_raw']}")
            print()