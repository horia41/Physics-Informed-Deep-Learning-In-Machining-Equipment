
import argparse
from pathlib import Path
from dataset.matwi.eda import (
    analyse_cutting_params,
    analyse_images,
    analyse_integrity,
    analyse_labels,
    analyse_modelling,
    analyse_sensors,
    analyse_trajectories
)

from dataset.matwi.dataset_loader import (
    resolve_data_dir,
    load_labels,
    load_sets
)

from utils.console_utils import section

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MATWI EDA script")
    p.add_argument("--data_dir",   type=str, default="./data/matwi",
                   help="Root directory of the MATWI dataset (contains labels.csv)")
    p.add_argument("--output_dir", type=str, default="./eda_output",
                   help="Directory to save all figures and outputs")
    p.add_argument("--skip_images",  action="store_true",
                   help="Skip image loading (faster if images not downloaded yet)")
    p.add_argument("--skip_sensors", action="store_true",
                   help="Skip sensor file loading (faster if sensors not downloaded yet)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MATWI — Exploratory Data Analysis")
    print(f"  Data dir   : {data_dir.resolve()}")
    print(f"  Output dir : {out_dir.resolve()}")
    print("=" * 70)

    # Load
    data_dir = resolve_data_dir(data_dir, "labels.csv")
    df = load_labels(data_dir)
    sets_df = load_sets(data_dir)

    # Run all analyses
    analyse_integrity(df, data_dir, out_dir)
    analyse_labels(df, out_dir)
    analyse_trajectories(df, out_dir)
    analyse_cutting_params(df, sets_df, out_dir)

    if not args.skip_sensors:
        analyse_sensors(df, data_dir, out_dir)
    else:
        print("\nSKIP Sensor analysis skipped (--skip_sensors flag set)")

    if not args.skip_images:
        analyse_images(df, data_dir, out_dir)
    else:
        print("\nSKIP Image analysis skipped (--skip_images flag set)")

    analyse_modelling(df, out_dir)

    section("DONE")
    print(f"  All figures saved to: {out_dir.resolve()}")
    print(f"  Figures produced:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()


# 1.  EDA: Sanity check, Visualize distributions, Find Correlations, Find Outliers.
# 2. Preprocessing Plan: Define steps based on #1.
# 3. Data Preprocessing: Clean, Encode, Scale, Engineer.
# 4. Final Validation: Ensure data is ready for Machine Learning.=