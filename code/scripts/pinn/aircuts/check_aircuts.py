#!/usr/bin/env python

import argparse
import importlib.util
import re
from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Make top-level project modules (dataset, constants, etc.) importable
# when this script is executed via file path.
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from constants.matwi_dataset_constants import (
    WAVELET_BASELINE_N,
    WAVELET_LEVEL,
    WAVELET_MIN_ACTIVE,
    WAVELET_NAME,
    WAVELET_TRIM,
    WAVELET_Z_THRESH,
    SENSOR_CHANNELS
)

from dataset.features import wavelet_cut_window, extract_sensor_features_v1

def _normalise_sensorfile(s: str) -> str:
    """Replicate DatasetClass._load_and_filter's path rewrite EXACTLY, so this
    check resolves files the same way training does:
        strip a leading 'MATWI/' , then 'Set7/...' -> 'Set7/Set7/...'
    (the on-disk layout has a doubled SetN/SetN/ directory)."""
    s = re.sub(r"^MATWI[\\/]", "", s)
    s = re.sub(r"^(Set\d+)[\\/]", r"\1/\1/", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dataset/matwi root")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--n", type=int, default=12, help="how many sensor files to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[check] WAVELET params: name={WAVELET_NAME} level={WAVELET_LEVEL} "
          f"z_thresh={WAVELET_Z_THRESH} baseline_n={WAVELET_BASELINE_N} "
          f"min_active={WAVELET_MIN_ACTIVE} trim={WAVELET_TRIM}")

    data_dir = Path(args.data_dir)
    labels = pd.read_csv(args.labels_csv)
    labels = labels[labels["SensorFile"].notna()]
    sample = labels.sample(n=min(args.n, len(labels)), random_state=args.seed)

    print(f"\n{'SensorFile':<40} {'n_total':>8} {'start':>7} {'end':>7} "
          f"{'kept':>7} {'kept%':>6} {'aircut%':>8} {'feats_changed':>13}")
    print("-" * 100)

    fracs, fellback = [], 0
    for _, row in sample.iterrows():
        sp = data_dir / _normalise_sensorfile(str(row["SensorFile"]))
        try:
            df = pd.read_csv(sp, header=None, usecols=[0, 1, 2, 3, 4],
                             dtype=np.float32, low_memory=False)
            df.columns = SENSOR_CHANNELS
        except Exception as e:
            print(f"{str(row['SensorFile']):<40}  LOAD FAILED: {e}")
            continue

        n = len(df)
        s, e = wavelet_cut_window(df)
        kept = e - s
        frac = kept / n if n else 0.0
        aircut_pct = 100.0 * (1 - frac)
        is_fullsignal = (s == 0 and e == n)      # detector fell back to full signal
        if is_fullsignal:
            fellback += 1

        # Do the gated vs ungated feature vectors actually differ?
        f_off = extract_sensor_features_v1(sp, gate_aircuts=False)
        f_on = extract_sensor_features_v1(sp, gate_aircuts=True)
        changed = not np.allclose(f_off, f_on, rtol=1e-4, atol=1e-6)

        fracs.append(frac)
        flag = "  <-- FULL (fallback)" if is_fullsignal else ""
        print(f"{str(row['SensorFile']):<40} {n:>8} {s:>7} {e:>7} {kept:>7} "
              f"{100*frac:>5.1f} {aircut_pct:>7.1f}  {str(changed):>13}{flag}")

    if fracs:
        fracs = np.array(fracs)
        print("-" * 100)
        print(f"[summary] files={len(fracs)}  "
              f"mean kept={100*fracs.mean():.1f}%  "
              f"mean aircut removed={100*(1-fracs.mean()):.1f}%  "
              f"min kept={100*fracs.min():.1f}%  max kept={100*fracs.max():.1f}%")
        print(f"[summary] full-signal fallbacks (no trim): {fellback}/{len(fracs)}")
        print("\nExpected per the README: ~28-46% of each recording is air cut, so a "
              "healthy 'aircut%' is roughly in that range and feats_changed=True. "
              "Many fallbacks / 0% aircut / feats_changed=False => detector not firing.")


if __name__ == "__main__":
    main()
