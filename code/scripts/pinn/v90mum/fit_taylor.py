import argparse
from pathlib import Path
import json
import numpy as np

from experiments.taylor.pinn90mum.fit import WEAR_FAILURE_UM, fit_set_trajectories, fit_taylor_constants
from experiments.taylor.fit import report_identifiability
from dataset.loader import load_labels_v3, load_sets_v2
from constants.matwi_dataset_constants import (
    WINDOW_HI,
    WINDOW_LO,
    PHYSICS_CAUTION_SETS,
    PHYSICS_EXCLUDED_SETS,
    WEAR_CAP,
    PINN_DEFAULT_TRAIN_SETS
)

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 Phase 1: fit Taylor constants offline.")
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--sets-csv",   type=Path, required=True)
    ap.add_argument("--out",        type=Path, default=Path("taylor_constants.json"))
    ap.add_argument("--train-sets", type=int, nargs="+", default=list(PINN_DEFAULT_TRAIN_SETS))
    ap.add_argument("--failure-um", type=float, default=WEAR_FAILURE_UM)
    args = ap.parse_args()

    df   = load_labels_v3(args.labels_csv)
    sets = load_sets_v2(args.sets_csv)

    report_identifiability(sets, args.train_sets)
    set_recs  = fit_set_trajectories(df, sets, args.train_sets)
    constants = fit_taylor_constants(set_recs, args.failure_um)

    # Add Taylor-predicted life / slope / re-anchored intercept per set for
    # the optional physics-slope mode. The intercept is re-fit with the
    # Taylor slope FIXED, so the line stays a valid trajectory through the
    # mid-life window (previously it kept the observed-slope intercept and
    # could drift far from the data when the slopes differed).
    for s, r in set_recs.items():
        c = constants.get(r["material"])
        if c and c["C"] and np.isfinite(c["C"]):
            T_taylor     = (c["C"] / r["Vc"]) ** (1.0 / c["n"])     # passes
            taylor_slope = float(args.failure_um / T_taylor)

            g = df[df["Set"] == s].sort_values("ImageID")
            t = g["ImageID"].to_numpy(float)
            w = g["wear"].to_numpy(float)
            t_max = float(t.max())
            lo, hi = WINDOW_LO * t_max, WINDOW_HI * t_max
            m = (t >= lo) & (t <= hi)
            tw, ww = (t[m], w[m]) if m.sum() >= 3 else (t, w)
            # Theil–Sen-style intercept with slope fixed: median residual.
            taylor_intercept = float(np.median(ww - taylor_slope * tw))

            r["T_taylor"]                  = float(T_taylor)
            r["taylor_slope_um_per_pass"]  = taylor_slope
            r["taylor_intercept_um"]       = taylor_intercept
        else:
            r["T_taylor"]                  = None
            r["taylor_slope_um_per_pass"]  = None
            r["taylor_intercept_um"]       = None

    payload = {
        "meta": {
            "form": "simple  Vc * T^n = C  (per material)",
            "failure_um": args.failure_um,
            "wear_cap_um": WEAR_CAP,
            "window": [WINDOW_LO, WINDOW_HI],
            "train_sets": list(args.train_sets),
            "excluded_sets": sorted(PHYSICS_EXCLUDED_SETS),
            "caution_sets": sorted(PHYSICS_CAUTION_SETS),
        },
        "constants": constants,
        "sets": {str(s): r for s, r in set_recs.items()},
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\n  [saved] {args.out}")
    print("  Next: pass this file to TaylorPhysicsLoss(constants_path=...).")


if __name__ == "__main__":
    main()