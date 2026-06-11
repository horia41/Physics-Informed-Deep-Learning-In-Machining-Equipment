"""
MATWI — Stage 3 Phase 1: Offline Taylor calibration
=====================================================
Fits the *simple* Taylor tool-life equation  V_c · T^n = C  per material and
derives a robust, early-stopping-tolerant expected-wear trajectory for every
training set. Run ONCE, offline, before training. Output is a frozen JSON the
PyTorch loss module loads.

WHY THE SIMPLE FORM (and not the volumetric V_c·T^n·f_z^m·A_p^u = C)
--------------------------------------------------------------------
The volumetric exponents m (feed) and u (depth) are NOT identifiable on the
MATWI training sets:
  • RVS 304 training sets {12,13}: f_z and A_p are CONSTANT (0.05, 0.5) → the
    log-design matrix is rank-deficient (rank 2 of 4). m, u cannot be fit.
  • CK45 training sets {2,5,7,8,10,11}: A_p varies in exactly one set (Set 10),
    V_c in exactly one (Set 2). Each exponent is pinned by a single point →
    perfect in-sample fit, zero generalisation.
This script prints the identifiability diagnostics so the decision is auditable.

WHY NOT THE 300 µm ISO FAILURE ANCHOR FOR TOOL LIFE
---------------------------------------------------
Across the CK45 training sets the final observed wear ranges from 90 µm (Set 7,
stopped at 30% of the threshold) to 750 µm (Set 11). Deriving tool life by
extrapolating each run linearly to 300 µm therefore means extrapolating 3×+ for
some sets and treating mid-run as "failure" for others — both corrupt the fit,
because the trajectories are non-linear (quantised steps + late-stage spikes).

Instead we estimate the *mid-life wear RATE* directly:
    k_s = d(wear)/d(ImageID)  [µm per pass], robustly (Theil–Sen) on the
    central 15–85% of each run.
The rate is an observed quantity — no extrapolation, no failure anchor needed to
get it. Taylor's n then governs how the rate scales with cutting speed:
    linear wear to a (notional) anchor W_f over life T:  k = W_f / T
    Taylor:  V·T^n = C  ⇒  log V = (log C − n·log W_f) + n·log k
    ⇒ regress log(V_s) on log(k_s): slope = n, intercept fixes C given W_f.
W_f only sets the absolute C scale (and the ceiling clamp); it does NOT affect n.

Usage
-----
    python fit_taylor.py \
        --labels-csv ./dataset/matwi/labels.csv \
        --sets-csv   ./dataset/matwi/sets.csv \
        --out        ./taylor_constants.json
"""

import argparse
from pathlib import Path
import numpy as np
import json

from constants.matwi_dataset_constants import (
    PINN_DEFAULT_TRAIN_SETS,
    WEAR_FAILURE_UM,
    WINDOW_HI,
    WINDOW_LO,
    WEAR_CAP,
    PHYSICS_CAUTION_SETS,
    PHYSICS_EXCLUDED_SETS
)

from dataset.loader import load_labels_v3, load_sets_v2
from experiments.taylor.fit import (
    fit_set_trajectories, 
    fit_taylor_constants, 
    report_identifiability
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
