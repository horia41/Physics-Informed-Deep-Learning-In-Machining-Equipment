

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from constants.matwi_dataset_constants import (
    PINN_DEFAULT_TRAIN_SETS,
    WINDOW_HI,
    WINDOW_LO,
    PHYSICS_CAUTION_SETS,
    PHYSICS_EXCLUDED_SETS,    
    WEAR_CAP
)
from experiments.taylor.v3.fit import WEAR_FAILURE_UM, fit_set_trajectories, fit_taylor_constants
from experiments.taylor.fit import report_identifiability
from dataset.loader import load_labels_v3, load_sets_v2

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 Phase 1: fit Taylor constants offline.")
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--sets-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("taylor_constants.json"))
    ap.add_argument("--train-sets", type=int, nargs="+", default=list(PINN_DEFAULT_TRAIN_SETS))
    ap.add_argument("--failure-um", type=float, default=WEAR_FAILURE_UM)
    args = ap.parse_args()

    df = load_labels_v3(args.labels_csv)
    sets = load_sets_v2(args.sets_csv)

    report_identifiability(sets, args.train_sets)
    set_recs = fit_set_trajectories(df, sets, args.train_sets, args.failure_um)
    constants = fit_taylor_constants(set_recs, args.failure_um)

    # Add Taylor-predicted life/slope per set for the optional physics-slope mode
    for s, r in set_recs.items():
        c = constants.get(r["material"])
        # Need a finite C and a non-zero n; if n collapsed to ~0 (e.g. constant Vc
        # after dropping sets), T is undefined -> leave taylor_slope unset and the
        # loss falls back to the observed per-set slope for that set.
        if (c and c.get("C") and np.isfinite(c["C"])
                and c.get("n") is not None and abs(c["n"]) > 1e-6):
            # Extended multi-variable Taylor form T = (C / (Vc * fz^m))^(1/n)
            # Derived in log-space to protect against extreme exponential overflows
            log_T = (np.log(c["C"]) - np.log(r["Vc"]) - c.get("m", 0.0) * np.log(r["fz"])) / c["n"]
            T_taylor = np.exp(log_T)  # passes
            if np.isfinite(T_taylor) and T_taylor > 0:
                r["T_taylor"] = float(T_taylor)
                r["taylor_slope_um_per_pass"] = float(args.failure_um / T_taylor)

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