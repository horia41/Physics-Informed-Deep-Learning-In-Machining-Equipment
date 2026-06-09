

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# ── Configuration matching the rest of the pipeline ──────────────────────────
WEAR_CAP_UM        = 450.0     # matches DatasetClass_Vision wear_cap
WEAR_FAILURE_UM    = 90.0     # ISO VB ceiling — used ONLY as the C/clamp anchor
DEFAULT_TRAIN_SETS = (1, 2, 5, 7, 8, 10, 11, 12, 13)   # paper train (1-13 range)
WINDOW_LO, WINDOW_HI = 0.15, 0.85   # mid-life band fraction of max ImageID

PHYSICS_EXCLUDED_SETS = {1}                 # Set 1: cutting params unknown
PHYSICS_CAUTION_SETS  = {6, 17}             # Set 6 variable feed; Set 17 z=2

def material_of(set_num: int) -> str:
    return "CK45" if set_num <= 11 else "RVS 304"


# ── Robust line fit (Theil–Sen) ──────────────────────────────────────────────
def theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Median-of-pairwise-slopes line fit. Robust to the adhesion spike
    outliers that dominate sets like 2, 7, 11, 17. Returns (slope, intercept)."""
    n = len(x)
    if n < 2:
        return 0.0, float(y[0]) if n else 0.0
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0:
                slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return 0.0, float(np.median(y))
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


# ── Load & prepare ────────────────────────────────────────────────────────────
def load_labels(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df.columns = df.columns.str.strip()
    df["wear"]    = pd.to_numeric(df["wear"],    errors="coerce")
    df["Set"]     = pd.to_numeric(df["Set"],     errors="coerce").astype("Int64")
    df["ImageID"] = pd.to_numeric(df["ImageID"], errors="coerce")
    df = df.dropna(subset=["wear", "Set", "ImageID"]).copy()
    df["wear"] = df["wear"].clip(upper=WEAR_CAP_UM)   # match training cap
    df["Set"]  = df["Set"].astype(int)
    return df


def load_sets(sets_csv: Path) -> pd.DataFrame:
    s = pd.read_csv(sets_csv)
    
    # --- BULLETPROOF ADDITION: Clean all column names ---
    # Removes hidden trailing spaces, leading spaces, and formatting artifacts (\r, \n)
    s.columns = s.columns.str.strip()
    
    # first column is the unnamed "Set N" label
    s = s.rename(columns={s.columns[0]: "SetLabel"})
    s["Set"] = s["SetLabel"].str.extract(r"(\d+)").astype(int)
    for c in ["Vc", "fz", "Ap", "z"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    return s.set_index("Set")


# ── Identifiability report (auditable formula decision) ──────────────────────
def report_identifiability(sets: pd.DataFrame, train_sets: Sequence[int]) -> None:
    print("=" * 74)
    print("  IDENTIFIABILITY CHECK  —  simple vs volumetric Taylor form")
    print("=" * 74)
    for mat in ["CK45", "RVS 304"]:
        ms = [s for s in train_sets
              if s in sets.index and material_of(s) == mat and s not in PHYSICS_EXCLUDED_SETS]
        if not ms:
            continue
        sub = sets.loc[ms, ["Vc", "fz", "Ap", "z"]]
        print(f"\n  {mat}  training sets {ms}")
        for c in ["Vc", "fz", "Ap", "z"]:
            v = sorted(sub[c].dropna().unique().tolist())
            note = ""
            if c in ("fz", "Ap") and len(v) < 2:
                note = "  <-- CONSTANT: volumetric exponent unidentifiable"
            print(f"     distinct {c:3s}: {len(v)}  {v}{note}")
        X = np.column_stack([np.ones(len(sub)), np.log(sub["Vc"]),
                             np.log(sub["fz"]), np.log(sub["Ap"])])
        rank = int(np.linalg.matrix_rank(X))
        print(f"     volumetric log-design rank = {rank}/4"
              f"{'  -> RANK-DEFICIENT, reject volumetric' if rank < 4 else '  (full rank but exponents pinned by single sets)'}")
    print("\n  DECISION: use the simple form  V_c · T^n = C  (per material).\n")


# ── Per-set robust trajectory fit ─────────────────────────────────────────────
def fit_set_trajectories(df: pd.DataFrame, sets: pd.DataFrame,
                         train_sets: Sequence[int], verbose: bool = True) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    if verbose:
        print("  PER-SET MID-LIFE WEAR-RATE FITS  (Theil–Sen, 15–85% window)")
        print("  " + "-" * 72)
        print(f"  {'Set':>4} {'mat':>7} {'Vc':>5} {'maxID':>6} {'maxWear':>8} "
              f"{'%of300':>7} {'slope':>8} {'icpt':>7} {'n_win':>6}")
    for s in sorted(train_sets):
        if s in PHYSICS_EXCLUDED_SETS or s not in sets.index:
            continue
        g = df[df["Set"] == s].sort_values("ImageID")
        if len(g) < 4:
            continue
        t = g["ImageID"].to_numpy(float)
        w = g["wear"].to_numpy(float)
        t_max = float(t.max())
        lo, hi = WINDOW_LO * t_max, WINDOW_HI * t_max
        m = (t >= lo) & (t <= hi)
        tw, ww = (t[m], w[m]) if m.sum() >= 3 else (t, w)
        slope, icpt = theil_sen(tw, ww)
        slope = max(slope, 0.0)   # wear does not decrease in expectation
        max_wear = float(w.max())
        rec = {
            "set": s, "material": material_of(s),
            "Vc": float(sets.loc[s, "Vc"]),
            "fz": float(sets.loc[s, "fz"]),
            "max_image_id": t_max,
            "max_wear_um": max_wear,
            "frac_of_failure": max_wear / WEAR_FAILURE_UM,
            "slope_um_per_pass": slope,
            "intercept_um": icpt,
            "window_lo_id": lo, "window_hi_id": hi,
            "n_window": int(m.sum()),
            "caution": s in PHYSICS_CAUTION_SETS,
        }
        out[s] = rec
        if verbose:
            print(f"  {s:>4} {material_of(s):>7} {rec['Vc']:>5.0f} {t_max:>6.0f} "
                  f"{max_wear:>8.0f} {rec['frac_of_failure']:>6.0%} "
                  f"{slope:>8.3f} {icpt:>7.1f} {rec['n_window']:>6}")
    return out


# ── Taylor n, C per material from the rates ──────────────────────────────────
def fit_taylor_constants(set_recs, failure_um, verbose=True):
    """
    Fits the extended logarithmic Taylor equation:
       ln(Vc) = ln(C) - n*ln(T) - m*ln(fz)
    strictly using the isolated data from the training sets.
    """
    from sklearn.linear_model import LinearRegression

    constants = {}
    
    # Group records by material type
    material_groups = {"CK45": [], "RVS 304": []}
    for s, r in set_recs.items():
        # Safeguard: Set 1 has unknown values ('?') in sets.csv, skip it to protect regression math
        if str(s) == "1" or s == 1:
            continue
        mat = r["material"]
        if mat in material_groups:
            material_groups[mat].append(r)

    for mat, recs in material_groups.items():
        if len(recs) < 2:
            if verbose:
                print(f"  [{mat}] Insufficient data points to fit exponents. Using standard fallback baseline.")
            constants[mat] = {"n": 0.25, "m": 0.50, "C": 300.0}
            continue
            
        # Build the logarithmic regression design matrix
        # Independent variables: X1 = ln(T), X2 = ln(fz)
        # Dependent target variable: Y = ln(Vc)
        X = []
        Y = []
        for r in recs:
            # T0 is the estimated passes to hit 90 micrometers from the steady-state window
            T0 = failure_um / r["slope_um_per_pass"]
            X.append([np.log(T0), np.log(r["fz"])])
            Y.append(np.log(r["Vc"]))
            
        # Run Ordinary Least Squares (OLS) Linear Regression
        reg = LinearRegression().fit(np.array(X), np.array(Y))
        
        # Extract the physical constants from the linear coefficients:
        # beta_1 = -n  =>  n = -beta_1
        # beta_2 = -m  =>  m = -beta_2
        # intercept = ln(C) => C = exp(intercept)
        n_fit = -reg.coef_[0]
        m_fit = -reg.coef_[1]
        C_fit = np.exp(reg.intercept_)
        
        constants[mat] = {
            "n": float(round(n_fit, 4)),
            "m": float(round(m_fit, 4)),
            "C": float(round(C_fit, 4)),
            "n_sets": len(recs),
            "source": "localized_training_fit_90um"
        }
        
        if verbose:
            print(f"  [{mat}] Custom Fit (90um Finish Line) -> n={n_fit:.4f}, m={m_fit:.4f}, C={C_fit:.2f}")

    return constants

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 Phase 1: fit Taylor constants offline.")
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--sets-csv",   type=Path, required=True)
    ap.add_argument("--out",        type=Path, default=Path("taylor_constants.json"))
    ap.add_argument("--train-sets", type=int, nargs="+", default=list(DEFAULT_TRAIN_SETS))
    ap.add_argument("--failure-um", type=float, default=WEAR_FAILURE_UM)
    args = ap.parse_args()

    df   = load_labels(args.labels_csv)
    sets = load_sets(args.sets_csv)

    report_identifiability(sets, args.train_sets)
    set_recs  = fit_set_trajectories(df, sets, args.train_sets)
    constants = fit_taylor_constants(set_recs, args.failure_um)

    # Add Taylor-predicted life/slope per set for the optional physics-slope mode
    for s, r in set_recs.items():
        c = constants.get(r["material"])
        if c and c["C"] and np.isfinite(c["C"]):
            # --- MODIFIED: Extended multi-variable Taylor form T = (C / (Vc * fz^m))^(1/n) ---
            # Derived in log-space to protect against extreme exponential overflows
            log_T = (np.log(c["C"]) - np.log(r["Vc"]) - c["m"] * np.log(r["fz"])) / c["n"]
            T_taylor = np.exp(log_T)      # passes
            
            r["T_taylor"] = float(T_taylor)
            r["taylor_slope_um_per_pass"] = float(args.failure_um / T_taylor)

    payload = {
        "meta": {
            "form": "simple  Vc * T^n = C  (per material)",
            "failure_um": args.failure_um,
            "wear_cap_um": WEAR_CAP_UM,
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
