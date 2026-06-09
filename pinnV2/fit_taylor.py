

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# ── Configuration matching the rest of the pipeline ──────────────────────────
WEAR_CAP_UM        = 450.0     # matches DatasetClass_Vision wear_cap
WEAR_FAILURE_UM    = 300.0     # ISO VB ceiling — used ONLY as the C/clamp anchor
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
        slope_raw, icpt = theil_sen(tw, ww)
        slope_clamped = slope_raw < 0.0
        slope = max(slope_raw, 0.0)   # wear does not decrease in expectation
        max_wear = float(w.max())
        rec = {
            "set": s, "material": material_of(s),
            "Vc": float(sets.loc[s, "Vc"]),
            "max_image_id": t_max,
            "max_wear_um": max_wear,
            "frac_of_failure": max_wear / WEAR_FAILURE_UM,
            "slope_um_per_pass": slope,
            "slope_raw_um_per_pass": float(slope_raw),
            "slope_clamped": bool(slope_clamped),
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
def fit_taylor_constants(set_recs: Dict[int, dict], failure_um: float,
                         verbose: bool = True) -> Dict[str, dict]:
    """
    Fit V_c · T^n = C per material using the noise-on-y direction:
        log T = (log C)/n  -  (1/n)·log V   →   regress logT on logV
    where T_s = W_f / k_s (W_f = failure_um, k_s = observed mid-life wear rate).
    V_c is the controlled experimental variable (no measurement error), so it
    belongs on the x-axis under OLS assumptions. Recovers n = -1/slope and
    log C = intercept · n. The previous direction (logV on logk) put noise on
    the x-axis and biased n toward zero.
    """
    constants: Dict[str, dict] = {}
    if verbose:
        print("\n  TAYLOR FIT  log T = a + b·log(Vc)   →   n = -1/b,  C = exp(a·n)")
        print("  " + "-" * 72)
    for mat in ["CK45", "RVS 304"]:
        recs = [r for r in set_recs.values()
                if r["material"] == mat and r["slope_um_per_pass"] > 1e-4]
        n_distinct_vc = len({round(r["Vc"], 3) for r in recs})

        # Fall back to literature n if the design is rank-deficient
        # (single Vc or single data point).
        if len(recs) < 2 or n_distinct_vc < 2:
            n_lit = 0.25 if mat == "CK45" else 0.20
            if recs:
                # Median-based anchor (was recs[0] — order-dependent)
                med_slope = float(np.median([r["slope_um_per_pass"] for r in recs]))
                med_vc    = float(np.median([r["Vc"] for r in recs]))
                T0 = failure_um / med_slope
                C  = med_vc * T0 ** n_lit
            else:
                C = float("nan")
            constants[mat] = {"n": n_lit, "C": round(float(C), 4),
                              "r2": None, "n_sets": len(recs),
                              "n_distinct_vc": n_distinct_vc, "source": "literature_n"}
            if verbose:
                print(f"  [{mat}] only {n_distinct_vc} distinct Vc among {len(recs)} sets"
                      f" -> using literature n={n_lit}, C anchored on median(Vc,k).")
            continue

        logV = np.array([np.log(r["Vc"]) for r in recs])
        logT = np.array([np.log(failure_um / r["slope_um_per_pass"]) for r in recs])
        A = np.column_stack([np.ones(len(logV)), logV])
        (a_int, b_slp), *_ = np.linalg.lstsq(A, logT, rcond=None)

        if b_slp >= -1e-9:
            # Non-physical: tool life increases (or is flat) with cutting speed.
            # Fall back to literature n rather than emit a nonsense exponent.
            n_lit = 0.25 if mat == "CK45" else 0.20
            med_slope = float(np.median([r["slope_um_per_pass"] for r in recs]))
            med_vc    = float(np.median([r["Vc"] for r in recs]))
            C = med_vc * (failure_um / med_slope) ** n_lit
            constants[mat] = {"n": n_lit, "C": round(float(C), 4),
                              "r2": None, "n_sets": len(recs),
                              "n_distinct_vc": n_distinct_vc,
                              "source": "literature_n",
                              "note": f"regression slope b={b_slp:.4f} >= 0 (non-physical)"}
            if verbose:
                print(f"  [{mat}] regression slope b={b_slp:.4f} >= 0 (T increases with V) "
                      f"-> using literature n={n_lit}.")
            continue

        n = -1.0 / b_slp
        log_C = a_int * n
        C = float(np.exp(log_C))

        pred   = A @ np.array([a_int, b_slp])
        ss_res = float(np.sum((logT - pred) ** 2))
        ss_tot = float(np.sum((logT - logT.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else None

        constants[mat] = {"n": round(float(n), 6), "C": round(float(C), 4),
                          "r2": round(r2, 4) if r2 is not None else None,
                          "n_sets": len(recs), "n_distinct_vc": n_distinct_vc,
                          "source": "fitted"}
        if verbose:
            warn = "  ** only 2 distinct Vc: n is weakly identified **" if n_distinct_vc == 2 else ""
            print(f"  [{mat}] n={n:.4f}  C={C:.2f}  R2={r2}  (sets={len(recs)}){warn}")
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