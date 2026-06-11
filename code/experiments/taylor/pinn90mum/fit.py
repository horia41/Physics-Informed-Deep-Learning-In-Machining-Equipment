import numpy as np
import pandas as pd
from typing import Dict, Sequence

from experiments.taylor.fit import theil_sen
from constants.matwi_dataset_constants import (
    PHYSICS_CAUTION_SETS, 
    PHYSICS_EXCLUDED_SETS,
    WINDOW_HI,
    WINDOW_LO,
    material_of    
)

WEAR_FAILURE_UM    = 90.0

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
