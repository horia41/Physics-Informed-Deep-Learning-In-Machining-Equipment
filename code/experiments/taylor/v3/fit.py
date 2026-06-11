import numpy as np
import pandas as pd
from typing import Sequence, Dict
from sklearn.linear_model import LinearRegression

from constants.matwi_dataset_constants import (
    PHYSICS_CAUTION_SETS,
    PHYSICS_EXCLUDED_SETS,
    WINDOW_HI,
    WINDOW_LO,
    material_of
)
from experiments.taylor.fit import theil_sen

WEAR_FAILURE_UM = 90.0

def life_at_threshold(t: np.ndarray, w: np.ndarray, threshold: float):
    """Observed tool life: the ImageID at which wear FIRST reaches `threshold`,
    linearly interpolated between the two bracketing samples. This is the real
    crossing point in the data (no rate division, no extrapolation) — every set
    is guaranteed to reach the 90um level, so all sets get a finite life.
    Returns None if the set never reaches the threshold."""
    t = np.asarray(t, float)
    w = np.asarray(w, float)
    order = np.argsort(t)
    t, w = t[order], w[order]
    if w.size == 0 or float(w.max()) < threshold:
        return None
    idx = int(np.argmax(w >= threshold))  # first index at/above threshold
    if idx == 0:
        return float(t[0])  # already past it at the start
    t0, w0 = float(t[idx - 1]), float(w[idx - 1])
    t1, w1 = float(t[idx]), float(w[idx])
    if w1 == w0:
        return float(t1)
    frac = (threshold - w0) / (w1 - w0)
    return float(t0 + frac * (t1 - t0))

# ── Per-set robust trajectory fit ─────────────────────────────────────────────
def fit_set_trajectories(df: pd.DataFrame, sets: pd.DataFrame,
                         train_sets: Sequence[int], failure_um: float = WEAR_FAILURE_UM,
                         verbose: bool = True) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    if verbose:
        print("  PER-SET MID-LIFE WEAR-RATE FITS  (Theil–Sen, 15–85% window)")
        print("  " + "-" * 72)
        print(f"  {'Set':>4} {'mat':>7} {'Vc':>5} {'maxID':>6} {'maxWear':>8} "
              f"{'%ofF':>6} {'slope':>8} {'icpt':>7} {'n_win':>6} {'life@F':>8}")
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
        slope = max(slope, 0.0)  # wear does not decrease in expectation
        max_wear = float(w.max())
        life_at_fail = life_at_threshold(t, w, failure_um)  # observed life: ImageID @ failure_um
        rec = {
            "set": s, "material": material_of(s),
            "Vc": float(sets.loc[s, "Vc"]),
            "fz": float(sets.loc[s, "fz"]),
            "max_image_id": t_max,
            "max_wear_um": max_wear,
            "frac_of_failure": max_wear / failure_um,
            "slope_um_per_pass": slope,
            "intercept_um": icpt,
            "life_at_failure": life_at_fail,
            "window_lo_id": lo, "window_hi_id": hi,
            "n_window": int(m.sum()),
            "caution": s in PHYSICS_CAUTION_SETS,
        }
        out[s] = rec
        if verbose:
            life_str = f"{life_at_fail:.1f}" if life_at_fail is not None else "none"
            print(f"  {s:>4} {material_of(s):>7} {rec['Vc']:>5.0f} {t_max:>6.0f} "
                  f"{max_wear:>8.0f} {rec['frac_of_failure']:>6.0%} "
                  f"{slope:>8.3f} {icpt:>7.1f} {rec['n_window']:>6} {life_str:>8}")
    return out


# ── Taylor n, m, C per material from OBSERVED life at the failure threshold ───
def fit_taylor_constants(set_recs, failure_um, verbose=True):
    """
    Fits the extended logarithmic Taylor equation per material:
        ln(Vc) = ln(C) - n*ln(T) - m*ln(fz)
    where T is the OBSERVED tool life: the ImageID at which each set first reaches
    `failure_um` (the 90um level every set is guaranteed to pass through). This is
    the true data crossing (life_at_failure), NOT 90/rate — so flat-window sets
    (2, 7) stay in, no extrapolation, and CK45 keeps its Vc=120 point (Set 2).
    """

    constants = {}
    material_groups = {"CK45": [], "RVS 304": []}
    for s, r in set_recs.items():
        if str(s) == "1" or s == 1:  # Set 1: cutting params unknown
            continue
        T = r.get("life_at_failure")
        if T is None or not np.isfinite(T) or T <= 0:  # never reached the threshold
            if verbose:
                print(f"  [skip Set {s}] never reaches {failure_um:.0f}um -> no life crossing")
            continue
        mat = r["material"]
        if mat in material_groups:
            material_groups[mat].append(r)

    for mat, recs in material_groups.items():
        if len(recs) < 2:
            if verbose:
                print(f"  [{mat}] <2 usable sets; using literature fallback baseline.")
            constants[mat] = {"n": 0.25, "m": 0.50, "C": 300.0,
                              "n_sets": len(recs), "source": "fallback"}
            continue

        # Y = ln(Vc); X = [ln(T_observed), ln(fz)]
        X = np.array([[np.log(r["life_at_failure"]), np.log(r["fz"])] for r in recs])
        Y = np.array([np.log(r["Vc"]) for r in recs])
        reg = LinearRegression().fit(X, Y)
        n_fit = -reg.coef_[0]  # coef on ln(T) is -n
        m_fit = -reg.coef_[1]  # coef on ln(fz) is -m
        C_fit = float(np.exp(reg.intercept_))

        n_distinct_vc = len({round(r["Vc"], 3) for r in recs})
        constants[mat] = {
            "n": float(round(n_fit, 4)),
            "m": float(round(m_fit, 4)),
            "C": float(round(C_fit, 4)),
            "n_sets": len(recs),
            "n_distinct_vc": n_distinct_vc,
            "source": f"life_at_{int(round(failure_um))}um_crossing",
        }
        if verbose:
            warn = "  ** fz constant: m not identifiable **" if len({round(r["fz"], 4) for r in recs}) < 2 else ""
            print(f"  [{mat}] life-crossing fit @{failure_um:.0f}um -> "
                  f"n={n_fit:.4f}, m={m_fit:.4f}, C={C_fit:.2f}  (sets={len(recs)}, distinctVc={n_distinct_vc}){warn}")

    return constants

