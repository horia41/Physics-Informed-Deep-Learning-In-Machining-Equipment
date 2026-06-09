

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# ── Configuration matching the rest of the pipeline ──────────────────────────
WEAR_CAP_UM = 450.0  # matches DatasetClass_Vision wear_cap
WEAR_FAILURE_UM = 90.0  # ISO VB ceiling — used ONLY as the C/clamp anchor
DEFAULT_TRAIN_SETS = (1, 2, 5, 7, 8, 10, 11, 12, 13)  # paper train (1-13 range)
WINDOW_LO, WINDOW_HI = 0.15, 0.85  # mid-life band fraction of max ImageID

PHYSICS_EXCLUDED_SETS = {1}  # Set 1: cutting params unknown
PHYSICS_CAUTION_SETS = {6, 17}  # Set 6 variable feed; Set 17 z=2


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


def life_at_threshold(t: np.ndarray, w: np.ndarray, threshold: float):
    """Observed tool life: the ImageID at which wear FIRST reaches `threshold`,
    linearly interpolated between the two bracketing samples. This is the real
    crossing point in the data (no rate division, no extrapolation) — every set
    is guaranteed to reach the 90um level, so all sets get a finite life.
    Returns None if the set never reaches the threshold."""
    t = np.asarray(t, float);
    w = np.asarray(w, float)
    order = np.argsort(t);
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


# ── Load & prepare ────────────────────────────────────────────────────────────
def load_labels(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df.columns = df.columns.str.strip()
    df["wear"] = pd.to_numeric(df["wear"], errors="coerce")
    df["Set"] = pd.to_numeric(df["Set"], errors="coerce").astype("Int64")
    df["ImageID"] = pd.to_numeric(df["ImageID"], errors="coerce")
    df = df.dropna(subset=["wear", "Set", "ImageID"]).copy()
    df["wear"] = df["wear"].clip(upper=WEAR_CAP_UM)  # match training cap
    df["Set"] = df["Set"].astype(int)
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
    from sklearn.linear_model import LinearRegression

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 Phase 1: fit Taylor constants offline.")
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--sets-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("taylor_constants.json"))
    ap.add_argument("--train-sets", type=int, nargs="+", default=list(DEFAULT_TRAIN_SETS))
    ap.add_argument("--failure-um", type=float, default=WEAR_FAILURE_UM)
    args = ap.parse_args()

    df = load_labels(args.labels_csv)
    sets = load_sets(args.sets_csv)

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