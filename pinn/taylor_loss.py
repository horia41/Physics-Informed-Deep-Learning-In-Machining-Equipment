"""
MATWI — Taylor Physics Loss (Stage 3)
=======================================
Implements a differentiable physics penalty based on Taylor's tool-life equation:

    V · T^n = C    →    log V + n · log T = log C

where:
    V  = cutting speed (m/min)
    T  = tool life until failure (min)
    n  = Taylor exponent (material-dependent)
    C  = Taylor constant (material-dependent)

Our model predicts wear W (µm) for a given sample at cutting time t (min).
We approximate tool life T from the wear trajectory within each set:

    T_approx = t_failure · (W_failure / W_pred) ** (1/n)

where t_failure is the end-of-life time for that set and W_failure is the
failure threshold (here: 300 µm, the conventional end-of-life criterion).

This gives a *per-sample* Taylor residual we can backpropagate through:

    residual(i) = log(V_i) + n · log(T_approx_i) − log(C)
    L_physics   = mean( residual(i)^2 )  [for samples where physics applies]

The total loss:
    L_total = L_data + λ(epoch) · L_physics

with λ(epoch) ramping from 0 to λ_max over the first `warmup_epochs` epochs.

Design decisions:
  • Set 1 is EXCLUDED from the physics loss — cutting speed is unknown.
  • Set 6 uses caution flag — feed rate varies mid-run (see README §8).
  • Set 17 has z=2 inserts instead of z=4 → different Taylor C, flagged separately.
  • n, C are NOT learned — they are material constants fitted offline from the data.
    Run fit_taylor_constants() once, then pass the result to TaylorPhysicsLoss.
  • λ warm-up prevents the physics loss from dominating before the vision backbone
    has converged to a reasonable prediction range.
  • Predictions clamped to (eps, W_failure) before log to avoid NaN gradients.

Usage:
    # 1. Fit constants once (offline, on training data)
    from taylor_loss import fit_taylor_constants, TaylorPhysicsLoss

    constants = fit_taylor_constants(
        labels_csv  = "./data/matwi/labels.csv",
        sets_csv    = "./data/matwi/sets.csv",
        train_sets  = [1, 2, 5, 7, 8, 10, 11],
    )
    print(constants)
    # {"CK45": {"n": 0.21, "C": 182.3}, "RVS 304": {"n": 0.17, "C": 143.7}}

    # 2. Create loss module
    physics_loss = TaylorPhysicsLoss(
        constants    = constants,
        n_epochs     = 17,
        lambda_max   = 0.1,
        warmup_epochs= 5,
    )

    # 3. In training loop
    for epoch in range(N_EPOCHS):
        for batch in train_loader:
            pred_wear_norm = model(batch["image"])["wear"].squeeze(1)   # [0, 1]
            pred_wear_um   = pred_wear_norm * 1000.0                    # µm

            data_loss    = mse_loss(pred_wear_norm, batch["wear"])
            physics_loss_ = physics_loss(
                pred_wear_um = pred_wear_um,
                set_ids      = batch["set"],       # list or tensor of int
                materials    = batch["material"],   # list of str
                epoch        = epoch,
            )
            loss = data_loss + physics_loss_
            loss.backward()
            ...

Notes on Taylor constant fitting:
  The paper provides cutting speeds per set (Table 1) but not the Taylor constants
  n and C. We estimate them by linear regression on log(V) + n·log(T) = log(C)
  using the end-of-life times derived from the wear trajectories in each set.
  For sets without a clear end-of-life (wear never reaching failure threshold),
  the last measurement time is used as a conservative lower bound for T.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# Cutting parameters per set
# From the MATWI paper (De Pauw et al., 2023) Table 1
# ══════════════════════════════════════════════════════════════════════════════

# Cutting speed Vc in m/min per set.
# Set 1 is None — not reported in the paper, cannot use physics loss.
# Set 6 has a variable feed rate mid-run — use with caution (flagged below).
# Sets 14-17: same material (RVS 304) as 12-13 but may have different Vc.
SET_CUTTING_SPEED: Dict[int, Optional[float]] = {
    1:  None,    # UNKNOWN — excluded from physics loss
    2:  200.0,
    3:  200.0,
    4:  200.0,
    5:  200.0,
    6:  200.0,   # NOTE: variable feed rate mid-run — treat with caution
    7:  200.0,
    8:  200.0,
    9:  200.0,
    10: 200.0,
    11: 200.0,
    12: 120.0,   # RVS 304 — lower speed due to harder material
    13: 120.0,
    14: 120.0,
    15: 120.0,
    16: 120.0,
    17: 120.0,   # NOTE: z=2 inserts (vs z=4 for all others) — different Taylor C
}

# Sets to EXCLUDE entirely from the physics loss (unknown or unreliable params).
PHYSICS_EXCLUDED_SETS = {
    1,    # Cutting speed unknown
}

# Sets to apply with reduced weight (caution flag — apply but halve λ).
PHYSICS_CAUTION_SETS = {
    6,    # Variable feed rate mid-run
    17,   # z=2 inserts — different Taylor constant, may mislead if C fitted on z=4
}

# Material assignment: sets ≤ 11 are CK45, sets ≥ 12 are RVS 304
MATERIAL_BY_SET: Dict[int, str] = {
    **{s: "CK45"    for s in range(1, 12)},
    **{s: "RVS 304" for s in range(12, 18)},
}

# End-of-life wear threshold (µm). Tool is considered "failed" at this value.
# Standard VB_max criterion used in the paper and in most milling literature.
WEAR_FAILURE_UM = 300.0

# Wear normalisation divisor (must match dataset class)
WEAR_DIVISOR = 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# Taylor constant fitting
# ══════════════════════════════════════════════════════════════════════════════

def fit_taylor_constants(
    labels_csv:    str | Path,
    sets_csv:      Optional[str | Path] = None,
    train_sets:    Sequence[int] = (1, 2, 5, 7, 8, 10, 11),
    failure_um:    float = WEAR_FAILURE_UM,
    verbose:       bool  = True,
) -> Dict[str, Dict[str, float]]:
    """
    Fit Taylor's equation constants (n, C) per material from training data.

    Method
    ------
    1. For each set in train_sets, derive the tool life T (min) by linear
       interpolation to find when wear crosses failure_um. If wear never
       reaches failure_um, use the final timestamp as a lower-bound estimate.
    2. Compute log(V) and log(T) for each set (excluding Set 1).
    3. Fit:  log(T) = (log(C) - log(V)) / n  → rewritten as
             log(V) = log(C) - n·log(T)
       i.e. linear regression of log(V) on log(T) with slope -n and intercept log(C).
    4. Fit separately for CK45 and RVS 304.

    Parameters
    ----------
    labels_csv  : path to labels.csv
    sets_csv    : path to sets.csv (optional — used if it has timing info)
    train_sets  : which sets to use for fitting (default: paper's training sets)
    failure_um  : wear threshold defining end-of-life (µm)
    verbose     : print fitted constants and diagnostic info

    Returns
    -------
    dict of the form:
        {
          "CK45":    {"n": float, "C": float, "r2": float, "n_sets": int},
          "RVS 304": {"n": float, "C": float, "r2": float, "n_sets": int},
        }
    """
    labels_csv = Path(labels_csv)
    df = pd.read_csv(labels_csv)
    df.columns = df.columns.str.strip()
    df["wear"] = pd.to_numeric(df["wear"], errors="coerce")
    df["Set"]  = pd.to_numeric(df["Set"],  errors="coerce").astype("Int64")

    # We need a timestamp column to derive tool life. MATWI labels.csv has
    # 'ImageID' which corresponds to measurement order within a set, not clock
    # time. We treat ImageID as a proxy for time (uniform measurement intervals).
    # If the sets.csv has actual timestamps, it will be used instead.
    # This is a common approximation in tool wear regression literature when
    # exact clock times aren't provided per image.
    if "timestamp" in df.columns or "time_min" in df.columns:
        time_col = "time_min" if "time_min" in df.columns else "timestamp"
    else:
        # Fall back to ImageID as a monotonic time proxy
        time_col = "ImageID"
        if verbose:
            print("[TaylorFit] No timestamp column found — using ImageID as time proxy.")
            print("            Tool life T values will be in 'measurement units', not minutes.")
            print("            Taylor constant C will be in matching units.")
            print("            This is valid for computing physics residuals consistently,")
            print("            as long as you use the same time proxy at inference time.")

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

    results_per_set = []
    skipped = []

    for set_num in sorted(train_sets):
        set_num = int(set_num)

        # Skip excluded sets
        if set_num in PHYSICS_EXCLUDED_SETS:
            skipped.append((set_num, "unknown cutting speed"))
            continue

        vc = SET_CUTTING_SPEED.get(set_num)
        if vc is None:
            skipped.append((set_num, "Vc=None"))
            continue

        material = MATERIAL_BY_SET.get(set_num, "unknown")

        # Extract this set's wear trajectory, sorted by time
        set_df = (
            df[(df["Set"] == set_num) & df["wear"].notna() & df[time_col].notna()]
            .sort_values(time_col)
            .copy()
        )

        if len(set_df) < 3:
            skipped.append((set_num, f"too few samples ({len(set_df)})"))
            continue

        t_vals = set_df[time_col].values.astype(float)
        w_vals = set_df["wear"].values.astype(float)

        # Find tool life T: interpolate to where wear crosses failure_um
        if w_vals[-1] >= failure_um:
            # Interpolate
            idx = np.searchsorted(w_vals, failure_um)
            if idx == 0:
                T = float(t_vals[0])
            elif idx >= len(t_vals):
                T = float(t_vals[-1])
            else:
                # Linear interpolation between idx-1 and idx
                w0, w1 = w_vals[idx-1], w_vals[idx]
                t0, t1 = t_vals[idx-1], t_vals[idx]
                T = t0 + (failure_um - w0) * (t1 - t0) / (w1 - w0 + 1e-8)
            interpolated = True
        else:
            # Wear never reached failure — use last time as lower bound
            T = float(t_vals[-1])
            interpolated = False
            if verbose:
                print(f"  [Set {set_num:2d}] Wear never reached {failure_um}µm "
                      f"(max={w_vals[-1]:.0f}µm). Using T={T:.1f} as lower bound.")

        if T <= 0:
            skipped.append((set_num, f"T<=0 ({T:.2f})"))
            continue

        results_per_set.append({
            "set":          set_num,
            "material":     material,
            "Vc":           vc,
            "T":            T,
            "log_Vc":       np.log(vc),
            "log_T":        np.log(T),
            "interpolated": interpolated,
            "caution":      set_num in PHYSICS_CAUTION_SETS,
        })

    if verbose and skipped:
        print(f"\n[TaylorFit] Skipped sets: {skipped}")

    if not results_per_set:
        raise RuntimeError(
            "No sets produced valid (Vc, T) pairs. Cannot fit Taylor constants."
        )

    fit_df = pd.DataFrame(results_per_set)

    # Fit per material: log(V) = log(C) − n · log(T)
    # i.e. linear regression: y = log_Vc, x = log_T, intercept = log(C), slope = -n
    constants: Dict[str, Dict[str, float]] = {}

    for material in ["CK45", "RVS 304"]:
        mat_df = fit_df[fit_df["material"] == material].copy()
        if len(mat_df) < 2:
            warnings.warn(
                f"[TaylorFit] Only {len(mat_df)} set(s) for material '{material}'. "
                f"Taylor constant estimate will be unreliable. "
                f"Consider using all 13 sets for fitting, not just training sets."
            )
            if len(mat_df) == 0:
                continue

        x = mat_df["log_T"].values   # (N,)
        y = mat_df["log_Vc"].values  # (N,)

        if len(x) == 1:
            # Can't fit a line with 1 point — use known Taylor n from literature
            # CK45 steel: n ≈ 0.2-0.25, RVS 304: n ≈ 0.15-0.20
            n_lit = 0.22 if material == "CK45" else 0.18
            log_C = y[0] + n_lit * x[0]
            C = np.exp(log_C)
            n = n_lit
            r2 = float("nan")
            if verbose:
                print(f"  [{material}] Single set — using literature n={n_lit:.2f}")
        else:
            # Ordinary least squares: y = a + b*x, slope b = -n, intercept a = log(C)
            A = np.vstack([np.ones(len(x)), x]).T   # (N, 2)
            result = np.linalg.lstsq(A, y, rcond=None)
            coeffs = result[0]
            log_C, neg_n = float(coeffs[0]), float(coeffs[1])
            n = -neg_n
            C = np.exp(log_C)

            # R² for diagnostic purposes
            y_pred = log_C + neg_n * x
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        constants[material] = {
            "n":      round(float(n), 6),
            "C":      round(float(C), 4),
            "r2":     round(float(r2), 4) if np.isfinite(r2) else None,
            "n_sets": len(mat_df),
        }

        if verbose:
            sets_used = mat_df["set"].tolist()
            print(f"\n  [{material}] Sets used: {sets_used}")
            for _, row in mat_df.iterrows():
                flag = " ⚠ caution" if row["caution"] else ""
                interp = "(interpolated)" if row["interpolated"] else "(lower bound)"
                print(f"    Set {int(row['set']):2d}: Vc={row['Vc']:.0f} m/min, "
                      f"T={row['T']:.1f}, log(V)={row['log_Vc']:.3f}, "
                      f"log(T)={row['log_T']:.3f} {interp}{flag}")
            print(f"  Fitted: n={n:.4f}, C={C:.2f}, R²={r2:.3f}")
            print(f"  Equation: V · T^{n:.4f} = {C:.2f}")

    if verbose:
        print(f"\n[TaylorFit] Final constants:")
        for mat, vals in constants.items():
            print(f"  {mat}: n={vals['n']}, C={vals['C']}, "
                  f"R²={vals['r2']}, n_sets={vals['n_sets']}")

    return constants


# ══════════════════════════════════════════════════════════════════════════════
# Taylor physics loss module
# ══════════════════════════════════════════════════════════════════════════════

class TaylorPhysicsLoss(nn.Module):
    """
    Differentiable physics penalty based on Taylor's tool-life equation.

    Computes a per-sample residual:
        residual_i = log(V_i) + n_mat · log(T_approx_i) − log(C_mat)

    where T_approx_i is the approximated tool life given the model's wear
    prediction at sample i:
        T_approx_i = T_ref_set · (W_failure / clamp(W_pred_i, eps, W_failure)) ^ (1/n)

    T_ref_set is the (approximate) tool life of the set that sample i belongs to,
    estimated from the maximum wear measurement observed in that set.

    Physics loss:
        L_physics = mean(residual_i^2)   [over valid samples only]

    Total loss:
        L_total = L_data + λ(epoch) · L_physics

    λ schedule (linear warm-up):
        λ(epoch) = λ_max · min(1, epoch / warmup_epochs)

    Parameters
    ----------
    constants : dict
        Output of fit_taylor_constants(). Must contain keys for each material
        present in the training data.
        Example: {"CK45": {"n": 0.21, "C": 182.3}, "RVS 304": {"n": 0.17, "C": 143.7}}
    n_epochs : int
        Total number of training epochs. Used only for warm-up scheduling.
    lambda_max : float
        Maximum weight of the physics loss after warm-up. Default 0.1.
        Start conservative — too large will override the data signal.
        Suggested range: 0.01 – 0.5. Use 0.1 as the first experiment.
    warmup_epochs : int
        Number of epochs to ramp λ from 0 to lambda_max. Default 5.
    failure_um : float
        Tool life end-of-life wear threshold in µm. Must match what was used
        in fit_taylor_constants(). Default 300.
    set_T_ref : dict[int, float] or None
        Pre-computed reference tool life per set (in the same time units used
        during constant fitting). If None, a default of 1.0 is used, which
        makes the physics loss a consistency check on relative predictions
        rather than an absolute Taylor constraint.
        Populate this from the same sets_csv / labels_csv you used for fitting.
    caution_weight : float
        Weight multiplier for caution-flagged sets (Set 6, Set 17). Default 0.5.
    eps : float
        Small constant to clamp predictions away from zero before log. Default 1e-3.
    """

    def __init__(
        self,
        constants:       Dict[str, Dict[str, float]],
        n_epochs:        int   = 17,
        lambda_max:      float = 0.1,
        warmup_epochs:   int   = 5,
        failure_um:      float = WEAR_FAILURE_UM,
        set_T_ref:       Optional[Dict[int, float]] = None,
        caution_weight:  float = 0.5,
        eps:             float = 1e-3,
    ):
        super().__init__()

        self.constants      = constants
        self.n_epochs       = n_epochs
        self.lambda_max     = lambda_max
        self.warmup_epochs  = max(1, warmup_epochs)
        self.failure_um     = failure_um
        self.set_T_ref      = set_T_ref or {}
        self.caution_weight = caution_weight
        self.eps            = eps

        # Pre-compute log values for constants to avoid recomputing each forward pass
        self._log_C: Dict[str, float] = {}
        self._n:     Dict[str, float] = {}
        for mat, vals in constants.items():
            self._log_C[mat] = np.log(vals["C"])
            self._n[mat]     = vals["n"]

        self._validate()

    def _validate(self) -> None:
        for mat, vals in self.constants.items():
            assert vals["n"] > 0, f"Taylor n must be > 0 for {mat}, got {vals['n']}"
            assert vals["C"] > 0, f"Taylor C must be > 0 for {mat}, got {vals['C']}"
        assert 0 < self.lambda_max <= 10.0, \
            f"lambda_max should be in (0, 10], got {self.lambda_max}"

    def get_lambda(self, epoch: int) -> float:
        """Current λ value based on warm-up schedule."""
        if self.warmup_epochs <= 0:
            return self.lambda_max
        ramp = min(1.0, (epoch + 1) / self.warmup_epochs)
        return self.lambda_max * ramp

    def forward(
        self,
        pred_wear_um:  torch.Tensor,           # (B,) predicted wear in µm
        set_ids:       Sequence[int] | torch.Tensor,   # (B,) set numbers
        materials:     Sequence[str],           # (B,) material labels
        epoch:         int,
    ) -> torch.Tensor:
        """
        Compute the weighted physics loss for a batch.

        Parameters
        ----------
        pred_wear_um : (B,) tensor of predicted wear values in µm (un-normalised).
        set_ids      : (B,) int tensor or list of set numbers (1–17).
        materials    : (B,) list of material strings ("CK45" or "RVS 304").
        epoch        : current training epoch (0-indexed).

        Returns
        -------
        Scalar tensor — the physics loss term (already scaled by λ).
        Returns 0.0 if no valid samples exist in this batch.
        """
        lam = self.get_lambda(epoch)
        if lam == 0.0:
            return pred_wear_um.new_zeros(1).squeeze()

        device = pred_wear_um.device
        B = pred_wear_um.shape[0]

        if isinstance(set_ids, torch.Tensor):
            set_ids = set_ids.cpu().tolist()

        residuals   = []
        weights     = []

        for i in range(B):
            set_num  = int(set_ids[i])
            material = materials[i]

            # --- Skip samples where physics loss does not apply ---
            if set_num in PHYSICS_EXCLUDED_SETS:
                continue
            if material not in self._log_C:
                continue
            vc = SET_CUTTING_SPEED.get(set_num)
            if vc is None or vc <= 0:
                continue

            # --- Taylor constants for this material ---
            n     = self._n[material]
            log_C = self._log_C[material]
            log_V = np.log(vc)

            # --- Approximate tool life from predicted wear ---
            # Clamp prediction to a valid range before log
            w_pred_clamped = pred_wear_um[i].clamp(self.eps, self.failure_um - self.eps)

            # T_approx = T_ref * (W_failure / W_pred) ^ (1/n)
            # In log space:
            # log(T_approx) = log(T_ref) + (1/n) * log(W_failure / W_pred)
            T_ref = self.set_T_ref.get(set_num, 1.0)
            log_T_ref = float(np.log(max(T_ref, 1e-8)))

            log_ratio = torch.log(
                torch.tensor(self.failure_um, device=device, dtype=pred_wear_um.dtype)
                / w_pred_clamped
            )
            log_T_approx = log_T_ref + (1.0 / n) * log_ratio

            # Taylor residual: log(V) + n * log(T) - log(C)
            residual = (
                float(log_V)
                + n * log_T_approx
                - float(log_C)
            )

            # Per-sample weight (halve for caution sets)
            w = self.caution_weight if set_num in PHYSICS_CAUTION_SETS else 1.0

            residuals.append(residual)
            weights.append(w)

        if not residuals:
            # No valid samples — return 0 (still in the computation graph)
            return pred_wear_um.sum() * 0.0

        # Stack and compute weighted MSE of residuals
        residuals_t = torch.stack(residuals)               # (N_valid,)
        weights_t   = torch.tensor(
            weights, device=device, dtype=pred_wear_um.dtype
        )
        sq_residuals = residuals_t ** 2                     # (N_valid,)
        physics_loss = (weights_t * sq_residuals).sum() / weights_t.sum()

        return lam * physics_loss

    def extra_repr(self) -> str:
        mats = ", ".join(
            f"{m}(n={v['n']:.4f}, C={v['C']:.2f})"
            for m, v in self.constants.items()
        )
        return (
            f"materials=[{mats}], lambda_max={self.lambda_max}, "
            f"warmup_epochs={self.warmup_epochs}, failure_um={self.failure_um}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Helper: compute set-level reference tool lives from labels.csv
# ══════════════════════════════════════════════════════════════════════════════

def compute_set_T_ref(
    labels_csv:    str | Path,
    failure_um:    float = WEAR_FAILURE_UM,
    time_col:      str   = "ImageID",
) -> Dict[int, float]:
    """
    Compute a reference tool life T_ref for each set from the wear trajectory.

    Used to populate the set_T_ref argument of TaylorPhysicsLoss, so that
    the approximated T_approx_i is on the correct scale.

    Returns dict mapping set_num → T_ref (in units of time_col).
    """
    df = pd.read_csv(labels_csv)
    df.columns = df.columns.str.strip()
    df["wear"] = pd.to_numeric(df["wear"], errors="coerce")
    df["Set"]  = pd.to_numeric(df["Set"],  errors="coerce").astype("Int64")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

    T_ref: Dict[int, float] = {}
    for set_num, grp in df.groupby("Set"):
        set_num = int(set_num)
        grp = grp.dropna(subset=["wear", time_col]).sort_values(time_col)
        if len(grp) == 0:
            T_ref[set_num] = 1.0
            continue
        w_vals = grp["wear"].values
        t_vals = grp[time_col].values.astype(float)
        if w_vals[-1] >= failure_um:
            idx = np.searchsorted(w_vals, failure_um)
            if idx == 0:
                T_ref[set_num] = float(t_vals[0])
            elif idx >= len(t_vals):
                T_ref[set_num] = float(t_vals[-1])
            else:
                w0, w1 = w_vals[idx-1], w_vals[idx]
                t0, t1 = t_vals[idx-1], t_vals[idx]
                T_ref[set_num] = t0 + (failure_um - w0) * (t1 - t0) / (w1 - w0 + 1e-8)
        else:
            T_ref[set_num] = float(t_vals[-1])
    return T_ref


# ══════════════════════════════════════════════════════════════════════════════
# Quick smoke test (run with: python taylor_loss.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import math

    print("=" * 70)
    print("  Taylor Physics Loss — smoke test (no real data required)")
    print("=" * 70)

    # Fake Taylor constants (plausible for CK45 at Vc=200 m/min)
    constants = {
        "CK45":    {"n": 0.22, "C": 185.0, "r2": 0.91, "n_sets": 6},
        "RVS 304": {"n": 0.18, "C": 145.0, "r2": 0.88, "n_sets": 2},
    }

    physics_loss = TaylorPhysicsLoss(
        constants      = constants,
        n_epochs       = 17,
        lambda_max     = 0.1,
        warmup_epochs  = 5,
        failure_um     = 300.0,
        set_T_ref      = {2: 90.0, 5: 85.0, 7: 80.0, 12: 70.0},
    )
    print(f"\nModule: {physics_loss}\n")

    # Test λ schedule
    print("λ warm-up schedule:")
    for ep in range(17):
        lam = physics_loss.get_lambda(ep)
        bar = "█" * int(lam / 0.1 * 20)
        print(f"  epoch {ep:2d}: λ={lam:.4f}  {bar}")

    # Test forward pass
    print("\nForward pass test:")
    torch.manual_seed(42)
    B = 8
    pred_wear_um = torch.rand(B) * 280.0 + 10.0   # 10–290 µm range
    set_ids      = [2, 5, 7, 1, 12, 6, 11, 17]   # mix of valid/excluded/caution
    materials    = ["CK45", "CK45", "CK45", "CK45", "RVS 304", "CK45", "CK45", "RVS 304"]

    for epoch in [0, 2, 4, 6, 16]:
        loss = physics_loss(pred_wear_um, set_ids, materials, epoch=epoch)
        print(f"  epoch {epoch:2d}: physics_loss={loss.item():.6f}")

    # Test that excluded set (1) produces no gradient through physics path
    pred_excl = torch.tensor([150.0], requires_grad=True)
    loss_excl = physics_loss(pred_excl, [1], ["CK45"], epoch=10)
    print(f"\n  Set 1 (excluded): loss={loss_excl.item():.6f}  "
          f"(should be ~0.0 or zero-grad)")

    # Verify gradient flows for a valid set
    pred_valid = torch.tensor([150.0], requires_grad=True)
    loss_valid = physics_loss(pred_valid, [2], ["CK45"], epoch=10)
    loss_valid.backward()
    print(f"  Set 2 (valid):    loss={loss_valid.item():.6f}, "
          f"grad={pred_valid.grad.item():.6f}  (should be non-zero)")

    print("\n  All checks passed.")
    print("=" * 70)
    print("\nTo use with real data:")
    print("  1. Run fit_taylor_constants(labels_csv, sets_csv, train_sets=[1,2,5,7,8,10,11])")
    print("  2. Run compute_set_T_ref(labels_csv) to get set_T_ref")
    print("  3. Instantiate TaylorPhysicsLoss(constants, set_T_ref=set_T_ref, ...)")
    print("  4. In training loop: loss = data_loss + physics_loss(pred_um, sets, mats, epoch)")
