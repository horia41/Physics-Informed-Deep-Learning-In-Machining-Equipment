"""
MATWI — Stage 3 Phase 2: Taylor physics-informed loss (PyTorch)
================================================================
Differentiable penalty that regularises the vision model toward a physically
plausible wear trajectory, targeting the adhesion *overprediction* failure mode
on RVS 304 (molten material welds to the tip and tricks the CNN into reading
high wear).

Formulation  (supervisor brief: "expected wear at elapsed time t, linear in
mid-life"):
    For sample i in set s at pass ImageID_i,
        VB_expected_i = clip( a_s + b_s · ImageID_i ,  0,  W_f )      [µm]
    where (a_s, b_s) is the frozen, spike-robust mid-life line fit offline
    (fit_taylor.py). Optionally b_s is replaced by the Taylor-predicted slope
    W_f / T_taylor(Vc_s) so that the cross-set rate ordering obeys Vc·T^n = C.

    L_physics = mean_i [ w_i · pen( VB_pred_i − VB_expected_i )^2 ]   (normalised /1000)
    pen(x) = x                (symmetric, default)
           = relu(x)          (one-sided "ceiling": only punish overprediction)
    L_total   = L_data + λ(epoch) · L_physics

Guardrails baked in:
    • Set 1 excluded (unknown cutting params).
    • Penalty applied ONLY inside the per-set mid-life window [15%, 85%] of the
      run — respects break-in / catastrophic-failure non-linearity and the
      early-stop distortion.
    • Caution sets (6 variable feed, 17 z=2) down-weighted.
    • Material gating: enforce on RVS only / CK45 only / both.
    • λ linear warm-up so physics does not dominate before the backbone settles.

This module loads the frozen constants produced by fit_taylor.py. It never
learns n / C / slopes (supervisors: learnable constants make it "no longer a
hybrid approach").
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

WEAR_DIVISOR = 1000.0
PHYSICS_EXCLUDED_SETS = {1}


class TaylorPhysicsLoss(nn.Module):
    def __init__(
        self,
        constants_path:   str | Path | None = None,
        payload:          Optional[dict] = None,
        lambda_max:       float = 0.05,
        warmup_epochs:    int   = 4,
        window:           tuple[float, float] | None = None,   # override JSON
        apply_to:         str   = "all",            # "all" | "rvs" | "ck45"
        one_sided:        bool  = False,            # True = ceiling (overpred only)
        use_taylor_slope: bool  = False,            # True = cross-set Taylor coupling
        caution_weight:   float = 0.5,
    ):
        super().__init__()
        if payload is None:
            if constants_path is None:
                raise ValueError("Provide constants_path or payload.")
            payload = json.loads(Path(constants_path).read_text())

        self.meta       = payload["meta"]
        self.constants  = payload["constants"]
        self.set_recs   = {int(k): v for k, v in payload["sets"].items()}
        self.failure_um = float(self.meta["failure_um"])
        self.win_lo, self.win_hi = window or tuple(self.meta["window"])
        self.lambda_max     = lambda_max
        self.warmup_epochs  = max(1, warmup_epochs)
        self.apply_to       = apply_to.lower()
        self.one_sided      = one_sided
        self.use_taylor_slope = use_taylor_slope
        self.caution_weight = caution_weight
        self.caution_sets   = set(self.meta.get("caution_sets", []))

        assert self.apply_to in ("all", "rvs", "ck45")
        assert 0 < self.lambda_max <= 10.0

    # ── λ warm-up ────────────────────────────────────────────────────────────
    def get_lambda(self, epoch: int) -> float:
        return self.lambda_max * min(1.0, (epoch + 1) / self.warmup_epochs)

    def _material_ok(self, mat: str) -> bool:
        if self.apply_to == "all":  return True
        if self.apply_to == "rvs":  return mat == "RVS 304"
        return mat == "CK45"

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        pred_norm: torch.Tensor,                 # (B,) or (B,1) in [0,1]
        set_ids:   Sequence[int] | torch.Tensor, # (B,)
        image_ids: Sequence[int] | torch.Tensor, # (B,)
        materials: Sequence[str],                # (B,)
        epoch:     int,
    ) -> torch.Tensor:
        lam = self.get_lambda(epoch)
        pred = pred_norm.squeeze(-1) if pred_norm.dim() > 1 else pred_norm
        if lam == 0.0:
            return pred.sum() * 0.0

        if isinstance(set_ids, torch.Tensor):   set_ids = set_ids.tolist()
        if isinstance(image_ids, torch.Tensor): image_ids = image_ids.tolist()

        dev, dt = pred.device, pred.dtype
        idxs, exp_um, weights = [], [], []
        for i in range(pred.shape[0]):
            s = int(set_ids[i]); mat = materials[i]; t = float(image_ids[i])
            rec = self.set_recs.get(s)
            if rec is None or s in PHYSICS_EXCLUDED_SETS:        continue
            if not self._material_ok(mat):                      continue
            if not (rec["window_lo_id"] <= t <= rec["window_hi_id"]):  continue
            if self.use_taylor_slope and rec.get("taylor_slope_um_per_pass"):
                # Pulls the pre-calculated extended slope directly from the fresh JSON file
                slope = rec["taylor_slope_um_per_pass"]
            elif self.use_taylor_slope:
                # --- MODIFIED: Live multi-variable backup calculator ---
                # Extracts the multi-variable exponents dynamically if not found in the record
                n_val = self.constants.get(mat, {}).get("n", 0.25)
                m_val = self.constants.get(mat, {}).get("m", 0.0)
                C_val = self.constants.get(mat, {}).get("C", 300.0)
                
                vc = rec.get("Vc", 174.0)
                fz = rec.get("fz", 0.05)
                
                log_T = (np.log(C_val) - np.log(vc) - m_val * np.log(fz)) / n_val
                t_life = np.exp(log_T)
                slope = self.failure_um / t_life
            else:
                slope = rec["slope_um_per_pass"]

            vb = slope * t + rec["intercept_um"]
            vb = min(max(vb, 0.0), self.failure_um)
            idxs.append(i)
            exp_um.append(vb)
            weights.append(self.caution_weight if s in self.caution_sets else 1.0)

        if not idxs:
            return pred.sum() * 0.0

        sel   = pred[idxs]                                   # (k,) normalised
        exp_n = torch.tensor(exp_um, device=dev, dtype=dt) / WEAR_DIVISOR
        w     = torch.tensor(weights, device=dev, dtype=dt)
        diff  = sel - exp_n
        if self.one_sided:
            diff = torch.clamp(diff, min=0.0)                # punish overprediction only
        physics = (w * diff.pow(2)).sum() / w.sum()
        return lam * physics

    def extra_repr(self) -> str:
        cs = ", ".join(f"{m}(n={v['n']},C={v['C']})" for m, v in self.constants.items())
        return (f"[{cs}] lambda_max={self.lambda_max} warmup={self.warmup_epochs} "
                f"window=({self.win_lo},{self.win_hi}) apply_to={self.apply_to} "
                f"one_sided={self.one_sided} taylor_slope={self.use_taylor_slope}")
