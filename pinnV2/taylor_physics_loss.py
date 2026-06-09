
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

WEAR_DIVISOR = 1000.0

# Canonical material strings (must match DatasetClass_VisionSensors.py).
# Encoded as integer codes (CK45=0, RVS=1, unknown=-1) at __init__ time so the
# forward pass works on tensors and never compares strings.
MATERIAL_CK45 = "CK45"
MATERIAL_RVS  = "RVS 304"
_MAT_CODE_CK45 = 0
_MAT_CODE_RVS  = 1
_MAT_CODE_UNKNOWN = -1


def _material_to_code(mat: str) -> int:
    if mat == MATERIAL_CK45: return _MAT_CODE_CK45
    if mat == MATERIAL_RVS:  return _MAT_CODE_RVS
    return _MAT_CODE_UNKNOWN


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

        # Window fraction: override > JSON > (0.15, 0.85) safety default.
        json_window = tuple(self.meta.get("window", (0.15, 0.85)))
        self.win_lo, self.win_hi = window if window is not None else json_window
        if not (0.0 <= self.win_lo < self.win_hi <= 1.0):
            raise ValueError(
                f"window fractions must satisfy 0 <= lo < hi <= 1, "
                f"got ({self.win_lo}, {self.win_hi})"
            )

        self.lambda_max       = lambda_max
        self.warmup_epochs    = max(1, warmup_epochs)
        self.apply_to         = apply_to.lower()
        self.one_sided        = one_sided
        self.use_taylor_slope = use_taylor_slope
        self.caution_weight   = caution_weight

        # Excluded / caution set lists from JSON meta (falls back to {1}
        # for older constants files that pre-date storing these).
        self.excluded_sets = set(self.meta.get("excluded_sets", [1]))
        self.caution_sets  = set(self.meta.get("caution_sets",  []))

        if self.apply_to not in ("all", "rvs", "ck45"):
            raise ValueError(f"apply_to must be 'all'/'rvs'/'ck45', got {apply_to!r}")
        if not (0.0 <= self.lambda_max <= 10.0):
            raise ValueError(f"lambda_max must be in [0, 10], got {self.lambda_max}")

        self._build_lookup_tables()

    # ── Per-set lookup tables (1-D buffers indexed by set id) ────────────────
    def _build_lookup_tables(self) -> None:
        """Precompute per-set (slope, intercept, win_lo_id, win_hi_id,
        weight, material_code, valid) into buffers. Missing or excluded
        sets get valid=False so they contribute zero to the loss."""
        max_set = max(self.set_recs.keys()) + 1 if self.set_recs else 1

        slope    = np.zeros(max_set, dtype=np.float32)
        intcpt   = np.zeros(max_set, dtype=np.float32)
        win_lo   = np.full(max_set,  np.inf,  dtype=np.float32)   # never inside
        win_hi   = np.full(max_set, -np.inf,  dtype=np.float32)
        weight   = np.zeros(max_set, dtype=np.float32)
        mat_code = np.full(max_set, _MAT_CODE_UNKNOWN, dtype=np.int64)
        valid    = np.zeros(max_set, dtype=bool)

        for s, rec in self.set_recs.items():
            if s in self.excluded_sets:
                continue

            # Slope + intercept: keep observed pair, or swap to Taylor-anchored
            # pair when use_taylor_slope=True AND a Taylor slope is available.
            taylor_slope = rec.get("taylor_slope_um_per_pass")
            taylor_icpt  = rec.get("taylor_intercept_um")
            use_taylor   = (
                self.use_taylor_slope
                and taylor_slope is not None
                and taylor_icpt  is not None
                and float(taylor_slope) > 0.0
            )
            if use_taylor:
                slope[s]  = float(taylor_slope)
                intcpt[s] = float(taylor_icpt)
            else:
                slope[s]  = float(rec["slope_um_per_pass"])
                intcpt[s] = float(rec["intercept_um"])

            # Window: re-derive from the (possibly overridden) fraction
            # using the per-set max_image_id stored in the JSON.
            t_max = float(rec.get("max_image_id", 0.0))
            win_lo[s] = self.win_lo * t_max
            win_hi[s] = self.win_hi * t_max

            weight[s]   = self.caution_weight if s in self.caution_sets else 1.0
            mat_code[s] = _material_to_code(rec["material"])
            valid[s]    = True

        self.register_buffer("_slope",    torch.from_numpy(slope))
        self.register_buffer("_intercept", torch.from_numpy(intcpt))
        self.register_buffer("_win_lo",   torch.from_numpy(win_lo))
        self.register_buffer("_win_hi",   torch.from_numpy(win_hi))
        self.register_buffer("_weight",   torch.from_numpy(weight))
        self.register_buffer("_mat_code", torch.from_numpy(mat_code))
        self.register_buffer("_valid",    torch.from_numpy(valid))

    # ── λ warm-up ────────────────────────────────────────────────────────────
    def get_lambda(self, epoch: int) -> float:
        return self.lambda_max * min(1.0, (epoch + 1) / self.warmup_epochs)

    # ── forward (fully vectorised) ───────────────────────────────────────────
    def forward(
        self,
        pred_norm: torch.Tensor,                 # (B,) or (B,1) in [0,1]
        set_ids:   Sequence[int] | torch.Tensor, # (B,)
        image_ids: Sequence[int] | torch.Tensor, # (B,)
        materials: Sequence[str],                # (B,) — unused (encoded at init)
        epoch:     int,
    ) -> torch.Tensor:
        lam = self.get_lambda(epoch)
        pred = pred_norm.squeeze(-1) if pred_norm.dim() > 1 else pred_norm
        if lam == 0.0:
            return pred.sum() * 0.0

        dev = pred.device
        dt  = pred.dtype

        # Per-set lookup buffers are registered on CPU at __init__; the
        # training scripts move the model to GPU but not this module, so
        # move the buffers onto the compute device once. Pure placement —
        # no effect on the computed loss.
        if self._slope.device != dev:
            self.to(dev)

        # Coerce ids to tensors on the same device.
        if not isinstance(set_ids, torch.Tensor):
            set_ids = torch.as_tensor(set_ids, dtype=torch.long, device=dev)
        sids = set_ids.to(device=dev, dtype=torch.long)
        if not isinstance(image_ids, torch.Tensor):
            image_ids = torch.as_tensor(image_ids, dtype=dt, device=dev)
        iids = image_ids.to(device=dev, dtype=dt)

        # Clamp to lookup range so out-of-table set ids (shouldn't happen) are
        # masked out rather than indexing past the buffer.
        in_range  = (sids >= 0) & (sids < self._slope.shape[0])
        safe_sids = torch.where(in_range, sids, torch.zeros_like(sids))

        slope    = self._slope[safe_sids]
        intcpt   = self._intercept[safe_sids]
        win_lo   = self._win_lo[safe_sids]
        win_hi   = self._win_hi[safe_sids]
        weight   = self._weight[safe_sids]
        mat_code = self._mat_code[safe_sids]
        valid    = self._valid[safe_sids]

        # Material gating
        if   self.apply_to == "all":  mat_ok = torch.ones_like(valid)
        elif self.apply_to == "rvs":  mat_ok = mat_code == _MAT_CODE_RVS
        else:                          mat_ok = mat_code == _MAT_CODE_CK45

        # Mid-life window
        win_ok = (iids >= win_lo) & (iids <= win_hi)

        use = in_range & valid & mat_ok & win_ok
        if not use.any():
            return pred.sum() * 0.0

        # Expected wear trajectory, clipped to [0, W_f] and normalised
        vb_um  = (slope * iids + intcpt).clamp(0.0, self.failure_um)
        exp_n  = vb_um / WEAR_DIVISOR

        diff = pred - exp_n
        if self.one_sided:
            diff = torch.clamp(diff, min=0.0)        # punish overprediction only

        w = weight * use.to(dt)
        physics = (w * diff.pow(2)).sum() / w.sum().clamp_min(1e-12)
        return lam * physics

    def extra_repr(self) -> str:
        cs = ", ".join(f"{m}(n={v['n']},C={v['C']})" for m, v in self.constants.items())
        return (f"[{cs}] lambda_max={self.lambda_max} warmup={self.warmup_epochs} "
                f"window=({self.win_lo},{self.win_hi}) apply_to={self.apply_to} "
                f"one_sided={self.one_sided} taylor_slope={self.use_taylor_slope}")