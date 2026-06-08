"""
MATWI — Stage 3 (HARD constraint): physics-anchored bounded-residual output
===========================================================================
Implements the hard-constraint reparametrisation:

    VB_pred(x, s, t) = VB_taylor(s, t)  +  C · tanh( g_theta(x) )

where
  • g_theta(x)  = the raw scalar output of the existing vision network
                  (out["wear"]; an unbounded real number).
  • VB_taylor(s,t) = the Taylor-predicted expected wear for set s at pass t,
                  a per-sample physics ANCHOR (µm).
  • C           = the band half-width (µm). tanh ∈ (-1,1) ⇒ the learned
                  correction is bounded to (-C, +C).

Because the correction is bounded by construction, the prediction is
*structurally* confined to the band [VB_taylor - C, VB_taylor + C]. No value
of theta can produce a wear estimate outside that physics-defined band — i.e.
the wear-law violation is made mathematically impossible, not merely penalised
(this is the "hard constraint" of project RQ3, as opposed to the soft λ·L_physics
penalty in taylor_physics_loss.py).

THE ANCHOR  (why it is computed from Vc, not from the per-set fitted line)
--------------------------------------------------------------------------
A hard constraint is part of the architecture, so VB_taylor must be defined for
*every* sample at train AND eval time, including the val/test sets (3,4,6,9) and
the excluded set 1 — none of which have a fitted per-set line in
taylor_constants.json. We therefore build the anchor purely from physics:

    k_s = W_f / T_taylor(Vc_s),   T_taylor(Vc_s) = (C_mat / Vc_s)^(1/n_mat)
    VB_taylor(s, t) = clip( k_s · t , 0 , W_f )                      [µm]

k_s (the Taylor wear RATE, µm/pass) needs only the set's cutting speed Vc_s
(a known machine setting from sets.csv) and the per-material Taylor constants
(n_mat, C_mat) — which were fitted on the TRAIN sets only. So the anchor is:
  • defined for every set (any set with a Vc in sets.csv),
  • LEAKAGE-FREE (it never uses a val/test set's own wear labels — unlike the
    per-set fitted intercept, which is fit on that set's data),
  • physically clean (wear starts ~0 on a fresh tool and ramps at the Taylor
    rate; the network only supplies the bounded correction).
A set with no usable Vc (or no material constant) gets k_s = 0 ⇒ anchor 0 ⇒
the prediction falls back to the pure bounded network output C·tanh(·).

Anchor is clipped to W_f (=failure_um, 300 µm). The final prediction can reach
up to W_f + C, so C must be ≳ (wear_cap − W_f) = 150 µm to be able to represent
the highest-wear samples — see recommend_C() and README_HARD.md.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

WEAR_DIVISOR = 1000.0


def material_of(set_num: int) -> str:
    return "CK45" if set_num <= 11 else "RVS 304"


def load_sets(sets_csv: str | Path) -> pd.DataFrame:
    """Read sets.csv → DataFrame indexed by integer Set with a numeric Vc column.
    Mirrors fit_taylor.load_sets so the Vc values match the calibration."""
    s = pd.read_csv(sets_csv)
    s = s.rename(columns={s.columns[0]: "SetLabel"})
    s["Set"] = s["SetLabel"].str.extract(r"(\d+)").astype(int)
    for c in ["Vc", "fz", "Ap", "z"]:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    return s.set_index("Set")


def build_taylor_rate_table(constants_path: str | Path,
                            sets_csv: str | Path) -> tuple[dict[int, float], float, float]:
    """Return ({set_id: k_s µm/pass}, failure_um, wear_cap_um).

    k_s = W_f / (C_mat / Vc_s)^(1/n_mat) for every set found in sets.csv.
    Sets with missing/invalid Vc or material constant get k_s = 0.0.
    """
    payload = json.loads(Path(constants_path).read_text())
    meta = payload["meta"]
    consts = payload["constants"]
    failure_um = float(meta["failure_um"])
    wear_cap_um = float(meta.get("wear_cap_um", 450.0))

    sets = load_sets(sets_csv)
    rates: dict[int, float] = {}
    for s in sets.index.tolist():
        mat = material_of(int(s))
        c = consts.get(mat)
        vc = sets.loc[s, "Vc"] if "Vc" in sets.columns else np.nan
        if c is None or not np.isfinite(vc) or vc <= 0 \
           or c.get("C") is None or c.get("n") in (None, 0):
            rates[int(s)] = 0.0
            continue
        T_taylor = (float(c["C"]) / float(vc)) ** (1.0 / float(c["n"]))   # passes
        rates[int(s)] = float(failure_um / T_taylor) if T_taylor > 0 else 0.0
    return rates, failure_um, wear_cap_um


def anchor_um_np(set_ids: np.ndarray, image_ids: np.ndarray,
                 rates: dict[int, float], failure_um: float) -> np.ndarray:
    """Vectorised VB_taylor (µm) for numpy arrays — used by recommend_C()."""
    k = np.array([rates.get(int(s), 0.0) for s in set_ids], dtype=np.float64)
    return np.clip(k * image_ids.astype(np.float64), 0.0, failure_um)


def recommend_C(labels_csv: str | Path, sets_csv: str | Path,
                constants_path: str | Path, train_sets: Sequence[int],
                percentile: float = 95.0, wear_cap: float = 450.0) -> float:
    """Suggest C (µm) as a percentile of |wear − VB_taylor| over the TRAIN sets.

    Rationale: C should make the band wide enough to contain (almost) all real
    wear so the constraint never *prevents* fitting a genuine value, while still
    excluding gross deviations. Computed on TRAIN only (leakage-free).
    """
    rates, failure_um, _ = build_taylor_rate_table(constants_path, sets_csv)
    df = pd.read_csv(labels_csv)
    df.columns = df.columns.str.strip()
    df["wear"] = pd.to_numeric(df["wear"], errors="coerce").clip(upper=wear_cap)
    df["Set"] = pd.to_numeric(df["Set"], errors="coerce")
    df["ImageID"] = pd.to_numeric(df["ImageID"], errors="coerce")
    df = df.dropna(subset=["wear", "Set", "ImageID"])
    df = df[df["Set"].astype(int).isin(list(train_sets))]
    anc = anchor_um_np(df["Set"].to_numpy(), df["ImageID"].to_numpy(), rates, failure_um)
    resid = np.abs(df["wear"].to_numpy() - anc)
    return float(np.percentile(resid, percentile))


class HardTaylorConstraint(nn.Module):
    """Reparametrise a raw network output into a physics-anchored, band-bounded
    wear prediction (normalised units, /WEAR_DIVISOR).

        VB_pred_norm = clip(k_s·t, 0, W_f)/D  +  (C/D)·tanh(raw)

    forward() returns (pred_norm, correction_um, anchor_um) so the caller can
    log how hard the band is binding (saturation of tanh ⇒ C too small)."""

    def __init__(self, constants_path: str | Path, sets_csv: str | Path,
                 C_um: float, wear_divisor: float = WEAR_DIVISOR):
        super().__init__()
        rates, failure_um, wear_cap_um = build_taylor_rate_table(constants_path, sets_csv)
        self.failure_um = failure_um
        self.wear_cap_um = wear_cap_um
        self.wear_divisor = wear_divisor
        self.C_um = float(C_um)

        max_set = (max(rates) + 1) if rates else 1
        k = np.zeros(max_set, dtype=np.float32)
        for s, v in rates.items():
            if 0 <= s < max_set:
                k[s] = v
        self.register_buffer("_k", torch.from_numpy(k))
        self.rates = rates   # keep python dict for logging/repr

    def forward(self, raw_pred, set_ids, image_ids):
        raw = raw_pred.squeeze(-1) if raw_pred.dim() > 1 else raw_pred
        dev, dt = raw.device, raw.dtype
        if self._k.device != dev:
            self._k = self._k.to(dev)

        if not isinstance(set_ids, torch.Tensor):
            set_ids = torch.as_tensor(set_ids, dtype=torch.long, device=dev)
        sids = set_ids.to(device=dev, dtype=torch.long)
        if not isinstance(image_ids, torch.Tensor):
            image_ids = torch.as_tensor(image_ids, dtype=dt, device=dev)
        iids = image_ids.to(device=dev, dtype=dt)

        in_range = (sids >= 0) & (sids < self._k.shape[0])
        safe = torch.where(in_range, sids, torch.zeros_like(sids))
        k = self._k[safe] * in_range.to(dt)                    # k=0 for OOR sets

        anchor_um = (k * iids).clamp(0.0, self.failure_um)
        corr_um = self.C_um * torch.tanh(raw)
        pred_norm = (anchor_um + corr_um) / self.wear_divisor
        return pred_norm, corr_um, anchor_um

    def extra_repr(self) -> str:
        nz = sum(1 for v in self.rates.values() if v > 0)
        return (f"C_um={self.C_um} failure_um={self.failure_um} "
                f"band=[VB_taylor±{self.C_um}]µm  sets_with_rate={nz}/{len(self.rates)} "
                f"max_reachable={self.failure_um + self.C_um:.0f}µm (cap={self.wear_cap_um:.0f})")
