

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from constants.matwi_dataset_constants import WEAR_DIVISOR, material_of
from dataset.loader import load_sets_v2

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

    sets = load_sets_v2(sets_csv)
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
