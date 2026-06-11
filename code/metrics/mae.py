from __future__ import annotations
import math
from typing import Any
import numpy as np

from constants.matwi_dataset_constants import WEAR_TYPES

def compute_mae_v1(
    pred_um:   np.ndarray,
    target_um: np.ndarray,
    types:     list[str],
) -> dict[str, Any]:
    abs_err = np.abs(pred_um - target_um)
    out: dict[str, Any] = {
        "mae_overall_um": float(abs_err.mean()) if len(abs_err) else math.nan,
        "n_total":        len(target_um),
    }
    for wt in WEAR_TYPES:
        mask = np.array([t == wt for t in types], dtype=bool)
        out[f"mae_{wt}_um"] = float(abs_err[mask].mean()) if mask.any() else math.nan
        out[f"n_{wt}"]      = int(mask.sum())
    return out

def compute_mae_v2(y_true, y_pred, types):
    """Compute overall and per-wear-type MAE."""
    abs_err = np.abs(y_true - y_pred)
    results = {"overall": float(abs_err.mean()), "n": len(y_true)}
    for wt in WEAR_TYPES:
        mask = np.array([t == wt for t in types])
        if mask.any():
            results[wt] = float(abs_err[mask].mean())
            results[f"n_{wt}"] = int(mask.sum())
        else:
            results[wt] = float("nan")
            results[f"n_{wt}"] = 0
    return results

