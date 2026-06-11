import numpy as np

from dataset.features import apply_delta, compute_set_baselines
from metrics.mae import compute_mae_v2
from model.sensor_only.lgbm import fit_predict_lgbm
from model.sensor_only.ridge import fit_predict_ridge

# ═════════════════════════════════════════════════════════════════════════════
# Leave-one-set-out CV
# ═════════════════════════════════════════════════════════════════════════════

def loso_cv(
    X:          np.ndarray,
    y:          np.ndarray,
    types:      np.ndarray,
    sets:       np.ndarray,
    sensor_ids: np.ndarray,
    use_delta:  bool,
    model_kind: str,     # "ridge" or "lgbm"
    seed:       int = 42,
) -> dict:
    """
    Leave-one-set-out on the provided sample matrix (typically training sets).

    Per fold:
      - Hold out set s
      - If use_delta: recompute baselines using only the remaining sets that
        appear in the in-fold data + the held-out set itself (so the
        held-out set still has its own setup baseline; no label leakage)
      - Train on remaining samples, predict on the held-out set
    """
    unique_sets = np.unique(sets)
    fold_results = []
    fold_preds   = np.full_like(y, fill_value=np.nan, dtype=np.float64)

    for s in unique_sets:
        ho   = sets == s
        keep = ~ho
        if not ho.any() or not keep.any():
            continue

        X_tr_raw, X_te_raw = X[keep], X[ho]
        y_tr,    y_te      = y[keep], y[ho]

        if use_delta:
            # Baselines on training portion + held-out set
            # (held-out uses its own SensorID, never its labels)
            base_tr = compute_set_baselines(X_tr_raw, sets[keep], sensor_ids[keep])
            base_te = compute_set_baselines(X_te_raw, sets[ho],   sensor_ids[ho])
            X_tr    = apply_delta(X_tr_raw, sets[keep], base_tr)
            X_te    = apply_delta(X_te_raw, sets[ho],   base_te)
        else:
            X_tr, X_te = X_tr_raw, X_te_raw

        if model_kind == "ridge":
            pred, _, _ = fit_predict_ridge(X_tr, y_tr, X_te)
        elif model_kind == "lgbm":
            # fit_predict_lgbm now carves its own in-distribution ES slice.
            pred, _ = fit_predict_lgbm(X_tr, y_tr, X_te, seed=seed + int(s))
        else:
            raise ValueError(f"Unknown model_kind={model_kind}")

        fold_preds[ho] = pred
        m = compute_mae_v2(y_te, pred, types[ho])
        m["set"] = int(s)
        fold_results.append(m)

    # Aggregate (sample-weighted) across folds — the per-fold "overall" is
    # already a per-sample average within a fold, so pool predictions.
    valid = ~np.isnan(fold_preds)
    overall = compute_mae_v2(y[valid], fold_preds[valid].astype(np.float32), types[valid])
    return {"per_fold": fold_results, "pooled": overall}
