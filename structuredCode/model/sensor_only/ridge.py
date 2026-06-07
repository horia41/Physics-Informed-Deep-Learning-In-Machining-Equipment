import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from constants import WEAR_CAP

def fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval:  np.ndarray,
    alpha:   float = 1.0,
) -> tuple[np.ndarray, Ridge, StandardScaler]:
    """Standardise → Ridge → predict; clipped to [0, WEAR_CAP].

    NaN-safe: any NaN columns (e.g. cutting parameters for Set 1) are
    replaced with the per-column training mean before standardisation.
    """
    # Impute NaN per column with train mean (fall back to 0 if all-NaN)
    col_means = np.nanmean(X_train, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_train_i = np.where(np.isnan(X_train), col_means, X_train)
    X_eval_i  = np.where(np.isnan(X_eval),  col_means, X_eval)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train_i)
    X_ev   = scaler.transform(X_eval_i)

    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_train)
    pred = model.predict(X_ev).clip(0.0, WEAR_CAP)
    return pred, model, scaler