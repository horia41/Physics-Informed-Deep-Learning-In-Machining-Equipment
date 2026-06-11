import numpy as np

from constants.matwi_dataset_constants import WEAR_CAP

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

def fit_predict_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval:  np.ndarray,
    seed:    int = 42,
    es_frac: float = 0.15,
) -> tuple[np.ndarray, "lgb.Booster"]:
    """
    LightGBM regression with early stopping on an IN-DISTRIBUTION slice of the
    training data. Uses small-data-friendly hyperparams.

    NOTE (bug fix): the previous version early-stopped on the passed val set
    (sets 3/6/12), which is out-of-distribution (RVS 304) relative to the
    CK45-heavy train. Val MAE plateaued instantly, so the model stopped after
    2-7 boosting rounds and never trained — drowning out any feature-quality
    differences. We now carve a random `es_frac` slice from the training data
    itself for early stopping, so the stopping signal is in-distribution and
    the model actually fits.
    """
    params = dict(
        objective         = "regression_l1",
        metric            = "mae",
        learning_rate     = 0.03,
        num_leaves        = 31,
        max_depth         = 5,
        min_data_in_leaf  = 10,
        feature_fraction  = 0.8,
        bagging_fraction  = 0.8,
        bagging_freq      = 1,
        lambda_l2         = 1.0,
        verbose           = -1,
        seed              = seed,
    )

    # Carve an in-distribution early-stopping slice from train.
    n      = len(y_train)
    rng    = np.random.RandomState(seed)
    order  = rng.permutation(n)
    n_es   = max(30, int(es_frac * n))
    es_idx = order[:n_es]
    fit_idx = order[n_es:]

    dtr  = lgb.Dataset(X_train[fit_idx], label=y_train[fit_idx])
    dval = lgb.Dataset(X_train[es_idx],  label=y_train[es_idx], reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round   = 3000,
        valid_sets        = [dval],
        callbacks         = [lgb.early_stopping(150, verbose=False),
                             lgb.log_evaluation(0)],
    )
    pred = model.predict(X_eval, num_iteration=model.best_iteration)
    return pred.clip(0.0, WEAR_CAP), model
