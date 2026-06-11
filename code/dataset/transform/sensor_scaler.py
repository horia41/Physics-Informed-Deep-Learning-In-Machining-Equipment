import pickle
from pathlib import Path
from typing import  Optional
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Sensor scaler
# ══════════════════════════════════════════════════════════════════════════════

class SensorScaler:
    """
    Simple zero-mean, unit-std standardiser for sensor feature vectors.
    Fit on training data only, applied to all splits.

    Always fits on all 40 features. Feature subset selection (raw25, top25)
    happens downstream — since standardisation is per-feature, selecting
    25 columns from a 40-feature scaler gives identical results to fitting
    a 25-feature scaler on those 25 columns.
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, features: np.ndarray) -> "SensorScaler":
        """features: (N, 40) array of raw sensor features from training set."""
        self.mean_ = features.mean(axis=0)
        self.std_  = features.std(axis=0)
        # Avoid division by zero for constant features
        self.mean_ = np.nan_to_num(self.mean_, nan=0.0)
        self.std_  = np.nan_to_num(self.std_, nan=1.0)
        self.std_[self.std_ < 1e-8] = 1.0
        self.fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("SensorScaler has not been fitted. Call fit() first.")
        return ((features - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_}, f)
        print(f"[SensorScaler] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SensorScaler":
        with open(path, "rb") as f:
            data = pickle.load(f)
        scaler        = cls()
        scaler.mean_  = data["mean"]
        scaler.std_   = data["std"]
        scaler.fitted = True
        print(f"[SensorScaler] Loaded from {path}")
        return scaler

    def __repr__(self) -> str:
        if not self.fitted:
            return "SensorScaler(not fitted)"
        return (f"SensorScaler(fitted, n_features={len(self.mean_)}, "
                f"mean_range=[{self.mean_.min():.3f}, {self.mean_.max():.3f}])")
