from __future__ import annotations
import random
import numpy as np
import torch

from constants.matwi_dataset_constants import WEAR_CAP

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed all RNGs. With deterministic=True (default) cuDNN runs in
    reproducible mode so small MAE differences between fusion configs reflect
    the design change, not run-to-run RNG noise. Pass deterministic=False to
    trade reproducibility for cuDNN autotuning speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark     = not deterministic


def seed_worker(worker_id: int) -> None:
    """Seed numpy/random per DataLoader worker from torch's base seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def predictions_to_um(pred: torch.Tensor) -> torch.Tensor:
    """Convert normalised predictions to µm, clamped to [0, WEAR_CAP]."""
    return (pred.squeeze(-1) * 1000.0).clamp(0.0, WEAR_CAP)

def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)