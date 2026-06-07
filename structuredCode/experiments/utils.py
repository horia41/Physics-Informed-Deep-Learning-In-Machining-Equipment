import random
import numpy as np
import torch
from structuredCode.constants import WEAR_CAP

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed all RNGs. With deterministic=True (default) cuDNN runs in
    reproducible mode so that small MAE differences between ablation runs
    reflect the design change being tested, not run-to-run RNG noise — this
    matters here because conclusions are drawn from 1-7 µm gaps on a
    647-sample dataset. Pass deterministic=False to trade reproducibility
    for the ~5-15% speed-up of cuDNN autotuning.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark     = not deterministic


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker init: seed numpy/random in each worker from torch's
    per-worker base seed so augmentation and any worker-side RNG are
    reproducible across runs (torch seeds its own RNG per worker already).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def predictions_to_um(pred: torch.Tensor) -> torch.Tensor:
    return (pred.squeeze(-1) * 1000.0).clamp(0.0, WEAR_CAP)

def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)
