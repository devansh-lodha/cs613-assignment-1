"""Reproducibility utilities for deterministic model inference and evaluation."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Configure deterministic random seeds across all supported execution backends.

    This function synchronizes seed values across:
    1. Python built-in `random` module.
    2. NumPy global random generator (`numpy.random`).
    3. PyTorch CPU manual seed (`torch.manual_seed`).
    4. PyTorch CUDA manual seed for all available GPUs (`torch.cuda.manual_seed_all`).
    5. Python runtime hash seed (`PYTHONHASHSEED`) for reproducible dictionary ordering.
    6. cuDNN deterministic algorithm selection to disable non-deterministic heuristics.

    Args:
        seed: Integer value applied as the pseudorandom baseline.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
