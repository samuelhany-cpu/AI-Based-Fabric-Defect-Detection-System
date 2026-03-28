from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional during non-training setup
    torch = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
