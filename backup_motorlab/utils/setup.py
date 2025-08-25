import random

import numpy as np
import torch


torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def fix_seed(seed: int = 0) -> None:
    """
    Fix random seed for reproducibility across random, numpy, and torch.

    Parameters
    ----------
    seed : int, optional
        Random seed value. Default is 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if DEVICE.type == "mps":
        torch.mps.manual_seed(seed)
