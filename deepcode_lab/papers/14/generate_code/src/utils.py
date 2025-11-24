import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seeds for reproducibility.
    Args:
        seed (int): Seed value.
        deterministic (bool): If True, set torch.backends.cudnn.deterministic and
            torch.backends.cudnn.benchmark to False for deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Ensure reproducibility for torch's random number generator
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_logger(log_dir: str = "logs") -> SummaryWriter:
    """Create a TensorBoard SummaryWriter.
    Args:
        log_dir (str): Directory where TensorBoard logs will be saved.
    Returns:
        SummaryWriter: TensorBoard writer instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer
