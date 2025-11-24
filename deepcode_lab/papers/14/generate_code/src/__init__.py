# src package initialization
"""Top-level package for Neural Sinkhorn Gradient Flow (NSGF) implementation.

This module sets up the package namespace and provides convenient imports for
common submodules such as utilities, models, training utilities, and inference
functions.
"""

# Expose key subpackages and modules for easy access
from . import utils
from . import models
from . import training
from . import inference
from . import sinkhorn
from . import particles

__all__ = [
    "utils",
    "models",
    "training",
    "inference",
    "sinkhorn",
    "particles",
]
