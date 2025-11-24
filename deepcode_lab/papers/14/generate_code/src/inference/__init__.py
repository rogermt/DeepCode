# Inference package initialization

"""Convenient imports for the inference submodule.

Provides easy access to the primary generation functions for Neural Sinkhorn
Gradient Flow (NSFG) and its NSGF++ variant.
"""

from .nsfg_infer import nsfg_generate, main as nsfg_main
from .nsfgpp_infer import nsfgpp_generate, main as nsfgpp_main

__all__ = [
    "nsfg_generate",
    "nsfg_main",
    "nsfgpp_generate",
    "nsfgpp_main",
]
