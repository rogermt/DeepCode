# Evaluation package initialization
"""nsfg.eval

Convenient import shortcuts for the evaluation utilities.

The module re‑exports the most commonly used functions and classes from
`nsfg.eval.metrics` so that users can simply write::

    from nsfg.eval import wasserstein2_distance, compute_fid, compute_inception_score,
        evaluate_generated, InceptionFeatureExtractor

All symbols are imported lazily at module import time.
"""

from .metrics import (
    wasserstein2_distance,
    compute_fid,
    compute_inception_score,
    evaluate_generated,
    InceptionFeatureExtractor,
)

__all__ = [
    "wasserstein2_distance",
    "compute_fid",
    "compute_inception_score",
    "evaluate_generated",
    "InceptionFeatureExtractor",
]
