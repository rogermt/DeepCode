'''Model package initialization.

This module exposes the primary neural network classes used throughout the
project so they can be imported directly via ``src.models``.

Available symbols:
- ``MLPVelocity`` – three‑layer MLP for 2‑D synthetic experiments.
- ``UNetVelocity`` – UNet architecture for image‑based velocity fields.
- ``UNetStraightFlow`` – identical UNet used for the straight‑flow network.
- ``sinusoidal_time_embedding`` – utility for sinusoidal time embeddings.
''' 

from .mlp import MLPVelocity
from .unet import (
    UNetVelocity,
    UNetStraightFlow,
    sinusoidal_time_embedding,
)

__all__ = [
    "MLPVelocity",
    "UNetVelocity",
    "UNetStraightFlow",
    "sinusoidal_time_embedding",
]
