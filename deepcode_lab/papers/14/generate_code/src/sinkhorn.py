'''sinkhorn.py
Implementation of a wrapper around geomloss.SinkhornLoss to compute potentials and their gradients.
'''

import torch
from torch import Tensor
from typing import Tuple

# geomloss provides SamplesLoss which can be used for Sinkhorn divergence.
# We import SinkhornLoss directly if available, otherwise fallback to SamplesLoss.
try:
    from geomloss import SinkhornLoss
except ImportError:
    # Older versions expose SamplesLoss
    from geomloss import SamplesLoss as SinkhornLoss


class SinkhornPotentials:
    """Wrapper for computing Sinkhorn potentials and their gradients.

    Given two empirical measures represented by point clouds ``X`` and ``Y`` (both
    of shape ``[B, d]``), this class computes the Sinkhorn divergence using
    ``geomloss.SinkhornLoss`` with debiasing and provides access to the dual
    potentials ``f`` (on ``X``) and ``g`` (on ``Y``). It also offers a method to
    obtain the gradient of ``f`` with respect to ``X`` which corresponds to the
    gradient of the potential needed for the velocity field.
    """

    def __init__(self, X: Tensor, Y: Tensor, blur: float, scaling: float):
        """Initialize the Sinkhorn potentials computation.

        Args:
            X (Tensor): Source point cloud of shape ``[B, d]``.
            Y (Tensor): Target point cloud of shape ``[B, d]``.
            blur (float): Entropy regularisation parameter (``ε``) passed as ``blur``.
            scaling (float): Scaling factor for the kernel (``σ``).
        """
        if X.requires_grad is False:
            X = X.clone().detach().requires_grad_(True)
        else:
            X = X
        # Ensure Y does not require grad (not needed for gradient of f)
        Y = Y.detach()
        self.X = X
        self.Y = Y
        self.blur = blur
        self.scaling = scaling
        # Initialise the Sinkhorn loss object with debiasing to obtain the Sinkhorn divergence.
        self.sinkhorn = SinkhornLoss(eps=blur, kernel_size=scaling, debias=True)
        # Compute loss (forces potentials to be computed internally)
        # The loss value itself is not used directly here.
        _ = self.sinkhorn(self.X, self.Y)
        # Retrieve the potential functions.
        # ``potential_fn`` returns a tuple of callables (f, g) that map points to potentials.
        self.f_fn, self.g_fn = self.sinkhorn.potential_fn()
        # Evaluate potentials on the provided point clouds.
        self.f = self.f_fn(self.X)
        self.g = self.g_fn(self.Y)

    def potential_X(self) -> Tensor:
        """Return the potential ``f`` evaluated on ``X``.
        """
        return self.f

    def potential_Y(self) -> Tensor:
        """Return the potential ``g`` evaluated on ``Y``.
        """
        return self.g

    def grad_X(self) -> Tensor:
        """Compute the gradient of the potential ``f`` with respect to ``X``.

        Returns:
            Tensor: Gradient of shape ``[B, d]``.
        """
        # ``self.f`` is a scalar per sample (shape [B]), we need gradient w.r.t each point.
        # Use torch.autograd.grad with create_graph=False.
        grads = torch.autograd.grad(
            outputs=self.f,
            inputs=self.X,
            grad_outputs=torch.ones_like(self.f),
            retain_graph=False,
            create_graph=False,
        )[0]
        return grads


def compute_potentials(
    X: Tensor, Y: Tensor, blur: float, scaling: float
) -> Tuple[Tensor, Tensor]:
    """Convenience function to compute Sinkhorn potentials.

    Args:
        X (Tensor): Source point cloud ``[B, d]``.
        Y (Tensor): Target point cloud ``[B, d]``.
        blur (float): Entropy regularisation parameter.
        scaling (float): Kernel scaling factor.

    Returns:
        Tuple[Tensor, Tensor]: ``(f, g)`` where ``f`` are potentials on ``X`` and ``g`` on ``Y``.
    """
    sp = SinkhornPotentials(X, Y, blur, scaling)
    return sp.potential_X(), sp.potential_Y()
