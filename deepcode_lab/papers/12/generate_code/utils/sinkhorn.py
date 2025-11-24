'''utils/sinkhorn.py
Utility module wrapping GeomLoss Sinkhorn potentials.

Provides a `SinkhornEngine` class that computes dual potentials `f` and `g`
for source and target point clouds. The potentials are differentiable and can
be used to obtain empirical velocity fields via their gradients.
''' 

import torch
from dataclasses import dataclass
from typing import Tuple

try:
    # GeomLoss provides the SamplesLoss class
    from geomloss import SamplesLoss
except ImportError as e:
    raise ImportError(
        "geomloss is required for Sinkhorn potentials. Install it via "
        "`pip install geomloss`."
    ) from e


@dataclass
class SinkhornEngine:
    """Engine for computing Sinkhorn dual potentials.

    Parameters
    ----------
    epsilon: float, default 0.1
        Entropy regularisation parameter.
    blur: float, default 1.0
        Gaussian blur parameter for the Sinkhorn loss.
    scaling: float, default 0.9
        Scaling factor for the Sinkhorn divergence.
    device: torch.device or str, default "cpu"
        Device on which computations are performed.
    """

    epsilon: float = 0.1
    blur: float = 1.0
    scaling: float = 0.9
    device: str = "cpu"

    def __post_init__(self):
        self.device = torch.device(self.device)
        # Initialise the GeomLoss SamplesLoss for Sinkhorn divergence
        self.loss_fn = SamplesLoss(
            "sinkhorn",
            blur=self.blur,
            scaling=self.scaling,
            epsilon=self.epsilon,
            backend="auto",
        )

    def compute_potentials(
        self, x_src: torch.Tensor, x_tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute dual potentials for source and target point clouds.

        Parameters
        ----------
        x_src: torch.Tensor of shape (N, d)
            Source samples.
        x_tgt: torch.Tensor of shape (M, d)
            Target samples.

        Returns
        -------
        f_src: torch.Tensor of shape (N,)
            Dual potential evaluated at source points for the self‑potential.
        f_tgt: torch.Tensor of shape (M,)
            Dual potential evaluated at target points for the self‑potential.
        g_src: torch.Tensor of shape (N,)
            Dual potential evaluated at source points for the cross‑potential.
        g_tgt: torch.Tensor of shape (M,)
            Dual potential evaluated at target points for the cross‑potential.
        """
        x_src = x_src.to(self.device)
        x_tgt = x_tgt.to(self.device)

        # Self‑potential: µ_t vs µ_t (both are x_src)
        # GeomLoss provides a method `potential` that returns the dual potentials
        # for the two point clouds. We call it with the same point set for both
        # arguments to obtain the self‑potential.
        f_src, f_tgt = self.loss_fn.potential(x_src, x_src)

        # Cross‑potential: µ_t vs µ*
        g_src, g_tgt = self.loss_fn.potential(x_src, x_tgt)

        # Ensure gradients flow through the potentials
        f_src = f_src.squeeze()
        f_tgt = f_tgt.squeeze()
        g_src = g_src.squeeze()
        g_tgt = g_tgt.squeeze()

        return f_src, f_tgt, g_src, g_tgt

    # Convenience method to compute the empirical velocity field ∇f_self - ∇f_cross
    def empirical_velocity(self, x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Compute the empirical velocity field at points `x`.

        This uses the gradients of the self‑potential and cross‑potential.
        """
        x = x.requires_grad_(True)
        f_src, _, g_src, _ = self.compute_potentials(x, x_target)
        # Gradient of f_src w.r.t. x
        grad_f = torch.autograd.grad(f_src.sum(), x, create_graph=True)[0]
        # Gradient of g_src w.r.t. x
        grad_g = torch.autograd.grad(g_src.sum(), x, create_graph=True)[0]
        # Empirical velocity as defined in the paper (Eq. 13)
        v_hat = grad_f - grad_g
        return v_hat

# Example usage (for debugging / unit‑test purposes)
if __name__ == "__main__":
    torch.manual_seed(0)
    src = torch.randn(8, 2)
    tgt = torch.randn(8, 2)
    engine = SinkhornEngine(epsilon=0.1, blur=1.0, scaling=0.9, device="cpu")
    f_src, f_tgt, g_src, g_tgt = engine.compute_potentials(src, tgt)
    print("Potentials shapes:", f_src.shape, f_tgt.shape, g_src.shape, g_tgt.shape)
    v = engine.empirical_velocity(src, tgt)
    print("Velocity shape:", v.shape)
