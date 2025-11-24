'''nsfg/eval/inference_nsfg.py
Implementation of the NSGF inference pipeline (Algorithm 2).

The core functionality is provided by the `run_nsfg` function, which takes a
learned velocity‑field network `v_theta`, an initial particle batch `X0`, a step
size `eta`, and a number of integration steps `T`.  It iteratively updates the
particles using the explicit Euler scheme described in the paper:

    X_{t+1} = X_t + eta * v_theta(X_t, t)

The function returns the final particle positions `X_T`.

A small command‑line interface is also provided for convenience.  It expects a
checkpoint file that contains a saved `torch.nn.Module` (the velocity‑field
network).  The script creates a Gaussian prior of the requested batch size and
dimensionality, runs the inference, and saves the generated samples to a NumPy
```.npz``` file.

Note: The surrounding package supplies data loaders, model definitions and the
`ode_solver` utilities, but this module only depends on PyTorch and the
`ode_solver` helper for the Euler step.
''' 

import argparse
import os
from typing import Tuple

import torch

from nsfg.utils.ode_solver import euler_step


def run_nsfg(
    v_theta: torch.nn.Module,
    X0: torch.Tensor,
    eta: float = 0.1,
    T: int = 10,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Run NSGF inference (Algorithm 2).

    Parameters
    ----------
    v_theta: torch.nn.Module
        Learned velocity‑field network.  It must accept two arguments ``(x, t)``
        where ``x`` is a tensor of shape ``[batch, *]`` and ``t`` is a 1‑D tensor
        of the same batch size containing the scalar time index.
    X0: torch.Tensor
        Initial particle positions (the prior). Shape ``[batch, dim]`` for
        point‑cloud data or ``[batch, C, H, W]`` for images.
    eta: float, default 0.1
        Fixed step size for the explicit Euler integration.
    T: int, default 10
        Number of integration steps.
    device: torch.device or str, optional
        Device on which to perform the computation.  If ``None`` the device of
        ``X0`` is used.

    Returns
    -------
    torch.Tensor
        Final particle positions after ``T`` Euler steps.
    """
    if device is None:
        device = X0.device
    else:
        device = torch.device(device)

    v_theta = v_theta.to(device)
    v_theta.eval()

    x = X0.to(device)
    batch_size = x.shape[0]

    with torch.no_grad():
        for step in range(T):
            # Time tensor: same scalar for the whole batch
            t_tensor = torch.full((batch_size,), float(step), device=device, dtype=x.dtype)
            # Compute velocity field
            v = v_theta(x, t_tensor)
            # Euler update
            x = x + eta * v
    return x


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSGF inference (Algorithm 2)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a torch checkpoint containing the velocity‑field model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Number of particles to generate (default: 1024).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Dimensionality of the prior (for point‑cloud data).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of Euler steps T (default: 10).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Euler step size (default: 0.1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (cpu or cuda).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="samples.npz",
        help="File to save generated samples (NumPy .npz).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load the model checkpoint.  The checkpoint is expected to contain a full
    # ``torch.nn.Module`` object (saved via ``torch.save(model, path)``).  This
    # keeps the implementation simple and avoids needing to import the exact
    # model class here.
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    v_theta = torch.load(args.checkpoint, map_location=args.device)

    # Create a Gaussian prior X0.
    # For image data the user would need to reshape accordingly; this script
    # focuses on the 2‑D synthetic setting described in the paper.
    X0 = torch.randn(args.batch_size, args.dim)

    # Run inference.
    X_T = run_nsfg(
        v_theta=v_theta,
        X0=X0,
        eta=args.eta,
        T=args.steps,
        device=args.device,
    )

    # Save the generated samples.
    samples = X_T.cpu().numpy()
    np.savez(args.output, samples=samples)
    print(f"Generated {samples.shape[0]} samples saved to {args.output}")


if __name__ == "__main__":
    main()
