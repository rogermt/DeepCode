# src/inference/nsfg_infer.py
"""
Inference script for Neural Sinkhorn Gradient Flow (NSFG).
Implements Algorithm 2 from the paper: given a trained velocity-field network vθ,
starting from source distribution μ₀, integrate particles using explicit Euler steps
with step size η (eta) for T steps.

The module provides a simple function `nsfg_generate` that returns generated samples
as a torch Tensor. It also includes a command‑line interface for quick testing.
"""

import argparse
import os
from typing import Callable, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from ..utils import set_seed, get_logger
from ..particles import euler_step

# The model is expected to have a forward signature: model(x, t) -> velocity

def nsfg_generate(
    model: torch.nn.Module,
    source_sampler: Callable[[int], torch.Tensor],
    eta: float,
    T: int,
    batch_size: int = 1024,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate samples using the trained NSFG velocity field.

    Args:
        model: Trained velocity‑field network (vθ).
        source_sampler: Callable that returns a tensor of shape (batch_size, d) sampled
            from the source distribution μ₀.
        eta: Euler step size.
        T: Number of integration steps.
        batch_size: Number of particles to generate.
        device: Torch device.

    Returns:
        Tensor of shape (batch_size, d) containing the generated particles after T steps.
    """
    model.eval()
    model.to(device)
    # Sample initial particles
    x = source_sampler(batch_size).to(device)
    # Integrate for T steps
    for t in range(T):
        # time tensor for the model – shape (batch_size, 1)
        t_tensor = torch.full((batch_size, 1), float(t), device=device)
        with torch.no_grad():
            v = model(x, t_tensor)
        x = euler_step(x, v, eta)
    return x.detach().cpu()


def main():
    parser = argparse.ArgumentParser(description="NSFG inference script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="generated.pt", help="File to save generated samples")
    parser.add_argument("--batch_size", type=int, default=1024, help="Number of particles to generate")
    parser.add_argument("--eta", type=float, default=0.2, help="Euler step size (η)")
    parser.add_argument("--T", type=int, default=10, help="Number of integration steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    logger = get_logger(os.path.join(os.path.dirname(args.checkpoint), "logs"))
    logger.add_text("args", str(vars(args)))

    # Load checkpoint – expect it contains 'model_state_dict'
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    # Determine model class from checkpoint metadata if present, else assume UNetVelocity
    # For simplicity we import the generic UNetVelocity and MLPVelocity and try both.
    from src.models import UNetVelocity, MLPVelocity
    # Heuristic: if the checkpoint contains a key 'model_type' we use it.
    model_type = checkpoint.get("model_type", None)
    if model_type == "mlp":
        # infer dimensionality from checkpoint metadata or default to 2
        dim = checkpoint.get("input_dim", 2)
        model = MLPVelocity(dim=dim)
    else:
        # default to UNetVelocity with typical params for image data
        model = UNetVelocity()
    model.load_state_dict(checkpoint["model_state_dict"])

    # Define a simple source sampler – for image data this is uniform noise in [-1,1]
    def source_sampler(n: int) -> torch.Tensor:
        # Assume image shape (C, H, W) = (3, 32, 32) for CIFAR‑10; for MNIST (1, 28, 28)
        # We infer from model's first conv input channels if possible.
        if isinstance(model, UNetVelocity):
            # Use model's in_channels attribute if present
            C = getattr(model, "in_channels", 3)
            H = getattr(model, "image_size", 32)  # fallback
            W = H
            return torch.rand(n, C, H, W) * 2 - 1  # uniform [-1,1]
        else:
            # MLP case – 2‑D points
            return torch.randn(n, dim)

    generated = nsfg_generate(
        model=model,
        source_sampler=source_sampler,
        eta=args.eta,
        T=args.T,
        batch_size=args.batch_size,
        device=args.device,
    )
    torch.save(generated, args.output)
    logger.add_text("status", f"Generated samples saved to {args.output}")
    logger.close()

if __name__ == "__main__":
    main()
