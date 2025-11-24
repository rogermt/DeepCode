'''nsfgpp_infer.py
Implementation of the NSGF++ inference pipeline (Algorithm 4).
Provides a function `nsfgpp_generate` that runs the two‑phase generation:
  1. NSGF phase – integrates particles using the learned velocity field `v_model` for up to `T` steps.
  2. Transition‑time prediction – per‑sample transition time `τ` is predicted by `t_model`.
  3. Neural straight‑flow refinement – applies the straight‑flow network `u_model`
     from `τ` to time 1 using explicit Euler steps of size `omega_nsf`.
A simple CLI entry point is also provided for quick generation from the command line.
''' 

import argparse
import os
from typing import Callable

import torch
from torch.utils.tensorboard import SummaryWriter

from ..utils import set_seed, get_logger
from ..particles import euler_step
from ..models import UNetVelocity, UNetStraightFlow, TimePredictor, MLPVelocity


def nsfgpp_generate(
    v_model: torch.nn.Module,
    t_model: torch.nn.Module,
    u_model: torch.nn.Module,
    source_sampler: Callable[[int], torch.Tensor],
    eta_nsfg: float,
    omega_nsf: float,
    T: int,
    batch_size: int = 1024,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate samples using the NSGF++ two‑phase pipeline.

    Parameters
    ----------
    v_model: torch.nn.Module
        Velocity‑field network `v_θ`.
    t_model: torch.nn.Module
        Phase‑transition predictor `t_φ`.
    u_model: torch.nn.Module
        Neural straight‑flow network `u_δ`.
    source_sampler: Callable[[int], torch.Tensor]
        Function that returns a batch of source particles on the given device.
    eta_nsfg: float
        Euler step size for the NSGF phase.
    omega_nsf: float
        Euler step size for the straight‑flow (NSF) phase.
    T: int
        Number of NSGF steps (≤ 5 for NSGF++ as per the paper).
    batch_size: int, default 1024
        Number of particles to generate.
    device: str, default "cpu"
        Device on which computation is performed.

    Returns
    -------
    torch.Tensor
        Generated samples on CPU, shape ``(batch_size, ...)`` matching the source sampler.
    """
    # Ensure models are on the correct device and in eval mode
    v_model.to(device).eval()
    t_model.to(device).eval()
    u_model.to(device).eval()

    # Sample initial particles
    X = source_sampler(batch_size).to(device)

    # ---------- Phase 1: NSGF ----------
    for t_step in range(T):
        # Time tensor of shape (batch_size, 1) – same for all particles at this step
        t_tensor = torch.full((batch_size, 1), float(t_step), device=device)
        with torch.no_grad():
            v = v_model(X, t_tensor)
        X = euler_step(X, v, eta_nsfg)

    # ---------- Phase 2: Predict transition time τ ----------
    with torch.no_grad():
        tau = t_model(X)  # Expected shape (batch_size, 1)
    # Clamp to [0, 1] for safety
    tau = torch.clamp(tau, 0.0, 1.0)

    # ---------- Phase 3: Neural Straight Flow (NSF) ----------
    # Determine the maximum number of refinement steps needed across the batch
    remaining = 1.0 - tau.squeeze(1)  # shape (batch_size,)
    max_steps = torch.ceil(remaining.max() / omega_nsf).int().item()

    for s in range(max_steps):
        # Current time for each particle: τ + s * ω
        cur_t = tau + s * omega_nsf
        # Mask particles that have already reached or passed t=1
        mask = (cur_t < 1.0).float()
        if mask.sum() == 0:
            break
        # Ensure shape (batch_size, 1)
        cur_t = cur_t * mask + (1.0 - mask)  # keep dummy value for masked entries
        with torch.no_grad():
            u = u_model(X, cur_t)
        # Apply Euler update only where mask == 1
        X = X + omega_nsf * u * mask.unsqueeze(-1)

    return X.detach().cpu()


def _default_source_sampler(batch_size: int, device: str = "cpu"):
    """Default source sampler used by the CLI.

    For image‑based models (UNet) we assume a 3‑channel 32×32 image with values in ``[-1, 1]``.
    For MLP‑based models (2‑D synthetic) we return a standard normal vector of dimension 2.
    The function inspects the environment variable ``SOURCE_TYPE`` to decide.
    """
    source_type = os.getenv("SOURCE_TYPE", "image")
    if source_type == "synthetic":
        # 2‑D synthetic data – standard normal
        return torch.randn(batch_size, 2, device=device)
    else:
        # Image data – uniform noise in [-1, 1]
        return (torch.rand(batch_size, 3, 32, 32, device=device) * 2.0 - 1.0


def main():
    parser = argparse.ArgumentParser(description="NSGF++ inference (two‑phase generation)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint containing model weights and metadata.")
    parser.add_argument("--output", type=str, default="generated.pt", help="File to save generated samples.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eta", type=float, default=0.2, help="Euler step size for NSGF phase.")
    parser.add_argument("--omega", type=float, default=0.02, help="Euler step size for NSF phase.")
    parser.add_argument("--T", type=int, default=5, help="Number of NSGF steps (≤5 for NSGF++).")
    args = parser.parse_args()

    set_seed(42)
    logger: SummaryWriter = get_logger(os.path.join(os.path.dirname(args.checkpoint), "logs"))
    logger.add_text("args", str(vars(args)))

    # Load checkpoint – we expect a dict with keys: 'v_model', 't_model', 'u_model', 'model_type'
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_type = checkpoint.get("model_type", "unet")

    if model_type == "mlp":
        v_model = MLPVelocity(dim=2)
        t_model = TimePredictor(in_channels=1)  # placeholder – not used for 2‑D synthetic
        u_model = MLPVelocity(dim=2)  # reuse MLP architecture for straight flow
    else:
        # Image models – infer channels from checkpoint if available, else defaults
        in_ch = checkpoint.get("in_channels", 3)
        v_model = UNetVelocity(in_channels=in_ch, out_channels=in_ch, base_channels=32, depth=1)
        t_model = TimePredictor(in_channels=in_ch)
        u_model = UNetStraightFlow(in_channels=in_ch, out_channels=in_ch, base_channels=32, depth=1)

    # Load state dicts if present
    if "v_state_dict" in checkpoint:
        v_model.load_state_dict(checkpoint["v_state_dict"])
    if "t_state_dict" in checkpoint:
        t_model.load_state_dict(checkpoint["t_state_dict"])
    if "u_state_dict" in checkpoint:
        u_model.load_state_dict(checkpoint["u_state_dict"])

    # Define source sampler – can be overridden via env var SOURCE_TYPE
    sampler = lambda n: _default_source_sampler(n, device=args.device)

    samples = nsfgpp_generate(
        v_model=v_model,
        t_model=t_model,
        u_model=u_model,
        source_sampler=sampler,
        eta_nsfg=args.eta,
        omega_nsf=args.omega,
        T=args.T,
        batch_size=args.batch_size,
        device=args.device,
    )

    torch.save(samples, args.output)
    logger.add_text("status", f"Generated {samples.shape[0]} samples saved to {args.output}")
    logger.close()
    print(f"Samples saved to {args.output}")


if __name__ == "__main__":
    main()
