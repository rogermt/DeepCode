# nsfg/eval/inference_nsfgpp.py
"""
Inference pipeline for NSGF++ (Algorithm 4).
Implements the two‑phase generation:
  1. Run the learned velocity‑field network v_θ for a small number of steps (T ≤ 5)
     to obtain an intermediate particle set \hat{X}.
  2. Predict the phase‑transition time τ using the trained predictor t_ϕ.
  3. Compute the number of refinement steps N = ceil((1-τ)/ω) where ω is the
     step size for the straight‑flow network u_δ.
  4. Integrate u_δ from τ to 1 using explicit Euler (or RK4) with N steps.

The module provides a ``run_nsfgpp`` function that can be called from scripts or
notebooks and a small CLI for end‑to‑end generation.
"""

import argparse
import os
from typing import Tuple

import torch

# Local imports
from nsfg.utils.ode_solver import euler_step, rk4_step, integrate
from nsfg.models.predictor_cnn import PredictorCNN
from nsfg.models.unet import UNet

__all__ = ["run_nsfgpp"]


def run_nsfgpp(
    v_theta: torch.nn.Module,
    u_delta: torch.nn.Module,
    predictor_phi: torch.nn.Module,
    X0: torch.Tensor,
    eta: float = 0.1,
    omega: float = 0.1,
    T: int = 5,
    device: torch.device | str | None = None,
    method: str = "euler",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the NSGF++ inference pipeline.

    Parameters
    ----------
    v_theta: torch.nn.Module
        Learned velocity‑field network (phase 1).
    u_delta: torch.nn.Module
        Learned straight‑flow network (phase 2).
    predictor_phi: torch.nn.Module
        Small CNN that predicts the transition time τ given the intermediate
        particles after phase 1.
    X0: torch.Tensor
        Prior particles of shape ``[batch, dim]`` (or ``[batch, C, H, W]`` for
        images).  The tensor is assumed to be on ``cpu``; it will be moved to the
        selected ``device``.
    eta: float, default 0.1
        Step size for the Euler integration of ``v_theta``.
    omega: float, default 0.1
        Step size for the straight‑flow integration of ``u_delta``.
    T: int, default 5
        Number of Euler steps for the first phase (must be ≤5 as described in
        the paper).
    device: torch.device | str | None, default None
        Device on which to run the computation.  If ``None`` the function will
        use ``torch.cuda.current_device()`` when a GPU is available, otherwise
        ``cpu``.
    method: str, default "euler"
        Integration method for the second phase – ``"euler"`` or ``"rk4"``.

    Returns
    -------
    X_T: torch.Tensor
        Final generated samples after the two‑phase procedure.
    tau: torch.Tensor
        Predicted transition time (scalar tensor) for each sample in the batch.
    """
    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Move models to device and set eval mode
    for model in (v_theta, u_delta, predictor_phi):
        model.to(device)
        model.eval()

    X = X0.to(device)
    batch_size = X.shape[0]

    # ------------------------------------------------------------------
    # Phase 1 – NSGF (few Euler steps)
    # ------------------------------------------------------------------
    with torch.no_grad():
        t = torch.zeros(batch_size, device=device, dtype=X.dtype)
        for step in range(T):
            # v_theta expects (x, t) where t is a tensor of shape [batch]
            v = v_theta(X, t)
            X = X + eta * v
            t = t + 1  # increment integer time (used only for conditioning)

        X_hat = X.clone()  # intermediate particles after phase 1

        # ------------------------------------------------------------------
        # Phase 2 – Predict transition time τ
        # ------------------------------------------------------------------
        # The predictor receives the intermediate particles and outputs a
        # scalar in (0, 1) per sample.  The implementation of ``PredictorCNN``
        # follows the plan (global average‑pool → linear → sigmoid).
        tau = predictor_phi(X_hat)
        # Clamp to a safe range to avoid division‑by‑zero later.
        tau = torch.clamp(tau, min=1e-6, max=1.0 - 1e-6)

        # ------------------------------------------------------------------
        # Phase 3 – Straight‑flow integration from τ to 1
        # ------------------------------------------------------------------
        # Number of refinement steps N = ceil((1-τ)/omega).  Because ``tau`` is a
        # tensor we compute a per‑sample N and then take the maximum to keep the
        # loop simple (all samples are advanced with the same number of steps).
        N_per = torch.ceil((1.0 - tau) / omega).long()
        N = int(N_per.max().item())

        # Prepare time tensor for the straight‑flow network.  The network expects
        # a scalar time ``t`` broadcastable to the batch dimension.  We will
        # increment ``t`` by ``omega`` each step starting from ``tau``.
        t_sf = tau.clone()
        for i in range(N):
            # Compute the straight‑flow velocity
            u = u_delta(X, t_sf)
            X = X + omega * u
            t_sf = t_sf + omega
            # Ensure we do not exceed 1.0 due to rounding errors.
            t_sf = torch.clamp(t_sf, max=1.0)

    # Return final particles and the predicted transition time (per‑sample).
    return X, tau


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NSGF++ inference (Algorithm 4)")
    parser.add_argument("--vtheta-ckpt", type=str, required=True, help="Path to v_theta checkpoint (.pt)")
    parser.add_argument("--udelta-ckpt", type=str, required=True, help="Path to u_delta checkpoint (.pt)")
    parser.add_argument("--phi-ckpt", type=str, required=True, help="Path to predictor_phi checkpoint (.pt)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Number of samples to generate")
    parser.add_argument("--dim", type=int, default=2, help="Dimensionality of the prior (2 for synthetic, 784 for MNIST, 3072 for CIFAR‑10)")
    parser.add_argument("--steps", type=int, default=5, help="Number of NSGF steps (T ≤ 5)")
    parser.add_argument("--eta", type=float, default=0.1, help="Step size for NSGF phase")
    parser.add_argument("--omega", type=float, default=0.1, help="Step size for straight‑flow phase")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., cuda:0 or cpu)")
    parser.add_argument("--output", type=str, default="samples_nsfgpp.npz", help="File to save generated samples")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load models – the checkpoints are assumed to contain the state_dict only.
    # Model architectures are reconstructed with default hyper‑parameters as
    # defined in the corresponding modules.
    # NOTE: The exact architecture (e.g., UNet depth) must match the training
    # configuration; for simplicity we instantiate with the defaults which are
    # compatible with the synthetic 2‑D experiments.  Users can adapt the code
    # for image experiments by editing the imports.
    v_theta = UNet()
    v_theta.load_state_dict(torch.load(args.vtheta_ckpt, map_location="cpu"))

    u_delta = UNet()
    u_delta.load_state_dict(torch.load(args.udelta_ckpt, map_location="cpu"))

    predictor_phi = PredictorCNN()
    predictor_phi.load_state_dict(torch.load(args.phi_ckpt, map_location="cpu"))

    # Prior generation – simple Gaussian with zero mean and unit variance.
    # For image data the dimensionality would be C×H×W; here we treat the prior
    # as a flat vector and reshape later if needed.
    prior = torch.randn(args.batch_size, args.dim)

    samples, tau = run_nsfgpp(
        v_theta=v_theta,
        u_delta=u_delta,
        predictor_phi=predictor_phi,
        X0=prior,
        eta=args.eta,
        omega=args.omega,
        T=args.steps,
        device=args.device,
        method="euler",
    )

    # Save results – ``samples`` is a tensor; we store it as a NumPy array for
    # easy downstream processing.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"samples": samples.cpu(), "tau": tau.cpu()}, args.output)
    print(f"Saved {samples.shape[0]} samples to {args.output}")


if __name__ == "__main__":
    main()
