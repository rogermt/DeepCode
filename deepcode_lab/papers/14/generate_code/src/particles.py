'''src/particles.py
Utility functions for particle integration used in NSGF algorithms.

The module provides a minimal explicit‑Euler integrator for particle trajectories.
It is deliberately lightweight – the heavy lifting (velocity field computation) is
performed elsewhere (e.g., in the training loops).

Functions
---------
- `euler_step(x, v, eta)`: one explicit‑Euler update step.
- `integrate_trajectory(x0, v_func, eta, T)`: integrates a particle cloud for `T`
  steps using the provided velocity function `v_func`.

Both functions operate on PyTorch tensors and are fully differentiable.
''' 

import torch
from typing import Callable


def euler_step(x: torch.Tensor, v: torch.Tensor, eta: float) -> torch.Tensor:
    """Perform a single explicit‑Euler update.

    Parameters
    ----------
    x : torch.Tensor
        Current particle positions, shape ``[B, d]``.
    v : torch.Tensor
        Velocity vectors evaluated at ``x``, same shape as ``x``.
    eta : float
        Step size (learning‑rate like scalar).

    Returns
    -------
    torch.Tensor
        Updated particle positions ``x + eta * v``.
    """
    return x + eta * v


def integrate_trajectory(
    x0: torch.Tensor,
    v_func: Callable[[torch.Tensor, int], torch.Tensor],
    eta: float,
    T: int,
) -> torch.Tensor:
    """Integrate particle positions for ``T`` Euler steps.

    Parameters
    ----------
    x0 : torch.Tensor
        Initial particle positions, shape ``[B, d]``.
    v_func : Callable[[torch.Tensor, int], torch.Tensor]
        Function that returns the velocity field given the current positions and the
        integer time step ``t``.  The signature matches the usage in the paper
        where ``vθ(x, t)`` is evaluated.
    eta : float
        Step size for the explicit Euler scheme.
    T : int
        Number of integration steps.

    Returns
    -------
    torch.Tensor
        Particle positions after ``T`` steps, same shape as ``x0``.
    """
    x = x0.clone()
    for t in range(T):
        v = v_func(x, t)
        x = euler_step(x, v, eta)
    return x

# Optional convenience wrapper used in some training scripts – returns the full
# trajectory (list of positions) if needed.
def integrate_trajectory_with_history(
    x0: torch.Tensor,
    v_func: Callable[[torch.Tensor, int], torch.Tensor],
    eta: float,
    T: int,
):
    """Integrate and record each intermediate state.

    Returns a list ``[x_0, x_1, ..., x_T]`` where ``x_t`` is the particle cloud at
    step ``t``.
    """
    history = [x0.clone()]
    x = x0.clone()
    for t in range(T):
        v = v_func(x, t)
        x = euler_step(x, v, eta)
        history.append(x.clone())
    return history
