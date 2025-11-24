'''nsfg.utils.ode_solver
Utility functions for integrating ordinary differential equations (ODEs) used in the
Neural Sinkhorn Gradient Flow (NSGF) and NSGF++ pipelines.

The module provides simple explicit Euler and classical Runge‑Kutta (RK4) steps
as well as a generic integration routine that repeatedly applies a chosen step
method.

All functions operate on PyTorch tensors and support CPU as well as CUDA
devices. The ``model`` argument is expected to be a callable that takes a
particle tensor ``x`` and a scalar time ``t`` (or a tensor of shape ``[batch]``)
and returns the velocity field (i.e. the time derivative of ``x``).
''' 

from __future__ import annotations

import torch
from typing import Callable, Iterable, List, Tuple

Tensor = torch.Tensor


def euler_step(
    model: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: float,
) -> Tensor:
    """Perform a single explicit Euler step.

    Args:
        model: Callable returning the derivative ``dx/dt`` given ``x`` and ``t``.
        x: Current state tensor of shape ``[batch, ...]``.
        t: Current time tensor of shape ``[batch]`` or a scalar tensor.
        dt: Time step size (float).

    Returns:
        Tensor representing the updated state ``x + dt * model(x, t)``.
    """
    with torch.no_grad():
        dx = model(x, t)
        return x + dt * dx


def rk4_step(
    model: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: float,
) -> Tensor:
    """Perform a single classical Runge‑Kutta (RK4) step.

    Args:
        model: Callable returning the derivative ``dx/dt`` given ``x`` and ``t``.
        x: Current state tensor.
        t: Current time tensor (scalar or per‑sample).
        dt: Time step size.

    Returns:
        Updated state after one RK4 step.
    """
    # Ensure tensors are on the same device and dtype
    dt_tensor = torch.tensor(dt, dtype=x.dtype, device=x.device)
    half_dt = dt_tensor * 0.5
    t = t.clone() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=x.dtype, device=x.device)

    with torch.no_grad():
        k1 = model(x, t)
        k2 = model(x + half_dt * k1, t + half_dt)
        k3 = model(x + half_dt * k2, t + half_dt)
        k4 = model(x + dt_tensor * k3, t + dt_tensor)
        return x + (dt_tensor / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(
    model: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    t0: float,
    t1: float,
    dt: float,
    method: str = "euler",
) -> Tuple[Tensor, List[Tensor]]:
    """Integrate ``dx/dt = model(x, t)`` from ``t0`` to ``t1``.

    Args:
        model: Callable returning the derivative given ``x`` and ``t``.
        x0: Initial state tensor.
        t0: Starting time (float).
        t1: End time (float).
        dt: Step size.
        method: Integration method – ``"euler"`` or ``"rk4"``.

    Returns:
        A tuple ``(x_T, trajectory)`` where ``x_T`` is the final state and
        ``trajectory`` is a list of intermediate states (including the initial
        state). The list can be useful for debugging or visualisation.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    steps = int((t1 - t0) / dt)
    if steps <= 0:
        raise ValueError("Number of integration steps must be positive; check t0, t1 and dt")

    step_fn = {
        "euler": euler_step,
        "rk4": rk4_step,
    }.get(method.lower())
    if step_fn is None:
        raise ValueError(f"Unsupported integration method '{method}'. Use 'euler' or 'rk4'.")

    trajectory: List[Tensor] = []
    x = x0.clone()
    t = torch.full((x.shape[0],), t0, dtype=x.dtype, device=x.device) if x.dim() > 0 else torch.tensor(t0, dtype=x.dtype, device=x.device)
    trajectory.append(x.clone())
    for _ in range(steps):
        x = step_fn(model, x, t, dt)
        t = t + dt
        trajectory.append(x.clone())
    return x, trajectory


__all__ = [
    "euler_step",
    "rk4_step",
    "integrate",
]
