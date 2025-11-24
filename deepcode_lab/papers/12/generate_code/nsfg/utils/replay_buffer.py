'''nsfg.utils.replay_buffer
================================
Implementation of a simple circular replay buffer used by the NSFG training
pipeline. The buffer stores tuples ``(x, t, v_hat)`` where

* ``x`` – particle positions (tensor of shape ``[batch, *]``)
* ``t`` – scalar time associated with each particle (tensor of shape ``[batch]``)
* ``v_hat`` – empirical velocity field (same shape as ``x``)

The buffer works in a FIFO manner: once the maximum capacity is reached the
oldest entries are discarded. It provides a ``sample`` method that returns a
random minibatch suitable for training the velocity‑field network.

The implementation is deliberately lightweight and does not depend on any
external libraries besides ``torch``. It stores the data as a list of
``torch.Tensor`` tuples; this is sufficient for the modest capacities used in
the paper (e.g. 1500–2500 batches). For larger workloads a pre‑allocated
tensor implementation would be preferable, but the list‑based approach keeps
the code simple and easy to understand.
'''"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch


class ReplayBuffer:
    """Circular replay buffer for storing trajectories.

    Parameters
    ----------
    capacity: int
        Maximum number of *batches* to retain in the buffer.
    device: str or torch.device, optional
        Device on which sampled tensors will be returned. Stored tensors are
        kept on the device they were provided with; ``sample`` moves them to
        ``device`` if specified.
    """

    def __init__(self, capacity: int, device: torch.device | str = "cpu"):
        self.capacity: int = capacity
        self.device = torch.device(device)
        # Internal storage: list of (x, t, v_hat) tuples.
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def __len__(self) -> int:
        """Return the number of stored batches."""
        return len(self.buffer)

    def add(self, x: torch.Tensor, t: torch.Tensor, v_hat: torch.Tensor) -> None:
        """Add a new batch to the buffer.

        Parameters
        ----------
        x: torch.Tensor
            Particle positions, shape ``[batch, *]``.
        t: torch.Tensor
            Time values, shape ``[batch]`` or ``[batch, 1]``.
        v_hat: torch.Tensor
            Empirical velocity, same shape as ``x``.
        """
        # Ensure tensors are detached to avoid retaining computation graphs.
        x = x.detach().clone()
        t = t.detach().clone()
        v_hat = v_hat.detach().clone()
        self.buffer.append((x, t, v_hat))
        if len(self.buffer) > self.capacity:
            # Remove the oldest entry (FIFO).
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random minibatch.

        Returns a tuple ``(x_batch, t_batch, v_batch)`` where each element is a
        concatenated tensor of the requested ``batch_size`` samples. If the
        buffer contains fewer than ``batch_size`` entries, all available data
        is returned.
        """
        if len(self.buffer) == 0:
            raise RuntimeError("ReplayBuffer is empty – cannot sample.")

        effective_batch = min(batch_size, len(self.buffer))
        # Randomly choose distinct indices.
        indices = random.sample(range(len(self.buffer)), effective_batch)
        xs, ts, vs = [], [], []
        for idx in indices:
            x, t, v = self.buffer[idx]
            xs.append(x)
            ts.append(t)
            vs.append(v)
        # Concatenate along the first dimension.
        x_batch = torch.cat(xs, dim=0).to(self.device)
        t_batch = torch.cat(ts, dim=0).to(self.device)
        v_batch = torch.cat(vs, dim=0).to(self.device)
        return x_batch, t_batch, v_batch

    def clear(self) -> None:
        """Remove all stored entries."""
        self.buffer.clear()

    # Convenience methods for debugging / inspection.
    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, size={len(self)})"


# Simple sanity‑check when the module is executed directly.
if __name__ == "__main__":
    rb = ReplayBuffer(capacity=3)
    for i in range(5):
        x = torch.randn(2, 4)
        t = torch.full((2,), i * 0.1)
        v = torch.randn_like(x)
        rb.add(x, t, v)
        print(rb)
    sample = rb.sample(2)
    print([s.shape for s in sample])
"""
