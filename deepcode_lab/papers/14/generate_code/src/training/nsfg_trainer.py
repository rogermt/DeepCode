# src/training/nsfg_trainer.py
"""NSFG (Neural Sinkhorn Gradient Flow) Trainer
Implements Algorithm 1 from the paper – builds a trajectory pool by simulating the
particle ODE (Eq.14) using empirical velocity estimates derived from Sinkhorn
potentials, then trains a neural network to match the empirical velocity field.

The implementation is deliberately lightweight and framework‑agnostic – it
expects the caller to provide:
* a model with signature ``model(x, t)`` returning a velocity tensor of the same
  shape as ``x``;
* a ``source_sampler`` and ``target_sampler`` callable that each accept an integer
  ``n`` and return a ``torch.Tensor`` of shape ``(n, d)``;
* hyper‑parameters such as ``blur``, ``scaling``, ``eta`` (Euler step size),
  ``T`` (max integration steps), ``num_pool_batches`` and training settings.

The trainer builds an in‑memory pool of ``(x, v̂, t)`` tuples, then performs
mini‑batch SGD on the MSE loss between the model prediction and the empirical
velocity.  Check‑pointing and TensorBoard logging are provided for convenience.
"""

import os
import random
from typing import Callable, List, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from ..utils import set_seed, get_logger
from ..sinkhorn import compute_potentials
from ..particles import euler_step


class NSFGTrainer:
    """Trainer for the NSFG velocity‑field network.

    Parameters
    ----------
    model: nn.Module
        Neural network approximating the velocity field. Must implement ``forward(x, t)``.
    source_sampler: Callable[[int], torch.Tensor]
        Function that returns ``n`` samples from the source distribution μ₀.
    target_sampler: Callable[[int], torch.Tensor]
        Function that returns ``n`` samples from the target distribution μ*.
    blur: float
        Entropy regularisation parameter ε for the Sinkhorn loss.
    scaling: float
        Kernel scaling σ for the Sinkhorn loss.
    eta: float
        Euler integration step size (η in the paper).
    T: int
        Number of integration steps for building the pool (Eq.14).
    num_pool_batches: int
        How many batches of particles to simulate when constructing the pool.
    batch_size: int
        Mini‑batch size for training the network.
    lr: float
        Learning rate for the Adam optimiser.
    total_iters: int
        Number of training iterations.
    device: str
        ``"cpu"`` or ``"cuda"``.
    seed: int, optional
        Random seed for reproducibility.
    log_dir: str, optional
        Directory for TensorBoard logs.
    checkpoint_dir: str, optional
        Directory where model checkpoints will be saved.
    """

    def __init__(
        self,
        model: nn.Module,
        source_sampler: Callable[[int], torch.Tensor],
        target_sampler: Callable[[int], torch.Tensor],
        blur: float,
        scaling: float,
        eta: float,
        T: int,
        num_pool_batches: int,
        batch_size: int,
        lr: float,
        total_iters: int,
        device: str = "cpu",
        seed: int = 42,
        log_dir: str = "logs/nsfg",
        checkpoint_dir: str = "checkpoints/nsfg",
    ):
        # reproducibility
        set_seed(seed)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.source_sampler = source_sampler
        self.target_sampler = target_sampler
        self.blur = blur
        self.scaling = scaling
        self.eta = eta
        self.T = T
        self.num_pool_batches = num_pool_batches
        self.batch_size = batch_size
        self.lr = lr
        self.total_iters = total_iters
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger: SummaryWriter = get_logger(self.log_dir)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        # pool will be a list of (x, v_hat, t) tuples
        self.pool: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    # ---------------------------------------------------------------------
    # Pool construction
    # ---------------------------------------------------------------------
    def _build_pool(self, n: int) -> None:
        """Build the trajectory pool.

        Parameters
        ----------
        n: int
            Number of particles per batch.
        """
        self.pool.clear()
        for batch_idx in range(self.num_pool_batches):
            # Sample initial particles from source and target
            X = self.source_sampler(n).to(self.device)
            Y = self.target_sampler(n).to(self.device)
            # Ensure gradient tracking for X (required for potential gradient)
            X.requires_grad_(True)
            for t_step in range(self.T + 1):
                # Compute potentials f_tt (self‑self) and f_tstar (self‑target)
                f_tt, _ = compute_potentials(X, X, self.blur, self.scaling)
                f_tstar, _ = compute_potentials(X, Y, self.blur, self.scaling)
                # Gradients w.r.t. X
                grad_f_tt = torch.autograd.grad(f_tt.sum(), X, retain_graph=True)[0]
                grad_f_tstar = torch.autograd.grad(f_tstar.sum(), X, retain_graph=True)[0]
                v_hat = grad_f_tt - grad_f_tstar  # empirical velocity (Eq.13)
                # Store a copy (detach to avoid graph retention)
                self.pool.append((X.detach().clone(), v_hat.detach().clone(), torch.tensor(t_step, dtype=torch.float32, device=self.device)))
                # Euler integration for next step (Eq.14)
                X = euler_step(X, v_hat, self.eta)
                X.requires_grad_(True)
        print(f"[NSFGTrainer] Built pool with {len(self.pool)} entries (batch size {n}, T={self.T})")

    # ---------------------------------------------------------------------
    # Training utilities
    # ---------------------------------------------------------------------
    def _sample_minibatch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a minibatch from the pool.
        Returns tensors ``x``, ``v`` and ``t`` each of shape ``(batch_size, ...)``.
        """
        indices = random.sample(range(len(self.pool)), self.batch_size)
        xs, vs, ts = zip(*[self.pool[i] for i in indices])
        x_batch = torch.stack(xs).to(self.device)
        v_batch = torch.stack(vs).to(self.device)
        t_batch = torch.stack(ts).unsqueeze(-1).to(self.device)  # shape (B,1)
        return x_batch, v_batch, t_batch

    def train(self, pool_particle_n: int = 1024, checkpoint_interval: int = 1000) -> None:
        """Run the full training loop.

        Parameters
        ----------
        pool_particle_n: int
            Number of particles per pool batch (the ``n`` in the paper).
        checkpoint_interval: int
            Save a checkpoint every ``checkpoint_interval`` iterations.
        """
        # Step 1: build the trajectory pool
        self._build_pool(pool_particle_n)
        # Step 2: training iterations
        for it in range(1, self.total_iters + 1):
            x_batch, v_batch, t_batch = self._sample_minibatch()
            # Model prediction
            pred = self.model(x_batch, t_batch)
            loss = ((pred - v_batch) ** 2).mean()
            # Optimisation step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Logging
            self.logger.add_scalar("train/loss", loss.item(), it)
            if it % 100 == 0:
                print(f"[NSFGTrainer] Iter {it}/{self.total_iters} - loss: {loss.item():.6f}")
            # Checkpointing
            if checkpoint_interval and (it % checkpoint_interval == 0):
                ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt_iter_{it}.pt")
                torch.save({
                    "iteration": it,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, ckpt_path)
        # Final checkpoint
        final_path = os.path.join(self.checkpoint_dir, "final.pt")
        torch.save({
            "iteration": self.total_iters,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, final_path)
        print(f"[NSFGTrainer] Training completed. Final model saved to {final_path}")

    # ---------------------------------------------------------------------
    # Utility for loading a checkpoint
    # ---------------------------------------------------------------------
    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[NSFGTrainer] Loaded checkpoint from {path} (iteration {ckpt.get('iteration')})")

    # ---------------------------------------------------------------------
    # Clean‑up
    # ---------------------------------------------------------------------
    def close(self) -> None:
        """Close the TensorBoard logger."""
        self.logger.close()

# End of file
