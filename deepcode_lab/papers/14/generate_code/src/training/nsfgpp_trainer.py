# NSFG++ Trainer implementation

"""
This module implements the training pipeline for the NSGF++ variant (Algorithm 3 in the paper).
It trains three components sequentially:
1. Velocity‑field network vθ (same as NSFG, but with a limited number of steps T≤5).
2. Phase‑transition predictor tφ (TimePredictor).
3. Neural straight‑flow network uδ (UNetStraightFlow).
The trainer builds a trajectory pool (experience replay) for the velocity field, then trains the predictor and the straight‑flow model on
straight‑line interpolations between source and target samples.
"""

import os
import random
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ..utils import set_seed, get_logger
from ..sinkhorn import compute_potentials
from ..particles import euler_step
from ..models import UNetVelocity, UNetStraightFlow, TimePredictor


class NSFGPPTrainer:
    """Trainer for the NSGF++ pipeline.

    Parameters
    ----------
    v_model: nn.Module
        Velocity‑field network (UNetVelocity) to be trained in phase 1.
    t_model: nn.Module
        Phase‑transition predictor (TimePredictor) to be trained in phase 2.
    u_model: nn.Module
        Straight‑flow network (UNetStraightFlow) to be trained in phase 3.
    source_sampler: Callable[[int], torch.Tensor]
        Function that returns ``n`` source particles (e.g., Gaussian noise or uniform image noise).
    target_sampler: Callable[[int], torch.Tensor]
        Function that returns ``n`` target samples (training data).
    blur: float
        Entropy regularisation parameter ε for Sinkhorn.
    scaling: float
        Kernel scaling σ for Sinkhorn.
    eta: float
        Euler step size for the NSGF phase.
    T: int
        Number of Euler steps in the NSGF phase (≤5 for NSGF++).
    num_pool_batches: int
        Number of batches used to build the trajectory pool.
    batch_size: int
        Mini‑batch size for training each component.
    lr_velocity: float
        Learning rate for the velocity‑field network.
    lr_time: float
        Learning rate for the phase‑transition predictor.
    lr_nsf: float
        Learning rate for the straight‑flow network.
    total_iters_velocity: int
        Training iterations for the velocity field.
    total_iters_time: int
        Training iterations for the predictor.
    total_iters_nsf: int
        Training iterations for the straight‑flow network.
    device: str, optional
        ``"cpu"`` or ``"cuda"``. Default ``"cpu"``.
    seed: int, optional
        Random seed for reproducibility. Default ``42``.
    log_dir: str, optional
        Directory for TensorBoard logs.
    checkpoint_dir: str, optional
        Directory where model checkpoints are saved.
    """

    def __init__(
        self,
        v_model: nn.Module,
        t_model: nn.Module,
        u_model: nn.Module,
        source_sampler: Callable[[int], torch.Tensor],
        target_sampler: Callable[[int], torch.Tensor],
        blur: float,
        scaling: float,
        eta: float,
        T: int,
        num_pool_batches: int,
        batch_size: int,
        lr_velocity: float,
        lr_time: float,
        lr_nsf: float,
        total_iters_velocity: int,
        total_iters_time: int,
        total_iters_nsf: int,
        device: str = "cpu",
        seed: int = 42,
        log_dir: str = "logs/nsfgpp",
        checkpoint_dir: str = "checkpoints/nsfgpp",
    ):
        # reproducibility
        set_seed(seed)
        self.device = torch.device(device)
        self.v_model = v_model.to(self.device)
        self.t_model = t_model.to(self.device)
        self.u_model = u_model.to(self.device)
        self.source_sampler = source_sampler
        self.target_sampler = target_sampler
        self.blur = blur
        self.scaling = scaling
        self.eta = eta
        self.T = T
        self.num_pool_batches = num_pool_batches
        self.batch_size = batch_size
        self.total_iters_velocity = total_iters_velocity
        self.total_iters_time = total_iters_time
        self.total_iters_nsf = total_iters_nsf

        # optimizers
        self.opt_v = optim.Adam(self.v_model.parameters(), lr=lr_velocity, betas=(0.9, 0.999))
        self.opt_t = optim.Adam(self.t_model.parameters(), lr=lr_time, betas=(0.9, 0.999))
        self.opt_u = optim.Adam(self.u_model.parameters(), lr=lr_nsf, betas=(0.9, 0.999))

        # logging / checkpointing
        self.logger: SummaryWriter = get_logger(log_dir)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # trajectory pool (list of tuples (x, v_hat, t))
        self.pool: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    # ---------------------------------------------------------------------
    # Helper: build trajectory pool (same logic as NSFGTrainer but limited T)
    # ---------------------------------------------------------------------
    def _build_pool(self, n: int) -> None:
        """Populate ``self.pool`` with (x, v̂, t) tuples.

        Parameters
        ----------
        n: int
            Number of particles per batch.
        """
        self.pool.clear()
        for _ in range(self.num_pool_batches):
            # sample source and target particles
            X = self.source_sampler(n).to(self.device)
            Y = self.target_sampler(n).to(self.device)
            # ensure gradients for X
            X.requires_grad_(True)
            for t_step in range(self.T + 1):
                # potentials for (X, X) and (X, Y)
                f_tt, _ = compute_potentials(X, X, self.blur, self.scaling)
                f_tstar, _ = compute_potentials(X, Y, self.blur, self.scaling)
                # gradient of f_tt w.r.t. X
                grad_f_tt = torch.autograd.grad(f_tt.sum(), X, create_graph=False)[0]
                grad_f_tstar = torch.autograd.grad(f_tstar.sum(), X, create_graph=False)[0]
                v_hat = grad_f_tt - grad_f_tstar  # Eq.13 empirical velocity
                self.pool.append((X.detach().clone(), v_hat.detach().clone(), torch.full((n, 1), t_step / self.T, device=self.device)))
                # Euler integration for next step (Eq.14)
                X = euler_step(X, v_hat, self.eta)
                X.requires_grad_(True)

    def _sample_minibatch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a minibatch from the trajectory pool.

        Returns
        -------
        x_batch, v_batch, t_batch: torch.Tensor
            Tensors of shape (batch_size, ...) on ``self.device``.
        """
        batch = random.sample(self.pool, self.batch_size)
        x_batch = torch.stack([item[0] for item in batch]).to(self.device)
        v_batch = torch.stack([item[1] for item in batch]).to(self.device)
        t_batch = torch.stack([item[2] for item in batch]).to(self.device)
        return x_batch, v_batch, t_batch

    # ---------------------------------------------------------------------
    # Phase 1 – train velocity field vθ
    # ---------------------------------------------------------------------
    def train_velocity(self, pool_particle_n: int = 1024, checkpoint_interval: int = 1000) -> None:
        """Train the velocity‑field network using the trajectory pool.

        Parameters
        ----------
        pool_particle_n: int
            Number of particles per pool batch.
        checkpoint_interval: int
            Save a checkpoint every ``checkpoint_interval`` iterations.
        """
        self._build_pool(pool_particle_n)
        for it in range(1, self.total_iters_velocity + 1):
            x_batch, v_batch, t_batch = self._sample_minibatch()
            pred = self.v_model(x_batch, t_batch)
            loss = ((pred - v_batch) ** 2).mean()
            self.opt_v.zero_grad()
            loss.backward()
            self.opt_v.step()
            self.logger.add_scalar("velocity/loss", loss.item(), it)
            if it % checkpoint_interval == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"v_model_iter_{it}.pt")
                torch.save({"iteration": it, "model_state_dict": self.v_model.state_dict(), "optimizer_state_dict": self.opt_v.state_dict()}, ckpt_path)
        # final checkpoint
        torch.save({"iteration": self.total_iters_velocity, "model_state_dict": self.v_model.state_dict(), "optimizer_state_dict": self.opt_v.state_dict()}, os.path.join(self.checkpoint_dir, "v_model_final.pt"))

    # ---------------------------------------------------------------------
    # Phase 2 – train phase‑transition predictor tφ
    # ---------------------------------------------------------------------
    def train_time_predictor(self, checkpoint_interval: int = 1000) -> None:
        """Train the TimePredictor network.

        The predictor is trained on straight‑line interpolations between source and target samples.
        """
        for it in range(1, self.total_iters_time + 1):
            X0 = self.source_sampler(self.batch_size).to(self.device)
            Y = self.target_sampler(self.batch_size).to(self.device)
            t = torch.rand(self.batch_size, 1, device=self.device)
            X_t = t * Y + (1 - t) * X0
            pred = self.t_model(X_t)
            loss = ((pred - t) ** 2).mean()
            self.opt_t.zero_grad()
            loss.backward()
            self.opt_t.step()
            self.logger.add_scalar("time_predictor/loss", loss.item(), it)
            if it % checkpoint_interval == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"t_model_iter_{it}.pt")
                torch.save({"iteration": it, "model_state_dict": self.t_model.state_dict(), "optimizer_state_dict": self.opt_t.state_dict()}, ckpt_path)
        torch.save({"iteration": self.total_iters_time, "model_state_dict": self.t_model.state_dict(), "optimizer_state_dict": self.opt_t.state_dict()}, os.path.join(self.checkpoint_dir, "t_model_final.pt"))

    # ---------------------------------------------------------------------
    # Phase 3 – train straight‑flow network uδ
    # ---------------------------------------------------------------------
    def train_nsf(self, checkpoint_interval: int = 1000) -> None:
        """Train the Neural Straight Flow (NSF) network.

        The target velocity is the constant vector ``Y - X0``.
        """
        for it in range(1, self.total_iters_nsf + 1):
            X0 = self.source_sampler(self.batch_size).to(self.device)
            Y = self.target_sampler(self.batch_size).to(self.device)
            t = torch.rand(self.batch_size, 1, device=self.device)
            X_t = t * Y + (1 - t) * X0
            target_vel = Y - X0
            pred = self.u_model(X_t, t)
            loss = ((pred - target_vel) ** plan
            # NOTE: The above line is intentionally left incomplete; the trainer will be refined in subsequent steps.
            self.opt_u
            # Placeholder
            pass
        # Placeholder for final checkpoint saving
        pass

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------
    def load_checkpoint(self, path: str, component: str) -> None:
        """Load a checkpoint for a specific component.

        Parameters
        ----------
        path: str
            Path to the checkpoint file.
        component: str
            One of ``"v"``, ``"t"`` or ``"u"`` indicating which model to load.
        """
        ckpt = torch.load(path, map_location=self.device)
        if component == "v":
            self.v_model.load_state_dict(ckpt["model_state_dict"])
            self.opt_v.load_state_dict(ckpt["optimizer_state_dict"])
        elif component == "t":
            self.t_model.load_state_dict(ckpt["model_state_dict"])
            self.opt_t.load_state_dict(ckpt["optimizer_state_dict"])
        elif component == "u":
            self.u_model.load_state_dict(ckpt["model_state_dict"])
            self.opt_u.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            raise ValueError(f"Unknown component '{component}'. Expected 'v', 't', or 'u'.")

    def close(self) -> None:
        """Close the TensorBoard logger."""
        self.logger.close()

""" End of NSFG++ trainer module.
"""
