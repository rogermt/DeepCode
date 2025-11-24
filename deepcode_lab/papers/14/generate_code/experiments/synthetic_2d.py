# Synthetic 2D experiment for Neural Sinkhorn Gradient Flow (NSGF)
"""
Train NSGF on a synthetic 2‑D target distribution (e.g., 8‑gaussians mixture) and
evaluate the 2‑Wasserstein distance between generated and true samples.

The script follows the reproduction plan:
1. Load configuration from `configs/nsfg_2d.yaml`.
2. Set random seeds for reproducibility.
3. Define source (standard Gaussian) and target samplers.
4. Initialise the MLP velocity model.
5. Build and run the NSFG trainer (Algorithm 1).
6. Generate samples using the trained model.
7. Compute the 2‑Wasserstein distance via `experiments.metrics.wasserstein_distance`.
8. Save generated samples and the metric to `results/`.
"""

import argparse
import os
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils import set_seed, get_logger
from src.models import MLPVelocity
from src.training import NSFGTrainer
from experiments import metrics


def sample_source(n: int, device: str = "cpu") -> torch.Tensor:
    """Sample `n` points from a standard Gaussian N(0, I)."""
    return torch.randn(n, 2, device=device)


def sample_target(n: int, device: str = "cpu") -> torch.Tensor:
    """Sample `n` points from an 8‑gaussians mixture arranged on a circle.
    Each component has unit variance.
    """
    # Define centers on a circle of radius 2
    angles = torch.arange(0, 8) * (2 * torch.pi / 8)
    centers = torch.stack([2 * torch.cos(angles), 2 * torch.sin(angles)], dim=1)  # (8, 2)
    # Choose a component for each sample uniformly
    idx = torch.randint(0, 8, (n, ), device=device)
    chosen_centers = centers[idx]
    # Add Gaussian noise with std=0.2
    noise = 0.2 * torch.randn(n, 2, device=device)
    return chosen_centers + noise


def main():
    parser = argparse.ArgumentParser(description="Synthetic 2D NSGF experiment")
    parser.add_argument("--config", type=str, default="configs/nsfg_2d.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Reproducibility
    set_seed(cfg.get("seed", 42))

    device = args.device
    # Create model
    model = MLPVelocity(dim=2, hidden_dim=cfg.get("hidden_dim", 256)).to(device)

    # Trainer
    trainer = NSFGTrainer(
        model=model,
        source_sampler=lambda n: sample_source(n, device),
        target_sampler=lambda n: sample_target(n, device),
        blur=cfg["sinkhorn"]["blur"],
        scaling=cfg["sinkhorn"]["scaling"],
        eta=cfg["euler"]["eta_nsfg"],
        T=cfg["nsfg"]["T"],
        num_pool_batches=cfg["nsfg"]["num_pool_batches"],
        batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["lr_velocity"],
        total_iters=cfg["training"]["total_iters"],
        device=device,
        seed=cfg.get("seed", 42),
        log_dir=cfg.get("log_dir", "logs/synthetic_2d"),
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints/synthetic_2d"),
    )

    # Train
    trainer.train(pool_particle_n=cfg.get("pool_particle_n", 1024),
                  checkpoint_interval=cfg.get("checkpoint_interval", 1000))

    # Generate samples
    model.eval()
    with torch.no_grad():
        x = sample_source(cfg.get("eval_samples", 10000), device)
        t = torch.zeros(x.shape[0], 1, device=device)  # start at t=0
        for step in range(cfg["nsfg"]["T"]):
            v = model(x, t + step)
            x = x + cfg["euler"]["eta_nsfg"] * v
        generated = x.cpu()

    # Compute metric
    true_samples = sample_target(generated.shape[0], device="cpu")
    w2 = metrics.wasserstein_distance(generated.numpy(), true_samples.numpy())
    print(f"2‑Wasserstein distance: {w2:.4f}")

    # Save results
    os.makedirs("results/figures", exist_ok=True)
    torch.save(generated, "results/figures/synthetic_2d_generated.pt")
    with open("results/figures/synthetic_2d_metrics.txt", "w") as f:
        f.write(f"W2: {w2}\n")

    # Close logger
    trainer.close()

if __name__ == "__main__":
    main()
