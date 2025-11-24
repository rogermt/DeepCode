# -*- coding: utf-8 -*-
"""
CIFAR-10 experiment script for NSGF++.
Implements the full NSGF++ pipeline on the CIFAR-10 dataset:
- Data loading and preprocessing
- Source sampler (uniform noise)
- Model instantiation (UNetVelocity, UNetStraightFlow, TimePredictor)
- Training phases via NSFGPPTrainer (velocity, time predictor, NSF)
- Inference to generate samples
- Evaluation using FID and Inception Score
- Saving generated samples and metrics
"""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils import set_seed, get_logger
from src.models import UNetVelocity, UNetStraightFlow, TimePredictor
from src.training import NSFGPPTrainer
from src.inference import nsfgpp_generate
from experiments import metrics


def get_data_loader(batch_size: int, device: torch.device) -> DataLoader:
    """Create a DataLoader for CIFAR-10 training data.

    Args:
        batch_size (int): Batch size for the DataLoader.
        device (torch.device): Device on which tensors will be placed.

    Returns:
        DataLoader: DataLoader yielding normalized CIFAR-10 images.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.CIFAR10(root="./data/cifar10", train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return loader


def source_sampler(batch_size: int, in_channels: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Sample uniform noise in [-1, 1] matching image dimensions.

    Args:
        batch_size (int): Number of samples.
        in_channels (int): Number of image channels (3 for CIFAR-10).
        height (int): Image height.
        width (int): Image width.
        device (torch.device): Device for the tensor.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, in_channels, height, width).
    """
    return (torch.rand(batch_size, in_channels, height, width, device=device) * 2.0 - 1.0


def main():
    parser = argparse.ArgumentParser(description="NSGF++ CIFAR-10 experiment")
    parser.add_argument("--config", type=str, default="configs/nsfgpp_cifar.yaml", help="Path to config YAML file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))

    # Prepare logger
    log_dir = cfg.get("logging", {}).get("log_dir", "logs/cifar10")
    logger = get_logger(log_dir)
    logger.add_text("config", yaml.dump(cfg))

    # Model hyper‑parameters
    model_cfg = cfg.get("model", {})
    in_channels = model_cfg.get("in_channels", 3)
    out_channels = model_cfg.get("out_channels", 3)
    base_channels = model_cfg.get("base_channels", 128)
    depth = model_cfg.get("depth", 2)
    time_emb_dim = model_cfg.get("time_emb_dim", 16)

    # Instantiate models
    v_model = UNetVelocity(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        depth=depth,
        time_emb_dim=time_emb_dim,
    ).to(device)
    t_model = TimePredictor(in_channels=in_channels)
    u_model = UNetStraightFlow(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        depth=depth,
        time_emb_dim=time_emb_dim,
    ).to(device)

    # Data loaders / samplers
    training_cfg = cfg.get("training", {})
    batch_size_velocity = training_cfg.get("batch_size_velocity", 128)
    batch_size_time = training_cfg.get("batch_size_time", 128)
    batch_size_nsf = training_cfg.get("batch_size_nsf", 128)

    target_loader = get_data_loader(batch_size_velocity, device)
    # Simple iterator to fetch a batch of target images
    target_iter = iter(target_loader)
    def target_sampler(n: int):
        try:
            batch = next(target_iter)
        except StopIteration:
            # Re‑initialize iterator if exhausted
            nonlocal target_iter
            target_iter = iter(target_loader)
            batch = next(target_iter)
        # batch is a tuple (images, labels)
        images = batch[0]
        if images.size(0) >= n:
            return images[:n].to(device)
        # If not enough, repeat samples
        repeats = (n + images.size(0) - 1) // images.size(0)
        expanded = images.repeat(repeats, 1, 1, 1)[:n].to(device)
        return expanded

    # Source sampler function
    src_sampler = lambda n: source_sampler(n, in_channels, 32, 32, device)

    # Trainer initialization
    trainer = NSFGPPTrainer(
        v_model=v_model,
        t_model=t_model,
        u_model=u_model,
        source_sampler=src_sampler,
        target_sampler=target_sampler,
        blur=training_cfg.get("blur", 1.0),
        scaling=training_cfg.get("scaling", 0.85),
        eta=training_cfg.get("eta_nsfg", 0.2),
        T=training_cfg.get("T_max", 5),
        num_pool_batches=training_cfg.get("num_pool_batches", 2500),
        batch_size=batch_size_velocity,
        lr_velocity=training_cfg.get("lr_velocity", 1e-4),
        lr_time=training_cfg.get("lr_time_predictor", 1e-4),
        lr_nsf=training_cfg.get("lr_nsf", 1e-4),
        total_iters_velocity=training_cfg.get("total_iters_velocity", 20000),
        total_iters_time=training_cfg.get("total_iters_time", 40000),
        total_iters_nsf=training_cfg.get("total_iters_nsf", 20000),
        device=device,
        log_dir=log_dir,
        checkpoint_dir=cfg.get("logging", {}).get("checkpoint_dir", "checkpoints/cifar10"),
    )

    # Training phases
    logger.info("Starting velocity field training...")
    trainer.train_velocity()
    logger.info("Starting time‑predictor training...")
    trainer.train_time_predictor()
    logger.info("Starting NSF training...")
    trainer.train_nsf()
    trainer.close()

    # Inference
    logger.info("Generating samples via NSGF++ inference...")
    generated = nsfgpp_generate(
        v_model=v_model,
        t_model=t_model,
        u_model=u_model,
        source_sampler=src_sampler,
        eta_nsfg=training_cfg.get("eta_nsfg", 0.2),
        omega_nsf=training_cfg.get("omega_nsf", 0.02),
        T=training_cfg.get("T_max", 5),
        batch_size=10000,
        device=device,
    )

    # Evaluation – use 10k real CIFAR-10 test images
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    test_dataset = datasets.CIFAR10(root="./data/cifar10", train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=4)
    real_images = next(iter(test_loader))[0].to(device)

    fid_score = metrics.compute_fid(real_images, generated, device=device)
    is_score = metrics.compute_is(generated, device=device)

    # Save results
    results_dir = "results/figures"
    os.makedirs(results_dir, exist_ok=True)
    torch.save(generated, os.path.join(results_dir, "cifar10_generated.pt"))
    with open(os.path.join(results_dir, "cifar10_metrics.txt"), "w") as f:
        f.write(f"FID: {fid_score:.4f}\n")
        f.write(f"Inception Score: {is_score:.4f}\n")
    logger.info(f"CIFAR‑10 experiment completed. FID: {fid_score:.4f}, IS: {is_score:.4f}")


if __name__ == "__main__":
    main()
