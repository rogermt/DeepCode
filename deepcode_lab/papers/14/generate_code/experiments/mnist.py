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


def get_data_loaders(batch_size, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),  # maps to [-1, 1]
    ])
    train_dataset = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform)
    target_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return target_loader


def source_sampler(batch_size, in_channels, height, width, device):
    # Uniform noise in [-1, 1]
    return (torch.rand(batch_size, in_channels, height, width, device=device) * 2 - 1)


def main():
    parser = argparse.ArgumentParser(description='NSGF++ MNIST experiment')
    parser.add_argument('--config', type=str, default='configs/nsfgpp_mnist.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))
    device = args.device

    # Model hyper‑parameters
    model_cfg = cfg['model']
    in_ch = model_cfg.get('in_channels', 1)
    out_ch = model_cfg.get('out_channels', 1)
    base_ch = model_cfg.get('base_channels', 32)
    depth = model_cfg.get('depth', 1)
    time_emb_dim = model_cfg.get('time_emb_dim', 16)

    # Instantiate models
    v_model = UNetVelocity(in_channels=in_ch, out_channels=out_ch, base_channels=base_ch,
                            depth=depth, time_emb_dim=time_emb_dim).to(device)
    t_model = TimePredictor(in_channels=in_ch).to(device)
    u_model = UNetStraightFlow(in_channels=in_ch, out_channels=out_ch, base_channels=base_ch,
                               depth=depth, time_emb_dim=time_emb_dim).to(device)

    # Data loaders / samplers
    train_loader = get_data_loaders(cfg['training']['batch_size_velocity'], device)
    # target sampler returns a batch from the loader
    def target_sampler(n):
        # simple iterator over loader
        for batch in train_loader:
            images, _ = batch
            return images[:n].to(device)

    # Source sampler function
    def src_sampler(n):
        return source_sampler(n, in_ch, 28, 28, device)

    # Trainer
    trainer = NSFGPPTrainer(
        v_model=v_model,
        t_model=t_model,
        u_model=u_model,
        source_sampler=src_sampler,
        target_sampler=target_sampler,
        blur=cfg['training']['blur'],
        scaling=cfg['training']['scaling'],
        eta=cfg['training']['eta_nsfg'],
        T=cfg['training']['T_max'],
        num_pool_batches=cfg['training']['num_pool_batches'],
        batch_size=cfg['training']['batch_size_velocity'],
        lr_velocity=cfg['training']['lr_velocity'],
        lr_time=cfg['training']['lr_time_predictor'],
        lr_nsf=cfg['training']['lr_nsf'],
        total_iters_velocity=cfg['training']['total_iters_velocity'],
        total_iters_time=cfg['training']['total_iters_time'],
        total_iters_nsf=cfg['training']['total_iters_nsf'],
        device=device,
        seed=cfg.get('seed', 42),
        log_dir=cfg['logging']['log_dir'],
        checkpoint_dir=cfg['logging']['checkpoint_dir'],
    )

    # Training phases
    trainer.train_velocity()
    trainer.train_time_predictor()
    trainer.train_nsf()
    trainer.close()

    # Inference
    generated = nsfgpp_generate(
        v_model=v_model,
        t_model=t_model,
        u_model=u_model,
        source_sampler=lambda n: src_sampler(n),
        eta_nsfg=cfg['training']['eta_nsfg'],
        omega_nsf=cfg['training']['omega_nsf'],
        T=cfg['training']['T_max'],
        batch_size=cfg['training']['batch_size_velocity'],
        device=device,
    )

    # Load a set of real images for evaluation
    real_images = []
    for batch in train_loader:
        imgs, _ = batch
        real_images.append(imgs)
        if len(real_images) * cfg['training']['batch_size_velocity'] >= 10000:
            break
    real_images = torch.cat(real_images)[:10000].to(device)

    # Compute metrics
    fid = metrics.compute_fid(real_images, generated, device=device)
    is_score = metrics.compute_is(generated, device=device)

    # Save results
    os.makedirs('results/figures', exist_ok=True)
    torch.save(generated, 'results/figures/mnist_generated.pt')
    with open('results/figures/mnist_metrics.txt', 'w') as f:
        f.write(f'FID: {fid:.4f}\nInception Score: {is_score:.4f}\n')

    print(f'Finished MNIST experiment. FID={fid:.4f}, IS={is_score:.4f}')

if __name__ == '__main__':
    main()
