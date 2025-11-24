import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor, device: str = "cpu") -> float:
    """Compute Frechet Inception Distance (FID) between real and generated images.

    Args:
        real_images (torch.Tensor): Tensor of shape (N, C, H, W) with values in [-1, 1].
        fake_images (torch.Tensor): Tensor of shape (M, C, H, W) with values in [-1, 1].
        device (str): Device to run the metric on.
    Returns:
        float: FID score.
    """
    fid = FrechetInceptionDistance(feature=2048).to(device)
    # Inception expects inputs in [0, 1] range
    real = (real_images + 1) / 2
    fake = (fake_images + 1) / 2
    fid.update(real, real=True)
    fid.update(fake, real=False)
    return float(fid.compute())


def compute_is(fake_images: torch.Tensor, device: str = "cpu") -> float:
    """Compute Inception Score (IS) for generated images.

    Args:
        fake_images (torch.Tensor): Tensor of shape (N, C, H, W) with values in [-1, 1].
        device (str): Device to run the metric on.
    Returns:
        float: Inception Score.
    """
    is_metric = InceptionScore().to(device)
    fake = (fake_images + 1) / 2
    is_metric.update(fake)
    # InceptionScore returns a tuple (mean, std)
    mean, _ = is_metric.compute()
    return float(mean)
