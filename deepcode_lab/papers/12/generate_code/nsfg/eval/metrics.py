'''nsfg/eval/metrics.py
Utility functions for evaluating generated samples.
Implements 2‑Wasserstein distance (via GeomLoss), Frechet Inception Distance (FID)
and Inception Score (IS).
''' 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
import numpy as np
from scipy import linalg
from typing import Tuple, Dict

# GeomLoss for Wasserstein distance
try:
    from geomloss import SamplesLoss
except ImportError as e:
    raise ImportError("geomloss is required for Wasserstein distance computation")

# ---------------------------------------------------------------------------
# Helper: Inception feature extractor
# ---------------------------------------------------------------------------
class InceptionFeatureExtractor(nn.Module):
    """Wraps a pretrained Inception‑v3 model and returns both the pool3
    features (2048‑dim) and the class logits.
    The model is set to evaluation mode and its parameters are frozen.
    """

    def __init__(self, device: torch.device | str = "cpu"):
        super().__init__()
        # Inception v3 expects input size 299x299 and values in [0,1] (or normalized)
        self.inception = models.inception_v3(pretrained=True, transform_input=False, aux_logits=False)
        self.inception.eval()
        for p in self.inception.parameters():
            p.requires_grad = False
        self.inception.to(device)
        # Feature extractor: everything except the final fully‑connected layer
        # The last two modules are: AdaptiveAvgPool2d and Linear (fc)
        self.feature_extractor = nn.Sequential(*list(self.inception.children())[:-1])
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.feature_extractor.to(device)
        self.device = device
        # Normalisation as used for Inception pretrained weights
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Resize to 299×299 and normalise.
        Assumes input in [0, 1] range.
        """
        # x: [N, C, H, W]
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # Normalise per channel
        # Apply transform per sample
        # transform works on (C, H, W) tensors
        x = torch.stack([self.transform(img) for img in x], dim=0)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (features, logits).
        Features are 2048‑dim vectors (after avg‑pool), logits are 1000‑dim.
        """
        x = self._preprocess(x.to(self.device))
        with torch.no_grad():
            # Get features
            feats = self.feature_extractor(x)  # shape [N, 2048, 1, 1]
            feats = feats.squeeze(-1).squeeze(-1)  # [N, 2048]
            # Get logits from the full model
            logits = self.inception(x)  # [N, 1000]
        return feats, logits

# ---------------------------------------------------------------------------
# 2‑Wasserstein distance via GeomLoss
# ---------------------------------------------------------------------------
def wasserstein2_distance(x: torch.Tensor, y: torch.Tensor, device: torch.device | str = "cpu") -> float:
    """Compute the squared 2‑Wasserstein distance between two point clouds.
    Uses GeomLoss with the ``energy`` loss (which corresponds to the 2‑Wasserstein).
    Returns a Python float.
    """
    loss_fn = SamplesLoss("energy", blur=0.0, scaling=0.0, epsilon=0.0)
    # Ensure tensors are on the same device and have shape [N, D]
    x = x.to(device)
    y = y.to(device)
    # The loss returns the squared distance (up to a constant factor). For
    # the energy loss, the value equals the squared 2‑Wasserstein distance.
    with torch.no_grad():
        dist = loss_fn(x, y).item()
    return dist

# ---------------------------------------------------------------------------
# Frechet Inception Distance (FID)
# ---------------------------------------------------------------------------
def _calculate_activation_statistics(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """Numpy implementation of the Frechet Distance.
    The formula is:
        ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    diff = mu1 - mu2
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        # Add epsilon to the diagonal of covariances
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)

def compute_fid(generated: torch.Tensor, real: torch.Tensor, device: torch.device | str = "cpu", batch_size: int = 64) -> float:
    """Compute Frechet Inception Distance between two sets of images.
    Both inputs are expected to be tensors of shape [N, C, H, W] with values in
    the range [0, 1].
    """
    extractor = InceptionFeatureExtractor(device)
    # Helper to get activations in batches
    def _get_activations(tensor: torch.Tensor) -> np.ndarray:
        activations = []
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        for (batch,) in loader:
            feats, _ = extractor(batch)
            activations.append(feats.cpu().numpy())
        return np.concatenate(activations, axis=0)

    act_gen = _get_activations(generated)
    act_real = _get_activations(real)
    mu_gen, sigma_gen = _calculate_activation_statistics(act_gen)
    mu_real, sigma_real = _calculate_activation_statistics(act_real)
    fid_value = _calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    return fid_value

# ---------------------------------------------------------------------------
# Inception Score (IS)
# ---------------------------------------------------------------------------
def compute_inception_score(generated: torch.Tensor, device: torch.device | str = "cpu", batch_size: int = 64, splits: int = 10) -> float:
    """Compute the Inception Score for generated images.
    Returns the mean IS over ``splits`` splits.
    """
    extractor = InceptionFeatureExtractor(device)
    # Collect logits
    logits_all = []
    dataset = TensorDataset(generated)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    for (batch,) in loader:
        _, logits = extractor(batch)
        logits_all.append(logits.cpu())
    logits = torch.cat(logits_all, dim=0)  # [N, 1000]
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=1).numpy()
    N = probs.shape[0]
    # Split into ``splits`` subsets
    split_size = N // splits
    scores = []
    for k in range(splits):
        part = probs[k * split_size : (k + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)  # marginal
        # KL divergence for each sample
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl_sum = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl_sum))
    return float(np.mean(scores))

# ---------------------------------------------------------------------------
# Combined evaluation helper
# ---------------------------------------------------------------------------
def evaluate_generated(
    generated: torch.Tensor,
    real: torch.Tensor,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute a dictionary of evaluation metrics for generated samples.
    Returns keys: ``wasserstein2``, ``fid``, ``inception_score``.
    """
    # Flatten point clouds for Wasserstein (assume same dimensionality)
    gen_flat = generated.view(generated.size(0), -1)
    real_flat = real.view(real.size(0), -1)
    w2 = wasserstein2_distance(gen_flat, real_flat, device=device)
    fid = compute_fid(generated, real, device=device, batch_size=batch_size)
    is_score = compute_inception_score(generated, device=device, batch_size=batch_size)
    return {"wasserstein2": w2, "fid": fid, "inception_score": is_score}

__all__ = [
    "wasserstein2_distance",
    "compute_fid",
    "compute_inception_score",
    "evaluate_generated",
]
