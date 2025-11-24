import torch
import torch.nn as nn
import math
from typing import List


def sinusoidal_time_embedding(t: torch.Tensor, dim: int = 16) -> torch.Tensor:
    """Create sinusoidal embeddings for a scalar time tensor.

    Args:
        t (torch.Tensor): Tensor of shape (B, 1) or (B,).
        dim (int): Embedding dimension (must be even).
    Returns:
        torch.Tensor: Embedding of shape (B, dim).
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)
    device = t.device
    half_dim = dim // 2
    # Compute the frequencies
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
    emb = t * emb  # (B, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb  # (B, dim)


class DoubleConv(nn.Module):
    """(conv => ReLU) * 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv. Supports both bilinear and transposed conv upsampling."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad if needed (in case of odd dimensions)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                    diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetVelocity(nn.Module):
    """UNet model for velocity field v_θ(x, t).

    The network takes an image tensor x of shape (B, C, H, W) and a scalar time t (B, 1).
    A sinusoidal time embedding is added to the bottleneck feature map.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        depth: int = 1,
        time_emb_dim: int = 16,
        bilinear: bool = True,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.bilinear = bilinear
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.downs: List[Down] = nn.ModuleList()
        chs = base_channels
        for _ in range(depth):
            self.downs.append(Down(chs, chs * 2))
            chs *= 2
        # Bottleneck
        self.bottleneck = DoubleConv(chs, chs * 2)
        # Project time embedding to match bottleneck channels
        self.time_proj = nn.Linear(time_emb_dim, chs * 2)
        # Decoder
        self.ups: List[Up] = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(chs * 2, chs // 2, bilinear=bilinear))
            chs //= 2
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Encoder path
        x1 = self.inc(x)
        enc_features = [x1]
        for down in self.downs:
            x1 = down(x1)
            enc_features.append(x1)
        # Bottleneck
        x1 = self.bottleneck(x1)
        # Add time embedding
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)  # (B, dim)
        # Project and reshape to (B, C, 1, 1)
        t_proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        x1 = x1 + t_proj
        # Decoder path (reverse order of encoder features)
        for i, up in enumerate(self.ups):
            skip = enc_features[-(i + 2)]  # corresponding skip connection
            x1 = up(x1, skip)
        out = self.outc(x1)
        return out

# Alias for straight‑flow network (same architecture, separate class name for clarity)
class UNetStraightFlow(UNetVelocity):
    """UNet used for the straight‑flow network u_δ.

    Inherits the same architecture as UNetVelocity.
    """
    pass
