'''time_predictor.py
Implementation of the phase‑transition predictor network t\phi.

The network follows the architecture described in the paper:
- 4 convolutional layers with filter sizes [32, 64, 128, 256]
- 3×3 kernels, stride 1, padding 1 (to preserve spatial dimensions)
- ReLU activation after each convolution
- 2×2 average‑pooling after each convolution (reduces H and W by a factor of 2 per layer)
- After the final pooling, the feature map is flattened and passed through a linear layer
  producing a single scalar per sample.
- A Sigmoid activation maps the output to the interval (0, 1), representing the
  predicted transition time τ.

The module can be used for both synthetic 2‑D data (where the input would be a
flattened vector) and image data (MNIST / CIFAR‑10). For image data the input
shape is (B, C, H, W). The implementation assumes image inputs; for non‑image
inputs the caller can reshape accordingly.
'''"""
import torch
import torch.nn as nn


class TimePredictor(nn.Module):
    """Phase‑transition predictor t\phi.

    Parameters
    ----------
    in_channels: int, default 3
        Number of input channels (e.g., 1 for MNIST, 3 for CIFAR‑10).
    hidden_dims: list[int], optional
        Number of filters for each convolutional layer. Defaults to
        ``[32, 64, 128, 256]`` as specified in the reproduction plan.
    """

    def __init__(self, in_channels: int = 3, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        assert len(hidden_dims) == 4, "TimePredictor expects exactly 4 conv layers"

        layers = []
        prev_ch = in_channels
        for i, out_ch in enumerate(hidden_dims):
            conv = nn.Conv2d(prev_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
            bn = nn.BatchNorm2d(out_ch)
            relu = nn.ReLU(inplace=True)
            pool = nn.AvgPool2d(kernel_size=2, stride=2)
            layers.extend([conv, bn, relu, pool])
            prev_ch = out_ch
        self.conv_body = nn.Sequential(*layers)

        # After four 2×2 poolings the spatial size is reduced by 2**4 = 16.
        # The exact size depends on the input image dimensions; we will infer it
        # lazily in the forward pass.
        self.fc = nn.Linear(0, 1)  # placeholder, will be replaced on first forward
        self.sigmoid = nn.Sigmoid()
        self._initialized = False

    def _initialize_fc(self, sample_tensor: torch.Tensor):
        """Create the final linear layer once the flattened size is known.
        """
        # sample_tensor shape: (B, C, H, W) after conv_body
        flat_dim = sample_tensor.shape[1] * sample_tensor.shape[2] * sample_tensor.shape[3]
        self.fc = nn.Linear(flat_dim, 1, bias=True)
        # Reset parameters (default initialization is fine)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, 1) with values in (0, 1).
        """
        # Apply convolutional body
        h = self.conv_body(x)
        if not self._initialized:
            self._initialize_fc(h)
        h = h.view(h.size(0), -1)  # flatten
        out = self.fc(h)
        out = self.sigmoid(out)
        return out

    def extra_repr(self) -> str:
        return f"in_channels={self.conv_body[0].in_channels}, " \
               f"hidden_dims={[self.conv_body[i].out_channels for i in range(0, len(self.conv_body), 4)]}"


__all__ = ["TimePredictor"]
"""
