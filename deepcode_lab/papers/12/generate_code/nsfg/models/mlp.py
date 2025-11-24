'''mlp.py
Implementation of a simple MLP for 2‑D synthetic experiments.

The network maps a 2‑D coordinate ``x`` and a scalar time ``t`` to a
2‑D velocity vector.  It follows the specification in the reproduction
plan:

* 3 hidden layers, each with 256 units
* ReLU activation
* Xavier uniform weight initialization, zero bias
* Input is the concatenation of ``x`` (shape ``[batch, 2]``) and ``t``
  (shape ``[batch, 1]`` or a scalar tensor broadcastable to ``[batch, 1]``)
* Output shape ``[batch, 2]``

The module is deliberately lightweight and has no external dependencies
besides ``torch``.
''' 

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPVelocity(nn.Module):
    """Simple MLP that predicts a 2‑D velocity from (x, t).

    Parameters
    ----------
    hidden_dim: int, default 256
        Number of units in each hidden layer.
    num_hidden: int, default 3
        Number of hidden layers.
    """

    def __init__(self, hidden_dim: int = 256, num_hidden: int = 3):
        super().__init__()
        self.input_dim = 2 + 1  # x (2‑D) + scalar time t
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden

        # Build layers list
        layers = []
        in_dim = self.input_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        # Final layer to 2‑D output
        layers.append(nn.Linear(hidden_dim, 2))
        self.layers = nn.ModuleList(layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform and biases to zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor of shape ``[batch, 2]``
            Spatial coordinates.
        t: torch.Tensor of shape ``[batch]`` or ``[batch, 1]``
            Scalar time values.  If a 1‑D tensor is provided it will be
            unsqueezed to ``[batch, 1]`` for concatenation.

        Returns
        -------
        torch.Tensor of shape ``[batch, 2]``
            Predicted velocity field.
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # Concatenate along feature dimension
        inp = torch.cat([x, t], dim=1)
        out = inp
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # Apply activation after all but the last layer
            if i < len(self.layers) - 1:
                out = F.relu(out)
        return out

    def __repr__(self) -> str:
        return (f"MLPVelocity(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
                f"num_hidden={self.num_hidden})")
