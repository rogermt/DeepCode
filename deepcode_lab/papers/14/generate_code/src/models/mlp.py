'''src/models/mlp.py
Implementation of the simple MLP used for 2‑D synthetic experiments.
The network approximates the time‑varying velocity field v_θ(x, t).
It takes a particle position ``x`` of shape ``[B, d]`` and a scalar time ``t``
(which can be broadcasted to ``[B, 1]``) and outputs a velocity vector of the
same dimension as ``x``.

Architecture (as described in the reproduction plan):
- 3 hidden layers, each with 256 units
- ReLU activation after each hidden layer
- No activation on the output layer (linear)
- Input dimension = d + 1 (position concatenated with time)
- Output dimension = d

The module is deliberately lightweight and has no external dependencies beyond
torch. It can be imported as ``from src.models.mlp import MLPVelocity``.
'''"""
import torch
import torch.nn as nn

class MLPVelocity(nn.Module):
    """Three‑layer MLP for learning the velocity field in 2‑D experiments.

    Parameters
    ----------
    dim : int
        Dimensionality of the input position vector ``x`` (e.g., 2 for synthetic
        2‑D data). The network will automatically infer the input size as
        ``dim + 1`` to accommodate the scalar time ``t``.
    hidden_dim : int, optional
        Number of hidden units per layer. Default is 256 as specified in the
        paper.
    """

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = dim + 1  # concatenate time scalar
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)  # output velocity same dim as x
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``[B, dim]`` representing particle positions.
        t : torch.Tensor
            Tensor of shape ``[B]`` or ``[B, 1]`` representing the scalar time.
            It will be broadcasted/reshaped to ``[B, 1]`` and concatenated to ``x``.

        Returns
        -------
        torch.Tensor
            Velocity prediction of shape ``[B, dim]``.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        # Ensure t matches batch size of x
        if t.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x has {x.shape[0]} but t has {t.shape[0]}"
            )
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

# Simple sanity check when the module is run directly
if __name__ == "__main__":
    B, d = 4, 2
    model = MLPVelocity(dim=d)
    x = torch.randn(B, d)
    t = torch.rand(B)
    v = model(x, t)
    print("Output shape:", v.shape)
    print(v)
"""
