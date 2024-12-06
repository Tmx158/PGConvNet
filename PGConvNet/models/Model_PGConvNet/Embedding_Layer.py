import torch
import torch.nn as nn
from typing import Tuple


class Embedding(nn.Module):
    """
    Embedding layer to project input features into a higher-dimensional space.

    Args:
        dmodel (int): Target dimension for each feature.
        alt_expand (bool): Whether to use an alternative method (Conv layer) to expand dimensions. Default: False
    """

    def __init__(self, dmodel: int, alt_expand: bool = False) -> None:
        super(Embedding, self).__init__()

        # Default Linear layer for dimension expansion from 1 to dmodel
        self.linear = nn.Linear(1, dmodel)

        # Alternative method using Conv1d layer for dimension expansion
        self.alt_conv = nn.Conv1d(in_channels=1, out_channels=dmodel, kernel_size=3, padding=1)

        # Store parameters for potential debug or future use
        self.dmodel = dmodel
        self.alt_expand = alt_expand

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, T).

        Returns:
            torch.Tensor: Projected tensor of shape (B, N, T, D).
        """

        # Check input dimensions to ensure it has 3 dimensions as expected
        if x.dim() != 3:
            raise ValueError(f"Expected input tensor with 3 dimensions (B,N,T), but got {x.dim()} dimensions")

        # Extract dimensions
        B, N, T = x.shape

        # Expand last dimension to apply the linear or convolutional layer
        x = x.unsqueeze(-1)  # Shape becomes (B, N, D, 1)

        if self.alt_expand:
            print("Using alternative dimension expansion method (Conv1D layer).")
            # Permute to match Conv1d expected input shape (B, C, L) where C is channels
            x = x.permute(0, 3, 1, 2).reshape(B, 1, N * T)  # Shape becomes (B, 1, N * T)
            x = self.alt_conv(x)  # Shape becomes (B, dmodel, N * T)
            x = x.view(B, N, T, self.dmodel)  # Reshape back to (B, N, T, D)
        else:
            x = self.linear(x)  # Default linear layer for dimension expansion (B, N, T, D)

        return x

    def reset_parameters(self) -> None:
        """
        Reset parameters of both expansion layers for reproducibility.
        """
        self.linear.reset_parameters()
        self.alt_conv.reset_parameters()

    def __repr__(self) -> str:
        """
        Custom representation for better readability in print statements or logs.
        """
        return f"Embedding(dmodel={self.dmodel}, alt_expand={self.alt_expand})"

    def extra_repr(self) -> str:
        """
        Additional information for model representation, useful in complex models.
        """
        return f"dmodel={self.dmodel}, alt_expand={self.alt_expand}"
