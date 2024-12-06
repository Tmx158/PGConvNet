import torch
import torch.nn as nn
from typing import Optional


class Imputation(nn.Module):
    """
    Imputation module for predicting missing values in multivariate time series.

    Args:
        nf (int): Number of features in the input data.
        pred_len (int): Prediction length for each variable.
        nvars (int): Number of variables in the input data.
        use_alt_method (bool): Whether to use the alternative complex method for prediction. Default: False.
    """

    def __init__(self, len: int, pred_len: int, nvars: int, use_alt_method: bool = False) -> None:
        super(Imputation, self).__init__()

        # Store configuration
        self.len = len
        self.pred_len = pred_len
        self.nvars = nvars
        self.use_alt_method = use_alt_method

        # Default method: Flatten + Linear layer
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Linear(len, pred_len)

        # Alternative complex method using multiple layers
        self.alt_conv1 = nn.Conv1d(in_channels=nvars, out_channels=64, kernel_size=3, padding=1)
        self.alt_conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.alt_fc = nn.Linear(128 * pred_len, pred_len * nvars)  # Final FC layer for prediction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Imputation module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, nvars, nf).

        Returns:
            torch.Tensor: Output tensor after prediction, shape (batch_size, nvars, pred_len).
        """

        if x.shape[1] != self.nvars:
            raise ValueError(f"Expected {self.nvars} variables, but got {x.shape[1]}")

        if self.use_alt_method:
            print("Using alternative complex method for imputation.")
            # Alternative complex method: Conv1d + FC layers
            x = self.alt_conv1(x)  # First convolution
            x = nn.ReLU()(x)
            x = self.alt_conv2(x)  # Second convolution
            x = nn.ReLU()(x)
            x = x.view(x.size(0), -1)  # Flatten for fully connected layer
            x = self.alt_fc(x)  # Final fully connected layer
            x = x.view(-1, self.nvars, self.pred_len)  # Reshape to (batch_size, nvars, pred_len)
        else:
            # Default method: Flatten + Linear
            x = self.flatten(x)
            x = self.linear(x)

        return x

    def reset_parameters(self) -> None:
        """
        Reset parameters of all layers for reproducibility.
        """
        self.linear.reset_parameters()
        self.alt_conv1.reset_parameters()
        self.alt_conv2.reset_parameters()
        self.alt_fc.reset_parameters()

    def __repr__(self) -> str:
        """
        Custom representation for better readability in print statements or logs.
        """
        return (f"Imputation(nf={self.nf}, pred_len={self.pred_len}, nvars={self.nvars}, "
                f"use_alt_method={self.use_alt_method})")

    def extra_repr(self) -> str:
        """
        Additional information for model representation, useful in complex models.
        """
        return f"nf={self.nf}, pred_len={self.pred_len}, nvars={self.nvars}, use_alt_method={self.use_alt_method}"
