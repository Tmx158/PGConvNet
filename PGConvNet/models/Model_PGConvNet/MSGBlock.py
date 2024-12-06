import torch
from torch import nn


class MSGBlock(nn.Module):
    """
    MSGBlock for multiscale grouping in time series.

    Args:
        updim (int): Dimension to project each variable.
        nvars (int): Number of variables in the input data.
        large_size (int): Kernel size for large scale convolution.
        small_size (int): Kernel size for small scale convolution.
        dropout (float): Dropout rate for activation layer.
    """

    def __init__(self, updim: int, nvars: int, large_size: int, small_size: int,dropout: float = 0.1) -> None:
        super(MSGBlock, self).__init__()

        self.updim = updim
        self.nvars = nvars
        self.large_size = large_size
        self.small_size = small_size
        self.channel = nvars * updim
        self.dropout = dropout
        # Define the large scale convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.channel, out_channels=self.channel, kernel_size=self.large_size, stride=1,
                      padding=self.large_size // 2, dilation=1, groups=self.channel),
            nn.Conv1d(in_channels=self.channel, out_channels=self.channel, kernel_size=self.large_size, stride=1,
                      padding=self.large_size // 2, dilation=1, groups=self.nvars),
        )
        # Activation block
        self.act = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.GELU()
        )
        # Define the small scale convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.channel, out_channels=self.channel, kernel_size=self.small_size, stride=1,
                      padding=self.small_size // 2, dilation=1, groups=self.channel),
            nn.Conv1d(in_channels=self.channel, out_channels=self.channel, kernel_size=self.small_size, stride=1,
                      padding=self.small_size // 2, dilation=1, groups=self.nvars),
        )

    def _reshape_for_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor to fit the Conv1d layers."""
        B, N, D, T = x.shape
        return x.reshape(B, N * D, T), (B, N, D, T)

    def _apply_conv(self, x: torch.Tensor, conv_layer: nn.Sequential) -> torch.Tensor:
        """Apply convolutional layer and activation function, then reshape back."""
        x = conv_layer(x)
        return self.act(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MSGBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D, T).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, D, T).
        """
        residual = x  # Store for residual connection

        # Reshape input for the first convolution
        x, original_shape = self._reshape_for_conv(x)
        # Apply large scale convolution and activation
        x = self._apply_conv(x, self.conv1)
        # Reshape back to (B, N, D, T) for further operations
        x = x.reshape(*original_shape).permute(0, 2, 1, 3)
        # Reshape again for the second convolution
        x, _ = self._reshape_for_conv(x)
        # Apply small scale convolution
        x = self.conv2(x)

        # Final reshape and residual connection
        x = x.reshape(original_shape[0], original_shape[2], original_shape[1], original_shape[3]).permute(0, 2, 1, 3)
        x = x + residual  # Residual connection

        return x