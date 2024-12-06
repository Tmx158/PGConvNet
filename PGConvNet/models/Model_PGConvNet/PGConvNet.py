import torch
from torch import nn
from models.Model_PGConvNet.Imputation_Layer import Imputation
from models.Model_PGConvNet.Embedding_Layer import Embedding
from models.Model_PGConvNet.MSGBlock import MSGBlock
from models.Model_PGConvNet.PGCBlock import PGCBlock


class Model(nn.Module):
    """
    Main Model class integrating embedding, MSG, PGC, and imputation layers for processing time series data.

    Args:
        configs: Configuration object containing model hyperparameters.
            task_name (str): Name of the task (e.g., imputation, forecasting).
            num_blocks (int): Number of MSG blocks to apply in the model.
            large_size (int): Kernel size for large-scale convolutions in MSGBlock.
            small_size (int): Kernel size for small-scale convolutions in MSGBlock.
            kernel_size (int): Kernel size for PGCBlock.
            num_experts (int): Number of experts for PGCBlock dynamic routing.
            updim (int): Dimensionality of the embedded features.
            nvars (int): Number of input variables (e.g., number of features).
            dropout (float): Dropout rate for regularization.
            revin (bool): Whether to apply reversible normalization in the input sequence.
            affine (bool): Whether to use affine transformation in normalization.
            subtract_last (bool): Option for normalization technique.
            freq (str): Frequency of the time series data.
            seq_len (int): Length of the input sequence.
            pred_len (int): Length of the prediction sequence.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # Model hyperparameters
        self.task_name = configs.task_name
        self.num_blocks = configs.num_blocks
        self.large_size = configs.large_size
        self.small_size = configs.small_size
        self.kernel_size = configs.kernel_size
        self.num_experts = configs.num_experts
        self.updim = configs.updim
        self.nvars = configs.enc_in
        self.dropout = configs.dropout
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last
        self.freq = configs.freq
        self.seq_len = configs.seq_len
        self.c_in = self.nvars
        self.pred_len = configs.pred_len

        # Embedding, imputation, and convolutional blocks
        self.Embedding = Embedding(dmodel=self.updim)
        self.Imputation = Imputation(len=self.updim * self.seq_len, pred_len=self.pred_len, nvars=self.nvars)
        self.MSGBlock = MSGBlock(updim=self.updim, nvars=self.nvars, large_size=self.large_size,
                                 small_size=self.small_size,dropout=self.dropout)
        self.PGCBlock = PGCBlock(in_channels=self.nvars, out_channels=self.nvars, groups=self.nvars,
                                 kernel_size=self.kernel_size, num_experts=self.num_experts)
        self.Gelu = nn.GELU()

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, nvars, seq_len).
            x_mark_enc (torch.Tensor): Time encoding for the encoder.
            x_dec (torch.Tensor): Decoder input tensor.
            x_mark_dec (torch.Tensor): Time encoding for the decoder.
            mask (torch.Tensor, optional): Mask tensor indicating valid data points.

        Returns:
            torch.Tensor: Output tensor after imputation, of shape (batch_size, pred_len, nvars).
        """

        # Permute dimensions for processing
        x = x.permute(0, 2, 1)  # Shape becomes (batch_size, seq_len, nvars)

        # Reversible normalization (RevIN) if enabled
        if self.revin:
            x, means, stdev = self._revin(x, mask, reverse=False)

        # Embedding layer
        x = self.Embedding(x)  # Shape becomes (batch_size, seq_len, nvars, updim)
        x = x.permute(0, 1, 3, 2)  # Shape becomes (batch_size, seq_len, updim, nvars)
        # MSGBlock processing for multiple blocks
        for i in range(self.num_blocks):
            x = self.MSGBlock(x)
            x = self.Gelu(x)
        #PGCBlock processing
        x = self.PGCBlock(x)
        # Imputation layer for final prediction
        x = self.Imputation(x)
        # Permute back to (batch_size, pred_len, nvars)
        x = x.permute(0, 2, 1)
        # Reversible normalization recovery if enabled
        if self.revin:
            x, _, _ = self._revin(x, mask, reverse=True, means=means, stdev=stdev)
        return x
    def _revin(self, x, mask, reverse=False, means=None, stdev=None):
        """
        Applies or reverses reversible normalization on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, nvars).
            mask (torch.Tensor): Mask tensor indicating valid data points.
            reverse (bool): Whether to reverse normalization.
            means (torch.Tensor, optional): Mean values for de-normalization.
            stdev (torch.Tensor, optional): Standard deviation values for de-normalization.

        Returns:
            torch.Tensor: Normalized or de-normalized tensor.
            torch.Tensor: Means used for normalization (if reverse is False).
            torch.Tensor: Standard deviations used for normalization (if reverse is False).
        """
        if not reverse:
            x_enc = x.permute(0, 2, 1)  # Shape becomes (batch_size, nvars, seq_len)
            means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)  # Compute mean for normalization
            means = means.unsqueeze(1).detach()
            x_enc = x_enc - means
            x_enc = x_enc.masked_fill(mask == 0, 0)  # Mask out invalid points

            # Standard deviation calculation for normalization
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x_enc /= stdev
            return x_enc.permute(0, 2, 1), means, stdev  # Shape returns to (batch_size, seq_len, nvars)
        else:
            dec_out = x
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
            return dec_out, means, stdev