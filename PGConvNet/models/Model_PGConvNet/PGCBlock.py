import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


# Routing module that is used to decide which experts (kernels) to use based on the input
class _routing(nn.Module):
    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)
        # Fully connected layer with 1x1 convolution to generate routing weights for each expert
        self.fc = nn.Conv2d(in_channels, num_experts, kernel_size=1)

    def forward(self, x):
        # Pass the input through the fully connected layer
        x = self.fc(x)
        # Flatten the output to a 1D tensor
        x = torch.flatten(x)
        # Apply sigmoid activation to obtain routing weights in the range [0, 1]
        return torch.sigmoid(x)


# PGCBlock, a custom convolutional block that includes routing to combine multiple experts (kernels)
class PGCBlock(_ConvNd):
    def __init__(self, in_channels, out_channels, groups, kernel_size, num_experts, dropout_rate=0.5, stride=1,
                 padding=None, dilation=1, bias=True, padding_mode='zeros'):
        # Convert kernel_size, stride, padding, and dilation into pairs if they are not already
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        # Calculate padding if not provided
        if padding is None:
            padding = tuple((dilation * (k - 1)) // 2 for k in kernel_size)
        else:
            padding = _pair(padding)
        dilation = _pair(dilation)

        # Initialize parent class (_ConvNd) with provided parameters
        super(PGCBlock, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # Define an adaptive average pooling operation that outputs a 1x1 feature map
        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        # Initialize routing function to calculate the routing weights for the experts
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)
        # Set number of groups for grouped convolution
        self.groups = groups
        # Define the weight parameter with dimensions (num_experts, out_channels, in_channels // groups, kernel_height, kernel_width)
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        # Reset parameters to initialize weights
        self.reset_parameters()
        # Store input channel count
        self.channels = in_channels

    # Function to perform convolution operation
    def _conv_forward(self, input, weight):
        # If padding mode is not zeros, pad the input accordingly
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        # Perform convolution using the provided weights and bias
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    # Function to process routing weights and generate kernels
    def _get_kernels(self, pooled_inputs):
        # Calculate routing weights using the pooled input
        routing_weights = self._routing_fn(pooled_inputs)
        # Compute the weighted sum of expert kernels
        return torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)

    # Function to apply pooling and get routing weights
    def _get_routing_weights(self, input):
        # Apply adaptive average pooling to generate a 1x1 feature map
        return self._avg_pooling(input)

    # Function to process a single input sample
    def _process_single_input(self, input):
        # Add a new dimension to input to represent the batch (batch size = 1)
        input = input.unsqueeze(0)
        # Get the pooled input
        pooled_inputs = self._get_routing_weights(input)
        # Get the kernels based on routing weights
        kernels = self._get_kernels(pooled_inputs)
        # Perform convolution with the computed kernel
        return self._conv_forward(input, kernels)

    def forward(self, inputs):
        # Extract batch size and other dimensions from input
        b, _, _, _ = inputs.size()
        # Process each input in the batch
        res = [self._process_single_input(input) for input in inputs]
        # Concatenate all results along the batch dimension and return
        return torch.cat(res, dim=0)