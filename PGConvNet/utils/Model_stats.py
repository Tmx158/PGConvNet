import torch
from torchprofile import profile_macs


def calculate_model_statistics(model, input_tensor, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
    """
    Calculate and print the number of parameters and FLOPs for a given model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        input_tensor (torch.Tensor): A dummy input tensor to pass through the model for FLOPs calculation.
        x_mark_enc, x_dec, x_mark_dec, mask: Additional inputs required by the model.

    Returns:
        dict: A dictionary containing 'params' and 'flops', where 'params' is in millions (M).
    """
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_m = total_params / 1e6  # Convert to millions (M)

    # Calculate FLOPs (MACs)
    model.eval()
    macs = profile_macs(model, (input_tensor, x_mark_enc, x_dec, x_mark_dec, mask))

    return {"params_m": total_params_m, "flops": macs}
