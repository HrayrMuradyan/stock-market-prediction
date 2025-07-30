import torch

def validate_device(device):
    """
    Validates and returns a PyTorch device based on the input string.

    Parameters
    ----------
    device : str
        The desired device for model training or inference.
        Expected values include:
        - 'cpu'
        - 'cuda'
        - 'cuda:0', 'cuda:1', etc.

    Returns
    -------
    torch.device
        A valid PyTorch device object corresponding to the input string.

    """

    # Device should be a string
    
    if not isinstance(device, str):
        raise TypeError(f"Expected `device` to be a string, got {type(device).__name__}")

    # Lower the device string
    device = device.lower()

    # If it's cpu, return cpu device
    if device == "cpu":
        return torch.device("cpu")

    # If starts with cuda
    if device.startswith("cuda"):

        # But not available, raise an error
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA device requested but CUDA is not available.")

        # If available, return it
        return torch.device(device)

    # If none of the conditions worked, something is wrong with the device
    raise ValueError(f"Unsupported device '{device}'. Use 'cpu' or 'cuda[:N]'.")