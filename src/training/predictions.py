import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from torch.utils.data import DataLoader, TensorDataset
from src.utils.checks import check_datatypes

def get_predictions_in_batches(model, X, batch_size=32, device="cpu"):
    """
    Generate predictions from a PyTorch model on a large dataset in batches.

    This function splits the input tensor `X` into batches and performs 
    inference using the provided model without computing gradients.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model used for inference.
    X : torch.Tensor
        Input features to predict on, shape (n_samples, n_features).
    batch_size : int, optional
        Batch size used during prediction (default is 32).
    device : str, optional
        The device on which computations will be performed (e.g., "cpu" or "cuda").

    Returns
    -------
    torch.Tensor
        Concatenated model predictions for all inputs, shape (n_samples, output_dim).

    Notes
    -----
    - The model is set to evaluation mode during inference.
    - Gradient tracking is disabled for efficiency.
    - Input tensor `X` is not shuffled to preserve order.
    """
    model.eval()
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (X_batch,) in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu())

    return torch.cat(all_preds, dim=0)
