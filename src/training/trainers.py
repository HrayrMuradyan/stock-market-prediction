import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.training.predictions import get_predictions_in_batches
from torch.utils.data import DataLoader, TensorDataset


def trainer(model, X_train, y_train, X_valid, y_valid, optimizer, loss_fn, n_epochs=50, device="cpu", batch_size=32, shuffle=True):
    """
    Train a PyTorch model using the provided training data loader and evaluate on a validation set.

    This function handles the full training loop, including forward pass, loss computation, 
    backpropagation, optimizer step, and periodic validation loss evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset, yielding (X_batch, y_batch) tuples.
    X_valid : torch.Tensor
        Validation features tensor.
    y_valid : torch.Tensor
        Validation target tensor.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    loss_fn : callable
        Loss function to compute training and validation loss. Should accept predictions and targets as input.
    n_epochs : int, optional
        Number of training epochs (default is 50).
    device : str, optional
        Device on which to perform computations, e.g., "cpu" or "cuda" (default is "cpu").

    Returns
    -------
    model : torch.nn.Module
        The trained model after completing all epochs.

    Notes
    -----
    - Training loss is computed as the average loss per batch over each epoch.
    - Validation loss is computed at the end of each epoch using the full validation set.
    - Training and validation losses are printed after each epoch.
    """

    # ---------------- Input Checks ----------------
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected model to be an instance of torch.nn.Module, got {type(model).__name__}.")
    
    if not isinstance(X_train, torch.Tensor):
        raise TypeError(f"Expected X_train to be a torch.Tensor, got {type(X_train).__name__}.")

    if not isinstance(y_train, torch.Tensor):
        raise TypeError(f"Expected y_train to be a torch.Tensor, got {type(y_train).__name__}.")

    if not isinstance(X_valid, torch.Tensor):
        raise TypeError(f"Expected X_valid to be a torch.Tensor, got {type(X_valid).__name__}.")

    if not isinstance(y_valid, torch.Tensor):
        raise TypeError(f"Expected y_valid to be a torch.Tensor, got {type(y_valid).__name__}.")

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(f"Expected optimizer to be an instance of torch.optim.Optimizer, got {type(optimizer).__name__}.")

    if not callable(loss_fn):
        raise TypeError(f"Expected loss_fn to be a callable function, got {type(loss_fn).__name__}.")

    if not isinstance(n_epochs, int) or n_epochs <= 0:
        raise ValueError(f"Expected n_epochs to be a positive integer, got {n_epochs}.")

    if device not in ["cpu", "cuda"] and not device.type.startswith("cuda"):
        raise ValueError(f"Device must be either 'cpu' or 'cuda', got '{device}'.")

    # Shape check for X_valid and y_valid
    if X_valid.size(0) != y_valid.size(0):
        raise ValueError(f"X_valid and y_valid must have the same number of samples, got {X_valid.size(0)} and {y_valid.size(0)}.")
    
    # Wrap training data into a DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Convert the model to training mode
    model.train()

    # Define predictions list to save each epoch
    train_preds_list = []
    valid_preds_list = []

    # For each epoch
    for epoch in range(n_epochs):

        # Define the total loss
        total_loss = 0

        # For each batch
        for X_batch, y_batch in train_loader:

            # Convert the input and output batch to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Zero the accumulated gradient of the optimizer
            optimizer.zero_grad()

            # Calculate the predictions of the batch
            preds = model(X_batch)

            # Calculate the loss
            loss = loss_fn(preds, y_batch)

            # Calculate the gradients
            loss.backward()

            # Update the weights
            optimizer.step()

            # Add the batch loss to the total loss
            total_loss += loss.item()

        # Calculate the average train loss per sample by dividing the total loss by the number of batches
        avg_train_loss = total_loss / len(train_loader)

        # Convert the model to evaluation mode
        model.eval()

        # Disable the gradient calculation
        with torch.no_grad():

            # Get the validation prediction and calculate the loss
            valid_preds = get_predictions_in_batches(model, X_valid, batch_size=batch_size, device=device)
            val_loss = loss_fn(valid_preds, y_valid).item()

            # Add the valid predictions to list
            valid_preds_list.append(valid_preds)      

            # Get the train prediction
            train_preds_list.append(
                get_predictions_in_batches(model, X_train, batch_size=batch_size, device=device)
            )

        # Convert the model to training mode
        model.train()

        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

    return model, val_loss, {"valid_preds_list": valid_preds_list, "train_preds_list": train_preds_list}