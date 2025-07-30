import torch
from datetime import datetime
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.checks import check_datatypes
from src.utils.path import verify_saving_path
from src.training.trainers import trainer
from src.training.loss_functions import masked_mse_loss

from torch.utils.data import DataLoader, TensorDataset


def train_model(model_architecture,
                X_train,
                y_train,
                X_valid,
                y_valid,
                loss_fn,
                prediction_window=7,
                batch_size=32,
                n_epochs=100,
                learning_rate=0.001,
                shuffle=True,
                model_params=None,
                device='cpu',
                save_path="./"):

    """
    Train a PyTorch model using the provided architecture and datasets, then save the traced model.

    Parameters
    ----------
    model_architecture : type
        A PyTorch module class (not instance) representing the model architecture. It must accept `input_dim` and 
        `output_dim` as keyword arguments in addition to any custom `model_params`.
    
    X_train : torch.Tensor
        Training input features of shape (n_samples, n_features).
    
    y_train : torch.Tensor
        Training target values of shape (n_samples, prediction_window).
    
    X_valid : torch.Tensor
        Validation input features of shape (n_samples, n_features).
    
    y_valid : torch.Tensor
        Validation target values of shape (n_samples, prediction_window).
    
    loss_fn : callable
        Loss function used during training. Must accept predicted and true outputs as inputs.
    
    prediction_window : int, optional
        Number of future steps to predict, by default 7.
    
    batch_size : int, optional
        Number of samples per training batch, by default 32.
    
    n_epochs : int, optional
        Number of epochs to train for, by default 100.
    
    learning_rate : float, optional
        Learning rate for the optimizer, by default 0.001.
    
    shuffle : bool, optional
        Whether to shuffle the training data during batching, by default True.
    
    model_params : dict or None, optional
        Additional keyword arguments to pass to the model constructor, by default None.
    
    device : str, optional
        Device on which to train the model (e.g., 'cpu' or 'cuda'), by default 'cpu'.
    
    save_path : str or pathlib.Path, optional
        Directory path where the traced model will be saved, by default "./".

    Returns
    -------
    model : torch.nn.Module
        The trained PyTorch model instance.

    """

    data_types_schema = [
        ("model_architecture", model_architecture, torch.nn.Module),
        ("X_train", X_train, torch.Tensor),
        ("y_train", y_train, torch.Tensor),
        ("X_valid", X_valid, torch.Tensor),
        ("y_valid", y_valid, torch.Tensor),
        ("loss_fn", loss_fn),
        ("prediction_window", prediction_window, int),
        ("batch_size", batch_size, int),
        ("n_epochs", n_epochs, int),
        ("learning_rate", learning_rate, float),
        ("shuffle", shuffle, bool),
        ("model_params", model_params, dict),
        ("device", device, str),
        ("save_path", save_path, (str, Path))
    ]

    # Check if all data types provided are correct
    check_datatypes(data_types_schema)

    # Check if the provided saving path is correctly provided
    save_path = Path(save_path)
    verify_saving_path(save_path)

    # Get todays date
    todays_date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

    # Model params is an empty dictionary if None is provided
    model_params = {} if model_params is None else model_params

    # Get the input size
    input_size = X_train.shape[1]

    # Wrap training data into a DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # Define the model
    model = model_architecture(input_dim=input_size,
                               output_dim=prediction_window,
                               **model_params).to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model, val_loss = trainer(
        model=model,
        train_loader=train_loader,
        X_valid=X_valid,
        y_valid=y_valid,
        optimizer=optimizer,
        loss_fn=masked_mse_loss,
        n_epochs=n_epochs,
        device=device
    )

    # Note that it's assumed that input is 1D here
    example_input = torch.randn(1, input_size)

    # Produce the computational graph
    traced_model = torch.jit.trace(model.cpu(), example_input)

    # Save the model
    full_save_path = save_path / f"{model_architecture.__name__}_{todays_date}_{val_loss}.pt"
    traced_model.save(full_save_path)

    print(f"Model successfully saved here: {full_save_path}")
