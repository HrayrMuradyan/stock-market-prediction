import torch
from datetime import datetime
from pathlib import Path
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.checks import check_datatypes
from src.utils.path import verify_saving_path
from src.training.trainers import trainer
from src.training.loss_functions import masked_mse_loss
from src.utils.config import load_config
from src.data.stocks import get_most_recent_processed_data_path, read_processed_data




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
                device='cpu'):

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
        ("device", device, str)
    ]

    # Check if all data types provided are correct
    check_datatypes(data_types_schema)

    # Model params is an empty dictionary if None is provided
    model_params = {} if model_params is None else model_params

    # Get the input size
    input_size = X_train.shape[1]

    # Define the model
    model = model_architecture(input_dim=input_size,
                               output_dim=prediction_window,
                               **model_params).to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model, val_loss, preds_dict = trainer(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        optimizer=optimizer,
        loss_fn=masked_mse_loss,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device
    )

    # Save the hyperparameters used
    hyperparams = {
        "training_process": {"input_size": input_size},
    }

    return model, val_loss, hyperparams, preds_dict

    


def train_model_from_config():
    """
    Load configuration, read training and validation data, prepare model and training parameters, 
    and train the model.

    This function serves as an end-to-end pipeline that:
      - Loads configuration from a YAML file.
      - Dynamically loads the model architecture and loss function based on config entries.
      - Loads preprocessed stock data for training and validation.
      - Trains a PyTorch model using the parameters specified in the configuration.
      - Saves the trained model to the path provided in the config.

    """

    import src.training.models as models
    import src.training.loss_functions as loss_functions
    
    config = load_config()

    model_architecture_str = config["train_config"]["model"]["architecture"]
    model_architecture = getattr(models, model_architecture_str)

    loss_fn_str = config["train_config"]["optimization"]["loss_fn"]
    loss_fn = getattr(loss_functions, loss_fn_str)

    prediction_window=config["train_config"]["target"]["prediction_window"]
    batch_size=config["train_config"]["optimization"]["batch_size"]
    n_epochs=config["train_config"]["optimization"]["n_epochs"]
    learning_rate=config["train_config"]["optimization"]["learning_rate"]
    shuffle=config["train_config"]["optimization"]["shuffle"]
    model_params=config["train_config"]["model"]["model_params"]
    device=config["train_config"]["general"]["device"]
    model_save_path=config["train_config"]["model"]["save_path"]

    most_recent_processed_data_path = get_most_recent_processed_data_path(config["stock"]["stock_data_processed_save_path"])

    all_target_folders = most_recent_processed_data_path.glob("*")

    if not all_target_folders:
        raise FileNotFoundError(f"There are no <Stock> folders in the processed data path: {most_recent_processed_data_path}. Please refer to the documentation.")
    
    # Get todays date
    todays_date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

    for target_folder in all_target_folders:
    
        X_train, X_valid, y_train, y_valid, _, _ = read_processed_data(target_folder)

        target_stock = target_folder.stem
        target_folder_str = str(target_folder)

        if target_stock not in config["train_config"]["target"]["target_columns"]:
            print(f"Stock {target_stock} is not in the target list defined in the config.")

        model, val_loss, hyperparams, preds_dict = train_model(model_architecture=model_architecture,
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    loss_fn=loss_fn,
                    prediction_window=prediction_window,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    learning_rate=learning_rate,
                    shuffle=shuffle,
                    model_params=model_params,
                    device=device)
        
        # Note that it's assumed that input is 1D here
        example_input = torch.randn(1, hyperparams["training_process"]["input_size"])

        # Produce the computational graph
        traced_model = torch.jit.trace(model.cpu(), example_input)

        # Main folder
        save_path_stock = Path(model_save_path) / todays_date / target_stock

        # Verify saving path
        verify_saving_path(save_path_stock)

        # Save the model
        full_save_path = save_path_stock / f"model_{model_architecture.__name__}_val_loss_{round(val_loss, 3)}.pt"
        traced_model.save(full_save_path)

        print(f"Model successfully saved here: {full_save_path}")

        # Save the hyperparameters
        hyperparams["data_from"] = target_folder_str
        hyperparams["model_architecture"] = model_architecture_str
        hyperparams["loss_fn"] = loss_fn_str
        hyperparams["prediction_window"] = prediction_window
        hyperparams["batch_size"] = batch_size
        hyperparams["n_epochs"] = n_epochs
        hyperparams["learning_rate"] = learning_rate
        hyperparams["shuffle"] = shuffle
        hyperparams["model_params"] = model_params
        hyperparams["device"] = device

        with open(save_path_stock / "hyperparams.json", "w") as f:
            json.dump(hyperparams, f, indent=4)

        # Save the preds list
        torch.save(preds_dict, save_path_stock / "predictions_list.pt")
