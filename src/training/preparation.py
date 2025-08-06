import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data.validation import validate_data
from src.utils.path import verify_existing_dir, verify_saving_path
from src.data.stocks import read_and_concat_all_stocks
from src.training.feature_engineering import add_date_as_feature, add_lagged_features


def time_series_train_test_split(X, y, validation_window=10):
    """
    Splits time-series data into training and validation sets.

    Parameters
    ----------
    X : pd.DataFrame, np.ndarray, or list
        Feature data.
    y : pd.DataFrame, np.ndarray, or list
        Target data.
    validation_window : int
        Number of most recent observations to use for validation.

    Returns
    -------
    tuple
        (X_train, X_valid, y_train, y_valid)
    """
    # Validate types
    valid_types = (pd.DataFrame, np.ndarray, list, pd.Series)
    for var, name in [(X, "X"), (y, "y")]:
        if not isinstance(var, valid_types):
            raise TypeError(f"Expected {name} to be one of {valid_types}, got {type(var).__name__}.")

    if not isinstance(validation_window, int):
        raise TypeError(f"Expected validation_window to be an integer. Got {type(validation_window).__name__}.")

    # Validate lengths
    if len(X) < validation_window or len(y) < validation_window:
        raise ValueError("Validation window is larger than the number of samples.")

    return X[:-validation_window], X[-validation_window:], y[:-validation_window], y[-validation_window:]


def separate_the_target_column(data, target_column):
    """
    Separates the target column from the input DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing features and the target column.
    
    target_column : str
        The name of the target column to separate.

    Returns
    -------
    X : pd.DataFrame
        A DataFrame containing all columns except the target column.
    
    y : pd.Series
        A Series containing the values of the target column.

    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected data to be a Pandas DataFrame. Got {type(data).__name__}.")

    if not isinstance(target_column, str):
        raise TypeError(f"Expected target_column to be a string. Got {type(target_column).__name__}.")

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame columns.")
        
    return data.drop(target_column, axis=1), data[target_column]



def prepare_nn_multistep_dataset(X_raw, y_raw, prediction_window=7):
    """
    Prepare a dataset for multi-step time series forecasting using neural networks.

    Each input sample corresponds to one timestamp's features, and each target sample 
    is a vector of future target values (next `prediction_window` steps).

    Parameters
    ----------
    X_raw : pd.DataFrame
        DataFrame containing the input features for each timestamp.
    y_raw : pd.Series or pd.DataFrame
        Series or single-column DataFrame containing the target variable.
    prediction_window : int, optional
        Number of future timesteps to predict for each input (default is 7).

    Returns
    -------
    X_tensor : torch.FloatTensor
        Tensor of shape (num_samples, num_features), representing input features.
    y_tensor : torch.FloatTensor
        Tensor of shape (num_samples, prediction_window), representing multi-step targets.
        Zero-padding is applied when fewer than `prediction_window` future values are available.

    Notes
    -----
    For samples near the end of the time series, if fewer than `prediction_window` target values 
    are available, the remaining positions are filled with zeros.
    """
    
    # Convert input features and target variable to NumPy arrays
    X_np = X_raw.to_numpy()
    y_np = y_raw.to_numpy()

    data_size = len(X_np)

    # Initialize target matrix with zeros: each row holds the next `prediction_window` target values
    y_targets = np.zeros((data_size, prediction_window), dtype=np.float32)

    # Populate the target matrix
    for i in range(data_size):
        # Determine the valid range of future target values from index i
        end = min(i + prediction_window, data_size)

        # Assign available future target values to the current row
        y_targets[i, :end - i] = y_np[i:end]

    return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_targets, dtype=torch.float32)


def prepare_data_for_training(data_folder_path, target_columns, n_lags=1, validation_size=30, prediction_window=7, save_path="./"):
    """
    Prepares time series data for training and validation in a neural network model.

    This function reads stock time series data from a folder, applies lagged feature generation,
    splits the data into training and validation sets using a time-aware split, and formats the
    data into multistep sequences suitable for neural network training.

    Args:
        data_folder_path (str): Path to the folder containing stock data files (e.g., CSVs).
        target_columns (list): The list of the target columns to be predicted.
        n_lags (int): Number of lagged time steps to include as features. Must be > 0.
        validation_size (int): Number of time steps to reserve for validation. Must be > 0.
        prediction_window (int): Number of future time steps to predict (multistep output). Must be > 0.
        save_path (str): Path where any intermediate files or outputs should be saved.

    Returns:
        tuple: A tuple containing four elements:
            - X_train (Tensor or np.ndarray): Input features for training.
            - y_train (Tensor or np.ndarray): Target values for training (multistep).
            - X_valid (Tensor or np.ndarray): Input features for validation.
            - y_valid (Tensor or np.ndarray): Target values for validation (multistep).

    Raises:
        TypeError: If input types are not as expected.
        ValueError: If validation size is too large, or required columns are missing,
                    or any of the integer arguments are invalid.
    """
    # Convert the paths to Path object first
    data_folder_path = Path(data_folder_path)
    save_path = Path(save_path)
    
    # Check input
    verify_existing_dir(data_folder_path)

    if not isinstance(target_columns, list):
        raise TypeError(f"Expected target_column to be a list. Got {type(target_columns).__name__}.")

    if not isinstance(n_lags, int):
        raise TypeError(f"Expected n_lags to be an integer. Got {type(n_lags).__name__}.") 

    if n_lags <= 0:
        raise ValueError(f"Expected n_lags to be an integer greater than 0. You have n_lags={n_lags}")

    if not isinstance(validation_size, int):
        raise TypeError(f"Expected validation_size to be an integer. Got {type(validation_size).__name__}.") 

    if validation_size <= 0:
        raise ValueError(f"Expected validation_size to be greater than 0. Got validation_size={validation_size}.")

    if not isinstance(prediction_window, int):
        raise TypeError(f"Expected prediction_window to be an integer. Got {type(prediction_window).__name__}.") 

    if prediction_window <= 0:
        raise ValueError(f"Expected prediction_window to be greater than 0. Got prediction_window={prediction_window}.")


    verify_saving_path(save_path)

    # Read and concat all stocks data
    # Ex. shape is (number of days, number of stocks) = (3500, 420)
    data = read_and_concat_all_stocks(data_folder_path)

    # Validate data
    data = validate_data(data)
    
    # The target column should be in the data
    if not any([col in data.columns for col in target_columns]):
        raise ValueError(f"None of the provided target columns is present in the data folder you provided. Please verify their presence.")

    # Check if the validation size is larger than the data size
    data_rows = data.shape[0]

    if validation_size >= data_rows:
        raise ValueError(f"Expected validation_size to be smaller than the size of the data. Data rows = {data_rows}, validation_size={validation_size}")
    
    # The most recent date in the data
    most_recent_date_in_data = (data.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Add date as feature
    # Ex. shape is (number of days, number of stocks+3) = (3500, 423)
    data = add_date_as_feature(data)

    # For each target column create a processed dataframe and save it
    for target_column in target_columns:
        if target_column not in data.columns:
            print(f"The target column {target_column} was not in the dataframe. Skipping...")
            continue

        # Added lagged features to the data (#new-features = #features x n_lags)
        data_lagged, last_row_lagged = add_lagged_features(data, target_column, n_lags=n_lags)

        # Separate train data from the target
        X, y = separate_the_target_column(data_lagged, target_column=target_column)

        # Split the data into train and validation
        X_train_raw, X_valid_raw, y_train_raw, y_valid_raw = time_series_train_test_split(X=X,
                                                                                          y=y,
                                                                                          validation_window=validation_size)
        
        # Save the dates of the train and validation sets for later use
        train_dates = list(y_train_raw.index)
        valid_dates = list(y_valid_raw.index)

        # Prepare the dataset into tensors. The target variable is converted to a prediction window
        # Ex. prediction_window=7, each timestamp prediction is a 7-length tensor
        X_train, y_train = prepare_nn_multistep_dataset(X_train_raw, y_train_raw, prediction_window=prediction_window)

        
        X_valid, y_valid = prepare_nn_multistep_dataset(X_valid_raw, y_valid_raw, prediction_window=prediction_window)

        # Get todays date
        todays_date = datetime.today().strftime("%Y-%m-%d")

        # Define the data folder
        save_folder = save_path / todays_date / target_column

        # Verify if the save folder exists, if not create it
        verify_saving_path(save_folder)

        # Save datasets
        torch.save(X_train, save_folder / "X_train.pt")
        torch.save(y_train, save_folder / "y_train.pt")
        torch.save(X_valid, save_folder / "X_valid.pt")
        torch.save(y_valid, save_folder / "y_valid.pt")

        # Save the dates
        torch.save(train_dates, save_folder / "y_train_dates.pt")
        torch.save(valid_dates, save_folder / "y_valid_dates.pt")

        # Save the last row for later real-time predictions
        torch.save(
            {most_recent_date_in_data: torch.tensor(last_row_lagged.to_numpy(), dtype=torch.float32)},
            save_folder / "last_row.pt"
        )

