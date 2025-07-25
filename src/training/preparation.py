import pandas as pd
import numpy as np
import torch

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