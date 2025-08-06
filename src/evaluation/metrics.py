import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.checks import check_datatypes
import torch
import numpy as np

def calc_mse(y_true, y_pred):
    """
    Calculate the element-wise squared error and mean squared error (MSE) along the last axis.

    This function supports both NumPy arrays and PyTorch tensors, and ensures that the inputs are
    of the same type and floating-point format. It computes the squared error and returns both the
    squared error tensor/array and the mean squared error per row (last axis).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values. Must be the same type and shape as `y_true`.

    Returns:
        Tuple:
            - se (np.ndarray or torch.Tensor): Element-wise squared errors.
            - mse (np.ndarray or torch.Tensor): Mean squared error along the last axis.
    """
    
    # Check types
    check_datatypes([
        ("y_true", y_true, (torch.Tensor, np.ndarray)),
        ("y_pred", y_pred, (torch.Tensor, np.ndarray)),
    ])

    # Both arguments should be of the same type
    if type(y_true) != type(y_pred):
        raise TypeError("y_true and y_pred must be of the same type")

    # Convert to float for torch
    if isinstance(y_true, torch.Tensor):
        y_true, y_pred = y_true.float(), y_pred.float()
        
    # Convert to float for Numpy
    else:
        y_true, y_pred = y_true.astype(np.float32), y_pred.astype(np.float32)

    # Calculate the squared error and mean squared error
    se = (y_true - y_pred) ** 2
    mse = se.mean(axis=-1)
    
    return se, mse


def calc_mae(y_true, y_pred):
    """
    Calculate the element-wise absolute error and mean absolute error (MAE) along the last axis.

    This function supports both NumPy arrays and PyTorch tensors, and ensures that the inputs are
    of the same type and floating-point format. It computes the absolute error and returns both the
    absolute error tensor/array and the mean absolute error per row (last axis).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values. Must be the same type and shape as `y_true`.

    Returns:
        Tuple:
            - ae (np.ndarray or torch.Tensor): Element-wise absolute errors.
            - mae (np.ndarray or torch.Tensor): Mean absolute error along the last axis.
    """
    
    # Check types
    check_datatypes([
        ("y_true", y_true, (torch.Tensor, np.ndarray)),
        ("y_pred", y_pred, (torch.Tensor, np.ndarray)),
    ])

    # Both arguments should be of the same type
    if type(y_true) != type(y_pred):
        raise TypeError("y_true and y_pred must be of the same type")

    # Convert to float for torch
    if isinstance(y_true, torch.Tensor):
        y_true, y_pred = y_true.float(), y_pred.float()
        
    # Convert to float for Numpy
    else:
        y_true, y_pred = y_true.astype(np.float32), y_pred.astype(np.float32)

    # Calculate the absolute error and mean absolute error
    ae = abs(y_true - y_pred)
    mae = ae.mean(axis=-1)
    
    return ae, mae


def calc_rmse(y_true, y_pred):
    """
    Calculate the element-wise squared error and root mean squared error (RMSE) along the last axis.

    This function supports both NumPy arrays and PyTorch tensors, and ensures that the inputs are
    of the same type and floating-point format. It computes the squared error and returns both the
    squared error tensor/array and the root mean squared error per row (last axis).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values. Must be the same type and shape as `y_true`.

    Returns:
        Tuple:
            - se (np.ndarray or torch.Tensor): Element-wise squared errors.
            - rmse (np.ndarray or torch.Tensor): Root mean squared error along the last axis.
    """
    
    # Check types
    check_datatypes([
        ("y_true", y_true, (torch.Tensor, np.ndarray)),
        ("y_pred", y_pred, (torch.Tensor, np.ndarray)),
    ])

    # Both arguments should be of the same type
    if type(y_true) != type(y_pred):
        raise TypeError("y_true and y_pred must be of the same type")

    # Convert to float
    if isinstance(y_true, torch.Tensor):
        y_true, y_pred = y_true.float(), y_pred.float()
        sqrt_fn = torch.sqrt
    else:
        y_true, y_pred = y_true.astype(np.float32), y_pred.astype(np.float32)
        sqrt_fn = np.sqrt

    # Calculate squared error and root mean squared error
    se = (y_true - y_pred) ** 2
    mse = se.mean(axis=-1)
    rmse = sqrt_fn(mse)

    return se, rmse



def calc_mape(y_true, y_pred, epsilon=1e-8):
    """
    Calculate the element-wise absolute percentage error and mean absolute percentage error (MAPE) along the last axis.

    This function supports both NumPy arrays and PyTorch tensors, and ensures that the inputs are
    of the same type and floating-point format. It computes the absolute percentage error and returns both the
    absolute percentage error tensor/array and the mean absolute percentage error per row (last axis).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values. Must be the same type and shape as `y_true`.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        Tuple:
            - ape (np.ndarray or torch.Tensor): Element-wise absolute percentage errors.
            - mape (np.ndarray or torch.Tensor): Mean absolute percentage error along the last axis.
    """

    # Check types
    check_datatypes([
        ("y_true", y_true, (torch.Tensor, np.ndarray)),
        ("y_pred", y_pred, (torch.Tensor, np.ndarray)),
    ])

    # Both arguments should be of the same type
    if type(y_true) != type(y_pred):
        raise TypeError("y_true and y_pred must be of the same type")

    # Convert to float
    if isinstance(y_true, torch.Tensor):
        y_true, y_pred = y_true.float(), y_pred.float()

    else:
        y_true, y_pred = y_true.astype(np.float32), y_pred.astype(np.float32)

    # Avoid division by zero by adding epsilon where y_true is zero
    denom = y_true.clone() if isinstance(y_true, torch.Tensor) else y_true.copy()
    zero_mask = denom == 0
    denom = denom + epsilon * zero_mask.float() if isinstance(denom, torch.Tensor) else denom + epsilon * zero_mask

    # Calculate absolute percentage error and mean absolute percentage error
    ape = abs((y_true - y_pred) / denom) * 100  
    mape = ape.mean(axis=-1)

    return ape, mape


def calc_directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy for stock price predictions.

    Args:
        y_true (np.ndarray or torch.Tensor): Actual prices, shape (n_samples,)
        y_pred (np.ndarray or torch.Tensor): Predicted prices, shape (n_samples,)

    Returns:
        float: Directional accuracy as a proportion between 0 and 1.
    """
    # Check types
    check_datatypes([
        ("y_true", y_true, (torch.Tensor, np.ndarray)),
        ("y_pred", y_pred, (torch.Tensor, np.ndarray)),
    ])

    if type(y_true) != type(y_pred):
        raise TypeError("y_true and y_pred must be of the same type")

    # Compute price changes (differences)
    if isinstance(y_true, torch.Tensor):
        actual_change = y_true[1:] - y_true[:-1]
        pred_change = y_pred[1:] - y_pred[:-1]
        actual_direction = torch.sign(actual_change)
        pred_direction = torch.sign(pred_change)
        correct = (actual_direction == pred_direction).float()
        accuracy = correct.mean().item()
    else:
        actual_change = np.diff(y_true)
        pred_change = np.diff(y_pred)
        actual_direction = np.sign(actual_change)
        pred_direction = np.sign(pred_change)
        correct = actual_direction == pred_direction
        accuracy = np.mean(correct)

    return correct, accuracy