import numpy as np

def collect_predictions_per_timestamp(y_pred):
    """
    Collect predictions for each timestamp by aggregating overlapping prediction windows.

    This function takes an array of predictions where each row corresponds to a prediction window
    (multiple future time steps predicted at once), and reassigns the predictions so that all
    predictions for each individual timestamp are grouped together. Predictions that are zero are ignored.
    The output is a 2D array where each row contains all predictions for a given timestamp,
    padded with NaNs if there are fewer predictions than the maximum window size.

    Parameters
    ----------
    y_pred : array-like of shape (n_windows, prediction_window)
        Array of predicted values where each row is a prediction window starting at a specific timestamp.

    Returns
    -------
    padded_preds : numpy.ndarray of shape (n_windows, prediction_window)
        Array where each row contains all collected predictions for that timestamp,
        padded with NaNs to maintain consistent shape.
    """
    # Convert predictions to a numpy array for easier processing
    y_pred = np.asarray(y_pred)

    # Get number of prediction windows and size of each window
    n_windows, prediction_window = y_pred.shape
    
    # Initialize a list of lists to hold predictions per timestamp
    all_preds = [[] for _ in range(n_windows)]

    # Loop over each prediction window and each predicted step in the window
    for i in range(n_windows):
        for j in range(prediction_window):
            # Calculate the actual timestamp the prediction corresponds to
            t = i + j

            # Only collect the prediction if timestamp is valid and prediction is non-zero
            if t < n_windows and y_pred[i, j] != 0:
                all_preds[t].append(y_pred[i, j])

    # Initialize output array padded with NaNs to handle varying number of predictions per timestamp
    padded_preds = np.full((n_windows, prediction_window), np.nan)

    # Assign collected predictions to their corresponding timestamp row in the output array
    for t, preds in enumerate(all_preds):
        padded_preds[t, :len(preds)] = preds

    return padded_preds