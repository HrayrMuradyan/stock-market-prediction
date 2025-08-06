import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.path import verify_existing_dir, verify_saving_path, get_project_root_path
from src.utils.config import load_config
from src.utils.checks import check_datatypes
from src.inference.models import get_most_recent_model
import torch
from pathlib import Path
from datetime import timedelta
import pandas_market_calendars as mcal
from src.data.stocks import read_processed_data
from src.evaluation.plots import plot_inference_results

def get_future_predictions(model_folder, data_path):
    """
    Load the latest TorchScript model and make predictions using the most recent input data.

    Args:
        model_folder (str or Path): Path to the folder containing model files named like 'model_*.pt'.
        data_path (str or Path): Path to the directory containing 'last_row.pt' with the latest input data.

    Returns:
        tuple:
            - prediction_date (str): The date corresponding to the input row used for prediction.
            - predictions (torch.Tensor): The model's predicted values, detached from the computation graph.
    
    """
    
    # Check the inputs
    check_datatypes([
        ("model_folder", model_folder, (str, Path)),
        ("data_path", data_path, (str, Path))
    ])

    # Convert the paths to Path object
    model_folder = Path(model_folder)
    data_path = Path(data_path) 

    # Read the model 
    model_path = sorted(model_folder.glob("model_*.pt"))[-1]
    model = torch.jit.load(model_path)

    # Read the latest date and the input (to use for real-time predictions)
    data_file = data_path / "last_row.pt"
    prediction_date_input_dict = torch.load(data_file)

    # Get the predictions from the input
    prediction_date, X = next(iter(prediction_date_input_dict.items()))
    predictions = model(X)
    
    return prediction_date, predictions.detach()


def get_next_trading_days(start_date, n_days):
    """
    Generate a list of the next n NYSE trading days (excluding weekends and market holidays), 
    starting strictly after the given start_date.

    Parameters
    ----------
    start_date : str or datetime-like
        The reference date (exclusive). The first trading day returned will be after this date.
    
    n_days : int
        The number of future trading days to return.

    Returns
    -------
    List[str]
        A list of date strings in "YYYY-MM-DD" format representing the next n trading days.

    Notes
    -----
    This function uses the official NYSE trading calendar via `pandas_market_calendars`.
    It skips both weekends and exchange holidays (e.g., New Year’s Day, Independence Day, etc.).
    """
    # Check the inputs
    check_datatypes([
        ("start_date", start_date, (str, pd.Timestamp)),
        ("n_days", n_days, int),
    ])
    
    # Get the NYSE trading calendar
    nyse = mcal.get_calendar('NYSE')

    # Define a date range to look ahead 
    schedule = nyse.schedule(
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(start_date) + pd.Timedelta(days=n_days)
    )

    # Extract the trading days strictly after the given start_date
    # (e.g., if start_date is Friday, next trading day would be Monday or later)
    future_trading_days = schedule.index[schedule.index > pd.to_datetime(start_date)][:n_days]

    # Convert the timestamps to strings in "YYYY-MM-DD" format and return as a list
    return future_trading_days.strftime("%Y-%m-%d").tolist()


def update_predictions_metadata(predictions,
                               prediction_date,
                               model_name,
                               predictions_folder_path,
                               predictions_data_file_name,
                               predictions_log_file_name):
    """
    Updates and saves prediction metadata and logs for a forecasting model.

    This function logs individual forecasts, overall prediction metadata and averaged predictions for each timestamp to CSV files.
    It appends new entries only if a prediction for the given prediction_date has not been recorded yet.

    Parameters
    ----------
    predictions : torch.Tensor
        A tensor containing the forecasted values, where each entry corresponds to a future day's prediction.
    
    prediction_date : str
        A string in "YYYY-MM-DD" format representing the date the prediction was made.
    
    model_name : str
        The name or identifier of the model that generated the predictions.
    
    predictions_folder_path : str or Path
        Path to the CSV file where detailed predictions and predictions logs for each forecasted day are stored.

    predictions_data_file_name : str
        Name of the CSV file where detailed predictions for each forecasted day are stored.
        Each row represents a single prediction.
    
    predictions_log_file_name : str
        Name of the CSV file where metadata for each prediction run is stored.
        Each row represents a full prediction run (batch).

    Returns
    -------
    None
        This function has no return value. It updates the given CSV files in place.

    """
    
    # Check the inputs
    check_datatypes([
        ("predictions", predictions, torch.Tensor),
        ("prediction_date", prediction_date, str),
        ("model_name", model_name, str),
        ("predictions_folder_path", predictions_folder_path, (str, Path)),
        ("predictions_data_file_name", predictions_data_file_name, str),
        ("predictions_log_file_name", predictions_log_file_name, str)
    ])

    # Convert the paths to Path object
    predictions_folder_path = Path(predictions_folder_path)

    verify_existing_dir(predictions_folder_path)

    # Define the predictions data and log file paths
    predictions_data_path = predictions_folder_path / predictions_data_file_name
    predictions_log_path = predictions_folder_path / predictions_log_file_name

    # Check the file extension
    if predictions_data_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected predictions_data_file_name to be a .csv file, got {predictions_data_path.suffix}.")

    if predictions_log_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected predictions_log_path to be a .csv file, got {predictions_log_path.suffix}.")

    # Load or initialize the metadata DataFrame (one row per prediction run)
    if predictions_data_path.is_file():
        predictions_data = pd.read_csv(predictions_data_path)
    else:
        predictions_data = pd.DataFrame(columns=[
            "prediction_done",
            "prediction_for_date",
            "horizon",
            "prediction",
            "model_name"
        ])

    # Load or initialize the per-prediction log (one row per forecasted day)
    if predictions_log_path.is_file():
        predictions_log = pd.read_csv(predictions_log_path)
    else:
        predictions_log = pd.DataFrame(columns=[
            "prediction_date",
            "prediction_saved",
            "forecast_horizon"
        ])

    # If that prediction already exists in the logs, do nothing
    if prediction_date in predictions_log["prediction_date"]:
        return

    # Define and append the row to the logs (one row per batch)
    log_row = {
        "prediction_date": prediction_date,
        "prediction_saved": True,
        "forecast_horizon": len(predictions)
    }
    
    predictions_log.loc[len(predictions_log)] = log_row

    # Get the trading days excluding holidays and weekends
    trading_days = get_next_trading_days(
        prediction_date,
        n_days=2*len(predictions)   # The worst case scenario is that first 7 days are all holiday
    )

    for index, pred in enumerate(predictions):
        data_row = {
            "prediction_done": prediction_date,
            "prediction_for_date": trading_days[index],
            "horizon": index + 1,
            "prediction": pred.item() if hasattr(pred, "item") else pred,
            "model_name": model_name,
        }
        
        predictions_data.loc[len(predictions_data)] = data_row
    

    # Save both the predictions data and the logs
    predictions_data.to_csv(predictions_data_path, index=False)
    predictions_log.to_csv(predictions_log_path, index=False)
    predictions_avg_per_timestamp = (
        predictions_data[["prediction_for_date", "prediction"]]
        .groupby("prediction_for_date")
        .agg({"prediction": "mean"})
        .reset_index()
    )

    predictions_avg_per_timestamp.to_csv(
        predictions_data_path.with_name(f"{predictions_data_path.stem}_averaged.csv"),
        index=False
    )

    print(f'Saved: {predictions_data_path.with_name(f"{predictions_data_path.stem}_averaged.csv")}')



def generate_and_save_future_predictions():
    """
    Generates and saves future stock price predictions using the most recent trained models.
    """

    # Load the config
    config = load_config()

    # Get the models folder and processed data folder from the config 
    models_folder = get_project_root_path(2) / config['train_config']['model']['save_path']
    processed_data_folder = get_project_root_path(2) / config['stock']['stock_data_processed_save_path']

    # Get the predictions folder and the predictions data and predictions log filenames
    predictions_folder = get_project_root_path(2) / config['deploy_config']['paths']['predictions_save_path']
    predictions_data_filename = config['deploy_config']['paths']['predictions_file']
    predictions_log_filename = config['deploy_config']['paths']['predictions_log_file']

    # Get the results save path
    prediction_results_folder_path = config['eval_config']['general']['results_save_path']
    prediction_results_folder_path = ".." / Path(prediction_results_folder_path)

    # Get the most recent model's folder from the models folder
    most_recent_model_path = get_most_recent_model(models_folder)

    # Get the name of the latest model
    latest_model_name = most_recent_model_path.stem

    # Get the latest model date in %Y-%m-%d format
    latest_model_data_date = "-".join(latest_model_name.split("_")[:3])

    # For each ticker
    for ticker_model_folder in most_recent_model_path.glob("*"):

        # Get the ticker name, the processed data path, and the predictions folder for the ticker
        ticker = ticker_model_folder.stem
        model_ticker_path = most_recent_model_path / ticker
        processed_data_ticker_path = processed_data_folder / latest_model_data_date / ticker
        predictions_ticker_path = predictions_folder / ticker

        verify_saving_path(predictions_ticker_path)

        # Get future predictions
        prediction_date, predictions = get_future_predictions(
            model_folder=model_ticker_path,
            data_path=processed_data_ticker_path
        )

        # Update prediction metadata
        update_predictions_metadata(
            predictions=predictions[0],
            prediction_date=prediction_date,
            model_name=latest_model_name,
            predictions_folder_path=str(predictions_ticker_path),
            predictions_data_file_name=predictions_data_filename,
            predictions_log_file_name=predictions_log_filename
        )

        # Define the path to the predictions data
        predictions_data_path = predictions_ticker_path / predictions_data_filename

        predictions_averaged_data_path = predictions_data_path.with_name(f"{predictions_data_path.stem}_averaged.csv")

        # Read the predictions data and update the real-time plot
        averaged_predictions_data = pd.read_csv(predictions_averaged_data_path)

        # Read the train and validation datasets
        X_train, X_valid, y_train, y_valid, y_train_dates, y_valid_dates = read_processed_data(processed_data_ticker_path)

        # Convert all date strings to datetime
        y_train_dates = pd.to_datetime(y_train_dates)
        y_valid_dates = pd.to_datetime(y_valid_dates)
        averaged_predictions_data['prediction_for_date'] = pd.to_datetime(averaged_predictions_data['prediction_for_date'])


        # Save the real-time stock plot
        plot_inference_results(y_train=y_train[:, 0],
                               y_train_dates=y_train_dates,
                               y_valid=y_valid[:, 0],
                               y_valid_dates=y_valid_dates,
                               y_pred=averaged_predictions_data["prediction"],
                               y_pred_dates=averaged_predictions_data["prediction_for_date"],
                               save_path=predictions_ticker_path
                              )
        
        print(f"PLOT SAVED: {str(predictions_ticker_path)}")