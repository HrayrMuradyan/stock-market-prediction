import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.config import load_config
from pathlib import Path
from src.utils.path import verify_existing_dir, verify_saving_path, get_project_root_path
import numpy as np
import json
import torch
from src.evaluation.utils import collect_predictions_per_timestamp  
from src.data.stocks import read_processed_data

from src.evaluation.metrics import (
    calc_mse,
    calc_mae,
    calc_rmse,
    calc_mape,
    calc_directional_accuracy
)

from src.evaluation.plots import (
    plot_interactive_predictions_with_slider,
    plot_error,
    plot_directional_accuracy_heatmap
)


def get_train_results():

    # Load the config
    config = load_config()

    # Get the model train results path
    # If there is a specific model provided in the eval_config, take that. If not, take the most recent train folder
    config_model_train_results_path = config['eval_config']['general']['model_train_results_path']
    train_results_save_path = get_project_root_path(2) / Path(config['train_config']['model']['save_path'])

    if not config_model_train_results_path:
        model_train_results_path = [p for p in train_results_save_path.iterdir() if p.is_dir()][-1]
    else:
        model_train_results_path = get_project_root_path(2) / Path(config_model_train_results_path)
        verify_existing_dir(model_train_results_path)

    # Get the save path for the results from the config
    results_save_path = config['eval_config']['general']['results_save_path']
    results_save_path = get_project_root_path(2) / Path(results_save_path) / model_train_results_path.stem

    # Get the valid stock folders
    stock_folders = [stock_folder for stock_folder in Path(model_train_results_path).iterdir() if stock_folder.is_dir()]

    # For each folder
    for stock_folder in stock_folders:

        # Get the ticker name
        ticker = stock_folder.stem 

        # Verify the existence of the save path
        results_save_path_ticker = results_save_path / ticker
        verify_saving_path(results_save_path_ticker)

        # Get the most recent model file
        model_folder = next(stock_folder.glob("model_*.pt"))
        
        # Read the hyperparameters used for the training
        with open(stock_folder / "hyperparams.json", "r") as f:
            hyperparams = json.load(f)
        
        # Load the predictions dictionary for both train and validation sets
        predictions_dict = torch.load(stock_folder / "predictions_list.pt")

        # Get the last predictions from the predictions dict
        y_train_pred_window = predictions_dict["train_preds_list"][-1]
        y_valid_pred_window = predictions_dict["valid_preds_list"][-1]

        # Get the average for each time stamp
        y_train_pred_window_per_timestamp = collect_predictions_per_timestamp(y_train_pred_window)    # np.array (n, prediction_window)
        y_valid_pred_window_per_timestamp = collect_predictions_per_timestamp(y_valid_pred_window)    # np.array (n, prediction_window)

        y_train_pred = np.nanmean(y_train_pred_window_per_timestamp, axis=-1)    # np.array (n, )
        y_valid_pred = np.nanmean(y_valid_pred_window_per_timestamp, axis=-1)    # np.array (n, )
        
        # Read the training data
        _, _, y_train, y_valid, y_train_dates, y_valid_dates = read_processed_data(get_project_root_path(2) / hyperparams['data_from'])    # torch.tensors

        # Get the single target by taking the first value for each timestamp's target window
        # ex. [y1, y2, y3, ..., y7], in this window, y1 is the target for that specific day
        y_train_target, y_valid_target = y_train[:, 0].numpy(), y_valid[:, 0].numpy()    # np.array (n, )

        # Calculate metrics
        train_se, train_mse = calc_mse(y_train_target, y_train_pred)
        train_ae, train_mae = calc_mae(y_train_target, y_train_pred)
        train_ape, train_mape = calc_mape(y_train_target, y_train_pred)
        _, train_rmse = calc_rmse(y_train_target, y_train_pred)
        train_correct_direction, train_directional_acc = calc_directional_accuracy(y_train_target, y_train_pred)

        valid_se, valid_mse = calc_mse(y_valid_target, y_valid_pred)
        valid_ae, valid_mae = calc_mae(y_valid_target, y_valid_pred)
        valid_ape, valid_mape = calc_mape(y_valid_target, y_valid_pred)
        _, valid_rmse = calc_rmse(y_valid_target, y_valid_pred)
        valid_correct_direction, valid_directional_acc = calc_directional_accuracy(y_valid_target, y_valid_pred)

        # Save metrics
        metrics_dict = {
            "mse": {"train": float(train_mse), "valid": float(valid_mse)},
            "mae": {"train": float(train_mae), "valid": float(valid_mae)},
            "mape": {"train": float(train_mape), "valid": float(valid_mape)},
            "rmse": {"train": float(train_rmse), "valid": float(valid_rmse)},
            "directional_acc": {"train": float(train_directional_acc), "valid": float(valid_directional_acc)},
        }

        metrics_file = results_save_path_ticker / "metrics.json"

        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        # Produce plots
        plots_save_path = results_save_path_ticker / "plots"

        plot_interactive_predictions_with_slider(
            y_train=y_train_target,
            y_train_dates=y_train_dates,
            y_valid=y_valid_target,
            y_valid_dates=y_valid_dates,
            predictions_dict=predictions_dict,
            save_path=plots_save_path,
            save_filename="predictions_with_slider.html"
        )

        # Plot errors for each metric
        plot_error(
            individual_errors=np.concatenate([train_se, valid_se]),
            dates=np.concatenate([y_train_dates, y_valid_dates]),
            train_val_split_date=str(y_train_dates[-1]),
            title="Squared Error of the Predictions Through time",
            xlabel="Date",
            metric="Squared Error",
            figsize=(1200, 500),
            save_path=plots_save_path
        )

        plot_error(
            individual_errors=np.concatenate([train_ae, valid_ae]),
            dates=np.concatenate([y_train_dates, y_valid_dates]),
            train_val_split_date=str(y_train_dates[-1]),
            title="Absolute Error of the Predictions Through time",
            xlabel="Date",
            metric="Absolute Error",
            figsize=(1200, 500),
            save_path=plots_save_path
        )

        plot_error(
            individual_errors=np.concatenate([train_ape, valid_ape]),
            dates=np.concatenate([y_train_dates, y_valid_dates]),
            train_val_split_date=str(y_train_dates[-1]),
            title="Percentage Error of the Predictions Through time",
            xlabel="Date",
            metric="Percentage Error",
            figsize=(1200, 500),
            save_path=plots_save_path
        )

        plot_directional_accuracy_heatmap(
            directional_accuracy=np.concatenate([train_correct_direction, valid_correct_direction]),
            dates=np.concatenate([y_train_dates, y_valid_dates]),
            title="Directional Accuracy of Predictions Over Time",
            save_path=plots_save_path,
            save_filename=None

        )       