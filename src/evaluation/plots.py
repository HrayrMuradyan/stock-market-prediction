import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.evaluation.utils import collect_predictions_per_timestamp
from src.utils.checks import check_datatypes
from src.utils.path import verify_saving_path
import plotly.graph_objects as go
from pathlib import Path
import torch
import pandas as pd

def plot_train_valid_with_preds(y_train,
                                y_train_pred,
                                y_valid,
                                y_valid_pred,
                                title="Train and Validation Sets Plot With Predictions",
                                x_label="Timestamp",
                                y_label="Value",
                                note=None,
                                figsize=(12, 5),
                                zoom_out=1,
                                save_path=None,
                                save_filename=None
                                ):

    """
    Plot training and validation target values along with their predicted values over time.

    This function plots the ground truth and predicted values for both training and validation sets.
    Predictions are averaged per timestamp if they come as windows of predictions. It also marks the
    boundary between training and validation data with a vertical line. The plot limits can be adjusted
    by zooming in or out around the validation set.

    Parameters
    ----------
    y_train : array-like of shape (n_train_samples,)
        Ground truth target values for the training set.

    y_train_pred : array-like
        Predicted values for the training set. Can be a list or array of prediction windows per timestamp.
        These will be averaged per timestamp using `collect_predictions_per_timestamp`.

    y_valid : array-like of shape (n_valid_samples,)
        Ground truth target values for the validation set.

    y_valid_pred : array-like
        Predicted values for the validation set. Can be a list or array of prediction windows per timestamp.
        These will be averaged per timestamp using `collect_predictions_per_timestamp`.

    title : str, optional, default="Train and Validation Sets Plot With Predictions"
        Title of the plot.

    x_label : str, optional, default="Timestamp"
        Label for the x-axis.

    y_label : str, optional, default="Value"
        Label for the y-axis.

    figsize : tuple of two ints, optional, default=(12, 5)
        Figure size for the plot in inches as (width, height).

    zoom_out : float, optional, default=1
        Zoom factor for the plot limits around the validation set. Higher values zoom out more.

    save_path : str or Path, optional, default=None
        File path to save the interactive HTML visualization.
        If None, the figure will not be saved.

    save_filename : str, optional, default=None
        Filename to use when saving the plot image (without extension).

    Returns
    -------
    None
        The function either shows the plot or saves it to disk depending on `save_path`.
    """
    if save_path:
        verify_saving_path(save_path)
        save_path = Path(save_path)
    
    plt.figure(figsize=figsize)

    # Get the size of the training set
    train_size = len(y_train)

    # Plot the target ground truth for training
    plt.plot(range(train_size), y_train, label="Train Target", color="tab:blue")

    # Get the train prediction's average from train predictions (prediction_window averaged per timestamp)
    train_predicted_per_timestamp = collect_predictions_per_timestamp(y_train_pred)
    plt.plot(range(train_size), np.nanmean(train_predicted_per_timestamp, axis=1), label="Train Predicted", color="tab:orange")

    # Get the size of the validation set
    valid_size = len(y_valid)

    # Validation set starts right after the training set
    valid_start = train_size + 1

    # Define the point where validation set ends
    valid_end = train_size + len(y_valid) + 1 

    # Plot the validation target ground truth
    plt.plot(range(valid_start, valid_end), y_valid, label="Validation Target", color="tab:blue")

    # Get the validation prediction's average from prediction window
    valid_predicted = collect_predictions_per_timestamp(y_valid_pred)

    # Plot the average and with gray color plot the individual predictions 
    plt.plot(range(valid_start, valid_end), np.nanmean(valid_predicted, axis=1), label="Validation Predicted", color="brown")
    plt.plot(range(valid_start, valid_end), valid_predicted, color="gray", alpha=0.3)

    # Add a straight vertical line showing the switch from train to validation
    plt.axvline(x=train_size, color='gray', linestyle='--', label='Change to validation set')

    # Add a note
    if note:
        plt.text(
            0.05, 0.95, note, 
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.5)
        )

    # Calculate the mean of the validation set
    validation_y = y_valid.mean()
    # Calculate the mean of the x axis
    validation_x = (train_size + valid_end) // 2

    # Calculate the ratio of the plot x/y
    figsize_x_over_y = figsize[0] / figsize[1]

    # Center the validation set on the plot. zoom_out controls how much zoom should be applied
    plt.xlim(validation_x-zoom_out*valid_size*figsize_x_over_y, validation_x+zoom_out*valid_size*figsize_x_over_y)
    plt.ylim(validation_y-zoom_out*valid_size, validation_y+zoom_out*valid_size)

    # Add the title, labels and legend
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    # If save path is defined, save the figure, return nothing. If not, plot the figure
    if save_path is not None:
        if not save_filename:
            save_filename = str(save_path / f"train_valid_with_predictions.png")
        plt.savefig(save_filename)  
        return
    
    plt.show()


def plot_interactive_predictions_with_slider(y_train,
                                             y_train_dates,
                                             y_valid,
                                             y_valid_dates,
                                             predictions_dict,
                                             save_path=None,
                                             save_filename=None
                                             ):
    """
    Create and optionally save an interactive Plotly animation that shows
    how training and validation predictions evolve across epochs.

    Parameters
    ----------
    y_train : torch.Tensor, np.ndarray
        1D tensor or numpy array of ground truth target values for the training set.

    y_train_dates : torch.Tensor, np.ndarray, list
        1D tensor or numpy array of dates corresponding to the y_train values.
    
    y_valid : torch.Tensor, np.ndarray
        1D tensor or numpy array of ground truth target values for the validation set.

    y_valid_dates : torch.Tensor, np.ndarray, list
        1D tensor or numpy array of dates corresponding to the y_valid values.
    
    predictions_dict : dict
        Dictionary with keys:
            - 'train_preds_list': list of tensors [T x S] for each epoch
            - 'valid_preds_list': list of tensors [T x S] for each epoch
          where T = number of time steps, S = number of samples/predictions.
    
    save_path : str or Path, optional, default=None
        File path to save the interactive HTML visualization.
        If None, the figure will not be saved.

    save_filename : str, optional, default=None
        Filename to use when saving the plot image.

    Returns
    -------
    None
        Displays the plot in the browser and optionally saves it to disk.
    """
    

    # Verify the input types
    schema_list = [
        ("y_train", y_train, (torch.Tensor, np.ndarray)),
        ("y_train_dates", y_train_dates, (torch.Tensor, np.ndarray, list)),
        ("y_valid", y_valid, (torch.Tensor, np.ndarray)),
        ("y_valid_dates", y_valid_dates, (torch.Tensor, np.ndarray, list)),
        ("predictions_list", predictions_dict, dict),
        ("save_filename", save_filename, (str, type(None)))
    ]

    # Verify the saving path
    if save_path:
        verify_saving_path(save_path)

    check_datatypes(schema_list)

    # Prepare basic properties
    num_epochs = len(predictions_dict["train_preds_list"])
    train_size = len(y_train)
    valid_size = len(y_valid)
    valid_start = train_size + 1
    valid_end = train_size + valid_size + 1

    # Initial epoch predictions
    initial_epoch = 0
    train_pred_0 = collect_predictions_per_timestamp(predictions_dict["train_preds_list"][initial_epoch])
    valid_pred_0 = collect_predictions_per_timestamp(predictions_dict["valid_preds_list"][initial_epoch])

    # Start the figure
    fig = go.Figure()

    # Ground truths
    fig.add_trace(go.Scatter(x=y_train_dates, y=y_train, mode="lines", name="Train Target", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=y_valid_dates, y=y_valid, mode="lines", name="Valid Target", line=dict(color="#1f77b4", dash="dot")))

    # Mean train prediction
    fig.add_trace(go.Scatter(
        x=y_train_dates,
        y=np.nanmean(train_pred_0, axis=1),
        mode="lines",
        name="Train Pred",
        line=dict(color="orange")
    ))

    # Individual valid predictions
    for i in range(valid_pred_0.shape[1]):
        fig.add_trace(go.Scatter(
            x=y_valid_dates,
            y=valid_pred_0[:, i],
            mode="lines",
            line=dict(color="gray", width=1),
            opacity=0.3,
            name="Valid Pred Sample" if i == 0 else None,
            showlegend=(i == 0)
        ))

    # Mean valid prediction
    fig.add_trace(go.Scatter(
        x=y_valid_dates,
        y=np.nanmean(valid_pred_0, axis=1),
        mode="lines",
        name="Valid Pred Mean",
        line=dict(color="brown", width=2)
    ))

    # Add vertical separator between train/valid
    separator_date = y_valid_dates[0]  # or y_train_dates[-1] + timedelta(days=1)

    fig.add_shape(
        type="line",
        x0=separator_date,
        y0=min(min(y_train), min(y_valid)),
        x1=separator_date,
        y1=max(max(y_train), max(y_valid)),
        line=dict(color="gray", dash="dash")
    )

    # Build animation frames
    frames = []
    for epoch in range(num_epochs):
        train_pred = collect_predictions_per_timestamp(predictions_dict["train_preds_list"][epoch])
        valid_pred = collect_predictions_per_timestamp(predictions_dict["valid_preds_list"][epoch])

        frame_data = [go.Scatter(x=y_train_dates, y=np.nanmean(train_pred, axis=1))]
        frame_data.extend([
            go.Scatter(x=y_valid_dates, y=valid_pred[:, i]) for i in range(valid_pred.shape[1])
        ])
        frame_data.append(go.Scatter(x=y_valid_dates, y=np.nanmean(valid_pred, axis=1)))

        trace_indices = [2] + list(range(3, 3 + valid_pred.shape[1])) + [3 + valid_pred.shape[1]]

        frames.append(go.Frame(data=frame_data, name=str(epoch), traces=trace_indices))

    fig.frames = frames

    # Slider layout
    fig.update_layout(
        title="Train and Validation Predictions Over Epochs",
        xaxis_title="Timestamp",
        yaxis_title="Value",
        sliders=[{
            "steps": [{
                "method": "animate",
                "args": [[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                "label": f"Epoch {k+1}"
            } for k in range(num_epochs)],
            "transition": {"duration": 0},
            "x": 0,
            "xanchor": "left",
            "y": -0.2,
            "yanchor": "top"
        }]
    )

    # Save plot to HTML
    if save_path is not None:
        if not save_filename:
            save_filename = f"training_process_plot.html"
        fig.write_html(str(save_path / save_filename), include_plotlyjs='cdn')
        print(f"[Saved] Plot saved to: {save_filename}")
        return

    # Show the plot in the browser
    fig.show()



def plot_error(individual_errors,
               dates,
               train_val_split_date=None,
               title="Prediction error",
               xlabel="Time",
               metric=None,
               figsize=(1200, 500),
               save_path=None,
               save_filename=None
               ):
    
    """
    Plot prediction error over time using Plotly with interactive zoom functionality.

    This function visualizes individual prediction errors as a line plot with interactive zoom.
    It includes a horizontal dashed line at y=0 to indicate perfect prediction.
    Optionally, it can also draw a vertical line at a user-specified split point (e.g., to mark
    the boundary between training and validation sets).

    Parameters
    ----------
    individual_errors : torch.Tensor, np.ndarray, list
        A 1D array representing prediction errors over time. Each value corresponds
        to a specific timestamp.
    
    dates : torch.Tensor, np.ndarray, list
        A 1D array representing dates corresponding to the individual errors.

    train_val_split_date : str, optional
        The index date at which to draw a vertical line to indicate a split (e.g., between training and
        validation data). If None (default), no vertical line is drawn.

    title : str, default="Prediction error"
        The title displayed at the top of the plot.

    xlabel : str, default="Time"
        The label for the x-axis (typically represents time or step index).

    metric : str or None, optional
        The name of the error metric (e.g., "MAE", "RMSE"). Used for the legend label and y-axis label.
        If None, the label will be left blank.

    figsize : tuple of int, default=(1200, 500)
        The size of the figure in pixels as (width, height).

    save_path : str or Path, optional, default=None
        File path to save the interactive HTML visualization.
        If None, the figure will not be saved.

    save_filename : str, optional, default=None
        Filename to use when saving the plot image (without extension).

    Returns
    -------
    None
        Displays an interactive Plotly line plot in the browser or notebook interface.

    """
    
    check_datatypes([
        ("individual_errors", individual_errors, (torch.Tensor, np.ndarray, list)),
        ("dates", dates, (torch.Tensor, np.ndarray, list)),
        ("train_val_split_date", train_val_split_date, str),
        ("title", title, str),
        ("xlabel", xlabel, str),
        ("metric", metric, (str, type(None))),
        ("figsize", figsize, tuple),
        ("save_path", save_path, (str, Path, type(None))),
        ("save_filename", save_filename, (str, type(None)))
    ])

    if save_path:
        save_path = Path(save_path)
        verify_saving_path(save_path)

    # Create figure
    fig = go.Figure()

    # Add prediction error trace
    fig.add_trace(go.Scatter(
        y=individual_errors,
        x=dates,
        mode='lines',
        name=metric,
        line=dict(color='#1f77b4')
    ))

    # If split point is provided
    if train_val_split_date:
        # Add vertical line indicating the split between train and validation
        fig.add_shape(
            type="line",
            x0=train_val_split_date,
            y0=min(individual_errors),
            x1=train_val_split_date,
            y1=max(individual_errors),
            line=dict(color="red", dash="dash")
        )

    # Set layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=metric,
        width=figsize[0],
        height=figsize[1],
        showlegend=True,
        hovermode='x unified',
    )

    # Zoom is enabled by default; optionally restrict to zoom-only mode
    fig.update_layout(dragmode="zoom")  

    # Save plot to HTML
    if save_path is not None:
        if not save_filename:
            save_filename = str(save_path / f"error_plot_{'_'.join(metric.lower().split(' '))}.html")
        fig.write_html(save_filename, include_plotlyjs='cdn')
        print(f"[Saved] Plot saved to: {save_filename}")
        return

    # Show the plot in the browser
    fig.show()  



def plot_inference_results(y_train,
                           y_train_dates,
                           y_valid,
                           y_valid_dates,
                           y_pred,
                           y_pred_dates,
                           save_path=None,
                           save_filename=None):
    """
    Plot training, validation, and real-time prediction results using Plotly.

    This function generates an interactive time series plot of:
    - Training data
    - Validation data
    - Real-time predictions

    The plot can be displayed directly or saved as an HTML file.

    Parameters
    ----------
    y_train : array-like
        Training target values.
    y_train_dates : array-like
        Corresponding datetime values for `y_train`.
    y_valid : array-like
        Validation target values.
    y_valid_dates : array-like
        Corresponding datetime values for `y_valid`.
    y_pred : array-like
        Future value predictions.
    y_pred_dates : array-like
        Corresponding datetime values for `y_pred`.
    save_path : str or pathlib.Path or None, optional
        Directory where the HTML file should be saved. If None, the plot is shown in a browser.
    save_filename : str or None, optional
        Name of the output HTML file. If None, defaults to `'real_time_inference.html'`.

    Returns
    -------
    None
        Displays or saves the plot. Does not return any object.
    """

    check_datatypes([
        ("y_train", y_train, (np.ndarray, torch.Tensor, list, pd.Series)),
        ("y_train_dates", y_train_dates, (np.ndarray, torch.Tensor, list, pd.Series, pd.DatetimeIndex)),
        ("y_valid", y_valid, (np.ndarray, torch.Tensor, list, pd.Series)),
        ("y_valid_dates", y_valid_dates, (np.ndarray, torch.Tensor, list, pd.Series, pd.DatetimeIndex)),
        ("y_pred", y_pred, (np.ndarray, torch.Tensor, list, pd.Series)),
        ("y_pred_dates", y_pred_dates, (np.ndarray, torch.Tensor, list, pd.Series, pd.DatetimeIndex)),
        ("save_path", save_path, (str, Path, type(None))),
        ("save_filename", save_filename, (str, type(None)))
    ])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_train_dates,
        y=y_train,
        mode='lines',
        name='Train',
        line=dict(color='#737e8c')
    ))

    fig.add_trace(go.Scatter(
        x=y_valid_dates,
        y=y_valid,
        mode='lines',
        name='Validation',
        line=dict(color='#1f77b4')
    ))

    fig.add_trace(go.Scatter(
        x=y_pred_dates,
        y=y_pred,
        mode='lines',
        name='Real-time Predictions',
        line=dict(color='#dc2330')
    ))

    fig.update_layout(
        title='Training, Validation, and Real-time Predictions',
        xaxis_title='Date',
        yaxis_title='Value',
        height=500,
        width=1200,
        template='plotly_white',
        legend=dict(x=0, y=1),
        hovermode='x unified',
    )

    # Save plot to HTML
    if save_path is not None:
        if not save_filename:
            save_filename = str(save_path / f"real_time_inference.html")
        fig.write_html(save_filename, include_plotlyjs='cdn')
        import json
        from plotly.utils import PlotlyJSONEncoder
        with open(str(save_path / f"plot.json"), "w") as f:
            json.dump(fig, f, cls=PlotlyJSONEncoder)
            
        print(f"[Saved] Plot saved to: {save_filename}")
        return

    # Show the plot in the browser
    fig.show()


def plot_directional_accuracy_heatmap(directional_accuracy,
                                      dates,
                                      title="Correct Stock Direction Predictions Over Time",
                                      save_path=None,
                                      save_filename=None):
    """
    Plot a binary heatmap showing correct (1) and incorrect (0) stock direction predictions over time.

    Parameters
    ----------
    directional_accuracy : array-like of bool or int
        Array indicating whether each prediction was correct (1/True) or incorrect (0/False).
        The shape should be (N,), where N is the number of predictions.

    dates : array-like of datetime or str
        Array of dates corresponding to each prediction. Must be the same length as `directional_accuracy`.

    title : str, optional
        Title of the heatmap. Default is "Correct Stock Direction Predictions Over Time".

    save_path : str or pathlib.Path or None, optional
        Path to directory where the plot should be saved. If None, the plot is shown instead of saved.

    save_filename : str or None, optional
        Custom filename (format `.html`) for the saved plot. If None and `save_path` is provided,
        the default filename "directional_accuracy_plot.html" is used.

    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        Plotly figure object. Returns None if the figure is saved to disk instead of shown.

    """

    # Check the input
    check_datatypes([
        ("directional_accuracy", directional_accuracy, (torch.Tensor, np.ndarray, list)),
        ("dates", dates, (torch.Tensor, np.ndarray, list)),
        ("title", title, str),
        ("save_path", save_path, (str, Path, type(None))),
        ("save_filename", save_filename, (str, type(None)))
    ])

    # Check if the saving path exists
    if save_path:
        save_path = Path(save_path)
        verify_saving_path(save_path)

    # Check if the filename is .html or not
    if save_filename:
        file_extension = Path(save_filename).suffix
        if file_extension != ".html":
            raise ValueError(f"Expected save_filename to have a .html extension, got {file_extension}.")
    
    # Ensure correct format
    directional_accuracy = np.array(directional_accuracy).astype(int).reshape(1, -1)
    dates = pd.to_datetime(dates) 

    # Colors for correct and incorrect
    red_color = "#c73938"
    green_color = "#60a659"

    # Base heatmap
    heatmap = go.Heatmap(
        z=directional_accuracy,
        x=dates,
        y=["Prediction"],
        colorscale=[[0, red_color], [1, green_color]],
        showscale=False,
        hoverinfo='x'
    )

    # Simulated legend using invisible scatter markers
    legend_correct = go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=green_color),
        name='Correct Prediction'
    )

    legend_incorrect = go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=red_color),
        name='Incorrect Prediction'
    )

    # Combine all in figure
    fig = go.Figure(data=[heatmap, legend_correct, legend_incorrect])

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="",
        yaxis=dict(showticklabels=False),
        margin=dict(t=40, b=20),
        legend=dict(
            orientation='h',
            x=0.7,
            xanchor='center',
            y=1.15
        )
    )

    # Save plot to HTML
    if save_path is not None:
        if not save_filename:
            save_filename = str(save_path / f"error_plot_directional_accuracy.html")
        fig.write_html(save_filename, include_plotlyjs='cdn')
        print(f"[Saved] Plot saved to: {save_filename}")
        return

    # Show the plot in the browser
    fig.show()  