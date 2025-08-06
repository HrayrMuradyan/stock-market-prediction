import pandas as pd

def add_date_as_feature(data):
    """
    Adds year, month, and weekday as separate features extracted from the DatetimeIndex of a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with additional columns: 'year', 'month', 'weekday'.

    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected data to be a Pandas DataFrame. Got {type(data).__name__}.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(f"Expected the DataFrame index to be a DatetimeIndex. Got {type(data.index).__name__}.")

    # Add year, month, and weekday as new columns
    return pd.concat([
        data,
        pd.DataFrame({
            'year': data.index.year,
            'month': data.index.month,
            'weekday': data.index.weekday
        }, index=data.index)
    ], axis=1)

def add_lagged_features(data, target_column, n_lags=1):
    """
    Create lagged features for time-series prediction and prepare the last
    lagged feature row for real-time inference.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing features and target column.
    target_column : str
        Name of the target column in `data`.
    n_lags : int, optional
        Number of lag periods to create (default is 1).

    Returns
    -------
    data_lagged : pandas.DataFrame
        DataFrame with lagged features and target column, cleaned of NaN rows.
    last_lagged_row : pandas.DataFrame
        Single-row DataFrame with lagged features corresponding to the last
        available data point, aligned with `data_lagged` columns (excluding target).

    Notes
    -----
    The lagged features are created by shifting the original features by 1 to `n_lags` rows.
    The `last_lagged_row` is constructed manually to be used for real-time prediction, 
    containing the most recent lagged feature values.
    """
    feature_cols = data.columns.drop(target_column)

    # Create lagged features
    lagged_parts = []
    for i in range(1, n_lags + 1):
        shifted = data[feature_cols].shift(i).add_suffix(f"_{i}")
        lagged_parts.append(shifted)
    
    data_lagged = pd.concat(lagged_parts, axis=1)
    
    # Create last lagged row manually, ensuring column order and names match
    last_rows = [
        data[feature_cols].iloc[[-i]].add_suffix(f"_{i}")
        for i in range(n_lags, 0, -1)
    ]
    last_lagged_row = pd.concat(last_rows, axis=1)

    # Reorder columns to match data_lagged
    lagged_feature_columns = data_lagged.columns
    last_lagged_row = last_lagged_row[lagged_feature_columns]

    assert data_lagged.columns.equals(last_lagged_row.columns), \
    "The columns of the lagged full data and the last row data should be the same"

    data_lagged[target_column] = data[target_column]
    data_lagged_clean = data_lagged.dropna()

    return data_lagged_clean, last_lagged_row
