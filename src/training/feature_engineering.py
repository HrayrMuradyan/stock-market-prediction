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
    ], axis=1).reset_index(drop=True)


def add_lagged_features(data, target_column, n_lags=1):
    """
    Create lagged features for time-series prediction.
    
    Parameters
    ----------
    data : pd.DataFrame
        Original DataFrame with all features including the target.
    target_column : str
        Name of the target column to be predicted.
    n_lags : int
        Number of time steps to look back (e.g., n_lags=1 means t-1).
    
    Returns
    -------
    data_lagged : pd.DataFrame
        DataFrame of lagged features.

    """
    # Create all lagged versions at once using list comprehension
    lagged_data = [data.shift(lag).add_suffix(f"_{lag}") for lag in range(1, n_lags + 1)]
    data_lagged = pd.concat(lagged_data, axis=1)
    
    # Add current target column
    data_lagged[target_column] = data[target_column]
    
    # Drop rows with NaNs and reset index
    data_lagged = data_lagged.dropna().reset_index(drop=True)

    return data_lagged
