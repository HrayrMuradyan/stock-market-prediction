from datetime import datetime
import yfinance as yf
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.path import verify_saving_path


def download_stock_data(ticker, stock_data_start_date, stock_data_end_date, column="Adj Close"):
    """
    Download historical stock data for a given ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str, list
        The stock ticker symbol or list of symbols (e.g., 'AAPL' or ['AAPL', 'A']).
    stock_data_start_date : str
        The start date for the data in 'YYYY-MM-DD' format.
    stock_data_end_date : str
        The end date for the data in 'YYYY-MM-DD' format.
    column : str, optional
        The column of interest to return from the downloaded data 
        (e.g., 'Adj Close', 'Close', 'Volume'). Default is 'Adj Close'.

    Returns
    -------
    pandas.Series
        Time series of the specified stock data column indexed by date.

    Raises
    ------
    TypeError
        If any of the `ticker`, `stock_data_start_date`, or `stock_data_end_date` arguments are not strings.
    KeyError
        If the specified column is not found in the downloaded DataFrame.
    """
    # Check the input
    if not isinstance(ticker, (str, list)):
        raise TypeError(f"Expected ticker argument to be a string or a list. Got {type(ticker).__name__}.")

    if not isinstance(stock_data_start_date, str):
        raise TypeError(f"Expected stock_data_start_date to be a string. Got {type(stock_data_start_date).__name__}.")

    if not isinstance(stock_data_end_date, str):
        raise TypeError(f"Expected stock_data_end_date to be a string. Got {type(stock_data_end_date).__name__}.")

    if not isinstance(column, str):
        raise TypeError(f"Expected column to be a string. Got {type(column).__name__}.")

    # Get the data from yfinance without adjustment
    sp500_data = yf.download(ticker, start=stock_data_start_date, end=stock_data_end_date, auto_adjust=False)

    # Get the Adjusted Close value of the stock
    return sp500_data[column]



def download_and_save_stock_data(tickers_list, stock_data_start_date, column="Adj Close", data_save_path="../data"):
    """
    Download stock data for multiple tickers, filter out tickers with missing data,
    and save each valid ticker's data as a separate CSV file.

    Parameters
    ----------
    tickers_list : list of str
        A list of stock ticker symbols to download data for.
    stock_data_start_date : str
        The start date for fetching stock data, in 'YYYY-MM-DD' format.
    column : str, optional
        The column of interest to extract from the stock data (default is "Adj Close").
    data_save_path : str or Path, optional
        Directory path where the ticker-wise CSV files should be saved (default is "../data").

    Raises
    ------
    TypeError
        If any of the inputs are not of the expected type.
    ValueError
        If none of the tickers have complete data (i.e., all contain missing values).

    Notes
    -----
    - Only tickers with complete data (no missing values) will be saved.
    - Each ticker's data is saved in a separate subfolder named after the ticker symbol.
    - The file name format is '{TICKER}_{YYYY-MM-DD}.csv', where the date is the current date.

    Examples
    --------
    >>> download_and_save_stock_data(["AAPL", "GOOG"], "2020-01-01")
    This will create:
        ../data/AAPL/AAPL_2025-07-19.csv
        ../data/GOOG/GOOG_2025-07-19.csv
    assuming today's date is July 19, 2025.
    """
    # Check the inputs
    if not isinstance(tickers_list, list):
        raise TypeError(f"Expected tickers_list argument to be a list. Got {type(tickers_list).__name__}.")

    if not isinstance(stock_data_start_date, str):
        raise TypeError(f"Expected stock_data_start_date to be a string. Got {type(stock_data_start_date).__name__}.")

    if not isinstance(column, str):
        raise TypeError(f"Expected column to be a string. Got {type(column).__name__}.")

    # Convert the save path to a Path object
    data_save_path = Path(data_save_path)

    # Verify saving path
    verify_saving_path(data_save_path)

    # Get today's date
    todays_date = datetime.today().strftime("%Y-%m-%d")

    # Download the stock data for all tickers
    ticker_data = download_stock_data(tickers_list,
                                      stock_data_start_date=stock_data_start_date,
                                      stock_data_end_date=todays_date,
                                      column=column)

    # Remove multi-level column labeling
    ticker_data.columns.name = None

    # Get only those columns which have no missing data
    ticker_data = ticker_data.loc[:, ticker_data.isna().sum() == 0]

    # Save each ticker data separately
    for ticker in ticker_data.columns:
        
        # Define and create (if doesn't exist) the folder for that specific ticker
        ticker_data_save_path = data_save_path / ticker
        ticker_data_save_path.mkdir(parents=True, exist_ok=True)

        # Save the stock data
        ticker_data[[ticker]].to_csv(str(ticker_data_save_path / f"{ticker}_{todays_date}.csv"), index=False)