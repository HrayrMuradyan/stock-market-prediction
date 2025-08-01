from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.path import verify_saving_path, verify_existing_dir
from src.utils.file import remove_file, copy_file


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
    downloaded_data = yf.download(ticker, start=stock_data_start_date, end=stock_data_end_date, auto_adjust=False)

    # Get the Adjusted Close value of the stock
    downloaded_data = downloaded_data[column]

    # Remove the two-level column naming
    downloaded_data.columns.name = None

    # Return the data
    return downloaded_data



def download_and_save_stock_data(tickers_list, stock_data_start_date, exclude_tickers, column="Adj Close", data_save_path="../data"):
    """
    Download stock data for multiple tickers, exclude unwanted tickers,
    filter out those with missing values, and save each valid ticker's data as a CSV.

    Parameters
    ----------
    tickers_list : list of str
        A list of stock ticker symbols to download data for.
    stock_data_start_date : str
        The start date for fetching stock data, in 'YYYY-MM-DD' format.
    exclude_tickers : set of str
        A set of ticker symbols to exclude from downloading and saving.
    column : str, optional
        The stock price column to extract (default is "Adj Close").
    data_save_path : str or Path, optional
        Directory where ticker-wise CSV files will be saved (default is "../data").

    Raises
    ------
    TypeError
        If any input is not of the expected type.
    ValueError
        If none of the tickers have complete data (i.e., all contain missing values).

    Notes
    -----
    - Only tickers with complete data (i.e., no missing values in the specified column) will be saved.
    - Each ticker's data is saved in a separate subfolder named after the ticker.
    - The file name format is '{TICKER}_{YYYY-MM-DD}.csv', where the date is the current date.
    - The exclude_tickers argument must be a set for performance reasons.

    Examples
    --------
    >>> download_and_save_stock_data(["AAPL", "GOOGL", "TSLA"], "2020-01-01", exclude_tickers={"TSLA"})
    This will create:
        ../data/AAPL/AAPL_2025-07-22.csv
        ../data/GOOGL/GOOGL_2025-07-22.csv
    assuming today's date is July 22, 2025, and TSLA is excluded.
    """
    # Check the inputs
    if not isinstance(tickers_list, list):
        raise TypeError(f"Expected tickers_list argument to be a list. Got {type(tickers_list).__name__}.")

    if not isinstance(exclude_tickers, set):
        raise TypeError(f"Expected exclude_tickers argument to be a set. Got {type(exclude_tickers).__name__}.")

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

    # Exclude tickers from tickers list given in the exclude_tickers set argument
    tickers_list = [ticker for ticker in tickers_list if ticker not in exclude_tickers]

    # Download the stock data for all tickers
    ticker_data = download_stock_data(tickers_list,
                                      stock_data_start_date=stock_data_start_date,
                                      stock_data_end_date=todays_date,
                                      column=column)

    # Get only those columns which have no missing data
    ticker_data = ticker_data.loc[:, ticker_data.isna().sum() == 0]

    # Save each ticker data separately
    for ticker in ticker_data.columns:
        
        # Define and create (if doesn't exist) the folder for that specific ticker
        ticker_data_save_path = data_save_path / ticker
        ticker_data_save_path.mkdir(parents=True, exist_ok=True)

        # Save the stock data
        ticker_data[[ticker]].to_csv(str(ticker_data_save_path / f"{ticker}_{todays_date}.csv"))
        import logging
        logger = logging.getLogger(__name__)
        print("Data saved in {ticker_data_save_path}")
        logger.info(f"Data saved in {ticker_data_save_path}")



def update_stock_data(data_folder, stock_data_start_date, keep_n_archived=3, column="Adj Close"):

    # Check the input
    verify_existing_dir(data_folder)

    if not isinstance(stock_data_start_date, str):
        raise TypeError(f"Expected stock_data_start_date to be a string. Got {type(stock_data_start_date).__name__}.")

    if not isinstance(keep_n_archived, int):
        raise TypeError(f"Expected keep_n_archived to be an integer. Got {type(keep_n_archived).__name__}.")

    if not isinstance(column, str):
        raise TypeError(f"Expected column to be a string. Got {type(column).__name__}.")
    

    # Convert the data folder to Path object
    data_folder_path = Path(data_folder)

    # Get todays date
    todays_date = datetime.today().strftime("%Y-%m-%d")
    print(f"Todays date: {todays_date}")

    # List to store tickers for update
    tickers_to_update = []

    # For each ticker folder
    for ticker_index, ticker_folder in enumerate(data_folder_path.glob("*")):

        # --------------------- Get ticker and data information ----------------------- 

        # Get the ticker name from the folder
        ticker = ticker_folder.stem

        print(f"Processing ticker {ticker_index}: {ticker}")

        # Get sorted non-archived csv files
        all_ticker_csv_files = sorted(ticker_folder.glob(f"{ticker}_*.csv"))


        # --------------------- If the most recent file exists and it's date is todays date, skip. Else, add it to the update list -----------------------

        if (all_ticker_csv_files and all_ticker_csv_files[-1].stem.split("_")[-1] == todays_date):
            continue

        else:
            tickers_to_update.append(ticker)
            
    # If no stock data found to be update, quit the function
    if not tickers_to_update:
        return


    # --------------------- Update the tickers in the list -----------------------
    updated_stock_data = download_stock_data(ticker=tickers_to_update,
                               stock_data_start_date=stock_data_start_date,
                               stock_data_end_date=todays_date,
                               column=column) 

    # Remove any stocks that have missing data
    updated_stock_data = updated_stock_data.loc[:, updated_stock_data.isna().sum() == 0]

    # Save each ticker data separately
    for ticker in updated_stock_data.columns:

        # Define the folder for that specific ticker
        ticker_data_save_path = data_folder_path / ticker

        # Get sorted ticker csv files
        all_ticker_csv_files = sorted(ticker_data_save_path.glob(f"{ticker}_*.csv"))

        # Save the updated stock data
        updated_stock_data[[ticker]].to_csv(str(ticker_data_save_path / f"{ticker}_{todays_date}.csv"))

        # Delete the old stock data
        remove_file(all_ticker_csv_files[-1])


    # --------------------- Create or update archives -----------------------

    # Iterate over the updated data
    for ticker_index, ticker_folder in enumerate(data_folder_path.glob("*")):

        # Get all archived csv files and ticker data
        all_archive_csv_files = sorted(ticker_folder.glob(f"archive_*.csv"))
        all_ticker_csv_files = sorted(ticker_folder.glob(f"{ticker}_*.csv"))

        # Get the most recent csv file
        most_recent_ticker_file = all_ticker_csv_files[-1]

        # Get the ticker name from the folder
        ticker = ticker_folder.stem
            
        # If no archived files
        if not all_archive_csv_files:
            
            print("No archived files")
            
            # Define a new filename to archive the current most recent ticker file
            save_a_copy = most_recent_ticker_file.with_name("archive_" + most_recent_ticker_file.stem + most_recent_ticker_file.suffix)
            
            # Copy the most recent ticker file
            copy_file(most_recent_ticker_file, save_a_copy)
    
            # Append the new archive file to the all_archive_csv_files
            all_archive_csv_files.append(save_a_copy)

        else:
        
            # --------------------- Check if the most recent archived data was a week ago, create a new archive file ----------------------- 
            most_recent_archived_file = all_archive_csv_files[-1]
        
            # The date of the most recent archived file
            most_recent_archived_file_date = most_recent_archived_file.stem.split("_")[-1]
            most_recent_archived_date = datetime.strptime(most_recent_archived_file_date, "%Y-%m-%d").date()

            # Check if the most recent archived file date was a week ago, create a new archive file
            if datetime.today().date() >= most_recent_archived_date + timedelta(weeks=1):

                print("There is an archived file and it's date is a week before. Time to create a new one")
        
                # Define a new filename to archive the current most recent ticker file
                new_archive = most_recent_ticker_file.with_name("archive_" + most_recent_ticker_file.stem + most_recent_ticker_file.suffix)
                
                # Copy the most recent ticker file
                copy_file(most_recent_ticker_file, new_archive)

                # Add the new archive to the archive list
                all_archive_csv_files.append(new_archive)

            
            # --------------------- If there are more than "keep_n_archived" archived data files, remove oldest ones ----------------------- 
            
            # If there are more than "keep_n_archived" archived files, remove all
            n_archived_files = len(all_archive_csv_files)
            while n_archived_files > keep_n_archived:
        
                # Get the oldest archive file
                oldest_archive_file = all_archive_csv_files[0]
        
                # Remove the oldest archive file
                remove_file(oldest_archive_file)
        
                # Check the number of archived files again
                all_archive_csv_files = sorted(ticker_folder.glob(f"archive_*.csv"))
                n_archived_files = len(all_archive_csv_files)







def read_and_concat_all_stocks(data_folder_path):
    """
    Read and concatenate the most recent stock data for all tickers in a given folder.

    This function searches through each subfolder in the specified directory,
    identifies the most recent stock data file (by name sorting), reads it into a DataFrame,
    and concatenates all such DataFrames along the column axis.

    Parameters
    ----------
    data_folder_path : str or pathlib.Path
        Path to the main directory containing subdirectories for each ticker.
        Each subdirectory is expected to contain CSV files with stock data named
        using the format: '<ticker>_<date>.csv', and having a 'Date' column as the index.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the concatenated stock data for all tickers,
        aligned by the 'Date' index.

    """

    # Input check
    if not isinstance(data_folder_path, (str, Path)):
        raise TypeError(f"Expected data_folder_path argument to be a string or a Path object. Got {type(data_folder_path).__name__}.")
    
    # Make data folder path a Path object if it's not already
    data_folder_path = Path(data_folder_path)
   
    # Define a list to load and append dataframes
    data_list = []

    # For ticket folder in the data folder
    for ticker_path in data_folder_path.glob("*"):

        # Get the ticker name
        ticker = ticker_path.stem

        # Get the most recent stock data from the tickern name
        most_recent_stock_data = sorted(ticker_path.glob(f"{ticker}_*"))[-1]


        # Read the data and append to the list
        data_list.append(pd.read_csv(str(most_recent_stock_data), index_col="Date", parse_dates=True))

    # Concatenate all dataframes of the list and return 
    return pd.concat(data_list, axis=1)




def read_processed_stock_data(data_folder):
    """
    Find the most recent dated subfolder in the given data folder,
    then load and return the PyTorch tensors stored in the files:
    'X_train.pt', 'X_valid.pt', 'y_train.pt', and 'y_valid.pt'.

    Assumes that the subfolder names are dates in the format 'YYYY-MM-DD'
    and that each subfolder contains the above four files.

    Parameters
    ----------
    data_folder : str or Path
        Path to the main data folder containing date-named subfolders.

    Returns
    -------
    tuple of torch.Tensor
        Returns a tuple with four tensors in this order:
        (X_train, X_valid, y_train, y_valid)

    Raises
    ------
    FileNotFoundError
        If the data_folder does not exist, is not a directory,
        or if any of the expected .pt files are missing.
    ValueError
        If no valid date-named subfolders are found or folder names do not match the date format.
    """
    import torch

    # Convert input to Path object if needed
    data_folder = Path(data_folder)

    # Verify that the directory exists
    verify_existing_dir(data_folder)

    # List all subfolders inside data_folder
    subfolders = data_folder.glob("*")

    # Get the most recent folder by date-parsing folder names
    most_recent_folder = max(
        subfolders,
        key=lambda f: datetime.strptime(f.name, "%Y-%m-%d")
    )

    # Define expected files inside the most recent folder
    expected_files = ["X_train.pt", "X_valid.pt", "y_train.pt", "y_valid.pt"]

    # Build full paths and check if all files exist
    file_paths = {}
    for fname in expected_files:
        fpath = most_recent_folder / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Expected file not found: {fpath}")
        file_paths[fname] = fpath

    # Load the tensors using torch.load
    X_train = torch.load(file_paths["X_train.pt"])
    X_valid = torch.load(file_paths["X_valid.pt"])
    y_train = torch.load(file_paths["y_train.pt"])
    y_valid = torch.load(file_paths["y_valid.pt"])

    return X_train, X_valid, y_train, y_valid