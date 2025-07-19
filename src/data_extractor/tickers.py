from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import json
import pandas as pd
from datetime import datetime
import logging
from src.utils.path import verify_existing_path, verify_saving_path


def get_most_recent_tickers_file(tickers_folder, tickers_list_filename_pattern):
    """
    Return the most recent file (based on sorted filename) from the given tickers folder.

    Parameters
    ----------
    tickers_folder : str or pathlib.Path
        Path to the folder containing ticker files.

    Returns
    -------
    pathlib.Path or None
        Path to the most recent file based on sorted filename order,
        or None if the folder is empty.

    """
    
    # Check if the input is a string or Path object
    if not isinstance(tickers_folder, (str, Path)):
        raise TypeError(f"Expected str or pathlib.Path, got {type(path).__name__}.")
        
    # Convert the string path to Path object
    tickers_folder_path = Path(tickers_folder)

    # Verify that the path exists
    verify_existing_path(tickers_folder_path)

    # Get sorted tickers' files
    tickers_sorted = sorted(tickers_folder_path.glob(f"{tickers_list_filename_pattern}_*.json"))

    # If something found, get the most recent one
    return tickers_sorted[-1] if tickers_sorted else None


def check_if_tickers_list_has_changed(new_tickers_list, tickers_list_save_path_str, tickers_list_filename_pattern):
    """
    Compares a new list of tickers with the most recent previously saved list (if any)
    to determine whether the list has changed.

    Parameters:
    ----------
    new_tickers_list : list
        The current list of tickers to check.

    Returns:
    -------
    changed : bool
        True if the list is different from the previous one or no previous list exists.
    difference : set or None
        The symmetric difference between the new and previous lists, or None if comparison wasn't possible.
    """
    
    # Convert the tickers list save path string to a Path object
    json_tickers_path = Path(tickers_list_save_path_str)

    # Get the most recent tickers list file
    most_recent_tickers_file = get_most_recent_tickers_file(json_tickers_path, tickers_list_filename_pattern)

    # If there are json files found
    if most_recent_tickers_file:

        # Try to open the most recent previously saved file
        try:
            with open(most_recent_tickers_file, "r") as f:
                prev_tickers = json.load(f)

        # If there was an error to load the json, raise a warning
        except json.JSONDecodeError:
            logging.warning(f"⚠️ Failed to decode JSON from {most_recent_tickers_file}. Skipping comparison.")

            # Return (Tickers changed, None)
            return True, None

        # Convert the new and previous tickers list to sets
        new_set = set(new_tickers_list)
        prev_set = set(prev_tickers)

        # If both sets are the same, there is no change
        if new_set == prev_set:
            logging.info("✅ No change in ticker list compared to previous run.")
            
            # Return (Tickers not changed, None)
            return False, None
            
        # If they are not the same
        else:
            # Get the difference
            diff = new_set.symmetric_difference(prev_set)
            logging.warning(f"⚠️ Ticker list changed: {len(diff)} differences found.")

            # Return (Tickers changed, the difference)
            return True, diff

    # If there were no previous ticker list files
    else:
        logging.info(f"No previous ticker list file found in directory: {json_tickers_path}")

        # Return (Tickers not changed, None)
        return True, None


def get_sp_500_tickers_from_wikipedia(url, tickers_list_save_path_str, tickers_list_filename_pattern):
    """
    Fetches the latest S&P 500 tickers from Wikipedia and updates saved records if changes are detected.

    The function performs the following:
    - Retrieves the S&P 500 ticker symbols from the specified Wikipedia page.
    - Compares the fetched list with the previously saved version (if any).
    - Saves the new tickers list to a timestamped JSON file if it has changed.
    - Optionally saves the difference between the new and previous lists.

    Parameters
    ----------
    url : str
        The URL of the Wikipedia page containing the S&P 500 tickers.

    Returns
    -------
    list of str
        The current list of S&P 500 ticker symbols.

    Raises
    ------
    RuntimeError
        If the Wikipedia page cannot be read or parsed properly.
    """

    # Convert the save path to Path object
    tickers_list_save_path = Path(tickers_list_save_path_str)

    # Check if it's a valid saving directory
    verify_saving_path(tickers_list_save_path)

    # Try to read the S&P500 Wikipedia page
    try:
        sp500 = pd.read_html(url)[0]
        tickers = sp500['Symbol'].tolist()

    # If there was a problem reading the list, raise an error
    except Exception as e:
        raise RuntimeError(f"There was an error reading S&P500 tickers from Wikipedia. The following error occured: {e}")

    # Check if tickers changed compared to the previous save
    tickers_changed, difference = check_if_tickers_list_has_changed(tickers, tickers_list_save_path_str, tickers_list_filename_pattern)

    # If tickers changed
    if tickers_changed:
        # Get today's date
        today = datetime.today().strftime('%Y_%m_%d')

        # Save new tickers in a json file
        with open(str(tickers_list_save_path / f"{tickers_list_filename_pattern}_{today}.json"), "w") as f:
            json.dump(tickers, f)

        # If there is a difference, save them as well
        if difference:
            
            # Save the difference in a json file
            with open(str(tickers_list_save_path / f"{tickers_list_filename_pattern}_{today}_difference.json"), "w") as f:
                json.dump(list(difference), f)

    # Return the tickers list
    return tickers


