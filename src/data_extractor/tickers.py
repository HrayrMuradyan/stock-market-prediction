from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import json
import pandas as pd
from datetime import datetime
import logging
from src.utils.path import verify_existing_path, verify_saving_path

def check_if_tickers_list_has_changed(config, new_tickers_list):
    """
    Compares a new list of tickers with the most recent previously saved list (if any)
    to determine whether the list has changed.

    Parameters:
    ----------
    config : dict
        Configuration dictionary with keys 'metadata' → 'tickers_list_save_path' and 'tickers_list_filename'.
    new_tickers_list : list
        The current list of tickers to check.

    Returns:
    -------
    changed : bool
        True if the list is different from the previous one or no previous list exists.
    difference : set or None
        The symmetric difference between the new and previous lists, or None if comparison wasn't possible.
    """
    # Extract required metadata from the config
    tickers_list_save_path_str = config['metadata']['tickers_list_save_path']
    tickers_list_filename_pattern = config['metadata']['tickers_list_filename']
    
    # Get the tickers list save path from the config and convert it to a Path object
    json_tickers_path = Path(tickers_list_save_path_str)

    # Check if the json tickers path exists or not
    verify_existing_path(json_tickers_path)

    # Find and sort matching JSON files
    all_jsons = sorted(json_tickers_path.glob(f"{tickers_list_filename_pattern}_*.json"))

    # If there are json files found
    if all_jsons:
        
        # Get the most recent previously saved file
        prev_file = all_jsons[-1]

        # Try to open the most recent previously saved file
        try:
            with open(prev_file, "r") as f:
                prev_tickers = json.load(f)

        # If there was an error to load the json, raise a warning
        except json.JSONDecodeError:
            logging.warning(f"⚠️ Failed to decode JSON from {prev_file}. Skipping comparison.")

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



def get_sp_500_tickers_from_wikipedia(url, config):
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
    config : dict
        A configuration dictionary with required keys:
            config['metadata']['tickers_list_save_path'] : str
                The directory where tickers JSON files are stored.
            config['metadata']['tickers_list_filename'] : str
                The filename prefix for saved tickers lists.

    Returns
    -------
    list of str
        The current list of S&P 500 ticker symbols.

    Raises
    ------
    RuntimeError
        If the Wikipedia page cannot be read or parsed properly.
    """
    
    # Get the necessary metadata from the config file
    save_path = config['metadata']['tickers_list_save_path']
    tickers_list_filename_pattern = config['metadata']['tickers_list_filename']

    # Convert the save path to Path object
    save_path = Path(save_path)

    # Check if it's a valid saving directory
    verify_saving_path(save_path)

    # Try to read the S&P500 Wikipedia page
    try:
        sp500 = pd.read_html(url)[0]
        tickers = sp500['Symbol'].tolist()

    # If there was a problem reading the list, raise an error
    except Exception as e:
        raise RuntimeError(f"There was an error reading S&P500 tickers from Wikipedia. The following error occured: {e}")

    # Check if tickers changed compared to the previous save
    tickers_changed, difference = check_if_tickers_list_has_changed(config, tickers)

    # If tickers changed
    if tickers_changed:
        # Get today's date
        today = datetime.today().strftime('%Y_%m_%d')

        # Save new tickers in a json file
        with open(str(save_path / f"{tickers_list_filename_pattern}_{today}.json"), "w") as f:
            json.dump(tickers, f)

        # If there is a difference, save them as well
        if difference:
            
            # Save the difference in a json file
            with open(str(save_path / f"{tickers_list_filename_pattern}_{today}_difference.json"), "w") as f:
                json.dump(list(difference), f)

    # Return the tickers list
    return tickers