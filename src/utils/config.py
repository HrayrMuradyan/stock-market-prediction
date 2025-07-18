from pathlib import Path
import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.path import verify_file_path

def load_config(config_path):
    """
    Loads a YAML configuration file.

    Parameters:
    ----------
    config_path : str or pathlib.Path
        Path to the configuration file.

    Returns:
    -------
    dict
        Parsed YAML configuration as a dictionary.

    Raises:
    ------
    TypeError
        If the input is not a string or Path.
    ValueError
        If the path does not exist, is not a file, or does not have a .yml or .yaml extension.
    RuntimeError
        If there is an error opening or reading the file.
    """
    # Convert the path to a Path class
    path = Path(config_path)

    # Verify if the argument is a file and it exists
    verify_file_path(path)

    # If the suffix of the file is not .yml or .yaml raise an error
    if path.suffix not in [".yml", ".yaml"]:
        raise ValueError(f"Path '{path}' should refer to a file with a .yml or .yaml extension.")

    # Try to open the config file
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # Raise an error if reading the file fails
    except Exception as e:
        raise RuntimeError(f"Failed to open or parse the config file '{path}': {e}")