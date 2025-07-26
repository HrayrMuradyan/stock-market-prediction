from pathlib import Path
import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.path import verify_file_path, get_project_root_path
import tomli

def get_config_path() -> str:
    """
    Load the path to the main project config file from pyproject.toml.

    This function assumes that the `pyproject.toml` file exists in the project root,
    and that it contains a section like:

        [tool.stock_prediction]
        config_path = "configs/main.yaml"

    Returns:
        str: The relative or absolute path to the main config file,
             as specified under [tool.stock_prediction] in pyproject.toml.

    """
    # Get the project root path (2 levels above this file)
    project_root_path = get_project_root_path(2)

    # Path to pyproject.toml
    pyproject_path = project_root_path / "pyproject.toml"

    # Check if pyproject.toml exists
    verify_file_path(pyproject_path)

    # Load the TOML data
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)

    # Check if tool.stock_prediction.config_path exists
    try:
        return data["tool"]["stock_prediction"]["config_path"]
    except KeyError as e:
        raise KeyError(
            "Missing 'tool.stock_prediction.config_path' in pyproject.toml"
        ) from e

def load_config():
    """
    Loads a YAML configuration file.

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

    # Get the config path
    config_path = get_config_path()

    # Convert the path to a Path class
    path = get_project_root_path(2) / config_path

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
    


