from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils.path import verify_existing_dir
from src.utils.checks import check_datatypes

def get_most_recent_model(models_folder):
    """
    Returns the most recently saved model file from the specified folder.

    Parameters:
    - models_folder (str or Path): Path to the folder containing saved model files.

    Returns:
    - Path: The path to the most recently saved model file, based on lexicographic sorting.
    """

    check_datatypes([
        ("models_folder", models_folder, (str, Path))
    ])

    # Convert the models folder to a Path object
    models_folder = Path(models_folder)

    # Check if the folder exists
    verify_existing_dir(models_folder)

    # Return the latest model from the folder
    return sorted(models_folder.glob("*"))[-1]