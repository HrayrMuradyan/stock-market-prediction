from pathlib import Path

def verify_saving_path(path):
    """
    Ensures that the given path is a valid directory for saving files.

    If the path exists and is a file, raises a ValueError.
    If the path does not exist, attempts to create it as a directory (including parent directories).
    If directory creation fails, raises a RuntimeError.

    Parameters:
    ----------
    path : pathlib.Path or str
        The target path to validate or create as a directory.

    Raises:
    ------
    ValueError
        If the path exists and is a file.
    RuntimeError
        If the directory cannot be created due to an underlying error.
    """
    # Check if the input is a string or Path object
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Expected str or pathlib.Path, got {type(path).__name__}.")

    # Convert the input path to Path
    path = Path(path)

    # If given path exists (is a file or a directory)
    if path.exists():
        # If the path is a file
        if path.is_file():

            # Raise an error
            raise ValueError(f"Path '{path}' is a file, not a directory.")

    # If given path doesn't exist
    else:
        # Try to create the directory
        try:
            path.mkdir(parents=True, exist_ok=True)

        # If there was an error, raise it
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {path}: {e}")


def verify_file_path(path):
    """
    Validates that the given path exists and is a file.

    Parameters:
    ----------
    path : str or pathlib.Path
        The path to verify.

    Raises:
    ------
    TypeError
        If the input is not a string or Path object.
    ValueError
        If the path does not exist or is not a file.
    """
    # Check if the path is a string or Path
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Expected str or pathlib.Path, got {type(path).__name__}.")

    # Convert the input path to Path
    path = Path(path)

    # If given path doesn't exist
    if not path.exists():
        # Raise an error
        raise ValueError(f"Path '{path}' doesn't exist.")
        
    # If given path is not a file
    if not path.is_file():
        # Raise an error
        raise ValueError(f"Path '{path}' is not a file.")


def verify_existing_path(path):
    """
    Validates that the given path exists and is a directory.

    Parameters:
    ----------
    path : str or pathlib.Path
        The path to validate.

    Raises:
    ------
    TypeError
        If the input is not a string or Path object.
    ValueError
        If the path does not exist or is not a directory.
    """

    # Check if the input is a string or Path object
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Expected str or pathlib.Path, got {type(path).__name__}.")

    # Convert the input path to Path
    path = Path(path)

    # If given path doesn't exist, raise an error:
    if not path.exists():
        raise ValueError(f"Path '{path}' doesn't exist.")
        
    # If the path is not a directory, raise an error
    if not path.is_dir():
        raise ValueError(f"Path '{path}' is not a directory.")
    

def get_project_root_path(levels_up):
    """
    Return the path to the project root by traversing up a given number of directory levels
    from the location of the current file.

    Parameters:
        levels_up (int): The number of parent directories to go up from the current file.

    Returns:
        Path: The resolved absolute path to the computed root directory.

    """
    # Get the absolute path of the file
    path = Path(__file__).resolve()

    # If the number of levels is larger than the number of parents raise an error
    if levels_up >= len(path.parents):
        raise ValueError(f"levels_up={levels_up} is too high for path {path}")
    
    # Get the project root
    return path.parents[levels_up]
