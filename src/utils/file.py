from pathlib import Path
import shutil

def remove_file(path):
    """
    Removes a file specified by the given path.

    Parameters
    ----------
    path : str or pathlib.Path
        Path of the file to be removed.

    Raises
    ------
    TypeError
        If `path` is not a string or pathlib.Path object.

    Notes
    -----
    If the file does not exist or there is a permission issue,
    an error message will be printed instead of raising an exception.
    """
    # Validate input type
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Expected path to be a string or a Path object. Got {type(path).__name__}.")

    # Ensure path is a Path object
    path = Path(path)

    # Attempt to remove the file
    try:
        path.unlink()
        print(f"Successfully removed file: {str(path)}")

    except Exception as e:
        # Print an informative error message if file removal fails
        print(f"Error removing file {str(path)}: {e}")


def copy_file(path1, path2):
    """
    Copies a file from source path (path1) to destination path (path2).

    Parameters
    ----------
    path1 : str or pathlib.Path
        Source file path to copy from.
    path2 : str or pathlib.Path
        Destination file path to copy to.

    Raises
    ------
    TypeError
        If `path1` or `path2` is not a string or pathlib.Path object.

    Notes
    -----
    If the source file does not exist, or there are permission or disk-space issues,
    an error message will be printed instead of raising an exception.
    """
    # Validate input types
    if not isinstance(path1, (str, Path)):
        raise TypeError(f"Expected path1 to be a string or a Path object. Got {type(path1).__name__}.")

    if not isinstance(path2, (str, Path)):
        raise TypeError(f"Expected path2 to be a string or a Path object. Got {type(path2).__name__}.")

    # Convert paths to Path objects
    path1 = Path(path1)
    path2 = Path(path2)

    # Attempt to copy the file
    try:
        shutil.copyfile(path1, path2)

    except Exception as e:
        # Print informative error message if file copy fails
        print(f"Error copying file {str(path1)} to {str(path2)}: {e}")