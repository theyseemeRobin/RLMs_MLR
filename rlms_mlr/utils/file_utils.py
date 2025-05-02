from pathlib import Path
import os
import shutil
from platformdirs import user_cache_dir


def get_cache_dir() -> Path:
    get_dl_dir = user_cache_dir(__package__.split('.')[0])
    return Path(get_dl_dir)


def copy_file(source_file: Path, destination_dir: Path) -> None:
    """
    Copies a file from the current location to the desired destination. Creates the destination directory if it does
    not yet exist.

    Parameters
    ----------
    source_file : str
        Current location of the file.
    destination_dir : str
        Destination location of the file.
    """

    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, os.path.basename(source_file))
    shutil.copy(source_file, destination_path)
