import os
import zipfile
from pathlib import Path


def unpack(file_path: Path, dest_dir: Path, remove_after: bool = False) -> None:
    """
    Unpacks a zip file to the specified directory. If remove_after is True, the zip file will be removed after
    unpacking.

    Args:
        file_path: Path to the zip file to unpack.
        dest_dir: Directory where the contents of the zip file will be extracted.
        remove_after: If True, the zip file will be removed after unpacking.
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    if remove_after:
        os.remove(file_path)