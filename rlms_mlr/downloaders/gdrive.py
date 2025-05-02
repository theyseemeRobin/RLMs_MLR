from pathlib import Path
import gdown
import os

from rlms_mlr.downloaders.common import unpack


def gdrive_download(source: str, dest: Path, force_zip: bool = True, remove_after: bool = True) -> None:
    """
    Downloads a file from Google Drive using gdown and unpacks it if it's a compressed file.

    Args:
        source: The Google Drive file ID or URL to download.
        dest: The destination path where the file will be saved.
        force_zip: If True, the file will be treated as a zip file and unpacked.
        remove_after: If True, the downloaded zip file will be removed after unpacking.
    """
    file_dir = Path(dest).parent
    os.makedirs(file_dir, exist_ok=True)
    if force_zip:
        dest = dest.with_suffix('.zip')
    try:
        gdown.download(id=source, output=str(dest))
    except Exception as e:
        raise FileNotFoundError(f"failed to download file for key: {source}") from e
    if dest.suffix in ['.zip', '.rar', '.tar', '.tar.gz']:
        unpack(dest, file_dir, remove_after=remove_after)