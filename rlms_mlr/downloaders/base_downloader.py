from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Downloader(Protocol):
    """
    Protocol for a downloader that can download files from a source to a destination.
    """
    def __call__(self, source: str, dest: Path) -> None:
        """
        Download a file from the given source to the destination.
        Args:
            source: The source URL or path to download from.
            dest: The destination path to save the downloaded file.
        """
        pass