"""Location of the local dataset cache."""

import os
import shutil
from importlib.resources import files
from pathlib import Path

__all__ = ["clear_data_home", "get_data_home"]


def get_data_home(data_home: str | os.PathLike[str] | None = None) -> Path:
    """Return the path of the skrecsys data directory, creating it if needed.

    Downloaded datasets are cached here so they are fetched only once.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Explicit location. If None, the ``SKRECSYS_DATA`` environment variable is used,
        falling back to ``~/skrecsys_data``.

    Returns
    -------
    data_home : Path
    """
    if data_home is None:
        data_home = os.environ.get("SKRECSYS_DATA", Path("~") / "skrecsys_data")
    path = Path(data_home).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_data_home(data_home: str | os.PathLike[str] | None = None) -> None:
    """Delete all the content of the data home cache.

    Parameters
    ----------
    data_home : str or path-like, default=None
        See :func:`get_data_home`.
    """
    shutil.rmtree(get_data_home(data_home))


def load_descr(name: str) -> str:
    """Read a dataset description shipped in ``skrecsys.datasets.descr``."""
    return files("skrecsys.datasets.descr").joinpath(name).read_text(encoding="utf-8")
