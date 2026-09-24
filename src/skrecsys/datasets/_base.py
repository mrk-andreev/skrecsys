"""Location of the local dataset cache, and the pieces every loader shares."""

import importlib
import os
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["clear_data_home", "get_data_home"]


def import_pandas() -> Any:
    """pandas, which ``as_frame=True`` needs and the rest of the package does not."""
    try:
        return importlib.import_module("pandas")
    except ImportError as exc:
        raise ImportError(
            "as_frame=True requires pandas. Install it with `pip install skrecsys[pandas]`."
        ) from exc


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


def group_bounds(user_ids: NDArray[Any]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Start and end row of each user's block, for rows already grouped by user.

    Every sequential loader sorts its rows by user and then by time, so a user's
    interactions are one contiguous block and the order inside it is chronological.
    """
    if len(user_ids) == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    boundaries = np.flatnonzero(user_ids[1:] != user_ids[:-1]) + 1
    starts = np.concatenate([[0], boundaries]).astype(np.intp)
    ends = np.concatenate([boundaries, [len(user_ids)]]).astype(np.intp)
    return starts, ends


def tail_rows(
    user_ids: NDArray[Any], max_sequence_length: int | None, *, held_out: bool
) -> NDArray[np.intp]:
    """Rows of the last interactions of each user, oldest first.

    ``max_sequence_length`` of them, and one more when the last one is held out, so
    that a model still sees a full history before the interaction it is scored on.
    """
    if max_sequence_length is None:
        return np.arange(len(user_ids), dtype=np.intp)
    starts, ends = group_bounds(user_ids)
    lengths = ends - starts
    group = np.repeat(np.arange(len(starts)), lengths)
    position = np.arange(len(user_ids)) - starts[group]
    keep = position >= lengths[group] - (max_sequence_length + held_out)
    return np.flatnonzero(keep).astype(np.intp)


def leave_one_out(user_ids: NDArray[Any]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Hold out the last interaction of every user; train on everything before it.

    This is the protocol the sequential-recommendation literature evaluates with. It is
    time ordered within a user but not across users: a training interaction of one user
    may be later in wall-clock time than a test interaction of another.
    """
    _, ends = group_bounds(user_ids)
    test = (ends - 1).astype(np.intp)
    train = np.setdiff1d(np.arange(len(user_ids), dtype=np.intp), test)
    return train, test
