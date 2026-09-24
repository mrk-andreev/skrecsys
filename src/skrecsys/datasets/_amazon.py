"""Amazon Books ratings dataset, preprocessed the way sequential recommenders use it.

https://jmcauley.ucsd.edu/data/amazon/
"""

import os
from pathlib import Path
from typing import Any, Literal, TypeAlias, overload

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets._base import RemoteFileMetadata, _fetch_remote
from sklearn.utils import Bunch

from skrecsys.datasets._base import (
    get_data_home,
    import_pandas,
    leave_one_out,
    load_descr,
    tail_rows,
)

__all__ = ["fetch_amazon_books"]

RATINGS = RemoteFileMetadata(
    filename="ratings_Books.csv",
    url="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv",
    checksum="87d923ddac2aacf40c1d0e0e056dbf4bf69c9c7d41852c6b434e781e759c0707",
)
_CACHE_NAME = "amazon_books.joblib"
_SUBSETS = ("all", "leave-one-out")
#: Interactions a user and an item each need to survive the k-core filter.
_MIN_INTERACTIONS = 5
#: Bytes read per parsing step. The file is ~900 MB, so it is streamed rather than
#: held in memory as one block of Python objects.
_CHUNK_BYTES = 8 << 20
#: The CSV columns, in file order, and the dtype each is stored as; the identifier
#: columns stay raw bytes until they are encoded.
_FIELDS: dict[str, Any] = {
    "user": None,
    "item": None,
    "rating": np.float64,
    "timestamp": np.int64,
}

Subset: TypeAlias = Literal["all", "leave-one-out"]


@overload
def fetch_amazon_books(
    *,
    data_home: str | os.PathLike[str] | None = ...,
    subset: Subset = ...,
    max_sequence_length: int | None = ...,
    download_if_missing: bool = ...,
    return_X_y: Literal[False] = ...,
    as_frame: bool = ...,
    n_retries: int = ...,
    delay: float = ...,
) -> Bunch: ...


@overload
def fetch_amazon_books(
    *,
    data_home: str | os.PathLike[str] | None = ...,
    subset: Literal["all"] = ...,
    max_sequence_length: int | None = ...,
    download_if_missing: bool = ...,
    return_X_y: Literal[True],
    as_frame: bool = ...,
    n_retries: int = ...,
    delay: float = ...,
) -> tuple[Any, Any]: ...


def fetch_amazon_books(
    *,
    data_home: str | os.PathLike[str] | None = None,
    subset: Subset = "all",
    max_sequence_length: int | None = 50,
    download_if_missing: bool = True,
    return_X_y: bool = False,
    as_frame: bool = False,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch | tuple[Any, Any]:
    """Load the Amazon Books ratings, downloading and preprocessing them if necessary.

    The 2014 Amazon product review dump, reduced to its 5-core and cut to the last
    ``max_sequence_length`` interactions of every user. The defaults reproduce the
    ``amzn-books-l50`` dataset that sequential recommenders such as HSTU and SASRec
    report on, so numbers measured here are comparable with theirs.

    =============================   ==============
    Interactions (l50, held out)    8,069,177
    Users                           694,897
    Items rated in those sequences  674,079
    Item identifier space           695,762
    Rating scale                    1-5 (integers)
    Collection period               May 1996 to July 2014
    =============================   ==============

    The download is ~900 MB and parsing it takes about half a minute and ~6 GB of
    memory; the result is cached, so this is paid once per ``data_home``.

    Read more in ``DESCR``.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Download and cache folder. By default data is stored in ``~/skrecsys_data``
        (or ``SKRECSYS_DATA``); see :func:`skrecsys.datasets.get_data_home`.

    subset : {"all", "leave-one-out"}, default="all"
        Split to expose. Both return every interaction; ``"leave-one-out"`` adds
        ``train_indices`` and ``test_indices`` into ``data``, holding out the last
        interaction of each user. That is the protocol the sequential-recommendation
        literature evaluates this dataset with.

    max_sequence_length : int or None, default=50
        Longest history a model may see, which is the ``50`` in ``amzn-books-l50``.
        Each user keeps their last this many interactions, plus the held-out one when
        ``subset="leave-one-out"`` - the window the reference implementation trains and
        evaluates on, where the target sits outside the history it is predicted from.
        None keeps the whole 5-core, 10,053,086 interactions. The truncation is applied
        to the cache, so changing it re-downloads and re-parses nothing.

    download_if_missing : bool, default=True
        If False, raise an ``OSError`` when the data is not cached locally instead of
        downloading it.

    return_X_y : bool, default=False
        If True, return ``(data, target)`` instead of a Bunch. Not allowed with
        ``subset="leave-one-out"``, since the split indices would be lost.

    as_frame : bool, default=False
        If True, ``data``, ``user_info`` and ``item_info`` are pandas DataFrames,
        ``target`` and ``timestamps`` are Series, and ``frame`` combines all
        interaction columns. Requires pandas.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        data : ndarray of int64 of shape (n_interactions, 2)
            ``[user_id, item_id]`` pairs, sorted by user, timestamp and item. The
            identifiers are contiguous codes, not the original Amazon strings.
        target : ndarray of float64 of shape (n_interactions,)
            Ratings.
        timestamps : ndarray of int64 of shape (n_interactions,)
            Rating times in Unix seconds.
        user_info : Bunch or DataFrame
            ``user_id`` and the ``reviewer_id`` it encodes.
        item_info : Bunch or DataFrame
            ``item_id`` and the product ``asin`` it encodes. Covers the whole
            identifier space, including items no surviving sequence rates; see
            ``DESCR``.
        feature_names : list of str
            ``["user_id", "item_id"]``.
        target_names : list of str
            ``["rating"]``.
        frame : DataFrame
            Only when ``as_frame=True``: ``user_id``, ``item_id``, ``rating``,
            ``timestamp``.
        train_indices, test_indices : ndarray of intp
            Only for ``subset="leave-one-out"``: row positions in ``data``. Use
            ``cv=[(train_indices, test_indices)]`` in scikit-learn model selection tools.
        DESCR : str
            Description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    Examples
    --------
    >>> from skrecsys.datasets import fetch_amazon_books
    >>> X, y = fetch_amazon_books(return_X_y=True)  # doctest: +SKIP
    >>> X.shape  # doctest: +SKIP
    (8044865, 2)
    """
    if subset not in _SUBSETS:
        raise ValueError(f"subset must be one of {_SUBSETS}, got {subset!r}.")
    if return_X_y and subset != "all":
        raise ValueError(
            "return_X_y=True is only supported with subset='all'; use the returned Bunch "
            "to access train_indices and test_indices."
        )
    if max_sequence_length is not None and max_sequence_length < 1:
        raise ValueError(f"max_sequence_length must be >= 1 or None, got {max_sequence_length!r}.")
    pd = import_pandas() if as_frame else None

    cached = _load_cache(
        data_home, download_if_missing=download_if_missing, n_retries=n_retries, delay=delay
    )
    interactions = cached["interactions"]
    held_out = subset == "leave-one-out"
    rows = tail_rows(interactions["user_id"], max_sequence_length, held_out=held_out)

    # The cache holds the narrowest dtype that fits; the returned pairs are int64,
    # like every other loader.
    data = np.column_stack([interactions["user_id"][rows], interactions["item_id"][rows]]).astype(
        np.int64
    )
    target = interactions["rating"][rows].astype(np.float64)
    timestamps = interactions["timestamp"][rows].astype(np.int64)
    users = Bunch(
        user_id=np.arange(len(cached["reviewer_ids"])), reviewer_id=cached["reviewer_ids"]
    )
    items = Bunch(item_id=np.arange(len(cached["asins"])), asin=cached["asins"])
    feature_names = ["user_id", "item_id"]

    bunch = Bunch(
        feature_names=feature_names,
        target_names=["rating"],
        DESCR=load_descr("amazon_books.rst"),
    )
    if pd is not None:
        data = pd.DataFrame(data, columns=feature_names)
        target = pd.Series(target, name="rating")
        timestamps = pd.Series(timestamps, name="timestamp")
        users = pd.DataFrame(dict(users))
        items = pd.DataFrame(dict(items))
        bunch.frame = pd.concat([data, target, timestamps], axis=1)
    bunch.update(data=data, target=target, timestamps=timestamps, user_info=users, item_info=items)

    if held_out:
        bunch.train_indices, bunch.test_indices = leave_one_out(interactions["user_id"][rows])

    if return_X_y:
        return data, target
    return bunch


def _load_cache(
    data_home: str | os.PathLike[str] | None,
    *,
    download_if_missing: bool,
    n_retries: int,
    delay: float,
) -> dict[str, Any]:
    """Return the parsed 5-core dataset, downloading and parsing it on a cache miss."""
    home = get_data_home(data_home)
    cache_path = home / _CACHE_NAME
    if not cache_path.exists():
        if not download_if_missing:
            raise OSError(f"Data not found in {home} and download_if_missing is False.")
        ratings_path = _fetch_remote(RATINGS, dirname=home, n_retries=n_retries, delay=delay)
        parsed = _parse_ratings(Path(ratings_path))
        joblib.dump(parsed, cache_path, compress=3)
        Path(ratings_path).unlink()
    return joblib.load(cache_path)


def _parse_ratings(path: Path) -> dict[str, Any]:
    """Parse the ratings CSV into the 5-core dataset, as contiguous integer codes."""
    columns = _read_columns(path)
    reviewer_ids, user_codes = _encode(columns.pop("user"))
    asins, item_codes = _encode(columns.pop("item"))

    dense = _k_core(user_codes, item_codes)
    user_codes, item_codes = user_codes[dense], item_codes[dense]
    # Re-encoding the survivors from their codes rather than from the strings is the
    # same relabelling - a code is the rank of its string - on a fraction of the bytes.
    kept_users, user_ids = np.unique(user_codes, return_inverse=True)
    kept_items, item_ids = np.unique(item_codes, return_inverse=True)
    reviewer_ids, asins = reviewer_ids[kept_users], asins[kept_items]

    # A user whose own count only cleared the floor thanks to rows the item filter
    # then dropped leaves the dataset, the way the reference preprocessing drops them:
    # after the identifiers are handed out, so the items they rated keep theirs.
    long_enough = np.bincount(user_ids)[user_ids] >= _MIN_INTERACTIONS
    user_ids, item_ids = user_ids[long_enough], item_ids[long_enough]
    ratings = columns.pop("rating")[dense][long_enough]
    timestamps = columns.pop("timestamp")[dense][long_enough]

    order = np.lexsort((item_ids, timestamps, user_ids))
    return {
        "interactions": {
            "user_id": user_ids[order].astype(np.int32),
            "item_id": item_ids[order].astype(np.int32),
            "rating": ratings[order],
            "timestamp": timestamps[order],
        },
        "reviewer_ids": reviewer_ids.astype("U"),
        "asins": asins.astype("U"),
    }


def _encode(values: NDArray[np.bytes_]) -> tuple[NDArray[np.bytes_], NDArray[np.int64]]:
    """Return the distinct values, sorted, and each row's index into them."""
    distinct, codes = np.unique(values, return_inverse=True)
    return distinct, codes


def _k_core(user_codes: NDArray[np.int64], item_codes: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Rows whose user and whose item each occur at least ``_MIN_INTERACTIONS`` times.

    One pass, over the counts of the raw file: dropping a row does not then drop the
    users and items it was propping up. This is the filter the reference preprocessing
    of ``amzn-books`` applies, and iterating it to a true k-core would leave a
    different, smaller dataset.
    """
    keep = np.ones(len(user_codes), dtype=bool)
    for codes in (user_codes, item_codes):
        keep &= np.bincount(codes)[codes] >= _MIN_INTERACTIONS
    return keep


def _read_columns(path: Path) -> dict[str, NDArray[Any]]:
    """Read the ``user,item,rating,timestamp`` CSV into one array per column."""
    chunks: dict[str, list[NDArray[Any]]] = {name: [] for name in _FIELDS}
    with path.open("rb") as stream:
        pending = b""
        while chunk := stream.read(_CHUNK_BYTES):
            pending += chunk
            cut = pending.rfind(b"\n")
            if cut < 0:
                continue
            pending, block = pending[cut + 1 :], pending[: cut + 1]
            _append_block(chunks, block)
        _append_block(chunks, pending)
    if not chunks["user"]:
        raise ValueError(f"{path} holds no ratings.")
    # Concatenating one column at a time, freeing its pieces as it goes, keeps the peak
    # at one extra copy of one column rather than of the whole file.
    return {name: _consume(pieces) for name, pieces in chunks.items()}


def _consume(pieces: list[NDArray[Any]]) -> NDArray[Any]:
    joined = np.concatenate(pieces)
    pieces.clear()
    return joined


def _append_block(chunks: dict[str, list[NDArray[Any]]], block: bytes) -> None:
    """Split one block of whole lines into its four fields."""
    rest = np.array(block.splitlines())
    if not len(rest):
        return
    fields = {}
    for name in list(_FIELDS)[:-1]:
        fields[name], _, rest = np.strings.partition(rest, b",")
    if np.any(np.strings.str_len(rest) == 0):
        raise ValueError("Malformed ratings row: expected user,item,rating,timestamp.")
    fields["timestamp"] = rest
    for name, dtype in _FIELDS.items():
        chunks[name].append(fields[name] if dtype is None else fields[name].astype(dtype))
